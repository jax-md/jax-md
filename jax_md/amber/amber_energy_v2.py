import sys
import jax
import jax.numpy as jnp
import numpy as onp
from jax.scipy.special import erf, erfc  # error function
from jax.scipy.linalg import cho_factor, cho_solve
#TODO check if this can be imported from jax_md
from jax_md.reaxff.reaxff_helper import safe_sqrt
from jax_md.reaxff.reaxff_energy import taper
from jax_md import dataclasses, space, smap, util, partition, simulate, minimize
from jax_md.amber.amber_helper import angle, torsion
from functools import wraps, partial
from typing import Callable, Tuple, TextIO, Dict, Any, Optional

# Types

f32 = util.f32
f64 = util.f64
Array = util.Array

PyTree = Any
Box = space.Box
DisplacementOrMetricFn = space.DisplacementOrMetricFn

NeighborFn = partition.NeighborFn
NeighborList = partition.NeighborList
NeighborListFormat = partition.NeighborListFormat

# also look into safe mask and safe sqrt for things
# TODO move these to constants file
# TODO make sure kB is correct in kj or kcal units
# TODO big task, but fully integrate with JAX MD units file
ONE_4PI_EPS = 138.935456
kB = 0.00831446267
CHARGE_SCALE_AMBER = 18.2223
# conversion factors from lrch_utils.mod.F90
BOHRS_TO_A = 0.529177249
AU_PER_AMBER_CHARGE = 1.0 / CHARGE_SCALE_AMBER
AU_PER_AMBER_ANGSTROM = 1.0 / BOHRS_TO_A
AU_PER_AMBER_KCAL_PER_MOL = AU_PER_AMBER_CHARGE * AU_PER_AMBER_CHARGE / AU_PER_AMBER_ANGSTROM
AU_PER_AMBER_FORCE = AU_PER_AMBER_KCAL_PER_MOL / AU_PER_AMBER_ANGSTROM

def amber_energy(ff, nonbonded_method="NoCutoff", charge_method="GAFF", ensemble="NVE", timestep=1e-3, init_temp=1e-3, return_charges=False, ffq_ff=None, backprop_solve=False):
    # TODO which functions should be generators and which should just return a value?
    # TODO remove this if it interferes with nb code and put large box vectors
    # TODO consider if wrapped=False + no MIC has significant performance benefits
    if nonbonded_method == "PME":
        disp_fn, shift_fn = space.periodic(ff.box_vectors)
    else:
        disp_fn, shift_fn = space.free()

    # TODO using canonicalize_, map_neighbor/bond, and creating partial(vmap)
    # functions may be wise with compatibility for fractional coordinates
    # there has to be a better way to balance mapping over neighbor objects
    # both sparse and dense, as well as ad hoc index lists also sparse or dense
    # in a way that is generalizable
    # TODO add fractional coordinate and non orthogonal box support
    disp_map = jax.vmap(disp_fn)
    metric_fn = space.metric(disp_fn)
    dist_fn = jax.vmap(metric_fn)
    map_metric = space.map_neighbor(metric_fn)
    map_disp = space.map_neighbor(disp_fn) # TODO this is confusing with the previous function
    # TODO rework these to take displacement and distance matrices/index lists
    angle_fn = jax.vmap(angle)
    torsion_fn = jax.vmap(torsion, (0,0,0,0))
    ffq_dist_fn = space.map_neighbor(metric_fn)

    # TODO named jax or nvtx annotations for each function? @named_call?
    bond_named = jax.named_call(harmonic_bond, name="Bond Function")
    angle_named = jax.named_call(harmonic_angle, name="Angle Function")
    torsion_named = jax.named_call(periodic_torsion, name="Torsion Function")
    if nonbonded_method == "PME":
        receivers = jnp.concatenate((ff.exclusions[:, 0], ff.exclusions[:, 1]))
        senders = jnp.concatenate((ff.exclusions[:, 1], ff.exclusions[:, 0]))
        idx = jnp.argsort(senders)
        receivers = receivers[idx]
        senders = senders[idx]
        # print(jnp.vstack((receivers, senders)))

        N = ff.atom_count
        count = jax.ops.segment_sum(jnp.ones(len(receivers), jnp.int32), receivers, N)
        max_count = jnp.max(count)
        offset = jnp.tile(jnp.arange(max_count), N)[:len(senders)]
        hashes = senders * max_count + offset
        dense_idx = N * jnp.ones((N * max_count,), jnp.int32)
        exclusions_dense = dense_idx.at[hashes].set(receivers).reshape((N, max_count))

        # TODO also add a named call to this mask function
        def mask_function(idx):
            # TODO test assumed unique here, not clear from the algorithm if the junk fill values matter
            idx = jax.vmap(lambda idx_r, mask_r: jnp.where(jnp.isin(idx_r, mask_r), ff.atom_count, idx_r), in_axes=(0,0))(idx, exclusions_dense)

            return idx

        if nonbonded_method == "PME" and charge_method == "FFQ":
            #TODO: add sparse support for this
            nbr_fmt = NeighborListFormat.Dense
            mask_fn = None
        else:
            nbr_fmt = NeighborListFormat.OrderedSparse
            mask_fn = mask_function

        neighbor_fn = partition.neighbor_list(
            space.canonicalize_displacement_or_metric(disp_fn),
            box=ff.box_vectors,
            r_cutoff=ff.cutoff,
            dr_threshold=ff.dr_threshold,
            disable_cell_list=False,
            custom_mask_function=mask_fn,
            fractional_coordinates=False,
            format=nbr_fmt, # TODO was ordered sparse, but need to figure out converting this from sparse to dense
        )

        # TODO this should be decoupled with the distance calculations
        ff = dataclasses.replace(ff, nbr_list=neighbor_fn.allocate(ff.positions))

        # TODO think more about the implications of this masking
        masked_direct_fn = lambda dr, **kwargs: jnp.where(
            dr < ff.cutoff,
            coulomb_direct(dr, **kwargs),
            0.)

        masked_lj_fn = lambda dr, **kwargs: jnp.where(
            dr < ff.cutoff,
            lennard_jones(dr, **kwargs),
            0.)

        pair_direct_fn = smap.pair_neighbor_list(
            masked_direct_fn,
            space.canonicalize_displacement_or_metric(disp_fn),
            charge_sq=(lambda q1, q2: q1 * q2, ff.charges),
            alpha=ff.ewald_alpha
        )

        pair_lj_fn = smap.pair_neighbor_list(
            masked_lj_fn,
            space.canonicalize_displacement_or_metric(disp_fn),
            sigma=(lambda s1, s2: 0.5*(s1 + s2), ff.sigma),
            epsilon=(lambda e1, e2: safe_sqrt(e1 * e2), ff.epsilon)
        )

        reciprocal_fn = coulomb_recip(ff.charges, ff.box_vectors, ff.grid_points, ff.ewald_alpha, fractional_coordinates=False, indiv_return=True)

        direct_named = jax.named_call(pair_direct_fn, name="Direct Function")
        reciprocal_named = jax.named_call(reciprocal_fn, name="Reciprocal Function")
        self_named = jax.named_call(coulomb_self, name="Self Function")
        correction_named = jax.named_call(coulomb_correction, name="Correction Function")
        lj_named = jax.named_call(pair_lj_fn, name="LJ Function")

    else:
        lj_named = jax.named_call(lennard_jones, name="LJ Function")
        coul_named = jax.named_call(coulomb, name="Coulomb Function")

    # TODO @partial(jax.jit, static_argnames=["debug"])
    def nrg_fn(positions, ff, nbr_list, debug=False):

        result_dict = dict()

        if charge_method == "FFQ":
            atom_mask = ff.species >= 0
            atm_mask = jnp.arange(len(ff.species))
            atm_mask = atm_mask < ff.solute_cut
            atom_mask = atom_mask * atm_mask # TODO consider if the solute cut portion should be there
            # i think it is for the last rows/columns of the array

            # TODO 3 cases to consider:
            # only solute, no cut
            # solvent present but not interacted with
            # solvent present AND interacting with it in the EEM matrix ***

            if nonbonded_method == "PME":
                nbr_inds = nbr_list.idx # expand dims to nxn
                print("nbr list", nbr_list.idx[0, :50])
                far_nbr_inds = nbr_inds
            else:
                nbr_inds = jnp.tile(jnp.arange(len(ff.masses)), (len(ff.masses), 1))
                far_nbr_inds = jnp.fill_diagonal(nbr_inds, ff.atom_count, inplace=False)

            row_inds = jnp.arange(len(ff.masses))
            nbr_mask = (row_inds >= ff.solute_cut).reshape(-1,1) & (far_nbr_inds >= ff.solute_cut)
            far_nbr_inds = jnp.where(nbr_mask, ff.atom_count, far_nbr_inds)

            R_far_nbr = positions[far_nbr_inds,:]
            # TODO add conversion constants to do this all in NM
            far_nbr_dists = ffq_dist_fn(positions, R_far_nbr) * 10.0
            far_nbr_mask = (far_nbr_inds != ff.atom_count) & (atom_mask.reshape(-1,1)
                                        & atom_mask[far_nbr_inds])
            far_nbr_dists = far_nbr_dists * far_nbr_mask

            tapered_dists = taper(far_nbr_dists, 0.0, 10.0)
            tapered_dists = jnp.where((far_nbr_dists > 10.0) | (far_nbr_dists < 0.001),
                                    0.0,
                                    tapered_dists)

            species = ff.species
            far_neigh_types = species[far_nbr_inds]

            far_nbr_mask = (far_nbr_inds != ff.atom_count) & (atom_mask.reshape(-1,1) # TODO look above, duplicate?
                                                    & atom_mask[far_nbr_inds])

            # TODO some of this can be precomputed
            gamma = jnp.power(ffq_ff.gamma.reshape(-1, 1), 3/2)
            gamma_mat = gamma * gamma.transpose()
            gamma_mat = gamma_mat[far_neigh_types, species.reshape(-1, 1)]
            hulp1_mat = far_nbr_dists ** 3 + (1/gamma_mat)
            hulp2_mat = jnp.power(hulp1_mat, 1.0/3.0) * far_nbr_mask

            charges = calculate_eem_charges(species,
                                      atom_mask,
                                      far_nbr_inds,
                                      hulp2_mat,
                                      tapered_dists,
                                      ffq_ff.hardness,
                                      ffq_ff.electronegativity,
                                      ff.atom_count,
                                      init_charges=None,
                                      total_charge=0.0,
                                      backprop_solve=backprop_solve,
                                      tol=1e-6,
                                      max_solver_iter=-1) # TODO add as param, same for lrch

            chg_mask = jnp.arange(len(charges))
            chg_mask = (chg_mask >= ff.solute_cut) & (chg_mask < (len(charges)-1))
            charges = jnp.where(chg_mask[:-1], ff.charges, charges[:-1])
            charges_14 = charges[ff.pairs_14[:, 0]] * charges[ff.pairs_14[:, 1]]
        else:
            charges = ff.charges
            charges_14 = ff.charges_14

        # TODO decide where to trim last value of array
        #charges = charges[:-1]

        ### Bond interaction
        bond_idx = ff.bond_idx
        b_pos = positions[bond_idx]
        bond_dist = dist_fn(b_pos[:, 0], b_pos[:, 1])
        bond_mask = bond_idx[:,0] != -1
        bond_pot = jnp.sum(bond_mask * bond_named(bond_dist, ff.bond_k, ff.bond_len))
        result_dict['bond_pot'] = bond_pot

        ### Angle interaction
        angle_idx = ff.angle_idx
        a_pos = positions[angle_idx]
        angle_theta = angle_fn(disp_map(a_pos[:, 0], a_pos[:, 1]), disp_map(a_pos[:, 2], a_pos[:, 1]))
        angle_mask = angle_idx[:,0] != -1
        # TODO decide if safe masking the angle pot is the best move, or safe masking inside the angle function
        angle_pot = jnp.sum(angle_mask * angle_named(angle_theta, ff.angle_k, ff.angle_equil))
        result_dict['angle_pot'] = angle_pot

        ### Torsion interaction
        torsion_idx = ff.torsion_idx
        t_pos = positions[torsion_idx]
        # TODO this also may not work over periodic boundaries, the displacements in the function are plain
        torsion_theta = torsion_fn(t_pos[:, 0], t_pos[:, 1], t_pos[:, 2], t_pos[:, 3])
        torsion_mask = torsion_idx[:,0] != -1
        torsion_pot = jnp.sum(torsion_mask * torsion_named(torsion_theta, ff.torsion_k, ff.torsion_phase, ff.torsion_period))
        result_dict['torsion_pot'] = torsion_pot

        ### Nonbonded interactions
        if nonbonded_method == "PME":
            lj_pot = 0.0
            direct_pot = 0.0
            self_pot = 0.0
            recip_pot = 0.0
            correction_pot = 0.0

            # print("nbr list {}", nbr_list.idx[:, :50])
            #print("nbr list shape", nbr_list.idx.shape)

            #nb_dist = map_metric(positions, positions[nbr_list.idx])

            i_nbrs = nbr_list.idx[0, :]
            j_nbrs = nbr_list.idx[1, :]

            #valid_mask = (i_nbrs < N) & (j_nbrs < N)
            #jax.debug.print("Valid sparse neighbor pairs: {}", jnp.sum(valid_mask))
            # sort_idx = jnp.argsort(i_nbrs)
            # i_nbrs = i_nbrs[sort_idx]
            # j_nbrs = j_nbrs[sort_idx]

            #lj_pot = nbr_list.idx[0,0] * 1e-10
            #lj_pot = jnp.sum(nbr_list.idx) * 1e-10 # TODO dummy example to force variable to be traced
            #lj_pot = jnp.sum(ff.charges[i_nbrs]) * 1e-10
            #lj_pot = jnp.sum(ff.charges[nbr_list.idx]) * 1e-10
            # print("chgs shape", ff.charges.shape)
            # print("pos shape", positions.shape)
            # sys.exit()

            # TODO look into making these functions use squared distances

            ### Alternative using take to see if semantics are better
            # pos_i = jnp.take(positions, i_nbrs, axis=0)
            # pos_j = jnp.take(positions, j_nbrs, axis=0)
            # nb_dist = dist_fn(pos_i, pos_j)

            # ### Attempt to directly manipulate lax.gather
            # def gather_rows(arr, idx):
            #     dnums = jax.lax.GatherDimensionNumbers(
            #         offset_dims=(1,),             # result dim 1 = slice dim 1
            #         collapsed_slice_dims=(0,),   # collapse dimension 0 in slice
            #         start_index_map=(0,)         # indices index into dimension 0
            #     )
            #     slice_sizes = (1, arr.shape[1])  # (1, 3) slice: one row of 3 values
            #     # print("slice_sizes", slice_sizes)
            #     idx = idx[:, None]               # make it shape (N, 1) as required
            #     gathered = jax.lax.gather(arr, idx, dimension_numbers=dnums, slice_sizes=slice_sizes)
            #     #return gathered[:, 0]            # remove length-1 gather 
            #     return gathered

            # # Now use this:
            # pos_i = gather_rows(positions, i_nbrs)  # shape: (N, 3)
            # pos_j = gather_rows(positions, j_nbrs)  # shape: (N, 3)
            # nb_dist = dist_fn(pos_i, pos_j)

            ### Original approach

            nb_dist = dist_fn(positions[i_nbrs], positions[j_nbrs])
            nb_mask = (nb_dist < ff.cutoff) & (i_nbrs < ff.atom_count)
            dir_chg = ff.charges[i_nbrs] * ff.charges[j_nbrs]

            #lj_pot = jnp.sum(positions[i_nbrs]) * 1e-10 + jnp.sum(positions[j_nbrs]) * 1e-10 # TODO dummy example to force variable to be traced
            # def nbr_sum(pos, idx):
            #     return jnp.sum(positions[nbr_list.idx]) * 1e-10
            # lj_pot = jnp.sum(jax.vmap(nbr_sum)(positions, nbr_list.idx)) * 1e-10
            #lj_pot = jnp.sum(positions[nbr_list.idx]) * 1e-10
            #lj_pot += jnp.sum(positions[i_nbrs]) * 1e-10
            #lj_pot += jnp.sum(positions[j_nbrs]) * 1e-10
            #lj_pot = jnp.sum(jnp.take(positions, nbr_list.idx, axis=0, mode="clip")) * 1e-10

            ###############################################
            ### Original approach .0057 s
            # lj_pot = jnp.sum(positions[i_nbrs]) * 1e-10 + jnp.sum(positions[j_nbrs]) * 1e-10

            ### Dual gather/split approach .0040

            # positions_x = positions[:, 0]
            # positions_y = positions[:, 1]
            # positions_z = positions[:, 2]

            # pxi = positions_x[i_nbrs]
            # pyi = positions_y[i_nbrs]
            # pzi = positions_z[i_nbrs]

            # #lj_pot += jnp.sum(px + py + pz) * 1e-10

            # positions_x = positions[:, 0]
            # positions_y = positions[:, 1]
            # positions_z = positions[:, 2]

            # pxj = positions_x[j_nbrs]
            # pyj = positions_y[j_nbrs]
            # pzj = positions_z[j_nbrs]

            # #lj_pot += jnp.sum(px + py + pz) * 1e-10

            # nb_dist = dist_fn(jnp.stack([pxi, pyi, pzi], axis=-1), jnp.stack([pxj, pyj, pzj], axis=-1))
            # nb_mask = (nb_dist < ff.cutoff) & (i_nbrs < ff.atom_count)
            # dir_chg = ff.charges[i_nbrs] * ff.charges[j_nbrs]

            ### Argsort approach - extremely slow

            # j_sorted = jnp.argsort(j_nbrs)
            # j_unsort = jnp.argsort(j_sorted)  # used later to restore order

            # # Now gather in sorted order
            # pj_sorted = positions[j_nbrs[j_sorted]]  # fast gather

            # # Reorder to original order
            # pj = pj_sorted[j_unsort]

            # pi = positions[i_nbrs]

            # lj_pot = jnp.sum(pi) * 1e-10 + jnp.sum(pj) * 1e-10

            ### Stack/single gather approach .0057

            # all_nbrs = jnp.concatenate([i_nbrs, j_nbrs])
            # pos_all = positions[all_nbrs]
            # lj_pot += jnp.sum(pos_all) * 1e-10

            ##################################################################

            #lj_pot = jnp.sum(nb_dist) * 1e-10 # TODO dummy example to force variable to be traced

            direct_pot = jnp.sum(coulomb_direct(nb_dist, dir_chg, ff.ewald_alpha) * nb_mask)

            self_pot = self_named(ff.charges, ff.ewald_alpha)

            corr_dr = dist_fn(positions[ff.exclusions[:, 0]], positions[ff.exclusions[:, 1]])
            charge_sq = ff.charges[ff.exclusions[:, 0]] * ff.charges[ff.exclusions[:, 1]]
            correction_pot = correction_named(corr_dr, charge_sq, ff.ewald_alpha)

            # TODO small modification to provide grid information for linear response code
            # a cleaner way of doing this would probably be to return the potential w.r.t
            # each atom, the potential grid, and the sum potential as in AMBER
            recip_grid = reciprocal_named(positions)
            pme_mask,pme_exp_m,pme_B,pme_Fgrid = recip_grid
            recip_pot = util.high_precision_sum(pme_mask * pme_exp_m * pme_B * jnp.abs(pme_Fgrid)**2)

            sigma = 0.5*(ff.sigma[i_nbrs] + ff.sigma[j_nbrs])
            epsilon = safe_sqrt(ff.epsilon[i_nbrs] * ff.epsilon[j_nbrs])
            lj_pot = jnp.sum(lennard_jones(nb_dist, sigma, epsilon) * nb_mask)

            result_dict["direct_pot"] = ONE_4PI_EPS * direct_pot
            result_dict["self_pot"] = ONE_4PI_EPS * self_pot
            result_dict["correction_pot"] = ONE_4PI_EPS * correction_pot
            result_dict["recip_pot"] = ONE_4PI_EPS * recip_pot

            disp_pot = ff.disp_coef/jnp.prod(ff.box_vectors)

            #TODO replace all sums with high precision sums - or is this only for mixed
            coul_pot = ONE_4PI_EPS * (direct_pot + self_pot + correction_pot + recip_pot) + disp_pot
        else:
            pos_nb_1 = positions[ff.pairs[:, 0]]
            pos_nb_2 = positions[ff.pairs[:, 1]]
            dist_nb = dist_fn(pos_nb_1, pos_nb_2)

            sigma = 0.5*(ff.sigma[ff.pairs[:, 0]] + ff.sigma[ff.pairs[:, 1]])
            epsilon = safe_sqrt(ff.epsilon[ff.pairs[:, 0]] * ff.epsilon[ff.pairs[:, 1]])
            charge_sq = charges[ff.pairs[:, 0]] * charges[ff.pairs[:, 1]]
            nb_mask = ff.pairs[:,0] != -1

            lj_pot = util.high_precision_sum(nb_mask * lj_named(dist_nb, sigma, epsilon))
            coul_pot = jnp.sum(nb_mask * coul_named(dist_nb, charge_sq))
            disp_pot = 0.0

        result_dict['lj_pot'] = lj_pot
        result_dict['coul_pot'] = coul_pot
        result_dict['disp_pot'] = disp_pot

        if charge_method == "LRCH":
            # TODO convert internal units from A -> nm
            dq, resp = linear_response(positions * 10.0, ff, nbr_list, 0, metric_fn, recip_grid, ff.box_vectors, ff.grid_points)
            result_dict['lrch_pot'] = resp
            lrch_pot = resp
        else:
            lrch_pot = 0.0

        ### 1-4 interactions
        # TODO change all the array names to follow xxx_14 xxx_bond for consistency
        pos_14_1 = positions[ff.pairs_14[:, 0]]
        pos_14_2 = positions[ff.pairs_14[:, 1]]
        dist_14 = dist_fn(pos_14_1, pos_14_2)

        mask_14 = ff.pairs_14[:,0] != -1
        lj_14_pot = util.high_precision_sum(mask_14 * lennard_jones(dist_14, ff.sigma_14, ff.epsilon_14))
        coul_14_pot = jnp.sum(mask_14 * coulomb(dist_14, charges_14))
        result_dict['lj_14_pot'] = lj_14_pot
        result_dict['coul_14_pot'] = coul_14_pot

        nb_pot = lj_pot + coul_pot + lj_14_pot + coul_14_pot + lrch_pot
        result_dict['nb_pot'] = nb_pot

        # TODO add restraint energies where applicable
        if debug:
            return result_dict
        elif return_charges:
            return (bond_pot + angle_pot + torsion_pot + nb_pot), charges
        else:
            return (bond_pot + angle_pot + torsion_pot + nb_pot)

    # TODO this may not be very safe
    # wrapping static argument via closure allows me to return non jit nrg function in this case
    # it also allows me to do this with debug and still respect the single return rule for jax md
    # i don't think i can get all of these things with @partial and static_argnames but i need to look more
    # this also might not be necessary actually
    if ensemble == "NVE":
        nrg_closure = lambda pos, ff, nbr_list: nrg_fn(pos, ff, nbr_list, False)
        init_fn, apply_fn = simulate.nve(nrg_closure, shift_fn, timestep)
        state = init_fn(jax.random.PRNGKey(0), ff.positions, mass=ff.masses, kT=init_temp*kB, ff=ff, nbr_list=ff.nbr_list)
    elif ensemble == "MIN":
        nrg_closure = lambda pos, ff, nbr_list: nrg_fn(pos, ff, nbr_list, False)
        init_fn, apply_fn = minimize.gradient_descent(nrg_closure, shift_fn, timestep)
        state = init_fn(ff.positions, ff=ff, nbr_list=ff.nbr_list)

    if nonbonded_method == "PME" and ensemble != None:
        def body_fn(i, state):
            state, ff, nbr_list = state
            nbr_list = nbr_list.update(state.position)
            state = apply_fn(state, ff=ff, nbr_list=nbr_list)

            # TODO example for COM motion removal
            # need to validate more with long simulation, not clear
            # if this is the correct formulation with momentum
            # total_mass = jnp.sum(state.mass)
            # v_com = jnp.sum(state.momentum, axis=0) / total_mass
            # state = state.set(momentum=state.momentum - state.mass * v_com)

            return state, ff, nbr_list
    elif ensemble != None:
        def body_fn(i, state):
            state, ff, _ = state
            state = apply_fn(state, ff=ff, nbr_list=None)

            return state, ff, _
    else:
        body_fn = None
        state = None


    return nrg_fn, ff, body_fn, state

# TODO partial decorator for this to make named call?
# does this work with mapping? wrap named call and vmap?
# @partial(jax.named_call, "bond_fn")
def harmonic_bond(dist, k, l):
    return k * jnp.power((dist - l), 2)

def harmonic_angle(theta, k, eq_angle):
    return k * jnp.power((theta - eq_angle), 2)

def periodic_torsion(theta, k, phase, period):
    return k * (1.0 + jnp.cos(period * theta - phase))

def cmap_torsion():
    '''
    Computes CMAP torsions for FF19SB
    '''
    return

# TODO consider accumulating LJ with coulomb direct for PME
# TODO is this robust, or should double where trick be used
def lennard_jones(dr, sigma, epsilon):
    dr = jnp.where(jnp.isclose(dr, 0.), 1, dr)
    idr = (sigma/dr)
    idr2 = idr*idr
    idr6 = idr2*idr2*idr2
    idr12 = idr6*idr6
    return 4.0*epsilon*(idr12-idr6)

def coulomb(dr, charge_sq):
    dr = jnp.where(jnp.isclose(dr, 0.), 1, dr)
    return ONE_4PI_EPS * charge_sq / dr
def coulomb_ewald():
    return

def coulomb_pme():
    # (direct_nrg + recip_nrg + self_nrg + corrective_nrg)
    return

def coulomb_direct(dr: Array, charge_sq: Array, alpha: float, **kwargs) -> Array:
    dr = jnp.where(jnp.isclose(dr, 0.), 1, dr)
    return charge_sq * erfc(alpha * dr) / dr

def coulomb_direct_neighbor():
    return

# TODO it may be more idiomatic to merge these two functions into 1 corrective term
def coulomb_self(charges, ewald_alpha):
    return jnp.sum(charges**2) * (-ewald_alpha/jnp.sqrt(jnp.pi))

def coulomb_correction(dr, charge_sq, ewald_alpha):
    return -jnp.sum(charge_sq * erf(ewald_alpha * dr) / dr)

def coulomb_recip(charges, box, grid_points, ewald_alpha, fractional_coordinates=False, indiv_return=False):
    box = jnp.diag(box) if (isinstance(box, jnp.ndarray) and box.ndim == 1) else box
    _ibox = space.inverse(box)

    if not isinstance(grid_points, jnp.ndarray):
        indexing = 'xy'
        dim = 3
        grid_dimensions = onp.array((grid_points,) * dim)
    else:
        indexing = 'ij'
        grid_dimensions = onp.array(grid_points)

    def energy_fn(R, **kwargs):
        q = kwargs.pop('charges', charges)
        _box = kwargs.pop('box', box)
        ibox = space.inverse(kwargs['box']) if 'box' in kwargs else _ibox

        # dim = R.shape[-1]
        # TODO consider if using onp calls here causes an issue
        # grid_dimensions = onp.array((grid_points,) * dim)

        map_named = jax.named_call(map_charges_to_grid, name="chg to grid")
        grid = map_named(R, q, ibox, grid_dimensions,
                                fractional_coordinates)
        Fgrid = jnp.fft.fftn(grid)
        mx, my, mz = jnp.meshgrid(*[jnp.fft.fftfreq(g) for g in grid_dimensions], indexing=indexing)

        if jnp.isscalar(_box):
            m_2 = (mx**2 + my**2 + mz**2) * (grid_dimensions[0] * ibox)**2
            V = (1.0 * _box)**dim
        else:
            m = (ibox[None, None, None, 0] * mx[:, :, :, None] * grid_dimensions[0] +
            ibox[None, None, None, 1] * my[:, :, :, None] * grid_dimensions[1] +
            ibox[None, None, None, 2] * mz[:, :, :, None] * grid_dimensions[2])
            m_2 = jnp.sum(m**2, axis=-1)
            V = jnp.linalg.det(_box)
        mask = m_2 != 0

        def exp_ret():
            exp_m = 1 / (2 * jnp.pi * V) * jnp.exp(-jnp.pi**2 * m_2 / ewald_alpha**2) / m_2
            if indiv_return:
                ret_val = (mask,exp_m,B(mx, my, mz),Fgrid)

            else:
                ret_val = util.high_precision_sum(mask * exp_m * B(mx, my, mz) * jnp.abs(Fgrid)**2)
            return ret_val
        ret_named = jax.named_call(exp_ret, name="exp and ret")
        final_ret = ret_named()
        return final_ret
    return energy_fn

# PME utility functions
# Copied from electrostatics.py

def structure_factor(g, R, q=1):
    if isinstance(q, jnp.ndarray):
        q = q[None, :]
    return util.high_precision_sum(
        q * jnp.exp(1j * jnp.einsum('id,jd->ij', g, R)),
        axis=1
    )

# B-Spline and charge (or structure factor) smearing code.
# TODO(schsam,  samarjeet): For now, we only include support for a fast fourth
# order spline. If you are interested in higher order b-splines or different
# interpolating functions, please raise an issue.

# TODO implement 5th order b-spline to match OMM

@partial(jnp.vectorize, signature='()->(p)')
def optimized_bspline_4(w):
  coeffs = jnp.zeros((4,))

  coeffs = coeffs.at[2].set(0.5 * w * w)
  coeffs = coeffs.at[0].set(0.5 * (1.0-w) * (1.0-w))
  coeffs = coeffs.at[1].set(1.0 - coeffs[0] - coeffs[2])

  coeffs = coeffs.at[3].set(w * coeffs[2] / 3.0)
  coeffs = coeffs.at[2].set(((1.0 + w) * coeffs[1] + (3.0 - w) * coeffs[2])/3.0)
  coeffs = coeffs.at[0].set((1.0 - w) * coeffs[0] / 3.0)
  coeffs = coeffs.at[1].set(1.0 - coeffs[0] - coeffs[2] - coeffs[3])

  return coeffs


def map_charges_to_grid(
    position: Array,
    charge: Array,
    inverse_box: Box,
    grid_dimensions: Array,
    fractional_coordinates: bool
  ) -> Array:
  """Smears charges over a grid of specified dimensions."""

  Q = jnp.zeros(grid_dimensions)
  N = position.shape[0]

  @partial(jnp.vectorize, signature='(),()->(p)')
  def grid_position(u, K):
    grid = jnp.floor(u).astype(jnp.int32)
    grid = jnp.arange(0, 4) + grid
    return jnp.mod(grid, K)

  @partial(jnp.vectorize, signature='(d),()->(p,p,p,d),(p,p,p)')
  def map_particle_to_grid(position, charge):
    if fractional_coordinates:
      u = transform_gradients(inverse_box, position) * grid_dimensions
    else:
      u = space.raw_transform(inverse_box, position) * grid_dimensions

    w = u - jnp.floor(u)
    coeffs = optimized_bspline_4(w)

    grid_pos = grid_position(u, grid_dimensions)

    accum = charge * (coeffs[0, :, None, None] *
                      coeffs[1, None, :, None] *
                      coeffs[2, None, None, :])
    grid_pos = jnp.concatenate(
        (jnp.broadcast_to(grid_pos[[0], :, None, None], (1, 4, 4, 4)),
         jnp.broadcast_to(grid_pos[[1], None, :, None], (1, 4, 4, 4)),
         jnp.broadcast_to(grid_pos[[2], None, None, :], (1, 4, 4, 4))), axis=0)
    grid_pos = jnp.transpose(grid_pos, (1, 2, 3, 0))

    return grid_pos, accum

  gp, ac = map_particle_to_grid(position, charge)
  gp = jnp.reshape(gp, (-1, 3))
  ac = jnp.reshape(ac, (-1,))

  return Q.at[gp[:, 0], gp[:, 1], gp[:, 2]].add(ac)


@partial(jnp.vectorize, signature='()->()')
def b(m, n=4):
  assert(n == 4)
  k = jnp.arange(n - 1)
  M = optimized_bspline_4(1.0)[1:][::-1]
  prefix = jnp.exp(2 * jnp.pi * 1j * (n - 1) * m)
  return prefix / jnp.sum(M * jnp.exp(2 * jnp.pi * 1j * m * k))


def B(mx, my, mz, n=4):
  """Compute the B factors from Essmann et al. equation 4.7."""
  b_x = b(mx)
  b_y = b(my)
  b_z = b(mz)
  return jnp.abs(b_x)**2 * jnp.abs(b_y)**2 * jnp.abs(b_z)**2


@jax.custom_jvp
def transform_gradients(box, coords):
  # This function acts as a no-op in the forward pass, but it transforms the
  # gradients into fractional coordinates in the backward pass.
  return coords


@transform_gradients.defjvp
def _(primals, tangents):
  box, coords = primals
  dbox, dcoords = tangents
  return coords, space.transform(dbox, coords) + space.transform(box, dcoords)

def calculate_eem_charges(species: Array,
                                atom_mask: Array,
                                nbr_inds: Array,
                                hulp2_mat: Array,
                                tapered_dists: Array,
                                idempotential: Array,
                                electronegativity: Array,
                                n_atoms: int,
                                init_charges: Array = None,
                                total_charge: float = 0.0,
                                backprop_solve: bool = False,
                                tol: float = 1e-06,
                                max_solver_iter: int = 500):
    '''
    EEM charge solver
    If max_solver_iter is set to -1, use direct solve
    Returns:
    an array of shape [n+1,] where first n entries are the charges and
    last entry is the electronegativity equalization value
    '''

    if backprop_solve == False:
        tapered_dists = jax.lax.stop_gradient(tapered_dists)
        hulp2_mat = jax.lax.stop_gradient(hulp2_mat)
    prev_dtype = tapered_dists.dtype
    N = len(species)
    # N = n_atoms
    # might cause nan issues if 0s not handled well
    # internal parameter is 14.4
    A = util.safe_mask(hulp2_mat != 0, lambda x: tapered_dists * 14.4 / x, hulp2_mat, 0.0)

    my_idemp = idempotential[species]
    my_elect = electronegativity[species] * atom_mask

    def to_dense():
        '''
        Create a dense matrix
        '''
        A_ = jax.vmap(lambda j: jax.vmap(lambda i: jnp.sum(A[i] * (nbr_inds[i] == j)))(jnp.arange(N)))(jnp.arange(N))
        A_ =  A_.at[jnp.diag_indices(N)].add(2.0 * my_idemp)
        matrix = jnp.zeros(shape=(N+1,N+1),dtype=prev_dtype)
        matrix = matrix.at[:N,:N].set(A_)
        matrix = matrix.at[N,:N].set(atom_mask)
        matrix = matrix.at[:N,N].set(atom_mask)
        matrix = matrix.at[N,N].set(0.0)
        return matrix

    mask = (nbr_inds != N)

    def SPMV_dense(vec):
        '''
        Matrix-free mat-vec
        '''
        res = jnp.zeros(shape=(N+1,), dtype=jnp.float64)
        s_vec = vec.astype(prev_dtype)[nbr_inds] * mask
        vals = jax.vmap(jnp.dot)(A, s_vec) + \
            (my_idemp * 2.0) * vec[:N] + vec[N]
        res = res.at[:N].set(vals * atom_mask)
        res = res.at[N].set(jnp.sum(vec[:N] * atom_mask))  # sum of charges
        return res

    b = jnp.zeros(shape=(N+1,), dtype=jnp.float64)
    b = b.at[:N].set(-1 * my_elect)
    b = b.at[N].set(total_charge)

    if max_solver_iter == -1:
        charges = jnp.linalg.solve(to_dense(), b)
    else:
        charges, conv_info = jax.scipy.sparse.linalg.cg(SPMV_dense, b, x0=init_charges, tol=tol, maxiter=max_solver_iter)
        # TODO look into other solvers like bicgstab or gmres
        # TODO figure out if there's a way to make convergence info work
        # jax.debug.print("Convergence Information {conv_info}", conv_info=conv_info)
    charges = charges.astype(prev_dtype)
    charges = charges.at[:-1].multiply(atom_mask)
    return charges

# linear response based fluctuating charge method based on lrch branch of amber gitlab
# TODO add type annotations and -> signature
def linear_response(positions, ff, nbr_list, intrascl=0.0, dist_fn=None, pme_recip_grid=None, box=None, grid_points=None):
    num_fluc = ff.solute_cut
    num_solvent = ff.atom_count - num_fluc

    cut = ff.cutoff
    cut = cut * AU_PER_AMBER_ANGSTROM

    bvec = jnp.zeros(num_fluc)

    pm_coords_bohr = positions[:num_fluc] * AU_PER_AMBER_ANGSTROM
    mm_coords_bohr = positions[num_fluc:] * AU_PER_AMBER_ANGSTROM
    pm_ref_charges = ff.charges[:num_fluc]
    mm_ref_charges = ff.charges[num_fluc:]
    pair_indices = None
    pair_scale = None
    # TODO what is the conversion factor for this from kcal/kj to atomic units
    # this is currently provided as the grid of potentials, need to interpolate back to the point charges

    # contribution at a given point will be equal to the spline weights times
    # the potentials at each of the grid points from 1..N where N = spline order
    # which in this case is 4

    # TODO most of this probably should be moved into the ewald routine itself eventually,
    # maybe with a mask to select which atoms to caluclate point potentials for to improve efficiency
    pme_mask,pme_exp_m,pme_B,pme_Fgrid = pme_recip_grid
    
    # TODO do you need to take the inverse fft if you're looking for the positions?
    # if so, does this move the coefficients and indexing back into real space?
    V_grid = jnp.fft.ifftn(pme_mask * pme_exp_m * pme_B * pme_Fgrid).real / (1.0/jnp.prod(jnp.array(grid_points)))

    grid_dimensions = grid_points

    ibox = space.inverse(box)

    pme_pos = positions
    pme_chg = ff.charges

    # TODO this doesn't work for fractional coordinates, consider this going forward
    u = space.raw_transform(ibox, pme_pos / 10.0) * grid_dimensions
    w = u - jnp.floor(u)
    coeffs = optimized_bspline_4(w)

    base_idx = jnp.floor(u).astype(jnp.int32)  # shape (n_atoms, 3)
    offsets = jnp.arange(4)

    # compute 4 indices per axis (broadcasted over atoms) for the spline order (could change in the future, not sure)
    # TODO likely need to add order 5 and 6 splines to match some other codes or explicitly require order 4 for lrch
    x_idx = (base_idx[:, 0, None] + offsets[None, :]) % grid_dimensions[0]
    y_idx = (base_idx[:, 1, None] + offsets[None, :]) % grid_dimensions[1]
    z_idx = (base_idx[:, 2, None] + offsets[None, :]) % grid_dimensions[2]

    # create all 64 combinations of x,y,z indices for each atom
    # same idea as the charge spread in the reciprocal function
    # TODO there are probably some better numpy routines for this
    x_idx = x_idx[:, :, None, None]  # (n_atoms, 4, 1, 1)
    y_idx = y_idx[:, None, :, None]  # (n_atoms, 1, 4, 1)
    z_idx = z_idx[:, None, None, :]  # (n_atoms, 1, 1, 4)

    idx = jnp.stack([x_idx + jnp.zeros_like(y_idx) + jnp.zeros_like(z_idx), 
                     y_idx + jnp.zeros_like(x_idx) + jnp.zeros_like(z_idx), 
                     z_idx + jnp.zeros_like(x_idx) + jnp.zeros_like(y_idx)], axis=-1)
    # shape: (n_atoms, 4, 4, 4, 3)

    # flatten indices for gathering
    idx_flat = idx.reshape(-1, 3)
    V_vals_flat = V_grid[idx_flat[:, 0], idx_flat[:, 1], idx_flat[:, 2]]  # (n_atoms * 64,)

    # reshape to (n_atoms, 4, 4, 4)
    V_vals = V_vals_flat.reshape(pme_pos.shape[0], 4, 4, 4)

    # compute tensor product of weights
    wx = coeffs[:, 0, :]  # shape (n_atoms, 4)
    wy = coeffs[:, 1, :]
    wz = coeffs[:, 2, :]

    # outer product of weights: (n_atoms, 4, 4, 4)
    weights = wx[:, :, None, None] * wy[:, None, :, None] * wz[:, None, None, :]

    V_atoms = jnp.sum(weights * V_vals, axis=(1, 2, 3))  # shape (n_atoms,)

    # TODO change this to only calculate relevant atoms if the MM point potentials aren't relevant
    pme_recip_pot = 2.0 * V_atoms[:ff.solute_cut] / 10.0 / AU_PER_AMBER_ANGSTROM

    # TODO have to deal with the case where eta is near 0
    eta = ff.hardness[:num_fluc]
    eta = jnp.where(eta < 1e-8, 10.0, eta)
    zetas = eta * eta * (jnp.pi/2.0)

    # final derived equation comes out to something like
    # Ecew = E0 + E1 + E2
    #
    # E0 = 1/2 <mm+qref|jn|mm+qref>
    # E1 = <dq|jn|mm+qref>
    #    =   <dq|(j<)|qref>  ! term 1
    #      - <dq|jg|qref>    ! term 2
    #      + <dq|(j<)|qref>  ! term 3
    #      - <dq|jg|qref>    ! term 4
    #      + <dq|j%|mm+qref> ! term 5
    # E2 = 1/2 <dq|j|dq>
    # where
    # E0 = regular MM energy
    # E1 = interaction of the response density with the periodic MM charges
    # E2 = nonperiodic interaction between the response charges
    # and
    # j< = short-range direct interaction
    # j% = reciprocal space contribution
    # jg = Ewald Gaussian potential
    # jn = periodic potential = ((j<) + j% - jg)|x>

    def build_bvec(
        pm_coords: jnp.ndarray,       # (npm, 3)
        mm_coords: jnp.ndarray,       # (nmm, 3)
        pm_ref_charges: jnp.ndarray,  # (npm, )
        mm_ref_charges: jnp.ndarray,  # (nmm, 3)
        zetas: jnp.ndarray,           # (npm, )
        # TODO consider if this really makes sense or if generating the matrices on the spot is easier, at least for pm-pm inter
        pair_indices: jnp.ndarray,    # (npairs, 2) for real-space pairs i<j assuming nb list is used
        pair_scale: jnp.ndarray,      # (npairs, ) scale factors (could be intramolecular scale, 1-4 SCEE, etc.)
        pme_recip_pot: jnp.ndarray,    # (npm, ) precomputed reciprocal potential at each PM site (if needed)
        cut: jnp.ndarray
        ): # TODO add type annotation
        """
        Construct the bvec for the linear-response model, adding real-space terms
        from reference charges and optionally a known reciprocal-space potential
        TODO how exactly do non-ewald cases work w.r.t. the recip vector
        """
        # TODO not sure if this is correct, this comes from the ewald routine in amber
        beta = ff.ewald_alpha / AU_PER_AMBER_ANGSTROM / 10.0 # TODO might be off by factor of 10
        #beta = 0.14558004294479038 # TODO calculated value from amber for test, not for production
        SQRT_PI = jnp.sqrt(jnp.pi)

        npm = pm_coords.shape[0]
        nmm = mm_coords.shape[0]
        ew_self = 2.0 * beta / SQRT_PI

        # TODO need to implement intrascl and do_cew options
        # particularly important for pbc vs nobc

        bvec = jnp.zeros((npm,), dtype=jnp.float64)

        # important to avoid NaN gradient due to evaluation of sqrt for i=j pairs
        pm_dr = jax.vmap(lambda x: jax.vmap(lambda y: jnp.sum((x-y)**2))(pm_coords))(pm_coords)
        pm_dr = util.safe_mask(pm_dr > 1e-12, lambda r: jnp.sqrt(r), pm_dr, 0.0)
        pm_mm_dr = jax.vmap(lambda x: jax.vmap(lambda y: jnp.sum((x-y)**2))(mm_coords))(pm_coords)
        pm_mm_dr = util.safe_mask(pm_mm_dr > 1e-12, lambda r: jnp.sqrt(r), pm_mm_dr, 0.0)

        # 1) + <dq|(j<)|qref>

        # this should be controlled by intrascl
        # in practice, for an MM simulation, this would be
        # 2 point charges right on top of each other
        # we should likely ignore the interaction of the response basis
        # with its own MM charge, this can just be treated as an additive term when
        # tuning our constant parameters, as this is what this would be in practice

        # 2) - <dq|jg|qref>
        # self contribution
        bvec = bvec - (pm_ref_charges * ew_self)

        # TODO should probably use jnp.where or maybe double where trick
        op = util.safe_mask(pm_dr > 1e-12, lambda r: jax.scipy.special.erf(beta*r)/r, pm_dr, 0.0)
        bvec = bvec - jnp.sum(pm_ref_charges * op, axis=1) #/ 2.0) # to avoid double counting?

        # 3) + <dq|(j<)|qref> TODO pretty sure this should be |mm>
        sqz = jnp.sqrt(zetas)
        # can just do something with the upper triangular portion of the matrix or directly manipulate the diag idx
        mask = (pm_mm_dr > 1e-12) & (pm_mm_dr < cut)
        pm_mm_dr_safe = jnp.where(mask, pm_mm_dr, 1.0)
        pm_mm_dr_sqz = jax.scipy.special.erf(sqz.reshape((-1,1)) * pm_mm_dr) # TODO this isn't a very efficient way of doing this
        op = jnp.where(mask, pm_mm_dr_sqz / pm_mm_dr_safe, 0.0)
        bvec = bvec + jnp.sum(mm_ref_charges * op, axis=1)

        # 4) - <dq|jg|qref> TODO pretty sure this should be |mm>
        mask = (pm_mm_dr > 1e-12) & (pm_mm_dr < cut)
        op = util.safe_mask(mask, lambda r: jax.scipy.special.erf(beta*r)/r, pm_mm_dr, 0.0)
        bvec = bvec - jnp.sum(mm_ref_charges * op, axis=1)

        # 5) + <dq|j%|mm+qref>
        bvec = bvec + pme_recip_pot

        # 6) TODO without intrascl do you add exclusions here?

        return bvec

    def build_Amat(
        pm_coords: jnp.ndarray,   # (npm,3)
        zetas: jnp.ndarray,       # (npm,)
        hardness: jnp.ndarray,    # (npm,) -> diagonal entries
        pair_indices: jnp.ndarray, # (npairs,2) but i don't think this will be used for now
        cut: jnp.ndarray):
        """
        construct A matrix - n_pm x n_pm
        A(i,i) = hardness[i]
        A(i,j) = erf( sqrt(zeta_ij)*r ) / r, for i<j
        """

        npm = pm_coords.shape[0]
        # Initialize Amat
        Amat = jnp.zeros((npm, npm), dtype=jnp.float64)

        # Put hardness on diag
        Amat = Amat.at[jnp.arange(npm), jnp.arange(npm)].set(hardness)

        # TODO consider if generating triangular matrix and then filling is better
        # https://stackoverflow.com/questions/36250729/how-to-convert-triangle-matrix-to-square-in-numpy
        # given the fact that you're essentially generating that for this and the pm-pm inters for the bvec,
        # this might be a good bit more efficient
        # also probably want to generate the matrix outside and pass in in to these functions for reuse
        # as both require the distance matrix
        # the pm-mm distance matrix should probably be built from the nb list eventually
        # in practice this doesn't seem to have too bad of a performance impact so far

        # important to mask nans from sqrt 0
        pm_dr = jax.vmap(lambda x: jax.vmap(lambda y: jnp.sum((x-y)**2))(pm_coords))(pm_coords)
        pm_dr = util.safe_mask(pm_dr > 1e-12, lambda r: jnp.sqrt(r), pm_dr, 0.0)

        zeta_mul = zetas * zetas.T
        zeta_sum = zetas + zetas.T
        zeta_mat = zeta_mul/zeta_sum # TODO might be a better way to implement this combining rule

        mask = (pm_dr > 1e-12) & (pm_dr < cut)
        pm_dr_safe = jnp.where(mask, pm_dr, 1.0)
        pm_dr_zeta = jax.scipy.special.erf(zeta_mat * pm_dr) 
        op = jnp.where(mask, pm_dr_zeta / pm_dr_safe, 0.0)


        Amat = Amat + op

        return Amat

    def solve_flucq(Amat: jnp.ndarray,
                bvec: jnp.ndarray,
                max_solver_iter: jnp.ndarray = -1) -> jnp.ndarray:
        """
        Solve A * dq = -b  for dq to get final delta from response vec
        For large N, iterative solver will be needed as this uses LU
        """

        # TODO it turns out this is a constrained problem
        # need to work out non LU based approach for larger systems
        # if max_solver_iter == -1:
        #     dq = jnp.linalg.solve(Amat, bvec)
        # else:
        #     dq, conv_info = jax.scipy.sparse.linalg.cg(SPMV_dense, bvec, x0=init_charges, tol=tol, maxiter=max_solver_iter)

        # attempt to replicate fortran code with constraint
        # use cholesky and lagrange to solve constrained system
        from jax.scipy.linalg import cho_factor, cho_solve
        d = jnp.ones(len(bvec))
        c = 0.0
        cho = cho_factor(Amat, lower=False)
        Ainv_b = cho_solve(cho, bvec)
        Ainv_d = cho_solve(cho, d)
        mu = (jnp.dot(d, Ainv_b) + c) / jnp.dot(d, Ainv_d)
        x = mu * Ainv_d - Ainv_b

        # TODO should also add options for things like spsolve, bicg/stab, gmres, etc
        # probably not necessary for well conditioned matrix in this case but useful for future constraints in linear system
        return x

    def compute_response_energy(Amat: jnp.ndarray,
                            bvec: jnp.ndarray,
                            dq: jnp.ndarray) -> float:
        """
        Compute linear response energy from bvec
        E = b·dq + 0.5 dq^T A dq
        """
        E1 = jnp.dot(bvec, dq)
        E2 = 0.5 * jnp.dot(dq, Amat @ dq)
        return E1 + E2

    # Build the real-space portion of bvec
    b_real = build_bvec(
        pm_coords_bohr, mm_coords_bohr, pm_ref_charges, mm_ref_charges, zetas,
        pair_indices, pair_scale, pme_recip_pot, cut
    )

    # Build the real-space portion of Amat
    A_real = build_Amat(pm_coords_bohr, zetas, ff.hardness[:ff.solute_cut], pair_indices, cut)

    # Combine with reciprocal block
    # TODO this may not be necessary, it seemed like the reciprocal interaction
    # should be incorporated into the hardness matrix but it doesn't look like
    # that's the case
    Amat_total = A_real

    # Solve for dq
    dq = solve_flucq(Amat_total, b_real)

    # Compute response energy
    # TODO check if this is correct, other terms are in kj/mol
    Eresp = compute_response_energy(Amat_total, b_real, dq) * 4.184

    # Return first-order response vector and response energy
    # TODO should extra information like second-order response be copied?
    return dq, Eresp