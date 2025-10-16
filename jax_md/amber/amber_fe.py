# wrapper for free energy tools

#import jax
# TODO remove this once done
#jax.config.update("jax_enable_x64", True)
#import jax.numpy as jnp
#from pymbar import testsystems, MBAR
import sys
import os
import numpy as np

def pymbar_test():
    x_n, u_kn, N_k, s_n = testsystems.HarmonicOscillatorsTestCase().sample()
    print(x_n) # (150,) oscilatior positions
    print(u_kn) # (5, 150) reduced potentials
    print(N_k) # (5,) nsamples
    print(s_n) # (150,) thermodynamic state each sample was drawn from


    mbar = MBAR(u_kn, N_k)

    results = mbar.compute_free_energy_differences()
    for key, value in results.items():
        print(key, value)
    # use position of harmonic oscillator as observable
    results = mbar.compute_expectations(x_n)
    for key, value in results.items():
        print(key, value)

#TODO
"""
need to write utility to run all the intermediate simulation windows at each lambda value
have to set up windows at each lambda and scale energy accordingly (LJ + coul?)


external amber/openmm interface for CI testing
"""

"""
&cntrl
  imin=0, ntb=1, ntt=3, tempi=300.0, temp0=300.0,
  nstlim=500000, dt=0.002,
  ntc=2, ntf=2,
  ntpr=1000, ntwx=1000, ntwr=5000,
  cut=10.0,
  icfe=1, ifsc=1, clambda=0.5,
  crgmask=':1',
  scmask=':1',
  scalpha=0.5, scbeta=12.0,
  logdvdl=1
/

icfe=1	Enable free energy perturbation
ifsc=1	Use softcore potentials (for LJ decoupling)
clambda=0.5	Current λ value (range 0.0 → 1.0)
crgmask=':1'	Decouple charges on solute (residue 1)
scmask=':1'	Decouple LJ on solute (residue 1)
scalpha, scbeta	Softcore parameters for LJ smoothing
logdvdl=1	Print dv/dλ (TI integrand) in output

just an example mdin script for a window to model after
"""

"""
Some thoughts for basic example setup

Ensemble NVT (canonical)
Thermostat	Langevin (recommended for small systems & stability)
Temperature	298.15 K (room temperature)
Time step	1–2 fs (0.001–0.002 ps)
100 ps overall simulation per lambda window
Sampling interval	1 ps (i.e., every 500–1000 steps)
λ windows	e.g., [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
Total samples	100 samples × 6 lambda windows → 600 samples
Box type	Periodic box with water and one solute molecule
"""

def run_window():
    # function runs lambda window and returns information needed for sampling (energy or other reduced potential)
    return

if __name__ == "__main__":
    num_points = 12
    #lambdas = [0.1, 0.3, 0.5, 0.7, 0.9]
    # for softcore potentials in sander, lambda should be
    # 0.01 < clambda < 0.99
    #lambdas = np.linspace(0.05, 0.95, 15)
    # in ali's code it's more like
    # linspace exclusive (0,1,12)
    # ideally though more like
    # points = np.linspace(0, 1, n + 2)[1:-1]
    # or with uneven spacing to better cover unstable ends

    #lambdas = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, num_points + 2)[1:-1])) # sine transformation approach

    # alternative approach with power biasing, slightly more ends-biased than sine approach with power = 2
    # power = 2
    # i = np.linspace(0, 1, num_points + 2)[1:-1]
    # lambdas = np.where(i < 0.5, 0.5 * (2 * i)**power, 1 - 0.5 * (2 * (1 - i))**power)

    # another alternative approach with gaussian quadrature
    from numpy.polynomial.legendre import leggauss
    nodes, weights = leggauss(num_points)

    print("Softcore potentials cannot be used with clambda < 0.005 or > 0.995")

    # Map from [-1, 1] to [0, 1]
    lambdas = 0.5 * (nodes + 1)
    weights = 0.5 * weights

    print(f"selected lambda values are: {lambdas}")

    #sys.exit()
    # inpcrd = "benzene.inpcrd"
    # prmtop = "benzene.prmtop"
    # res_mask = ":1" # first residue is decoupled, can also just use solute cut
    # l_windows = np.arange(0.0, 1.0, 0.2)

    # TODO figure out testing environment
    # and installing ambertools along with jax, omm, and everything else

    #base_dir = "/mnt/home/betanc18/JAX-ParamOpt/Datasets/test_fe"

    if len(sys.argv) != 2 or sys.argv[1] not in ["jax", "amber", "analysis"]:
        sys.exit("Script should be run as \"amber_fe.py jax/amber/analysis\" for testing purposes")

    run_mode = sys.argv[1]

    if run_mode == "jax":
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        #from pymbar import testsystems, MBAR
        import openmm as omm
        import openmm.app as app
        from amber_helper import load_amber_ff, load_ffq_ff
        from amber_energy_v2 import amber_energy
        import parmed as pmd
        import jax_md
        #pymbar_test()

        test_lambda = 0.5
        timestep = 1e-3
        init_temp = 1e-3 # can set to low non-zero value e.g. 1e-3
        # TODO should this be a deterministic thermostat? nose hoover?
        ensemble = "NVT" # NVT or NPT? probably NPT for solvated, nvt for gas phase
        ffq_file = None
        ffq_ff = None
        nonbonded_method = "NoCutoff"
        charge_method = "GAFF"
        dr_threshold = 0.2
        dtype = jnp.float64
        cutoff = 0.8
        coul_scaling = "linear"
        vdw_scaling = None
        sc_mask = ":1" # only for solvated systems
        chg_mask = "" # only for decoupling step, convenient for single topology

        inpcrds = ["benzene.inpcrd", "benzene_solvated.inpcrd",]
        prmtops = ["benzene.prmtop", "benzene_decharged.prmtop", "benzene_decoupled.prmtop", 
                    "benzene_solvated.prmtop", "benzene_solvated_decharged.prmtop", "benzene_solvated_decoupled.prmtop"]
        # TODO for test
        # use the decharging leg of the gas structure

        # load parmed mask
        structure = pmd.load_file('benzene.prmtop', 'benzene.inpcrd')
        atom_mask = ":1"
        selected_atoms = pmd.amber.AmberMask(structure, atom_mask).Selection()
        #selected_atoms = pmd.amber.AmberMask.Selection(structure, atom_mask)
        vacuum_selected_atoms = jnp.array(selected_atoms, dtype=bool)
        print("vacuum parmed mask", vacuum_selected_atoms)

        vacuum_positions = structure.coordinates

        structure = pmd.load_file('benzene_solvated.prmtop', 'benzene_solvated.inpcrd')
        atom_mask = ":1"
        selected_atoms = pmd.amber.AmberMask(structure, atom_mask).Selection()
        #selected_atoms = pmd.amber.AmberMask.Selection(structure, atom_mask)
        solvated_selected_atoms = jnp.array(selected_atoms, dtype=bool)
        print("solvated parmed mask", solvated_selected_atoms)

        solvated_positions = structure.coordinates

        # TODO decide if this is safe
        # it seems like using np arrays in some situations can avoid tracing issues
        # but then will cause indexing issues because of dummy values and such in the nrg fn
        vacuum_positions = jnp.array(vacuum_positions)
        solvated_positions = jnp.array(solvated_positions)

        # steps to get dvdl values
        # have to write either softcore potential, or linear scaling approach
        # ideally this should be something like E0 to E1 * 1/lambda for more complex transformations
        # but in the simplest case is just a scaling factor, may be unstable for annhilation like ahfe

        # TODO some architecture considerations
        # need to figure out flexible approach, pmemd has single topology and 2 masks
        # as long as only absolute free energy is being run right now, having one mask and mimicing this will work:
        #     icfe = 1, ifsc = 1,
        #     timask1=':1', scmask1=':1',
        #     timask2='', scmask2='',
        # the appearing potential is handled slightly differently

        # so use parmed or omm to get flat mask of atoms
        # add vdw_scaling = [None, "linear", "softcore"] and coul_scaling = [None, "linear", "softcore"]
        # mask = scmask(nbr[:,0]) | scmask(nbr[:,1])
        # remember that sc/cc and sc/sc have different handling depending on gti_add_sc_switch
        # vdw = ~mask * vdw_vec and vdw_sc = mask * vdw_vec
        # same applies for 1-4 and other ixns

        # so compile energy function 4 times, then run each window
        # this is the decharging step

        vacuum_amber_ff = load_amber_ff(inpcrd_file="benzene.inpcrd", prmtop_file="benzene.prmtop", 
                        ffq_file=ffq_file, nonbonded_method="NoCutoff",
                        charge_method=charge_method, dr_threshold=dr_threshold, dtype=dtype, cutoff=cutoff)

        solvated_amber_ff = load_amber_ff(inpcrd_file="benzene_solvated.inpcrd", prmtop_file="benzene_solvated.prmtop", 
                        ffq_file=ffq_file, nonbonded_method="PME",
                        charge_method=charge_method, dr_threshold=dr_threshold, dtype=dtype, cutoff=cutoff)

        # solvated_amber_ff_de = load_amber_ff(inpcrd_file="benzene_solvated.inpcrd", prmtop_file="benzene_solvated_decoupled.prmtop", 
        #                 ffq_file=ffq_file, nonbonded_method="PME",
        #                 charge_method=charge_method, dr_threshold=dr_threshold, dtype=dtype, cutoff=cutoff)

        # amber_ff_decharged = load_amber_ff(inpcrd_file=None, prmtop_file=prmtops[1], 
        #                 ffq_file=ffq_file, nonbonded_method=nonbonded_method,
        #                 charge_method=charge_method, dr_threshold=dr_threshold, dtype=dtype, cutoff=cutoff)

        vacuum_chg_nrg_fn, vacuum_chg_amber_ff, vacuum_chg_body_fn, vacuum_chg_state = amber_energy(ff=vacuum_amber_ff, nonbonded_method="NoCutoff",
                                                charge_method=charge_method, ensemble=ensemble,
                                                timestep=timestep, init_temp=init_temp, ffq_ff=ffq_ff, debug=True, coul_scaling="linear")

        vacuum_vdw_nrg_fn, vacuum_vdw_amber_ff, vacuum_vdw_body_fn, vacuum_vdw_state = amber_energy(ff=vacuum_amber_ff, nonbonded_method="NoCutoff",
                                                charge_method=charge_method, ensemble=ensemble,
                                                timestep=timestep, init_temp=init_temp, ffq_ff=ffq_ff, debug=True, vdw_scaling="softcore", sc_mask=vacuum_selected_atoms)

        # solvated_chg_nrg_fn, solvated_chg_amber_ff, body_fn, state = amber_energy(ff=solvated_amber_ff, nonbonded_method="PME",
        #                                         charge_method=charge_method, ensemble=ensemble,
        #                                         timestep=timestep, init_temp=init_temp, ffq_ff=ffq_ff, coul_scaling="linear")

        # print(amber_energy.__closure__)
        
        # solvated_chg_nrg_fn, solvated_chg_amber_ff, body_fn, state = amber_energy(ff=solvated_amber_ff, nonbonded_method="PME",
        #                                         charge_method=charge_method, ensemble="NVT",
        #                                         timestep=timestep, init_temp=init_temp, ffq_ff=ffq_ff, debug=True)

        # print(amber_energy.__closure__)
        # print(solvated_chg_nrg_fn.__closure__)

        # solvated_chg_de_nrg_fn, solvated_chg_de_amber_ff, body_fn, state = amber_energy(ff=solvated_amber_ff, nonbonded_method="PME",
        #                                         charge_method="GAFF", ensemble="NVT",
        #                                         timestep=1e-3, init_temp=1e-3, ffq_ff=None, debug=True)

        from jax_md import space, simulate

        # def ff_test(ref_amber_ff):
        #     nrg_fn, amber_ff, body_fn, state, nbr_fn = amber_energy(ff=ref_amber_ff, nonbonded_method="PME",
        #                                         # charge_method="GAFF", ensemble="NVT",
        #                                         charge_method="GAFF", ensemble=None,
        #                                         #timestep=1e-3, init_temp=1e-3, ffq_ff=None, debug=True)
        #                                         timestep=1e-3, init_temp=1e-3, ffq_ff=None, debug=True, return_mode="simple")
        #     return nrg_fn, amber_ff, nbr_fn

        kB = 0.00831446267 # TODO need to fix this

        solvated_chg_nrg_fn, solvated_chg_amber_ff, body_fn, state, nbr_chg_fn = amber_energy(ff=solvated_amber_ff, nonbonded_method="PME",
                                                charge_method=charge_method, ensemble=None,
                                                timestep=timestep, init_temp=init_temp, ffq_ff=ffq_ff, coul_scaling="linear", return_mode="simple")
        # solvated_chg_nrg_fn, solvated_chg_amber_ff, nbr_chg_fn = ff_test(solvated_amber_ff)
        # solvated_chg_nb_list = nbr_chg_fn.allocate(solvated_chg_amber_ff.positions) # TODO can also try using external positions instead of ff, probably better policy
        
        # TODO this shouldnt be necessary
        #solvated_chg_nrg_fn = jax.jit(solvated_chg_nrg_fn)
        #box_vecs = solvated_chg_amber_ff.box_vectors # there is still the shift function in the energy function that may cause issues
        
        solvated_chg_nb_list = nbr_chg_fn.allocate(solvated_positions)
        disp_fn, shift_fn = space.periodic(solvated_chg_amber_ff.box_vectors)
        #periodic(box_vecs)

        solvated_chg_init_fn, solvated_chg_apply_fn = simulate.nvt_langevin(solvated_chg_nrg_fn, shift_fn, 1e-3, kT=1e-3*kB) # should probably be * kB
        solvated_chg_state = solvated_chg_init_fn(jax.random.PRNGKey(0), solvated_chg_amber_ff.positions, mass=solvated_chg_amber_ff.masses, kT=init_temp*kB, ff=solvated_chg_amber_ff, nbr_list=solvated_chg_nb_list)

        def solvated_chg_body_fn(i, state):
            state, ff, nbr_list = state
            nbr_list = nbr_list.update(state.position)
            state = solvated_chg_apply_fn(state, ff=ff, nbr_list=nbr_list)

            return state, ff, nbr_list

        #print(solvated_amber_ff)
        
        solvated_vdw_nrg_fn, _solvated_vdw_amber_ff, body_fn, state, nbr_vdw_fn = amber_energy(ff=solvated_amber_ff, nonbonded_method="PME",
                                                charge_method=charge_method, ensemble=None,
                                                timestep=timestep, init_temp=init_temp, ffq_ff=ffq_ff, vdw_scaling="softcore", sc_mask=solvated_selected_atoms, return_mode="simple")
        # solvated_vdw_nrg_fn, solvated_vdw_amber_ff, nbr_vdw_fn = ff_test(solvated_amber_ff)
        #solvated_vdw_nrg_fn, _solvated_vdw_amber_ff, nbr_vdw_fn = ff_test(solvated_amber_ff)
        #solvated_vdw_nb_list = nbr_vdw_fn.allocate(solvated_vdw_amber_ff.positions)
        solvated_vdw_nb_list = solvated_chg_nb_list # works
        #solvated_vdw_nb_list = nbr_vdw_fn.allocate(solvated_chg_amber_ff.positions) # doesn't work
        #solvated_vdw_nb_list = nbr_chg_fn.allocate(solvated_vdw_amber_ff.positions) # works
        #solvated_vdw_nb_list = nbr_chg_fn.allocate(solvated_chg_amber_ff.positions) # works
        solvated_vdw_amber_ff = solvated_chg_amber_ff
        #shift_fn = space.periodic(solvated_vdw_amber_ff.box_vectors)
        solvated_vdw_init_fn, solvated_vdw_apply_fn = simulate.nvt_langevin(solvated_vdw_nrg_fn, shift_fn, 1e-3, kT=1e-3*kB) # should probably be * kB
        solvated_vdw_state = solvated_vdw_init_fn(jax.random.PRNGKey(0), solvated_vdw_amber_ff.positions, mass=solvated_vdw_amber_ff.masses, kT=init_temp*kB, ff=solvated_vdw_amber_ff, nbr_list=solvated_vdw_nb_list)

        def solvated_vdw_body_fn(i, state):
            state, ff, nbr_list = state
            nbr_list = nbr_list.update(state.position)
            state = solvated_vdw_apply_fn(state, ff=ff, nbr_list=nbr_list) #TODO is passing debug like this valid?

            return state, ff, nbr_list




        # TODO next thing to try is mimicing cagri's code, don't use ensemble and just call those things externally
        # also try to separate neighbor logic where possible and rethink passing by closure and where to do it
        # also consider replacing pairs with non cell, non periodic nb list and then standardizing allocation logic
        # this is just amortized as setup time anyways
        # or just wrap these in functions to limit scope and force copying
        # also print ff object before and after and look for tracers, maybe this goes back to the helper file parser and tracing

        # 

        # inpcrd = app.AmberInpcrdFile(inpcrds[0])
        # positions = inpcrd.positions._value

        # print(positions)

        # vacuum tests
        nrg = vacuum_chg_nrg_fn(vacuum_chg_amber_ff.positions, vacuum_chg_amber_ff, vacuum_chg_amber_ff.nbr_list, debug=True, cl_lambda=lambdas[0])
        jax.debug.print("lambda 0 vacuum chg nrg {} {}", nrg["coul_pot"]/4.184, nrg["coul_14_pot"]/4.184)

        # nrg = vacuum_chg_nrg_fn(vacuum_chg_amber_ff.positions, vacuum_chg_amber_ff, vacuum_chg_amber_ff.nbr_list, debug=True, cl_lambda=lambdas[10])
        # jax.debug.print("nrg {} {}", nrg["coul_pot"]/4.184, nrg["coul_14_pot"]/4.184)

        nrg = vacuum_vdw_nrg_fn(vacuum_vdw_amber_ff.positions, vacuum_vdw_amber_ff, vacuum_vdw_amber_ff.nbr_list, debug=True, cl_lambda=lambdas[0])
        jax.debug.print("lambda 0 vacuum vdw nrg {} {} {} {}", nrg["lj_pot"]/4.184, nrg["lj_14_pot"]/4.184, nrg["SC_VDW"]/4.184, nrg["SC_14NB"]/4.184)

        # nrg = vacuum_vdw_nrg_fn(vacuum_vdw_amber_ff.positions, vacuum_vdw_amber_ff, vacuum_vdw_amber_ff.nbr_list, debug=True, cl_lambda=lambdas[10])
        # jax.debug.print("nrg {} {}", nrg["coul_pot"]/4.184, nrg["coul_14_pot"]/4.184)

        # solvated tests
        # from jax import make_jaxpr
        # nrg_jaxpr = make_jaxpr(solvated_chg_nrg_fn)(solvated_chg_amber_ff.positions, solvated_chg_amber_ff, solvated_chg_amber_ff.nbr_list, cl_lambda=lambdas[0])
        # #print(nrg_jaxpr)
        # with open("jaxpr_dump.txt", "w") as f:
        #     f.write(nrg_jaxpr.pretty_print())

        # nrg = solvated_chg_nrg_fn(solvated_chg_amber_ff.positions, solvated_chg_amber_ff, solvated_chg_amber_ff.nbr_list, debug=True, cl_lambda=lambdas[0])
        # jax.debug.print("nrg {} {}", nrg["coul_pot"]/4.184, nrg["coul_14_pot"]/4.184)

        # nrg = solvated_chg_nrg_fn(solvated_chg_amber_ff.positions, solvated_chg_amber_ff, solvated_chg_amber_ff.nbr_list, debug=True, cl_lambda=lambdas[10])
        # jax.debug.print("nrg {} {}", nrg["coul_pot"]/4.184, nrg["coul_14_pot"]/4.184)

        # nrg = solvated_vdw_nrg_fn(solvated_vdw_amber_ff.positions, solvated_vdw_amber_ff, solvated_vdw_amber_ff.nbr_list, debug=True, cl_lambda=lambdas[0])
        # jax.debug.print("lambda 0 solvated vdw nrg {} {}", nrg["coul_pot"]/4.184, nrg["coul_14_pot"]/4.184)

        # nrg = solvated_vdw_nrg_fn(solvated_vdw_amber_ff.positions, solvated_vdw_amber_ff, solvated_vdw_amber_ff.nbr_list, debug=True, cl_lambda=lambdas[10])
        # jax.debug.print("nrg {} {}", nrg["coul_pot"]/4.184, nrg["coul_14_pot"]/4.184)

        # new test with all 2 of the transformations at 2 different lambdas
        # TODO also need to capture dv/dl values w/ debug false
        # e.g. jax.value_and_grad(argnums=4)

        nrg = solvated_chg_nrg_fn(solvated_chg_amber_ff.positions, solvated_chg_amber_ff, solvated_chg_nb_list, debug=True, cl_lambda=lambdas[0])
        jax.debug.print("lambda 0 solvated chg nrg {} {}", nrg["coul_pot"]/4.184, nrg["coul_14_pot"]/4.184)

        # nrg = solvated_chg_nrg_fn(solvated_chg_amber_ff.positions, solvated_chg_amber_ff, solvated_chg_nb_list, debug=True, cl_lambda=lambdas[10])
        # jax.debug.print("nrg {} {}", nrg["coul_pot"]/4.184, nrg["coul_14_pot"]/4.184)

        # print(solvated_vdw_state)
        # print(solvated_positions)
        # sys.exit()

        # using solvated_positions doesn't work
        # using solvated_vdw_state.position works

        nrg = solvated_vdw_nrg_fn(solvated_vdw_state.position, solvated_vdw_amber_ff, solvated_vdw_nb_list, debug=True, cl_lambda=lambdas[0])
        jax.debug.print("lambda 0 solvated vdw nrg {} {} {} {}", nrg["lj_pot"]/4.184, nrg["lj_14_pot"]/4.184, nrg["SC_VDW"]/4.184, nrg["SC_14NB"]/4.184)

        sys.exit()

        # nrg = solvated_vdw_nrg_fn(solvated_positions, solvated_vdw_amber_ff, solvated_vdw_nb_list, debug=True, cl_lambda=lambdas[10])
        # jax.debug.print("nrg {} {}", nrg["lj_pot"]/4.184, nrg["lj_14_pot"]/4.184)

        # print(type(solvated_vdw_state.position))
        # print(type(solvated_positions))

        # run 100ps NVT md at each lambda value determined by 12 point gaussian quadrature (or 6 to start)
        # sample dv/dl every x steps and average
        # TODO something to consider
        # this is a lot of simulations and really should use multiple GPUs
        # it might be worth setting up the full simulation interface with logs amber style
        # and then using this process to run a series of subprocesses or a pmap/shard map
        # as far as what will work with the optimization machinery, that's also an important question

        # perform TI on all averages to get deltaG
        def run_simulation(nrg_fn, body_fn, state, ff, nbr_list, cl_lambda, num_steps=10, output_freq=2):
            dv_dl_vals = []
            body_fn_jit = jax.jit(body_fn)
            nrg_fn = jax.jit(nrg_fn)
            for i in range(int(num_steps/output_freq)):
                new_state, new_ff, nbr_list = jax.lax.fori_loop(0, output_freq, body_fn_jit, (state, ff, nbr_list))
                # TODO this may be bad logic, new_ff is the ff being used but nbr list is separate, taken from test_v2.py
                if nonbonded_method == "PME" and nbr_list.did_buffer_overflow:
                    print('Neighbor list overflowed, reallocating.')
                    #amber_ff = dataclasses.replace(amber_ff, nbr_list=ff.nbr_list.allocate(state.positions))
                    nbr_list = nbr_list.allocate(state.position)
                else:
                    state = new_state
                    #step += 1
                
                pE = nrg_fn(state.position, ff, nbr_list)
                kE = jax_md.quantity.kinetic_energy(momentum=state.momentum, mass=state.mass)
                temp = jax_md.quantity.temperature(momentum=state.momentum, mass=state.mass)/kB
                current_step = (i + 1) * output_freq
                jax.debug.print("{step}, {pE}, {kE}, {pEkE}, {temp}", step=current_step, pE=pE, kE=kE, pEkE=(pE+kE), temp=temp)
            return

        jax.debug.print("vacuum chg simulation")
        run_simulation(vacuum_chg_nrg_fn, vacuum_chg_body_fn, vacuum_chg_state, vacuum_chg_amber_ff, None)
        jax.debug.print("vacuum vdw simulation")
        run_simulation(vacuum_vdw_nrg_fn, vacuum_vdw_body_fn, vacuum_vdw_state, vacuum_vdw_amber_ff, None)

        #res = jax.jit(solvated_chg_body_fn)(0, (solvated_chg_state, solvated_chg_amber_ff, solvated_chg_nb_list))

        jax.debug.print("solvated chg simulation")
        run_simulation(solvated_chg_nrg_fn, solvated_chg_body_fn, solvated_chg_state, solvated_chg_amber_ff, solvated_chg_nb_list)
        jax.debug.print("solvated vdw simulation")
        run_simulation(solvated_vdw_nrg_fn, solvated_vdw_body_fn, solvated_vdw_state, solvated_vdw_amber_ff, solvated_vdw_nb_list)

        sys.exit()

    elif run_mode == "amber":
        amber_imports = None

    elif run_mode == "analysis":
        import re
        def parse_dvdl(filepath):
            with open(filepath) as f:
                lines = f.readlines()
            for line in reversed(lines):
                if "DV/DL" in line:
                    match = re.search(r'DV/DL\s*=\s*(-?\d+\.\d+)', line)
                    if match:
                        return float(match.group(1))
            raise ValueError(f"DV/DL not found in {filepath}")

        # vacuum decharging dv/dl
        vacuum_decharging_dvdl = []
        for idx, lam in enumerate(lambdas):
            #lam_str = f"{lam:.2f}"
            dvdl = parse_dvdl(f"vacuum/mdout_prod_{idx}_v0_decharged.out") # TODO should this be 0 or 1?
            vacuum_decharging_dvdl.append(dvdl)
        vacuum_decharging_dvdl = np.array(vacuum_decharging_dvdl)

        # vacuum decoupling dv/dl
        vacuum_decoupling_dvdl = []
        for idx, lam in enumerate(lambdas):
            #lam_str = f"{lam:.2f}"
            dvdl = parse_dvdl(f"vacuum/mdout_prod_{idx}_v0_decoupled.out") # TODO should this be 0 or 1?
            vacuum_decoupling_dvdl.append(dvdl)
        vacuum_decoupling_dvdl = np.array(vacuum_decoupling_dvdl)

        # solvent decharging dv/dl
        solvent_decharging_dvdl = []
        for idx, lam in enumerate(lambdas):
            #lam_str = f"{lam:.2f}"
            dvdl = parse_dvdl(f"solvated/mdout_prod_{idx}_v0_decharged.out") # TODO should this be 0 or 1?
            solvent_decharging_dvdl.append(dvdl)
        solvent_decharging_dvdl = np.array(solvent_decharging_dvdl)

        # solvent decoupling dv/dl
        solvent_decoupling_dvdl = []
        for idx, lam in enumerate(lambdas):
            #lam_str = f"{lam:.2f}"
            dvdl = parse_dvdl(f"solvated/mdout_prod_{idx}_v0_decoupled.out") # TODO should this be 0 or 1?
            solvent_decoupling_dvdl.append(dvdl)
        solvent_decoupling_dvdl = np.array(solvent_decoupling_dvdl)

        # TODO is this correct? it intuitively seems like it should be
        # solvent - solute represents the change to hydrate
        deltaG_vacuum_decharging = np.sum(weights * np.array(vacuum_decharging_dvdl))
        deltaG_vacuum_decoupling = np.sum(weights * np.array(vacuum_decoupling_dvdl))
        deltaG_solvent_decharging = np.sum(-1.0 * weights * np.array(solvent_decharging_dvdl))
        deltaG_solvent_decoupling = np.sum(-1.0 * weights * np.array(solvent_decoupling_dvdl))

        print("deltaG Vacuum Decharge:", deltaG_vacuum_decharging, vacuum_decharging_dvdl)
        print("deltaG Vacuum Decouple:", deltaG_vacuum_decoupling, vacuum_decoupling_dvdl)
        print("deltaG Solvent Decharge:", deltaG_solvent_decharging, solvent_decharging_dvdl)
        print("deltaG Solvent Decouple:", deltaG_solvent_decoupling, solvent_decoupling_dvdl)
        print("deltaG total", deltaG_vacuum_decharging + deltaG_vacuum_decoupling + deltaG_solvent_decharging + deltaG_solvent_decoupling)


        

        # deltaG_vacuum = np.sum(weights * np.array(dvdl_values))
        # print("dvdl vacuum", dvdl_values)
        # print(f"deltaG vacuum = {deltaG_vacuum:.2f} kcal/mol")

        # dvdl_values = []
        # for idx, lam in enumerate(lambdas):
        #     #lam_str = f"{lam:.2f}"
        #     dvdl = parse_dvdl(f"mdout_prod_{idx}_v0_solvated.out") # TODO should this be 0 or 1?
        #     dvdl_values.append(dvdl)

        # Trapezoidal integration over lambda
        # deltaG = sum(
        #     0.5 * (dvdl_values[i] + dvdl_values[i + 1]) * (lambdas[i + 1] - lambdas[i])
        #     for i in range(len(lambdas) - 1)
        # )
        # deltaG_solvated = np.sum(weights * np.array(dvdl_values))
        # print("dvdl solvated", dvdl_values)
        # print(f"deltaG solvated = {deltaG_solvated:.2f} kcal/mol")

        # print("overall value (invert weights?)", deltaG_vacuum - deltaG_solvated)

        #################
        # alternative with gaussian quadrature, requires specific lambdas
        #deltaG = sum(w * dvdl for dvdl, w in zip(dvdl_values, weights))


        #################

        # TODO for benzene in water, this value should be something like -.87 to -.9 ish

        #print(dvdl_values)

        # TODO add mbar here as well

        sys.exit()


    # for each lambda window
    # compute the energy with lambda as an additional parameter
    # collect frames from a simulation for some number of steps

    # maybe start without PME support as i'm not sure how slowly scaling that potential works
    # shouldnt require modification
    # maybe a hacky way to do the scaling by modifying epsilon and charges in ff object?
    # it's also really easy to add a default scaling factor of 1 to the force field file and then explicitly modify it right before
    # the simulation to change it to lambda

    # mbar = MBAR(u_kn, N_k)
    # remember to use B * Uk(Xn) for boltzman weighted energies where B = 1/kBT


    # TODO pysander reference?
    # convert to mol2 with obabel or maybe RDKit
    obabel_command_string = "obabel -:\"c1ccccc1\" -O benzene.mol2 --gen3d"

    os.system(obabel_command_string)

    # Generate GAFF parameters
    antechamber_command_string = "antechamber -i benzene.mol2 -fi mol2 -o benzene_gaff.mol2 -fo mol2 -c bcc -s 2 -nc 0"

    os.system(antechamber_command_string)

    # generate frcmod file (missing parameters)
    parmchk_command_string = "parmchk2 -i benzene_gaff.mol2 -f mol2 -o benzene.frcmod"

    os.system(parmchk_command_string)

    #sys.exit()

    leap_file_text = """source leaprc.gaff
source leaprc.water.tip3p

mol = loadmol2 benzene_gaff.mol2
loadamberparams benzene.frcmod

saveamberparm mol benzene.prmtop benzene.inpcrd

solvated = copy mol

solvatebox solvated TIP3PBOX 12.0
saveamberparm solvated benzene_solvated.prmtop benzene_solvated.inpcrd
quit
"""

# don't need to remove solvent, need to turn off interactions
# solvent = copy solvated
# remove solvent solvent.1
# saveamberparm solvent benzene_decoupled.prmtop benzene_decoupled.inpcrd
# quit

    with open(f'tleap.in', 'w') as f:
        f.write(leap_file_text)

    leap_command_string = "tleap -f tleap.in"

    os.system(leap_command_string)

    #sys.exit()

#     mdin_base = """sander run for lambda window
# &cntrl
#     imin=0, irest=0, ntx=1,
#     nstlim=50000, dt=0.002,
#     ntc=2, ntf=2,
#     temp0=300.0, tempi=300.0, ntt=3, gamma_ln=1.0,
#     ntpr=500, ntwx=500, ntwr=5000,
#     cut=10.0,
#     ntb=1, ntp=0,
#     icfe=1, ifsc=1, clambda=REPLACE_LAMBDA,
#     crgmask=':1', scmask=':1',
#     scalpha=0.5, scbeta=12.0,
#     logdvdl=1
# /
# """

# original template, doesn't work with dual topology because of softcore
#     mdin_equil_template = """Equilibration run
# &cntrl
#   imin=0, irest=0, ntx=1,
#   nstlim=50000, dt=0.002,
#   ntc=2, ntf=1,
#   temp0=300.0, tempi=300.0, ntt=3, gamma_ln=1.0,
#   ntpr=5000, ntwx=0, ntwr=0,
#   cut=10.0,
#   ntb=REPLACE_NTB, ntp=0,
#   icfe=1, ifsc=1, clambda=REPLACE_LAMBDA,
#   scmask=":1",
#   crgmask=":1",
#   scalpha=0.5, scbeta=12.0,
# /
# """

#     mdin_prod_template = """Production run
# &cntrl
#   imin=0, irest=1, ntx=5,
#   nstlim=250000, dt=0.002,
#   ntc=2, ntf=1,
#   temp0=300.0, tempi=300.0, ntt=3, gamma_ln=1.0,
#   ntpr=500, ntwx=500, ntwr=5000,
#   cut=10.0,
#   ntb=REPLACE_NTB, ntp=0,
#   icfe=1, ifsc=1, clambda=REPLACE_LAMBDA,
#   scmask=":1",
#   crgmask=":1",
#   scalpha=0.5, scbeta=12.0,
# /
# """

# nst lim should be 50,000 and 250,000, reducing for speed

    mdin_equil_template = """Equilibration run
&cntrl
  imin=0, irest=0, ntx=1,
  nstlim=100000, dt=0.001,
  ntc=2, ntf=1,
  temp0=300.0, tempi=300.0, ntt=3, gamma_ln=1.0,
  ntpr=5000, ntwx=0, ntwr=0,
  cut=10.0,
  ntb=REPLACE_NTB
  icfe=1, clambda=REPLACE_LAMBDA,
  REPLACE_SC
/
"""

    mdin_prod_template = """Production run
&cntrl
  imin=0, irest=1, ntx=5,
  nstlim=1000000, dt=0.001,
  ntc=2, ntf=1,
  temp0=300.0, tempi=300.0, ntt=3, gamma_ln=1.0,
  ntpr=500, ntwx=500, ntwr=5000,
  cut=10.0,
  ntb=REPLACE_NTB
  icfe=1, clambda=REPLACE_LAMBDA,
  REPLACE_SC
/
"""
#   ifsc=1,
#   scmask=":1",

#     mdin_base = """sander run for lambda window
# &cntrl
#     imin=0, irest=0, ntx=1,
#     nstlim=50000, dt=0.002,
#     ntc=2, ntf=1,
#     temp0=300.0, tempi=300.0, ntt=3, gamma_ln=1.0,
#     ntpr=500, ntwx=500, ntwr=5000,
#     cut=10.0,
#     ntb=1, ntp=0,
#     icfe=1, ifsc=1, clambda=REPLACE_LAMBDA,
#     scmask=":1",
#     crgmask=":1",
#     scalpha=0.5, scbeta=12.0
# /
# """

    ####################################################################################

    # just deleting the solute for the decoupled version doesn't work, going to try switching off the interactions
    import parmed as pmd

    # First decoupling the gas phase interactions

    def parmed_decouple(prmtop, inpcrd, decharge_output, decouple_output):
        # Load the full solvated system
        parm = pmd.load_file(prmtop, inpcrd)

        # Zero charges and LJ on solute atoms (assume residue is BEN) [':1'] or [':BEN'] i think
        # solute = parm[':BEN']
        # for atom in solute:
        #     atom.charge = 0.0
        #     atom.epsilon = 0.0
        #     atom.rmin = 0.0

        first_residue = parm.residues[0]
        atom_indices = [atom.idx for atom in first_residue.atoms]

        for idx in atom_indices:
            parm.atoms[idx].charge = 0.0
            #parm.atoms[idx].epsilon = 0.0 #this doesn't work
            #parm.atoms[idx].rmin = 0.0

        parm.save(decharge_output, overwrite=True)

        # Get LJ A and B coefficient arrays
        lj_a = parm.parm_data['LENNARD_JONES_ACOEF']
        lj_b = parm.parm_data['LENNARD_JONES_BCOEF']

        # Get atom types and their mapping to LJ params
        atom_types = parm.parm_data['ATOM_TYPE_INDEX']

        type_indices = [atom_types[atom.idx] for atom in first_residue.atoms]

        # Convert to 0-based indices
        type_indices = [i - 1 for i in type_indices]

        # Zero out all pairwise A and B values involving these atom types
        #print(parm.parm_data['NONBONDED_PARM_INDEX'])
        n_types = len(parm.parm_data['NONBONDED_PARM_INDEX'])  # usually square of number of types
        nbidx = parm.parm_data['NONBONDED_PARM_INDEX']  # Matrix of indices into LJ A/B
        n_atom_types = int(np.sqrt(len(nbidx)))

        # sanity check to print all combinations of types
        # print("n_atom_types", n_atom_types)
        # # solute_types = [1,2]
        # for i in range(1,n_atom_types+1):
        #     #for j in solute_types:
        #     for j in range(1,n_atom_types+1):
        #         #i = i + 1
        #         idx_ij = (i - 1) * n_atom_types + (j-1)
        #         #print(i,j,idx_ij)
        #         idx = parm.parm_data["NONBONDED_PARM_INDEX"][idx_ij] - 1
        #         ai = parm.parm_data["LENNARD_JONES_ACOEF"][idx]
        #         bi = parm.parm_data["LENNARD_JONES_BCOEF"][idx]
        #         print(f"i: {i} j: {j} idx: {idx} ai: {ai} bi: {bi}")

        # this removes all pairs between solute-solute and solute-solvent
        for i in type_indices:
            for j in range(n_atom_types):
                idx1 = nbidx[i * n_atom_types + j] - 1  # 1-based to 0-based
                idx2 = nbidx[j * n_atom_types + i] - 1
                lj_a[idx1] = 0.0
                lj_b[idx1] = 0.0
                lj_a[idx2] = 0.0
                lj_b[idx2] = 0.0

        # in reality, the correct approach may involve leaving solute-solute intact
        # print(type_indices)
        # for i in type_indices:
        #     print("ti", i)
        #     for j in [2,3]:
        #         idx1 = nbidx[i * n_atom_types + j] - 1  # 1-based to 0-based
        #         idx2 = nbidx[j * n_atom_types + i] - 1
        #         lj_a[idx1] = 0.0
        #         lj_b[idx1] = 0.0
        #         lj_a[idx2] = 0.0
        #         lj_b[idx2] = 0.0



        # this approach doesn't seem to work completely
        # from parmed.tools.actions import changeLJPair

        # print("type indices", type_indices)

        # for i in type_indices:
        #     for j in range(n_atom_types):
        #         type1 = atom_types[i]
        #         type2 = atom_types[j]
        #         action = changeLJPair(parm, f'@%{type1}', f'@%{type2}', 0.0, 0.0)
        #         action.execute()

        # output = "test_" + output
        
        parm.save(decouple_output, overwrite=True)

        # sanity check to print all combinations of types
        # print("n_atom_types", n_atom_types)
        # # solute_types = [1,2]
        # for i in range(1,n_atom_types+1):
        #     #for j in solute_types:
        #     for j in range(1,n_atom_types+1):
        #         #i = i + 1
        #         idx_ij = (i - 1) * n_atom_types + (j-1)
        #         #print(i,j,idx_ij)
        #         idx = parm.parm_data["NONBONDED_PARM_INDEX"][idx_ij] - 1
        #         ai = parm.parm_data["LENNARD_JONES_ACOEF"][idx]
        #         bi = parm.parm_data["LENNARD_JONES_BCOEF"][idx]
        #         print(f"i: {i} j: {j} idx: {idx} ai: {ai} bi: {bi}")

        # print(output)
        #sys.exit()


    # Save as decoupled prmtop
    # parm.save("benzene_decoupled.prmtop", overwrite=True)

    # # Then decoupling the condensed phase interactions

    # # Load the full solvated system
    # parm = pmd.load_file("benzene_solvated.prmtop", "benzene_solvated.inpcrd")

    # # Zero charges and LJ on solute atoms (assume residue is BEN)
    # solute = parm[':BEN']
    # for atom in solute:
    #     atom.charge = 0.0
    #     atom.epsilon = 0.0
    #     atom.rmin = 0.0

    # # Save as decoupled prmtop
    # parm.save("benzene_solvated_decoupled.prmtop", overwrite=True)

    parmed_decouple("benzene.prmtop", "benzene.inpcrd", "benzene_decharged.prmtop", "benzene_decoupled.prmtop")
    parmed_decouple("benzene_solvated.prmtop", "benzene_solvated.inpcrd", "benzene_solvated_decharged.prmtop", "benzene_solvated_decoupled.prmtop")

    #sys.exit()

    ##############################################

    #replace lambda with actual lambda for window

    #lambdas = [0.1, 0.3, 0.5, 0.7, 0.9]
    #lambdas = np.arange(0.1, 0.9)
    #template = open('mdin_template.in').read()
    #template = mdin_base

    # for idx, lam in enumerate(lambdas):
    #     with open(f'lambda_{idx}.in', 'w') as f:
    #         f.write(template.replace("REPLACE_LAMBDA", f"{lam:.2f}"))

    #sys.exit()

#     sander_command_string = """sander -O -p benzene_solvated.prmtop -c benzene_solvated.inpcrd \
#     -i test.in -o test.out -r test.rst \
#     -x test.nc -inf test.mdinfo
# # """

    #mpirun -np 2 sander.MPI -ng 2 -groupfile groupfiles/groupfile_$lam

    #create necessary directory structure
    os.mkdir("vacuum")
    os.mkdir("vacuum/in_files")
    os.mkdir("vacuum/restrt_files")
    os.mkdir("vacuum/group_files")
    os.mkdir("vacuum/slm_files")
    os.mkdir("vacuum/slm_out_files")
    os.mkdir("solvated")
    os.mkdir("solvated/in_files")
    os.mkdir("solvated/restrt_files")
    os.mkdir("solvated/group_files")
    os.mkdir("solvated/slm_files")
    os.mkdir("solvated/slm_out_files")

    # subprocess based approach for efficiency, move this import TODO
    import subprocess

    for idx, lam in enumerate(lambdas):
        # generate mdin file
        
        # gas phase decharging
        with open(f'vacuum/in_files/equil_{idx}_decharged.in', 'w') as f:
            #it may not be best to limit precision, or to set it higher
            #template = mdin_equil_template.replace("REPLACE_LAMBDA", f"{lam:.2f}")
            template = mdin_equil_template.replace("REPLACE_LAMBDA", f"{lam}")
            template = template.replace("REPLACE_NTB", "0, ntp=0,")
            template = template.replace("REPLACE_SC", "ifsc=0,")
            f.write(template)

        with open(f'vacuum/in_files/prod_{idx}_decharged.in', 'w') as f:
            template = mdin_prod_template.replace("REPLACE_LAMBDA", f"{lam}")
            template = template.replace("REPLACE_NTB", "0, ntp=0,")
            template = template.replace("REPLACE_SC", "ifsc=0,")
            f.write(template)

        # gas phase vdw decoupling

        with open(f'vacuum/in_files/equil_{idx}_decoupled.in', 'w') as f:
            template = mdin_equil_template.replace("REPLACE_LAMBDA", f"{lam}")
            template = template.replace("REPLACE_NTB", "0, ntp=0,")
            template = template.replace("REPLACE_SC", "ifsc=1, scmask=\":1\",")
            f.write(template)

        with open(f'vacuum/in_files/prod_{idx}_decoupled.in', 'w') as f:
            template = mdin_prod_template.replace("REPLACE_LAMBDA", f"{lam}")
            template = template.replace("REPLACE_NTB", "0, ntp=0,")
            template = template.replace("REPLACE_SC", "ifsc=1, scmask=\":1\",")
            f.write(template)

        # solvent phase decharging
        with open(f'solvated/in_files/equil_{idx}_decharged.in', 'w') as f:
            template = mdin_equil_template.replace("REPLACE_LAMBDA", f"{lam}")
            template = template.replace("REPLACE_NTB", "2, ntp = 1, nscm = 1000, pres0 = 1.0, taup = 2.0,") # was originally 1 before adding pressure control
            template = template.replace("REPLACE_SC", "ifsc=0,")
            f.write(template)

        with open(f'solvated/in_files/prod_{idx}_decharged.in', 'w') as f:
            template = mdin_prod_template.replace("REPLACE_LAMBDA", f"{lam}")
            template = template.replace("REPLACE_NTB", "2, ntp = 1, nscm = 1000, pres0 = 1.0, taup = 2.0,")
            template = template.replace("REPLACE_SC", "ifsc=0,")
            f.write(template)

        # solvent phase vdw decoupling
        with open(f'solvated/in_files/equil_{idx}_decoupled.in', 'w') as f:
            template = mdin_equil_template.replace("REPLACE_LAMBDA", f"{lam}")
            template = template.replace("REPLACE_NTB", "2, ntp = 1, nscm = 1000, pres0 = 1.0, taup = 2.0,")
            template = template.replace("REPLACE_SC", "ifsc=1, scmask=\":1\",")
            f.write(template)

        with open(f'solvated/in_files/prod_{idx}_decoupled.in', 'w') as f:
            template = mdin_prod_template.replace("REPLACE_LAMBDA", f"{lam}")
            template = template.replace("REPLACE_NTB", "2, ntp = 1, nscm = 1000, pres0 = 1.0, taup = 2.0,")
            template = template.replace("REPLACE_SC", "ifsc=1, scmask=\":1\",")
            f.write(template)
        
        # groupfile = f"-O -i lambda_equil_{idx}.in -p benzene_solvated.prmtop -c benzene_solvated.inpcrd -o mdout_equil_{idx}_v0.out -r restrt_equil_{idx}_v0.ncrst\n" + \
        #         f"-O -i lambda_equil_{idx}.in -p benzene_decoupled.prmtop -c benzene_solvated.inpcrd -o mdout_equil_{idx}_v1.out -r restrt_equil_{idx}_v1.ncrst"

        # generate group files
        # TODO same coordinates for both?
        
        # gas phase decharging
        groupfile = f"-O -i in_files/equil_{idx}_decharged.in -p ../benzene.prmtop -c ../benzene.inpcrd -o mdout_equil_{idx}_v0_decharged.out -r restrt_files/restrt_equil_{idx}_v0_decharged.ncrst\n" + \
                f"-O -i in_files/equil_{idx}_decharged.in -p ../benzene_decharged.prmtop -c ../benzene.inpcrd -o mdout_equil_{idx}_v1_decharged.out -r restrt_files/restrt_equil_{idx}_v1_decharged.ncrst"
        with open(f"vacuum/group_files/groupfile_{idx}_equil_decharged", "w") as f:
            f.write(groupfile)

        groupfile = f"-O -i in_files/prod_{idx}_decharged.in -p ../benzene.prmtop -c restrt_files/restrt_equil_{idx}_v0_decharged.ncrst -o mdout_prod_{idx}_v0_decharged.out -r restrt_files/restrt_prod_{idx}_v0_decharged.ncrst\n" + \
                f"-O -i in_files/prod_{idx}_decharged.in -p ../benzene_decharged.prmtop -c restrt_files/restrt_equil_{idx}_v1_decharged.ncrst -o mdout_prod_{idx}_v1_decharged.out -r restrt_files/restrt_prod_{idx}_v1_decharged.ncrst"
        with open(f"vacuum/group_files/groupfile_{idx}_prod_decharged", "w") as f:
            f.write(groupfile)

        # gas phase vdw decoupling
        groupfile = f"-O -i in_files/equil_{idx}_decoupled.in -p ../benzene_decharged.prmtop -c ../benzene.inpcrd -o mdout_equil_{idx}_v0_decoupled.out -r restrt_files/restrt_equil_{idx}_v0_decoupled.ncrst\n" + \
                f"-O -i in_files/equil_{idx}_decoupled.in -p ../benzene_decoupled.prmtop -c ../benzene.inpcrd -o mdout_equil_{idx}_v1_decoupled.out -r restrt_files/restrt_equil_{idx}_v1_decoupled.ncrst"
        with open(f"vacuum/group_files/groupfile_{idx}_equil_decoupled", "w") as f:
            f.write(groupfile)

        groupfile = f"-O -i in_files/prod_{idx}_decoupled.in -p ../benzene_decharged.prmtop -c restrt_files/restrt_equil_{idx}_v0_decoupled.ncrst -o mdout_prod_{idx}_v0_decoupled.out -r restrt_files/restrt_prod_{idx}_v0_decoupled.ncrst\n" + \
                f"-O -i in_files/prod_{idx}_decoupled.in -p ../benzene_decoupled.prmtop -c restrt_files/restrt_equil_{idx}_v1_decoupled.ncrst -o mdout_prod_{idx}_v1_decoupled.out -r restrt_files/restrt_prod_{idx}_v1_decoupled.ncrst"
        with open(f"vacuum/group_files/groupfile_{idx}_prod_decoupled", "w") as f:
            f.write(groupfile)

        # condensed phase decharging
        groupfile = f"-O -i in_files/equil_{idx}_decharged.in -p ../benzene_solvated.prmtop -c ../benzene_solvated.inpcrd -o mdout_equil_{idx}_v0_decharged.out -r restrt_files/restrt_equil_{idx}_v0_decharged.ncrst\n" + \
                f"-O -i in_files/equil_{idx}_decharged.in -p ../benzene_solvated_decharged.prmtop -c ../benzene_solvated.inpcrd -o mdout_equil_{idx}_v1_decharged.out -r restrt_files/restrt_equil_{idx}_v1_decharged.ncrst"
        with open(f"solvated/group_files/groupfile_{idx}_equil_decharged", "w") as f:
            f.write(groupfile)

        groupfile = f"-O -i in_files/prod_{idx}_decharged.in -p ../benzene_solvated.prmtop -c restrt_files/restrt_equil_{idx}_v0_decharged.ncrst -o mdout_prod_{idx}_v0_decharged.out -r restrt_files/restrt_prod_{idx}_v0_decharged.ncrst\n" + \
                f"-O -i in_files/prod_{idx}_decharged.in -p ../benzene_solvated_decharged.prmtop -c restrt_files/restrt_equil_{idx}_v1_decharged.ncrst -o mdout_prod_{idx}_v1_decharged.out -r restrt_files/restrt_prod_{idx}_v1_decharged.ncrst"
        with open(f"solvated/group_files/groupfile_{idx}_prod_decharged", "w") as f:
            f.write(groupfile)

        # condensed phase vdw decoupling
        groupfile = f"-O -i in_files/equil_{idx}_decoupled.in -p ../benzene_solvated_decharged.prmtop -c ../benzene_solvated.inpcrd -o mdout_equil_{idx}_v0_decoupled.out -r restrt_files/restrt_equil_{idx}_v0_decoupled.ncrst\n" + \
                f"-O -i in_files/equil_{idx}_decoupled.in -p ../benzene_solvated_decoupled.prmtop -c ../benzene_solvated.inpcrd -o mdout_equil_{idx}_v1_decoupled.out -r restrt_files/restrt_equil_{idx}_v1_decoupled.ncrst"
        with open(f"solvated/group_files/groupfile_{idx}_equil_decoupled", "w") as f:
            f.write(groupfile)
        
        groupfile = f"-O -i in_files/prod_{idx}_decoupled.in -p ../benzene_solvated_decharged.prmtop -c restrt_files/restrt_equil_{idx}_v0_decoupled.ncrst -o mdout_prod_{idx}_v0_decoupled.out -r restrt_files/restrt_prod_{idx}_v0_decoupled.ncrst\n" + \
                f"-O -i in_files/prod_{idx}_decoupled.in -p ../benzene_solvated_decoupled.prmtop -c restrt_files/restrt_equil_{idx}_v1_decoupled.ncrst -o mdout_prod_{idx}_v1_decoupled.out -r restrt_files/restrt_prod_{idx}_v1_decoupled.ncrst"
        with open(f"solvated/group_files/groupfile_{idx}_prod_decoupled", "w") as f:
            f.write(groupfile)

        #sys.exit()

        # cmd = [
        #     "sander", "-O",
        #     "-p", "benzene_solvated.prmtop",
        #     "-c", "benzene_solvated.inpcrd",
        #     "-i", f"lambda_{idx}.in",
        #     "-o", f"lambda_{idx}.out",
        #     "-r", f"lambda_{idx}.rst",
        #     "-x", f"lambda_{idx}.nc",
        #     "-inf", f"lambda_{idx}.mdinfo"
        # ]

        # subprocess.run(cmd, check=True)
        print(f"Running lambda = {idx}")

        # TODO how to run, wait, run, wait?

        slurm_template = """#!/bin/bash --login
#SBATCH --job-name=ti_lambda_REPLACE_LAMBDA
#SBATCH --output=./slm_out_files/lambda_REPLACE_LAMBDA.out
#SBATCH --error=./slm_out_files/lambda_REPLACE_LAMBDA.err
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --time=2:00:00
#SBATCH --account=hmakmm
#SBATCH --constraint=amr

module purge
module load Amber

mpirun -np 2 sander.MPI -ng 2 -groupfile ./group_files/groupfile_REPLACE_LAMBDA

scontrol show job $SLURM_JOB_ID
module load powertools
js -j $SLURM_JOB_ID
"""

        
        equil_script_decharged = f'slm_files/{idx}_equil_decharged.slm'
        prod_script_decharged = f'slm_files/{idx}_prod_decharged.slm'
        equil_script_decoupled = f'slm_files/{idx}_equil_decoupled.slm'
        prod_script_decoupled = f'slm_files/{idx}_prod_decoupled.slm'
        
        for state_dir in ["vacuum/", "solvated/"]:
            with open(state_dir + equil_script_decharged, 'w') as f:
                f.write(slurm_template.replace("REPLACE_LAMBDA", f"{idx}_equil_decharged"))

            with open(state_dir + prod_script_decharged, 'w') as f:
                f.write(slurm_template.replace("REPLACE_LAMBDA", f"{idx}_prod_decharged"))

            with open(state_dir + equil_script_decoupled, 'w') as f:
                f.write(slurm_template.replace("REPLACE_LAMBDA", f"{idx}_equil_decoupled"))

            with open(state_dir + prod_script_decoupled, 'w') as f:
                f.write(slurm_template.replace("REPLACE_LAMBDA", f"{idx}_prod_decoupled"))

        # submit all jobs
        def submit_slurm(equil_script, prod_script):
                result = subprocess.run(['sbatch', equil_script], capture_output=True, text=True)
                # example output: "Submitted batch job 123456"
                #print("sbatch output", result)
                print(result.stdout, end="")
                job_id = result.stdout.strip().split()[-1]
                result = subprocess.run([
                    'sbatch', f'--dependency=afterok:{job_id}', prod_script
                ])
                return result
                #return job_id
        
        for state_dir in ["vacuum/", "solvated/"]:
            os.chdir(state_dir)
            result = submit_slurm(equil_script_decharged, prod_script_decharged)
            result = submit_slurm(equil_script_decoupled, prod_script_decoupled)
            os.chdir("../")

        #os.system(f"sbatch lambda_{idx}.slm")

        # cmd = ["mpirun", "-np", "2", "sander.MPI", "-ng", "2", "-groupfile", f"groupfile_{idx}"]
        # mpirun -np 2 sander.MPI -ng 2 -groupfile groupfile_0
        # subprocess.run(cmd, check=True)

    sys.exit()

    # TODO good HFE reference for benzene is: -0.87 kcal/mol (Mobley et al., FreeSolv, 2014) (double check this)
    # TODO add amber edgembar or ti to compare and then jax simulations + internal pymbar and pymbar ti if it exists

    # TODO need to wait for slurm scripts or make this a separate options e.g. jax, amber, amb_analysis, jax_analysis
    # trapezoidal rule over <dV/dL> values
    dVdL = [parse_from_output(f'lambda_{lam:.1f}.out') for lam in lambdas]
    deltaG = 0.2 * sum(0.5*(dVdL[i]+dVdL[i+1]) for i in range(len(dVdL)-1))

    import re

    def parse_dvdl(filepath):
        with open(filepath) as f:
            lines = f.readlines()
        for line in reversed(lines):
            if "DV/DL" in line:
                match = re.search(r'DV/DL\s*=\s*(-?\d+\.\d+)', line)
                if match:
                    return float(match.group(1))
        raise ValueError(f"DV/DL not found in {filepath}")

    dvdl_values = []
    for lam in lambdas:
        lam_str = f"{lam:.2f}"
        dvdl = parse_dvdl(f"outputs/mdout_v0_{lam_str}.out")
        dvdl_values.append(dvdl)

    # Trapezoidal integration over lambda
    deltaG = sum(
        0.5 * (dvdl_values[i] + dvdl_values[i + 1]) * (lambdas[i + 1] - lambdas[i])
        for i in range(len(lambdas) - 1)
    )
    print(f"deltaG_hydration = {deltaG:.2f} kcal/mol")

    #or
    # def parse_dvdl(fname):
    # with open(fname, 'r') as f:
    #     lines = f.readlines()
    # dvdl_vals = [float(line.split('=')[1].strip()) for line in lines if "DV/DL =" in line]
    # return np.mean(dvdl_vals)

    # def compute_deltaG(lambdas):
    #     dvdl = [parse_dvdl(f"lambda_{lam:.1f}.out") for lam in lambdas]
    #     dg = 0.0
    #     for i in range(len(lambdas) - 1):
    #         dg += 0.5 * (dvdl[i] + dvdl[i+1]) * (lambdas[i+1] - lambdas[i])
    #     print("Estimated ΔG (TI, kcal/mol):", dg)

    # TODO does pymbar have TI/FEP/BAR/other FE methods?


