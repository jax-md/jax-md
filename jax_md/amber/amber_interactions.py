import jax
import jax.numpy as jnp
from jax_md import space
import openmm.app as app
import time

"""
There's a few things to consider here
reaxff and reax-opt have 2 different interaction list counters
they vary because of vectorization during optimization and handling of passing params

in the case where you have multiple systems, some periodic, some not, what do you do?
lax.switch(p_fn, free_fn, dR, box) works, but may be less efficient
the plain form is just p_fn = space.periodic(box) -> p_fn(dR) which may bake this in as a compile time constant

just doing periodic calculation for all but appending huge periodic box [9999.9] or 1000*AABB might work but feels hacky

the current way of fully switching between the two works but isn't vectorizable

lax switch is probably the cleanest way of doing this going forward but i still hesitate
nve still needs a shift function, you can probably pass shift_fn as a kwarg for step_fn but this again
may hurt performance

some other things to look at are the periodic_general implementation (still MIC, also fractional coords which is a plus)
cagri's code (not MIC as far as i can tell, orth matrix and image shifts)
online sources to see alternatives to that bookkeeping (no MIC means same shift function both ways)
https://en.wikipedia.org/wiki/Periodic_boundary_conditions
part of the issue with this is that non minimum image convention makes ewald summation a lot more complicated

i think the most sane path going forward is going to be something with MIC, fractional coordinates (later)
for the shift function, some options include doing something like np.iinfo(dtype).max and passing as box/side kwarg
or dynamically lax switching the function that is passed, which may be dangerous for jit

the max float approach probably works better because of the above point, at risk of overflows?
something like jnp.where(condition, jnp.mod(x, y), x) is also possible but with redundant computation

cagri's code only uses energy minimization anyways, so you don't need to deal with the shift fn issue explicitly
the emin code doesn't even apply shifts when updating positions, which is probably dangerous but works assuming there's enough padding
it might help to throw an error statement in if the padding for a system is less than some value
"""

def calculate_dist_and_angles_amber(positions, force_field, nbr_lists=None):
    # return nbr_dist, nbr_disps, bond_dist, angles, torsions, ffq_dist
    # or return all nbr_dists, bond, angles, torsions
    # how to index exclusions and ffq into nbr dists?
    # look at far nbr list in ffq code for example

    # would jnp.diff(bond_pos, axis=1) work? any more efficient?
    # add optional flag for periodic for paramopt support

    # also add dense flag, this may end up being more efficient
    # something to the effect of map_neighbor vs map_bond/vmap

    pos_b = positions[ff.bond_idx]
    dist_bond = dist_fn(pos_b[:, 0], pos_b[:, 1])

    pos_a = positions[ff.angle_idx]
    theta_angle = angle_fn(disp_map(pos_a[:, 0], pos_a[:, 1]), disp_map(pos_a[:, 2], pos_a[:, 1]))

    pos_t = positions[ff.torsion_idx] # TODO is doing this and then taking 4 slices less efficient?
    theta_torsion = torsion_fn(pos_t[:, 0], pos_t[:, 1], pos_t[:, 2], pos_t[:, 3]) # TODO redo this and angle signature

    pos_nb = positions[ff.nb_idx]
    # or if pme nb_idx = nbr_lists.idx
    # TODO consider how the indices are reversed nbr_list.idx[0, :]
    # i think i tested nb.idx.T but didn't see any performance increase under jit
    dist_nb = dist_fn(pos_nb[:, 0], pos_nb[:, 1])

    pos_14 = positions[ff.nb_idx]
    dist_14 = dist_fn(pos_14[:, 0], pos_14[:, 1])

    # if pme

    #if ffq

    #exclusions distance for corr_term


    # TODO test jax lax switch here to see if you can cleanly switch between periodic and nonperiodic distance func
    # otherwise assigning really large box (e.g. 10,000 A) may also be a realistic solution

    return (dist_bond, theta_angle, theta_torsion, dist_nb, dist_14)

##########################################################################################################

#disp_fn, shift_fn = space.periodic_general(jnp.identity(3), fractional_coordinates=False)
pd_fn, ps_fn = space.periodic(jnp.float32(1))
fd_fn, fs_fn = space.free()
m = jnp.finfo(jnp.float32).max/10000
pos = jnp.array([10.32, 4.23, 6.32], dtype=jnp.float32)
def periodic_fn(side, dR):
    return jnp.mod(dR + side * jnp.float32(0.5), side) - jnp.float32(0.5) * side

print("m", m)
print("p", pd_fn(pos, .5 * pos))
print("p max", pd_fn(pos, .5 * pos, side=100))
dR = pos - (.5 * pos)
side = m
print(jnp.mod(dR + side * jnp.float32(0.5), side))
print(jnp.float32(0.5) * side)
print("periodic max", periodic_fn(m, dR)) # -> [0., 0., 0.] if m is used, overflow

#print("norm", jnp.linalg.norm(dR))
print("free", fd_fn(pos, .5 * pos))

sys.exit()

##########################################################################################################

inpcrd_file = "data/diala_pbc/diala_pbc.inpcrd"
prmtopFile = "data/diala_pbc/diala_pbc.prmtop"

inpcrd = app.AmberInpcrdFile(inpcrd_file)
positions = jnp.array(inpcrd.getPositions(asNumpy=True))

#print(positions)

idx = jnp.tile(jnp.arange(1000000), 2).reshape((1000000,2))

mask = jnp.arange(1000000)
mask = jnp.where((mask % 2) == 0, 1, 0)

# side = jnp.arange(1000000) / 1000
side = positions[idx[:, 0]] * 100


#print(mask[:50])
# perodic_fn = space.periodic_displacement
# free_fn = space.pairwise_displacement
f32 = jnp.float32
def periodic_fn(side, dR):
    return jnp.linalg.norm(5 + jnp.mod(dR + side * f32(0.5), side) - f32(0.5) * side)

def free_fn(side, dR):
    return jnp.linalg.norm(1 + jnp.mod(dR + side * f32(0.5), side) - f32(0.5) * side)

#part_fn = partial(switch)

def dist_fn_switch(dist, mask, side):
    return jax.lax.switch(mask, [periodic_fn, free_fn], side, dist)

jit_dist = jax.jit(jax.vmap(dist_fn_switch))
jit_peri = jax.jit(jax.vmap(periodic_fn))

#print(idx[:, 0].shape)

print(periodic_fn(10, positions[0]-positions[1]))

print(positions[idx[:, 0]] - positions[idx[:, 1]])

print(mask)

print(side)

res = jnp.sum(jit_dist(positions[idx[:, 0]] - positions[idx[:, 1]], mask, side)).block_until_ready()
res2 = jnp.sum(jit_peri(side, positions[idx[:, 0]] - positions[idx[:, 1]])).block_until_ready()

start = time.time()
total = 0
for i in range(1000):
    total += jnp.sum(mask*jit_peri(side, positions[idx[:, 0]] - positions[idx[:, 1]])).block_until_ready()
    
end = time.time()
print("periodic total time is", end-start, total)

start = time.time()
total = 0
for i in range(1000):
    total += jnp.sum(jit_dist(positions[idx[:, 0]] - positions[idx[:, 1]], mask, side)).block_until_ready()
    
end = time.time()
print("switch total time is", end-start, total)

def switch_test():
    return