# Imports

import os
# TODO: Re-enable x64 mode after XLA bug fix.
# from jax.config import config ; config.update('jax_enable_x64', True)
import warnings

import numpy as onp
from jax import jit, vmap, grad
from jax import lax

warnings.simplefilter('ignore')
import jax.numpy as np

from jax import random

import jax

jax.config.update('jax_platform_name', 'cpu')

import optax

from jax_md import energy, space

# Plotting.

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style(style='white')
sns.set(font_scale=1.6)


def format_plot(x, y):
    plt.xlabel(x, fontsize=20)
    plt.ylabel(y, fontsize=20)


def finalize_plot(shape=(1, 1)):
    plt.gcf().set_facecolor('white')
    plt.gcf().set_size_inches(
        shape[0] * 1.5 * plt.gcf().get_size_inches()[1],
        shape[1] * 1.5 * plt.gcf().get_size_inches()[1])
    plt.tight_layout()


def draw_training(params, iteration):
    plt.subplot(1, 2, 1)
    plt.semilogy(train_energy_error)
    plt.semilogy(test_energy_error)
    plt.xlim([0, train_epochs])
    format_plot('Epoch', '$L$')
    plt.subplot(1, 2, 2)
    predicted = vectorized_energy_fn(params, example_positions)
    plt.plot(example_energies, predicted, 'o')
    plt.plot(np.linspace(-400, -300, 10), np.linspace(-400, -300, 10), '--')
    format_plot('$E_{label}$', '$E_{prediction}$')
    finalize_plot((2, 1))
    plt.savefig(f'training_plot_{iteration}.png')


# Data Loading.

def MD_trajectory_reader(f, no_skip=20):
    filename = os.path.join('Supplementary/', f)
    fo = open(filename, 'r')
    samples = fo.read().split('iter= ')[1:]
    steps = []
    lattice_vectors = []
    positions = []
    forces = []
    temperatures = []
    energies = []
    for sample in samples[::no_skip]:
        entries = sample.split('\n')
        steps.append(int(entries[0]))
        lattice_vectors.append(onp.array([list(map(float, lv.split())) for lv in entries[1:4]]))
        assert entries[4] == '64'
        temp = onp.array([list(map(float, lv.split()[1:])) for lv in entries[5:69]])
        positions.append(temp[:, :3])
        forces.append(temp[:, 3:])
        remaining_lines = entries[69:]
        temperatures.append(float([entry for entry in entries[69:] if 'Temp' in entry][0].split('=')[1].split()[0]))
        energies.append(float([entry for entry in entries[69:] if 'el-ion E' in entry][0].split('=')[1].split()[0]))
    assert (len(set(steps)) - (steps[-1] - steps[0] + 1) / no_skip) < 1
    return np.array(positions), np.array(energies), np.array(forces)


def build_dataset():
    no_skip = 15
    data300, energies300, forces300 = MD_trajectory_reader(
        'MD_DATA.cubic_300K', no_skip=no_skip)
    data600, energies600, forces600 = MD_trajectory_reader(
        'MD_DATA.cubic_600K', no_skip=no_skip)
    data900, energies900, forces900 = MD_trajectory_reader(
        'MD_DATA.cubic_900K', no_skip=no_skip)
    dataliq, energiesliq, forcesliq = MD_trajectory_reader(
        'MD_DATA.liq_1', no_skip=no_skip)

    all_data = np.vstack((data300, data600, data900))
    all_energies = np.hstack((energies300, energies600, energies900))
    all_forces = np.vstack((forces300, forces600, forces900))
    noTotal = all_data.shape[0]

    onp.random.seed(0)
    II = onp.random.permutation(range(noTotal))
    all_data = all_data[II]
    all_energies = all_energies[II]
    all_forces = all_forces[II]
    noTr = int(noTotal * 0.65)
    noTe = noTotal - noTr
    train_data = all_data[:noTr]
    test_data = all_data[noTr:]

    train_energies = all_energies[:noTr]
    test_energies = all_energies[noTr:]

    train_forces = all_forces[:noTr]
    test_forces = all_forces[noTr:]

    return ((train_data, train_energies, train_forces),
            (test_data, test_energies, test_forces))


train, test = build_dataset()

positions, energies, forces = train
test_positions, test_energies, test_forces = test

energy_mean = np.mean(energies)
energy_std = np.std(energies)

print('positions.shape = {}'.format(positions.shape))
print('<E> = {}'.format(energy_mean))

box_size = 10.862  # The size of the simulation region.
displacement, shift = space.periodic(box_size)

neighbor_fn, init_fn, energy_fn = energy.graph_network_neighbor_list(
    displacement, box_size, r_cutoff=3.0, dr_threshold=0.0)

neighbor = neighbor_fn.allocate(positions[0], extra_capacity=6)

print('Allocating space for at most {} edges'.format(neighbor.idx.shape[1]))


@jit
def train_energy_fn(params, R):
    _neighbor = neighbor.update(R)
    return energy_fn(params, R, _neighbor)


# Vectorize over states, not parameters.
vectorized_energy_fn = vmap(train_energy_fn, (None, 0))

grad_fn = grad(train_energy_fn, argnums=1)
force_fn = lambda params, R, **kwargs: -grad_fn(params, R)
vectorized_force_fn = vmap(force_fn, (None, 0))



key = random.PRNGKey(0)

params = init_fn(key, positions[0], neighbor)

n_predictions = 500
example_positions = positions[:n_predictions]
example_energies = energies[:n_predictions]
example_forces = forces[:n_predictions]

print(f'example_energies: {example_energies.shape}')

predicted = vmap(train_energy_fn, (None, 0))(params, example_positions)

print(f'predicted: {predicted.shape}')

plt.plot(example_energies, predicted, 'o')

format_plot('$E_{label}$', '$E_{predicted}$')
finalize_plot((1, 1))

print('Here!--------------------')

plt.savefig('initial_plot.png')

print('Here!--------------------')

@jit
def energy_loss(params, R, energy_targets):
    return np.mean((vectorized_energy_fn(params, R) - energy_targets) ** 2)


@jit
def force_loss(params, R, force_targets):
    dforces = vectorized_force_fn(params, R) - force_targets
    return np.mean(np.sum(dforces ** 2, axis=(1, 2)))


@jit
def loss(params, R, targets):
    return energy_loss(params, R, targets[0]) + force_loss(params, R, targets[1])


opt = optax.chain(optax.clip_by_global_norm(1.0),
                  optax.adam(1e-3))


@jit
def update_step(params, opt_state, R, labels):
    updates, opt_state = opt.update(grad(loss)(params, R, labels),
                                    opt_state)
    return optax.apply_updates(params, updates), opt_state


@jit
def update_epoch(params_and_opt_state, batches):
    def inner_update(params_and_opt_state, batch):
        params, opt_state = params_and_opt_state
        b_xs, b_labels = batch

        return update_step(params, opt_state, b_xs, b_labels), 0

    return lax.scan(inner_update, params_and_opt_state, batches)[0]


dataset_size = positions.shape[0]
batch_size = 128

lookup = onp.arange(dataset_size)
onp.random.shuffle(lookup)


@jit
def make_batches(lookup):
    batch_Rs = []
    batch_Es = []
    batch_Fs = []

    for i in range(0, len(lookup), batch_size):
        if i + batch_size > len(lookup):
            break

        idx = lookup[i:i + batch_size]

        batch_Rs += [positions[idx]]
        batch_Es += [energies[idx]]
        batch_Fs += [forces[idx]]

    return np.stack(batch_Rs), np.stack(batch_Es), np.stack(batch_Fs)


batch_Rs, batch_Es, batch_Fs = make_batches(lookup)

train_epochs = 20

opt_state = opt.init(params)

train_energy_error = []
test_energy_error = []

for iteration in range(train_epochs):
    train_energy_error += [float(np.sqrt(energy_loss(params, batch_Rs[0], batch_Es[0])))]
    test_energy_error += [float(np.sqrt(energy_loss(params, test_positions, test_energies)))]

    draw_training(params, iteration)

    params, opt_state = update_epoch((params, opt_state),
                                     (batch_Rs, (batch_Es, batch_Fs)))

    onp.random.shuffle(lookup)
    batch_Rs, batch_Es, batch_Fs = make_batches(lookup)
