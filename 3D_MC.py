import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool
from tqdm import tqdm


def init_spins_3d(size_x, size_y, size_z, p_up=0.5):
    """Initialize a 3D lattice of spins (+1 or -1)."""
    return np.random.choice([-1, 1], size=(size_x, size_y, size_z), p=[1 - p_up, p_up])


def energy_ising_3d(spins, J=1.0):
    """Compute the energy per spin of a 3D Ising configuration."""
    neighbors = (
            np.roll(spins, 1, axis=0) + np.roll(spins, -1, axis=0) +  # x-axis neighbors
            np.roll(spins, 1, axis=1) + np.roll(spins, -1, axis=1) +  # y-axis neighbors
            np.roll(spins, 1, axis=2) + np.roll(spins, -1, axis=2)  # z-axis neighbors
    )
    E = -J * np.sum(spins * neighbors)
    return E / spins.size  # Energy per spin


def magnetization_ising_3d(spins):
    """Compute the magnetization per spin."""
    return np.mean(spins)


def metropolis_step(spins, kT, J=1.0):
    """Perform one Metropolis-Hastings step on the lattice."""
    size_x, size_y, size_z = spins.shape
    for _ in range(spins.size):  # Attempt N spin flips (N = lattice size)
        # Randomly select a spin
        x, y, z = np.random.randint(0, size_x), np.random.randint(0, size_y), np.random.randint(0, size_z)
        spin = spins[x, y, z]

        # Compute energy change if flipped
        neighbors = (
                spins[(x + 1) % size_x, y, z] + spins[(x - 1) % size_x, y, z] +
                spins[x, (y + 1) % size_y, z] + spins[x, (y - 1) % size_y, z] +
                spins[x, y, (z + 1) % size_z] + spins[x, y, (z - 1) % size_z]
        )
        dE = 2 * J * spin * neighbors

        # Flip if energetically favorable or with Boltzmann probability
        if dE <= 0 or np.random.rand() < np.exp(-dE / kT):
            spins[x, y, z] *= -1
    return spins


def simulate_3d_ising(size_x, size_y, size_z, kT_values, mc_steps, J=1.0, parallel=False):
    """Run Metropolis algorithm for 3D Ising model."""
    results = {
        'E_mean': np.zeros((len(kT_values), len(mc_steps))),
        'M_mean': np.zeros((len(kT_values), len(mc_steps))),
        'spins': np.zeros((size_x, size_y, size_z, len(kT_values)), dtype=int)
    }

    def worker(params):
        temp_idx, mc_idx = params
        spins = init_spins_3d(size_x, size_y, size_z)
        for _ in range(mc_steps[mc_idx]):
            spins = metropolis_step(spins, kT_values[temp_idx], J)
        return (
            temp_idx, mc_idx,
            energy_ising_3d(spins, J),
            magnetization_ising_3d(spins),
            spins
        )

    tasks = [(ti, mi) for ti in range(len(kT_values)) for mi in range(len(mc_steps))]

    if parallel:
        with Pool() as pool:
            for ti, mi, E, M, S in tqdm(pool.imap(worker, tasks), total=len(tasks)):
                results['E_mean'][ti, mi] = E
                results['M_mean'][ti, mi] = M
                results['spins'][..., ti] = S
    else:
        for ti, mi in tqdm(tasks):
            _, _, E, M, S = worker((ti, mi))
            results['E_mean'][ti, mi] = E
            results['M_mean'][ti, mi] = M
            results['spins'][..., ti] = S

    return results

# Parameters (adjust as needed)
size_x, size_y, size_z = 10, 10, 10
kT_values = np.linspace(1.0, 7.0, 100)  # Temperature range
mc_steps = [100, 200]  # Monte Carlo steps

# Run simulation (set parallel=True if needed)
results = simulate_3d_ising(size_x, size_y, size_z, kT_values, mc_steps, parallel=False)


