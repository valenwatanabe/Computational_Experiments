#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 11:43:32 2024

@author: vwitch
"""

import numpy as np
import random
import matplotlib.pyplot as plt

class IsingModel:
    def __init__(self, size, temperature, magnetic_field=0):
        self.size = size
        self.temperature = temperature
        self.magnetic_field = magnetic_field
        self.lattice = np.random.choice([-1, 1], size=(size, size))

    def energy(self):
        """Calculate the total energy of the current state."""
        energy = 0
        for i in range(self.size):
            for j in range(self.size):
                spin = self.lattice[i, j]
                neighbors = (
                    self.lattice[(i + 1) % self.size, j]
                    + self.lattice[i, (j + 1) % self.size]
                    + self.lattice[(i - 1) % self.size, j]
                    + self.lattice[i, (j - 1) % self.size]
                )
                energy += -spin * neighbors
        return energy / 2  # Each pair counted twice

    def magnetization(self):
        """Calculate the total magnetization of the current state."""
        return np.sum(self.lattice)

    def metropolis_step(self):
        """Perform one Metropolis step."""
        for _ in range(self.size**2):
            i = random.randint(0, self.size - 1)
            j = random.randint(0, self.size - 1)
            spin = self.lattice[i, j]
            neighbors = (
                self.lattice[(i + 1) % self.size, j]
                + self.lattice[i, (j + 1) % self.size]
                + self.lattice[(i - 1) % self.size, j]
                + self.lattice[i, (j - 1) % self.size]
            )
            delta_energy = 2 * spin * neighbors

            if delta_energy < 0 or random.random() < np.exp(-delta_energy / self.temperature):
                self.lattice[i, j] *= -1

    def simulate(self, steps):
        """Simulate the Ising model for a given number of steps."""
        magnetizations = []
        energies = []
        # Run steps for thermalization
        for _ in range(steps // 2):
            self.metropolis_step()
        # Collect data after thermalization
        for _ in range(steps // 2):
            self.metropolis_step()
            magnetizations.append(self.magnetization())
            energies.append(self.energy())
        return np.array(magnetizations), np.array(energies)

# Parameters
sizes = [10, 20, 30, 40, 50]
temperature = 2.27
steps = 50000  # Increase steps for better statistics

# Calculate properties
susceptibilities = []
heat_capacities = []

for size in sizes:
    model = IsingModel(size, temperature)
    mags, ens = model.simulate(steps)

    # Susceptibility calculation
    avg_m = np.mean(mags)
    susceptibility = (np.mean(mags**2) - avg_m**2) / (temperature * size * size)
    susceptibilities.append(susceptibility)

    # Heat capacity calculation
    avg_e = np.mean(ens)
    heat_capacity = (np.mean(ens**2) - avg_e**2) / (temperature**2 * size * size)
    heat_capacities.append(heat_capacity)

# Plotting susceptibility and heat capacity as a function of system size
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(sizes, susceptibilities, 'o-', label='Magnetic Susceptibility')
plt.xlabel('System Size (L)')
plt.ylabel('Susceptibility')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(sizes, heat_capacities, 'o-', label='Heat Capacity')
plt.xlabel('System Size (L)')
plt.ylabel('Heat Capacity')
plt.legend()

plt.tight_layout()
plt.show()

# Extract critical exponents
susceptibility_exponent = np.polyfit(np.log(sizes), np.log(susceptibilities), 1)[0]
heat_capacity_exponent = np.polyfit(np.log(sizes), np.log(heat_capacities), 1)[0]

print("Susceptibility Exponent (γ/ν):", susceptibility_exponent)
print("Heat Capacity Exponent (α/ν):", heat_capacity_exponent)


