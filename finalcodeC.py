#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:13:50 2024

@author: vwitch
"""

import numpy as np
import random
import matplotlib.pyplot as plt

class IsingModel:
    def __init__(self, size, temperature, field=0):
        self.size = size
        self.temperature = temperature
        self.field = field
        self.lattice = np.random.choice([-1, 1], size=(size, size))

    def energy(self):
        """Calculate the total energy of the current state."""
        energy = 0
        for i in range(self.size):
            for j in range(self.size):
                spin = self.lattice[i, j]
                neighbors = self.lattice[(i+1)%self.size, j] + self.lattice[i, (j+1)%self.size] + \
                            self.lattice[(i-1)%self.size, j] + self.lattice[i, (j-1)%self.size]
                energy += -spin * neighbors
        return energy / 2  - self.field * np.sum(self.lattice)

    def magnetization(self):
        """Calculate the total magnetization of the current state."""
        return np.sum(self.lattice)
    def metropolis_step(self, sweep='random'):
        """Perform one Metropolis step."""
        if sweep == 'random':
            for _ in range(self.size**2):
                i = random.randint(0, self.size-1)
                j = random.randint(0, self.size-1)
                self._attempt_flip(i, j)
        elif sweep == 'sequential':
            for i in range(self.size):
                for j in range(self.size):
                    self._attempt_flip(i, j)

    def _attempt_flip(self, i, j):
        """Attempt to flip a spin at position (i, j)."""
        spin = self.lattice[i, j]
        neighbors = (self.lattice[(i+1) % self.size, j] + self.lattice[(i-1) % self.size, j] +
                     self.lattice[i, (j+1) % self.size] + self.lattice[i, (j-1) % self.size])
      
        delta_energy = 2 * spin * (neighbors + float(self.field))
        
        if delta_energy < 0 or random.random() < np.exp(-delta_energy / self.temperature):
            self.lattice[i, j] *= -1
    

    def simulate(self, steps):
        """Simulate the Ising model for a given number of steps."""
        for _ in range(steps):
            self.metropolis_step()

def calculate_properties(size, temperatures, steps):
    magnetizations = []
    energies = []
    susceptibilities = []
    heat_capacities = []

    for T in temperatures:
        model = IsingModel(size, T)
        model.simulate(steps)
        magnetizations.append(model.magnetization() / (size * size))
        energies.append(model.energy() / (size * size))

    magnetizations = np.array(magnetizations)
    energies = np.array(energies)
    susceptibilities = np.gradient(magnetizations, temperatures)
    heat_capacities = np.gradient(energies, temperatures)

    return magnetizations, energies, susceptibilities, heat_capacities

# Parameters
size = 10
temperatures = np.linspace(1.0, 4.0, 50)
steps = 1000

# Calculate properties
magnetizations, energies, susceptibilities, heat_capacities = calculate_properties(size, temperatures, steps)

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(temperatures, magnetizations, label='Magnetization')
plt.xlabel('Temperature')
plt.ylabel('Magnetization')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(temperatures, energies, label='Energy')
plt.xlabel('Temperature')
plt.ylabel('Energy')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(temperatures, susceptibilities, label='Magnetic Susceptibility')
plt.xlabel('Temperature')
plt.ylabel('Susceptibility')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(temperatures, heat_capacities, label='Heat Capacity')
plt.xlabel('Temperature')
plt.ylabel('Heat Capacity')
plt.legend()

plt.tight_layout()
plt.show()












