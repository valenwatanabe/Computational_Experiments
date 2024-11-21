#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 18:07:51 2024

@author: vwitch
"""

import numpy as np
import matplotlib.pyplot as plt

class IsingModel:
    def __init__(self, size, temperature, field):
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
        return (energy / 2) - self.field * np.sum(self.lattice)  # Each pair counted twice

    def magnetization(self):
        """Calculate the total magnetization of the current state."""
        return np.sum(self.lattice)

        
    def metropolis_step(self, sweep='random'):
        """Perform one Metropolis step."""
        if sweep == 'random':
            for _ in range(self.size**2):
                i = np.random.randint(0, self.size)
                j = np.random.randint(0, self.size)
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
        delta_energy = 2 * spin * (neighbors + self.field)
    
        if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / self.temperature):
            self.lattice[i, j] *= -1


    def simulate(self, steps):
        """Simulate the Ising model for a given number of steps."""
        for _ in range(steps):
            self.metropolis_step()
            
    def plot_lattice(self):
        """Plot the current state of the lattice."""
        plt.imshow(self.lattice, cmap='Greys', interpolation='nearest')
        plt.title(f"2D Ising Model (T={self.temperature})")
        plt.colorbar(label='Spin')
        plt.show()


def calculate_properties(size, temperature, field, steps):
    magnetizations = []
    energies = []
    susceptibilities = []
    heat_capacities = []

    for h in field:
        model = IsingModel(size, temperature, h)
        model.simulate(steps)
        magnetizations.append(model.magnetization() / (size * size))
        energies.append(model.energy() / (size * size))

    magnetizations = np.array(magnetizations)
    energies = np.array(energies)
    susceptibilities = np.gradient(magnetizations, field)
    heat_capacities = np.gradient(energies, field)

    return magnetizations, energies, susceptibilities, heat_capacities
    
size = 10
temperatures = [1.0, 4.0]
magnetic_fields = np.linspace(-2, 2, 50)
steps = 1000

ising_model = IsingModel(size=10, temperature=2.5, field =0)
ising_model.simulate(steps=1000)
ising_model.plot_lattice()
print("Final Energy:", ising_model.energy())
print("Final Magnetization:", ising_model.magnetization())

plt.figure(figsize=(12, 8))

for T in temperatures:
    magnetizations, energies, susceptibilities, heat_capacities = calculate_properties(size, T, magnetic_fields, steps)

    plt.subplot(2, 2, 1)
    plt.plot(magnetic_fields, magnetizations, label=f'T={T}')
    plt.xlabel('Magnetic Field (h)')
    plt.ylabel('Magnetization')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(magnetic_fields, energies, label=f'T={T}')
    plt.xlabel('Magnetic Field (h)')
    plt.ylabel('Energy')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(magnetic_fields, susceptibilities, label=f'T={T}')
    plt.xlabel('Magnetic Field (h)')
    plt.ylabel('Magnetic Susceptibility')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(magnetic_fields, heat_capacities, label=f'T={T}')
    plt.xlabel('Magnetic Field (h)')
    plt.ylabel('Heat Capacity')
    plt.legend()

plt.tight_layout()
plt.show()
















