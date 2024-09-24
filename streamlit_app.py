import torch
import torch.nn.functional as F

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
import time

import streamlit as st
plt.rcParams["font.family"] = "serif"

# Initializing the device
device = torch.device("cpu")

# Convolutional filter for 2D Ising with J=1.0
kernel = torch.tensor([[0.0, 1.0, 0.0],
                       [1.0, 0.0, 1.0],
                       [0.0, 1.0, 0.0]], device=device).reshape(1, 1, 3, 3)

# Function for performing the Metropolis step on one sublattice
def metropolis_sublattice_step(system, T, mask):
    # Calculating the sum of neighbors via convolution
    neighbor_sum = F.conv2d(system, kernel, padding=1)

    # Energy change calculation
    dE = 2 * system * neighbor_sum

    # Calculating the probability of adoption and updating the system
    acceptance_prob = torch.exp(-dE / T).clamp(max=1)
    random_prob = torch.rand_like(system)
    update_mask = (dE <= 0) | (random_prob < acceptance_prob)
    system[update_mask & mask] *= -1
    
    return system

def calculate_total_energy(system):
    # Calculating the sum of neighbors via convolution
    neighbor_sum = F.conv2d(system, kernel, padding=1, groups=1)

    # Calculation of the total energy of the system
    total_energy = - torch.sum(system * neighbor_sum, axis=(1,2,3)) / 2
    return total_energy

# system initialization
def upd_system(batch_size, L, device):
    return torch.randint(0, 2, (batch_size, 1, L, L), dtype=torch.float32, device=device) * 2 - 1

# Creating masks for subgrids
def upd_mask(system):
    mask_A = torch.zeros_like(system, dtype=torch.bool)
    mask_B = torch.zeros_like(system, dtype=torch.bool)
    mask_A[:, :, ::2, ::2] = 1 # Even rows, even columns
    mask_A[:, :, 1::2, 1::2] = 1  # Odd rows, odd columns
    mask_B = ~mask_A
    return mask_A, mask_B

def ising_evol(T, L, n_steps=100, batch_size=100, system_IC="empty"):
    """
    :param T: system temperature
    :param L: lattice linear size
    :param n_steps: number of equilibration steps before the measurements start
    :param batch_size: number of bins use for the error analysis
    :param system_IC: initial configuration of the grid
    :retrun: (C_mean, C_std)
    :retrun: (E_mean, E_std)
    """
    
    # Initializing the spin system
    if system_IC == "empty":
        system = upd_system(batch_size, L, device)
    else:
        system = system_IC
    mask_A, mask_B = upd_mask(system)
    
    # Running the simulation
    for step in range(n_steps):
        system = metropolis_sublattice_step(system, T, mask_A) # Обновление подрешётки A
        system = metropolis_sublattice_step(system, T, mask_B) # Обновление подрешётки B
    
    return system


# 2D Ising crytical temperature
Tc = 2 / np.log(1 + np.sqrt(2))




st.button("Reset")

n_steps = st.slider('Steps', min_value=1, max_value=40, value=20, step=1)
T       = st.slider('Temperature', min_value=0.1, max_value=4.0, value=Tc, step=0.1)
L       = st.slider('System size', min_value=10, max_value=1000, value=100, step=1)
system1 = upd_system(1, L, device)
system1 = ising_evol(T, L, n_steps=n_steps, batch_size=1, system_IC=system1)

st.image((torch.squeeze(system1).numpy()+1)/2)
