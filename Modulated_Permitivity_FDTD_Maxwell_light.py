import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
c = 1  # Speed of light (normalized)
dx = 0.001  # Spatial resolution
dt = dx / c*0.5  # Temporal resolution
num_steps = 10000  # Number of time steps
num_cells = 1000  # Number of cells in the simulation domain
v = 0.1  # Velocity of the wave

# Define functions for permittivity and permeability (modify as needed)
def epsilon_r(z, t):
    # Define your function for permittivity here
   # return 2.0  # Constant permittivity for simplicity
     return 1.0 + 0.2 * np.sin(2 * np.pi * (z - v * t)) 

def mu_r(x):
    # Define your function for permeability here (default: constant permeability)
    return 1.0

# Define electric and magnetic fields
Ex = np.zeros(num_cells)
Hy = np.zeros(num_cells)

# Initialize a series of Gaussian pulses in the E Field
pulse_positions = [num_cells // 4, 3 * num_cells // 4]  # Positions of the Gaussian pulses
pulse_amplitudes = [1.0, 0.5]  # Amplitudes of the Gaussian pulses
pulse_sigmas = [10, 15]  # Standard deviations of the Gaussian pulses

for pos, ampl, sigma in zip(pulse_positions, pulse_amplitudes, pulse_sigmas):
    Ex += ampl * np.exp(-((np.arange(num_cells) - pos) ** 2) / (2 * sigma ** 2))


# Function to update fields
def update_fields(frame):
    global Ex, Hy
    # Update electric field
    Ex[1:] += c * dt / (dx * epsilon_r(np.arange(1, num_cells) * dx - v * frame * dt, frame * dt)) * (Hy[1:] - Hy[:-1])
    # Update magnetic field
    Hy[:-1] += c * dt / (dx * mu_r(np.arange(num_cells - 1) * dx)) * (Ex[1:] - Ex[:-1])

    # Apply boundary conditions
    Ex[0] = 0
    Ex[-1] = 0
    Hy[0] = 0
    Hy[-1] = 0

    # Plot fields
    plt.cla()
    plt.plot(Ex, label='Electric Field (Ex)')
    plt.plot(Hy, label='Magnetic Field (Hy)')
    plt.xlabel('Cell index')
    plt.ylabel('Field Strength')
    plt.title(f'Simulation of Electromagnetic Wave (Step {frame+1}/{num_steps})')
    plt.legend()
    plt.ylim(-2, 2)

# Create the animation
fig = plt.figure(figsize=(10, 6))
ani = FuncAnimation(fig, update_fields, frames=num_steps, interval=50)
plt.show()
# Save the animation as a GIF
ani.save('electromagnetic_wave_modulated_permittivity.gif', writer='pillow')
