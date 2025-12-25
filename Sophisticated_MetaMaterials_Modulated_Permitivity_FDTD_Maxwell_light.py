import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
c = 1  # Speed of light (normalized)
dx = 0.001  # Spatial resolution
dt = dx / c * 0.5  # Temporal resolution
num_steps = 10000  # Number of time steps
num_cells = 1000  # Number of cells in the simulation domain
v = 0.1  # Velocity of the wave (controls permittivity variation)
omega = 2.0 * np.pi  # Light frequency
alpha = 0.01  # Absorption coefficient
n0 = 1.0  # Refractive index of surrounding medium
PML_width = 20  # Width of Perfectly Matched Layers (PMLs)

# Define functions for permittivity and permeability
def epsilon_r_pml(z, t, omega, PML_width):
  if z < PML_width or z > num_cells - PML_width:
    distance_from_boundary = min(z, num_cells - z)
    grading_factor = (distance_from_boundary / PML_width) ** 2
    return n0**2 + 0.2 * np.sin(2 * np.pi * (z - v * t)) + grading_factor * (10.0 - (n0**2 + 0.2 * np.sin(2 * np.pi * (z - v * t))))  # Adjust factor for appropriate absorption
  else:
    return n0**2 + 0.2 * np.sin(2 * np.pi * (z - v * t))

epsilon_r = lambda z, t: epsilon_r_pml(z, t, omega, PML_width)

def mu_r(x):
  return 1.0

# Define electric and magnetic fields
Ex = np.zeros(num_cells)
Hy = np.zeros(num_cells)

# Initialize a Gaussian pulse in the E Field
pulse_center = num_cells // 2
pulse_width = 100

Ex = np.exp(-((np.arange(num_cells) - pulse_center) ** 2) / (2 * pulse_width ** 2))


# Function to update fields
def update_fields(frame):
  global Ex, Hy
  # Update electric field
  Ex[1:] += c * dt / (dx * epsilon_r(np.arange(1, num_cells) * dx - v * frame * dt, frame * dt, omega)) * (Hy[1:] - Hy[:-1])
  # Apply absorption
  Ex *= np.exp(-alpha * dx)

  # Update magnetic field
  Hy[:-1] += c * dt / (dx * mu_r(np.arange(num_cells - 1) * dx)) * (Ex[1:] - Ex[:-1])

  # Apply boundary conditions (assuming perfectly conducting boundaries)
  Ex[0] = 0
  Ex[-1] = 0
  Hy[0] = 0
  Hy[-1] = 0

  # Calculate intensity
  intensity = np.abs(Ex) ** 2

  # Plot intensity
  plt.cla()
  plt.plot(intensity, label='Electric Field Intensity')
  plt.xlabel('Cell index')
  plt.ylabel('Intensity (Ex^2)')
  plt.title(f'Simulation (Step {frame+1}/{num_steps})')
  plt.legend()
  plt.ylim(-0.1, 1.1)  # Adjust y-axis limits for better visualization

# Create the animation
fig = plt.figure(figsize=(10, 6))
ani = FuncAnimation(fig, update_fields, frames=num_steps, interval=50)
plt.show()
