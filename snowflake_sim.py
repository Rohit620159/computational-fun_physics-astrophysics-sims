import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from numba import jit, prange

# --- Simulation Parameters (High Res for Cluster) ---
GRID_SIZE = 800       # Increased resolution since we have speed now
STEPS = 8000          # Longer duration
DT = 0.0001
DX = 0.03

# Phase Field Parameters
TAU = 0.0002
EPSILON_BAR = 0.01
ANISOTROPY = 0.05
ALPHA = 0.9
GAMMA = 10.0
TEQ = 0.9
K = 1.5

# --- Numba Optimized Physics Kernels ---
# @jit compiles this function to machine code. 
# parallel=True allows it to use multiple CPU cores automatically.
# fastmath=True allows aggressive floating point optimizations.

@jit(nopython=True, parallel=True, fastmath=True)
def compute_next_step(Phi, T, new_Phi, new_T):
    """
    Computes the Laplacian and updates fields for the entire grid in parallel.
    """
    rows, cols = Phi.shape
    
    # We ignore the boundary (1 pixel) to avoid checking bounds inside the loop
    # prange enables parallel execution of this loop
    for i in prange(1, rows - 1):
        for j in range(1, cols - 1):
            
            # 1. Five-point Stencil Laplacian
            # 
            lap_phi = (Phi[i+1, j] + Phi[i-1, j] + Phi[i, j+1] + Phi[i, j-1] - 4.0 * Phi[i, j]) / (DX**2)
            lap_T   = (T[i+1, j]   + T[i-1, j]   + T[i, j+1]   + T[i, j-1]   - 4.0 * T[i, j])   / (DX**2)

            # 2. Anisotropy (Gradients)
            # Central difference for gradients
            dy_phi = (Phi[i+1, j] - Phi[i-1, j]) / (2 * DX)
            dx_phi = (Phi[i, j+1] - Phi[i, j-1]) / (2 * DX)
            
            theta = np.arctan2(dy_phi, dx_phi)
            
            # Anisotropy function (6-fold symmetry for snow)
            epsilon = EPSILON_BAR * (1.0 + ANISOTROPY * np.cos(6.0 * theta))
            
            # 3. Phase Field Evolution (Kobayashi Model)
            m = ALPHA / np.pi * np.arctan(GAMMA * (TEQ - T[i, j]))
            
            # Term for the phase change driving force
            term = Phi[i, j] * (1.0 - Phi[i, j]) * (Phi[i, j] - 0.5 + m)
            
            dPhi = (epsilon**2 * lap_phi + term) / TAU
            
            # Update Phi
            new_Phi[i, j] = Phi[i, j] + dPhi * DT

            # 4. Heat Diffusion
            # Latent heat release only happens where phase is changing
            latent_heat = K * dPhi
            new_T[i, j] = T[i, j] + (lap_T + latent_heat) * DT

    return new_Phi, new_T

def run_simulation():
    # Initialize Fields
    T = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
    Phi = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
    
    # Double buffering (needed for parallel safety)
    new_T = np.zeros_like(T)
    new_Phi = np.zeros_like(Phi)
    
    # Seed the crystal
    center = GRID_SIZE // 2
    seed_rad = 4
    y, x = np.ogrid[-center:GRID_SIZE-center, -center:GRID_SIZE-center]
    mask = x**2 + y**2 <= seed_rad**2
    Phi[mask] = 1.0
    new_Phi[mask] = 1.0
    
    # Pre-compile the Numba function by running it once with dummy data
    # This prevents the compilation time from appearing in the progress bar
    print("Compiling physics kernel with Numba JIT...")
    compute_next_step(Phi, T, new_Phi, new_T)
    print("Compilation complete. Starting simulation.")

    frame_skip = 15  # Save every 15th frame
    
    for n in tqdm(range(STEPS), desc="Simulating Crystal Growth"):
        # Run the parallel physics kernel
        # We swap buffers: compute into 'new', then 'new' becomes 'current'
        compute_next_step(Phi, T, new_Phi, new_T)
        
        # Add slight noise to tip generation (Thermal fluctuations)
        # We do this outside the JIT loop for simplicity using numpy
        if n % 5 == 0:
            noise = np.random.normal(0, 0.005, Phi.shape)
            # Only add noise at the interface (where Phi is between 0 and 1)
            interface_mask = (new_Phi > 0.01) & (new_Phi < 0.99)
            new_Phi[interface_mask] += noise[interface_mask]
        
        # Update references for next step (Pointer swap, very fast)
        Phi, new_Phi = new_Phi, Phi
        T, new_T = new_T, T
        
        if n % frame_skip == 0:
            # Copy data for the frame yield
            yield np.copy(Phi), np.copy(T)



if __name__ == '__main__':
    # Setup the plot
    fig, ax = plt.subplots(figsize=(12, 12), dpi=150)
    fig.patch.set_facecolor('black') # True Black background
    ax.set_axis_off()

    # --- UPGRADE: Create a Starfield Background ---
    # Generate random stars
    num_stars = 150
    star_x = np.random.randint(0, GRID_SIZE, num_stars)
    star_y = np.random.randint(0, GRID_SIZE, num_stars)
    star_brightness = np.random.uniform(0.5, 1.0, num_stars)
    # Scatter plot for stars (zorder=0 puts them behind the crystal)
    ax.scatter(star_x, star_y, s=star_brightness*2, c='white', alpha=0.8, zorder=0)

    # --- UPGRADE: Custom "Cosmic Ice" Colormap ---
    # Colors: Transparent/Black -> Deep Blue -> Cyan -> White (Heat)
    colors = [
        (0.0, 0.0, 0.0, 0.0), # Start Transparent (so we see stars)
        (0.0, 0.0, 0.5, 1.0), # Deep Blue (Ice)
        (0.0, 0.8, 1.0, 1.0), # Cyan (Edge)
        (1.0, 1.0, 1.0, 1.0)  # White (Latent Heat/Hot Tips)
    ]
    cosmic_ice = LinearSegmentedColormap.from_list("cosmic_ice", colors, N=256)

    # Initialize Image with new colormap
    im = ax.imshow(np.zeros((GRID_SIZE, GRID_SIZE)), cmap=cosmic_ice, vmin=0, vmax=1.1, zorder=1)

    # Sophisticated Title
    ax.set_title("Stochastic Phase-Field Crystallization", color='white', fontsize=16, y=0.92)
    ax.text(0.5, 0.05, "M.Sc. Astrophysics Simulation • $\epsilon=0.01, \delta=0.05$",
            transform=ax.transAxes, ha='center', color='cyan', fontsize=8, alpha=0.7)

    def update(data):
        Phi, T = data
        # Combine Structure (Phi) and Heat (T)
        # The crystal body is Phi, the 'glow' is T
        visual_field = Phi + (T * 0.5)
        im.set_data(visual_field)
        return im,

    ani = animation.FuncAnimation(fig, update, run_simulation,
                                  save_count=int(STEPS/15), blit=True)

    output_file = 'cosmic_snowflake_final.mp4'
    print(f"Saving {output_file}...")

    # Save command (keep using ffmpeg via conda)
    ani.save(output_file, writer='ffmpeg', fps=30, bitrate=4000) # Increased bitrate for quality
    print("Done! ❄️")






























