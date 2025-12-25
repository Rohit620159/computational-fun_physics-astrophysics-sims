#!/usr/bin/env python
# Save this as blob_animation_solar.py

import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import glob
import imageio.v3 as iio
from natsort import natsorted

print("--- Starting 'Toy Model Sun' Animation Script ---")

# --- 1. USER-ADJUSTABLE PARAMETERS ---
SIM_DATA_DIR = 'simulation_output'
N_RANKS = 8 
VARIABLE_TO_PLOT = 'temp' # 'temp', 'rho', or 'vz'
SLICE_Y_INDEX = 32 
OUTPUT_FILE = 'blob_ejection_v6.gif' 
FPS = 10 

# !!!!!!!!!!!!!!!!!!!!! THE FIX IS HERE !!!!!!!!!!!!!!!!!!!!!
# We define the grid dimensions so the script knows them.
GRID_DIMS = (64, 64, 128)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# --- 2. Find and sort all snapshot files ---
print(f"Scanning for files in '{SIM_DATA_DIR}'...")
rank_0_files = natsorted(glob.glob(os.path.join(SIM_DATA_DIR, 'snap_*_rank000.npz')))
snap_steps = [os.path.basename(f).split('_')[1] for f in rank_0_files]

if not snap_steps:
    print(f"Error: No snapshot files found in '{SIM_DATA_DIR}'.")
    exit()

print(f"Found {len(snap_steps)} total timesteps.")

# --- 3. Setup Plotting ---
if VARIABLE_TO_PLOT == 'temp':
    cmap = 'hot'
    vmin = 5000.0  # 5,000 K (to see the 6000K "sun")
    vmax = 5.0e6   # 5,000,000 K (to see the blob)
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    label = "Temperature (K)"

elif VARIABLE_TO_PLOT == 'rho':
    cmap = 'viridis'
    vmin = 1.0e-16 # See the "corona"
    vmax = 1.0e-9  # See the "blob"
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    label = "Density (g/cm^3)"

else: # 'vz'
    cmap = 'RdBu_r'
    vmax_abs = 1e8 # Clip velocity to +/- 1000 km/s (1e8 cm/s)
    norm = mcolors.Normalize(vmin=-vmax_abs, vmax=vmax_abs)
    label = "Vertical Velocity (cm/s)"

print(f"Plotting '{VARIABLE_TO_PLOT}' with clipped range: vmin={norm.vmin:.2e}, vmax={norm.vmax:.2e}")

# --- 4. Loop, Stitch, and Plot ---
frame_files = []
print("Generating frames...")

for i, step in enumerate(snap_steps):
    rank_data_list = []
    for rank in range(N_RANKS):
        filename = os.path.join(SIM_DATA_DIR, f"snap_{step}_rank{rank:03d}.npz")
        try:
            data = np.load(filename)[VARIABLE_TO_PLOT]
            rank_data_list.append(data)
        except FileNotFoundError:
            if rank_data_list:
                rank_data_list.append(np.zeros_like(rank_data_list[0]))
            else:
                continue 

    if not rank_data_list: continue
        
    try:
        global_data = np.concatenate(rank_data_list, axis=2)
    except ValueError as e:
        print(f"Error concatenating step {step}. Skipping.")
        continue

    plot_data = global_data[:, SLICE_Y_INDEX, :].T 
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(plot_data, origin='lower', aspect='auto', 
                   cmap=cmap, norm=norm)
                   
    ax.set_title(f"{label} (Step: {step})")
    ax.set_xlabel("X Position (grid cells)")
    ax.set_ylabel("Z Position (grid cells)")
    
    # Draw a line for the "surface"
    # Z=0 is halfway up the 128-cell grid (cell 64)
    z_surface_cell = GRID_DIMS[2] / 2
    ax.axhline(z_surface_cell, color='cyan', linestyle='--', label='Z=0 (Sun Surface)')
    ax.legend(loc='upper left')
    
    fig.colorbar(im, ax=ax, label=label)
    
    frame_filename = os.path.join(SIM_DATA_DIR, f"_frame_{i:04d}.png")
    fig.savefig(frame_filename)
    plt.close(fig)
    frame_files.append(frame_filename)
    
    print(f"  Generated frame {i+1}/{len(snap_steps)}", end='\r')

print("\nAll frames generated.")

# --- 5. Animate ---
print(f"Creating animation '{OUTPUT_FILE}'...")
frames = [iio.imread(f) for f in frame_files]
iio.imwrite(OUTPUT_FILE, frames, fps=FPS)
print("Animation saved.")

# --- 6. Clean up ---
print("Cleaning up temporary frame files...")
for f in frame_files:
    os.remove(f)

print("--- Done ---")
