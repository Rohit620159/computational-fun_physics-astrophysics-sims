import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# --- Simulation Parameters ---
NUM_STARS = 2500       # Number of stars in the tree
DURATION = 20          # Seconds
FPS = 30
ROTATION_SPEED = 1.5   # Base rotation multiplier

def generate_galaxy_tree():
    """
    Generates stars in a conical volume (Christmas Tree shape).
    """
    # 1. Height (z): Random distribution from 0 (bottom) to 1 (top)
    z = np.random.triangular(0, 0.3, 1.0, NUM_STARS)
    
    # 2. Radius (r): Tapers as height increases (Cone shape)
    # The cone gets narrower as z goes up.
    # We add some randomness to make it look like "branches" not a solid cone.
    max_radius_at_z = 1.0 - z
    r = np.random.uniform(0, 1, NUM_STARS) * max_radius_at_z
    
    # 3. Angle (theta): Random starting angle
    theta = np.random.uniform(0, 2 * np.pi, NUM_STARS)
    
    # 4. Velocities (The Physics!)
    # Keplerian-ish velocity: inner stars spin faster.
    # We add a small 'softening' parameter to avoid infinity at r=0
    # angular_velocity = v/r ~ (1/sqrt(r))/r = 1/r^1.5
    angular_velocity = ROTATION_SPEED / (r + 0.1)**1.5
    
    # 5. Colors
    # We'll assign colors based on "Temperature" (Radius)
    # Inner/Top stars = Hot/Blue, Outer/Bottom stars = Red/Gold
    colors = np.zeros((NUM_STARS, 4)) # RGBA
    
    # Simple logic: R varies, G is high, B varies. 
    # Creates a "Green-Gold-White" mix.
    colors[:, 0] = 1.0 - r # Red channel
    colors[:, 1] = 0.8 + 0.2*np.random.random(NUM_STARS) # Green (always high)
    colors[:, 2] = 0.2 + 0.8*z # Blue (more at top)
    colors[:, 3] = 0.8 # Alpha (transparency)
    
    return z, r, theta, angular_velocity, colors

def update(frame, z, r, theta, angular_velocity, scat, ax):
    """
    Updates the position of every star based on its orbital speed.
    """
    # Physics Update: theta = theta_0 + omega * t
    # We use 'frame' as time.
    current_theta = theta + angular_velocity * (frame * 0.02)
    
    # Convert Polar (r, theta) back to Cartesian (x, y)
    x = r * np.cos(current_theta)
    y = r * np.sin(current_theta)
    
    # Update the scatter plot
    # matplotlib 3D scatter requires specific format
    scat._offsets3d = (x, y, z)
    
    # Rotate the camera slowly to show 3D depth
    ax.view_init(elev=15, azim=frame * 0.5)
    
    return scat,

# --- Setup ---
z, r, theta_0, omega, star_colors = generate_galaxy_tree()

# Create Figure
fig = plt.figure(figsize=(10, 10), facecolor='black')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')

# Hide axes/grid (Space look)
ax.axis('off')
ax.grid(False)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Plot Initial Stars
# s=sizes. Taper sizes so top stars are smaller points.
sizes = 5 + 15 * (1-z) 
scat = ax.scatter(r*np.cos(theta_0), r*np.sin(theta_0), z, 
                  c=star_colors, s=sizes, depthshade=True)

# Add the "Topper" (The Supernova)
ax.scatter([0], [0], [1.02], c='white', s=200, marker='*', edgecolors='gold', linewidth=1.5)

# Add "Space Dust" (Background stars)
bg_x = np.random.uniform(-2, 2, 200)
bg_y = np.random.uniform(-2, 2, 200)
bg_z = np.random.uniform(0, 1.5, 200)
ax.scatter(bg_x, bg_y, bg_z, c='white', s=1, alpha=0.3)

# Title
plt.title("Keplerian Differential Rotation", color='white', fontfamily='monospace', y=0.95)

# Run Animation
ani = animation.FuncAnimation(fig, update, frames=int(DURATION*FPS), 
                              fargs=(z, r, theta_0, omega, scat, ax), 
                              interval=1000/FPS, blit=False)

print("Simulating Orbital Mechanics... 🎄")
# Save
ani.save('galactic_tree.mp4', writer='ffmpeg', fps=FPS, bitrate=3000)
print("Done! Check galactic_tree.mp4")
