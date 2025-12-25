import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def apply_collatz_rule(number):
    """Applies the Collatz rule to a given number."""
    if number % 2 == 0:
        return number // 2
    else:
	return 3 * number + 1

def collatz_simulation(start_number):
    """Simulates the Collatz sequence for a given starting number."""
    sequence = [start_number]
    current_number = start_number

    while current_number != 1:
        current_number = apply_collatz_rule(current_number)
        sequence.append(current_number)

    return sequence

def update(frame):
    """Updates the animation frame with the current sequence data."""
    plt.cla()  # Clear the current axes

    if frame >= max_steps:
        anim.event_source.stop()  # Stop the animation when all steps are reached

    for i, sequence in enumerate(all_sequences):
        if frame < len(sequence):
            plt.plot(range(frame + 1), sequence[:frame + 1], linestyle='-', marker='o', markersize=5, label=f'Starting number: {start_numbers[i]}')

    plt.title('Collatz Conjecture Animation')
    plt.xlabel('Step number')
    plt.ylabel('Number')
    plt.legend(loc='upper right')
    plt.grid(True)

# Generate 10 random starting numbers
start_numbers = [random.randint(1, 100) for _ in range(10)]
# Perform Collatz simulation for each starting number
all_sequences = [collatz_simulation(start_number) for start_number in start_numbers]

# Determine the maximum number of steps across all sequences
max_steps = max(len(sequence) for sequence in all_sequences)

# Create the animation
fig = plt.figure(figsize=(10, 6))
anim = FuncAnimation(fig, update, frames=max_steps + 1, interval=350)

# Save the animation as a GIF file due to college cluster limitations
anim.save('collatz_animation.gif', fps=2, writer='pillow')

print("Animation saved as 'collatz_animation.gif'.")
