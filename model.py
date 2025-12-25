## Will try to explain code much, as asked while introducing the project
#importing required librairies
import numpy as np
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap

#Defining class - Ising model & its properties
class Ising2D:
    def __init__(                        #initializes
        itself, Nx: int, Ny: int, J: float, beta: float,                #defines size N of lattice , interaction energy J , inverse temp Beta
        start_type=None
    ):
        itself.Nx, itself.Ny = Nx, Ny                                   #size of grid
        itself.J, itself.beta = J, beta

        if start_type is None:                                   #starting state alignes- all spin up or down
            start_type = 'aligned'

        if start_type == 'aligned':
            itself.spins = np.ones([Nx, Ny], dtype=int) * np.random.choice([-1, 1])       #grid of spins of spins mostly alined using np ones
        else:
            itself.spins = np.random.choice([-1, 1], size=[Nx, Ny])          #random grid of spins - not aligned
        itself.energy = itself._energy()
        itself.magnetisation = itself.spins.sum()                             #initial energy and magnetisation
         
    def _energy(itself):                                       #total energy of system by iterating over all NEIGHBORS and adding J
        interaction_energy = -itself.J * (
            (itself.spins[:-1, :] * itself.spins[1:, :]).sum() +
            (itself.spins[:, :-1] * itself.spins[:, 1:]).sum() +               #boundary conditions - wrapping lattice around
            (itself.spins[0, :] * itself.spins[-1, :]).sum() +
            (itself.spins[:, 0] * itself.spins[:, -1]).sum()
        )
        return interaction_energy
    def delta_energy(itself, i, j):                                       #difference in energy due to each flip
        delta_interaction = 2 * itself.J * itself.spins[i, j] * (    #to avoid double counting reff in notes 
            itself.spins[i-1, j] + itself.spins[i, j-1] +
            itself.spins[(i+1) if i+1 < itself.Nx else 0, j] +
            itself.spins[i, (j+1) if j+1 < itself.Ny else 0]
        )
        return delta_interaction

    def random_update(itself):
        i = np.random.randint(0, itself.Nx)           #randomly selected
        j = np.random.randint(0, itself.Ny)
        deltaE = itself.delta_energy(i, j)      #change in energy due single flip
        if deltaE <= 0 or np.exp(-itself.beta * deltaE) > np.random.rand():        # criteria for accepting that change
            itself.spins[i, j] *= -1
            itself.energy += deltaE
            itself.magnetisation += 2 * itself.spins[i, j]                     #adding new values if spin accepted
#Parameters
N_trials = 5        #No of times to run the simulation for each temp 
N_equilibrate, N_compute = 100000, 100000    # steps to reach equilibrium then computing steps
Nx, Ny = 10, 10         #lattice size
N = Nx * Ny
J = 1                   #interaction factor
betas = np.geomspace(0.05, 2, 101)       #geometric space/power of 10 

fig, axs = plt.subplots(2, 2, sharex=True, figsize=(10, 5))
axs = [ax for row in axs for ax in row]

for beta in tqdm(betas, leave=False):
    energies, magnetisations, heatcapacities, susceptibilities = [], [], [], []      
    quantities = [energies, magnetisations, heatcapacities, susceptibilities]
    for trial in range(N_trials):           #MC runs
        system = Ising2D(Nx, Ny, J, beta, 'hot')
        for i in range(N_equilibrate):            #to reach equilibrium, properties not measured 
            system.random_update()
            
        energy = np.zeros([N_compute])
        magnetisation = np.zeros([N_compute])
        for i in range(N_compute):            # now measuring properties for n compute steps
            system.random_update()
            energy[i] = system.energy
            magnetisation[i] = system.magnetisation
                                                          #using formula to calculate values of these properties now
        energies.append(np.mean(energy) / N)
        magnetisations.append(np.abs(np.mean(magnetisation)) / N)
        heatcapacities.append(beta ** 2 * (np.mean(energy ** 2) - np.mean(energy) ** 2) / N)
        susceptibilities.append(beta * (np.mean(magnetisation ** 2) - np.mean(magnetisation) ** 2) / N)
    for ax, quantity in zip(axs, quantities):
        ax.scatter([1/beta], [np.median(quantity)], color='k', s=5)    #plotting median of quantities vs temperature.

axs[0].set_xscale('log')          #x axis is on log scale 
axs[0].set_ylabel('Energy')
axs[1].set_ylabel('Magnetisation')
axs[2].set_ylabel('Heat capacity')
axs[3].set_ylabel('Susceptibility')
for ax in axs:
    ax.set_xlabel(r'$k_B T$')
    
plt.savefig("2D_ising_model_plot.png", bbox_inches="tight")   #save plot


