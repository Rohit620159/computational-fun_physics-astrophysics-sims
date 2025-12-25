# Computational Astrophysics Simulations

A collection of physics-based simulations and data pipelines developed for exploring stellar atmospheres, magnetohydrodynamics (MHD), and non-equilibrium thermodynamics.

## 📂 Project Modules

### 1. Non-Equilibrium Phase Transition (Snowflake)
**Files:** `snowflake_sim.py`, `cosmic_snowflake_final.mp4`
* **Physics:** Solves the **Ginzburg-Landau phase-field equations** coupled with a heat diffusion equation.
* **Method:** Finite-difference scalar field evolution using **Numba** for JIT-compiled parallelization on HPC clusters.
* **Key Feature:** Simulates dendritic solidification where macroscopic symmetry (6-fold) emerges from microscopic surface tension anisotropy.

### 2. Solar Atmosphere Dynamics (Blob Ejection)
**Files:** `blob_animation_solar.py`, `run_solar_blob_sim.py`
* **Physics:** Simulation of plasma instabilities (likely Rayleigh-Taylor or convective overshooting) in the solar photosphere/chromosphere.
* **Visualization:** Animated evolution of plasma density/temperature perturbations ("blobs") moving through stratified solar layers.

### 3. FALC Model Atmosphere Pipeline
**Files:** `parse_falc.py`, `prepare_atmosphere.py`, `falc_model.csv`
* **Context:** Data pipeline for the **Fontenla-Avrett-Loeser (FALC)** semi-empirical model of the solar atmosphere.
* **Function:** Parses, cleans, and structures raw FALC data (Temperature, Density, Ionization fractions) for use in radiative transfer simulations.

### 5. Electrodynamics & Photonics (FDTD Maxwell Solvers) ⚡
**Files:** `Modulated_Permitivity_FDTD_Maxwell_light.py`, `Sophisticated_MetaMaterials...py`
* **Physics:** Finite-Difference Time-Domain (FDTD) simulations of light propagating through materials with modulated permittivity.
* **Application:** Modeling **metamaterials** and optical wave behavior in complex media.

### 6. Keplerian Differential Rotation (Galactic Christmas Tree) 🎄
**Files:** `galactic_tree.py`, `galactic_tree.mp4`
* **Physics:** Simulates **orbital mechanics** and **differential rotation** within a conical volume. It demonstrates **Kepler's 3rd Law** ($v \propto r^{-1/2}$), where inner particles orbit significantly faster than outer ones, mimicking the rotation curves of spiral galaxies and accretion disks.
* **Visualization:** A 3D particle system that starts as a cone (Christmas Tree) and naturally shears into spiral arms due to the conservation of angular momentum.

### 7. Classical Mechanics & Number Theory 🎲
**Files:** `Gravity_influenced_motion.py`, `Collatz_conjecture.py`
* **Physics:** N-body gravity simulations and visualizations of the Collatz (3n+1) stopping times.

## 🛠️ Tech Stack
* **Compute:** Python 3.11, NumPy, SciPy
* **Acceleration:** Numba (JIT), Multiprocessing
* **Visualization:** Matplotlib, FFmpeg
* **Environment:** HPC Cluster (Hydra/CFA)

---
*Author: Rohit Raj*
