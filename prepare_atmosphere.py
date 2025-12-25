# Save this as prepare_atmosphere.py
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
import scipy.constants as const

# --- Physical Constants (in CGS units) ---
KB_CGS = const.k  # Boltzmann constant in erg/K
MP_CGS = const.m_p * 1000.0  # Proton mass in g
ME_CGS = const.m_e * 1000.0  # Electron mass in g
G_CGS = const.G * 1000.0 * 100**2 # Gravitational constant in cm^3 g^-1 s^-2
MSUN_CGS = 1.98847e33  # Sun mass in g (CGS units)# Sun mass in g
RSUN_CGS = 6.957e10  # Sun radius in cm

def prepare_1d_atmosphere(model_file, hydro_file, output_file):
    """
    Loads the FALC model data, computes height, pressure, and density,
    and saves a processed file for the simulation.
    """
    print("Loading data...")
    df_model = pd.read_csv(model_file)
    df_hydro = pd.read_csv(hydro_file)

    # The parsed files have a 1-row mismatch. Truncate the longer one.
    min_rows = min(len(df_model), len(df_hydro))
    df_model = df_model.iloc[:min_rows]
    df_hydro = df_hydro.iloc[:min_rows]

    # --- 1. Calculate Mass Density (rho) ---
    # We assume a 10% Helium abundance by number relative to Hydrogen
    he_abundance = 0.1
    m_H = 1.008 * MP_CGS  # Mass of Hydrogen
    m_He = 4.0026 * MP_CGS # Mass of Helium

    # Total hydrogen number density (neutral + ionized)
    n_H_total = (
        df_hydro['nh(1)'] + df_hydro['nh(2)'] + df_hydro['nh(3)'] +
        df_hydro['nh(4)'] + df_hydro['nh(5)'] + df_hydro['np']
    )
    # Helium number density
    n_He = he_abundance * n_H_total
    
    # Mass density (rho) = mass_H + mass_He
    rho = (n_H_total * m_H) + (n_He * m_He)
    df_model['rho'] = rho
    print(f"Calculated density. Min: {rho.min():.2e}, Max: {rho.max():.2e} g/cm^3")

    # --- 2. Calculate Gas Pressure (P_gas) ---
    # Total particle number density (electrons + H + He)
    # Assume He is mostly neutral or singly ionized at these temps,
    # and contributes n_He particles. This is a simplification.
    n_total = df_model['Ne'] + n_H_total + n_He
    P_gas = n_total * KB_CGS * df_model['Temperature']
    df_model['P_gas'] = P_gas
    
    # --- 3. Calculate Height (z) ---
    # We integrate the equation of hydrostatic equilibrium: dP/dz = -rho * g
    # Or, in column mass (m): dm = -rho * dz  =>  dz = -dm / rho
    # log_g = 4.44 => g = 10**4.44 cm/s^2
    g_sun = 10**4.44
    
    # Column mass 'm' from 'lg_col_mass'
    m_col = 10**df_model['lg_col_mass']
    
    # We need to integrate dz = -dm/rho. We'll use trapezoidal rule.
    # We sort by m_col to integrate from top-down (low m_col to high m_col)
    df_model = df_model.sort_values(by='lg_col_mass')
    
    # Re-calculate m_col and rho after sorting
    m_col = 10**df_model['lg_col_mass']
    rho = df_model['rho']
    
    # `cumtrapz` needs dx and y. We'll use m_col as 'x' and (1/rho) as 'y'.
    # We are integrating -1/rho(m) dm
    # The integral is from m_top (smallest m) to m_current.
    heights = cumulative_trapezoid(-1.0 / rho, x=m_col, initial=0.0)
    
    # This gives height relative to the top boundary.
    # We'll set the z=0 point at the photosphere, often defined
    # where T = 5770K. Let's find that index.
    photosphere_idx = (df_model['Temperature'] - 5770).abs().idxmin()
    photosphere_height = heights[photosphere_idx]
    
    # Shift all heights so z=0 is at the photosphere
    df_model['z_height_cm'] = heights - photosphere_height
    df_model['z_height_km'] = df_model['z_height_cm'] / 1e5 # Convert to km
    
    print(f"Calculated heights. Range: {df_model['z_height_km'].min():.1f} km to {df_model['z_height_km'].max():.1f} km")
    print(f"Photosphere (T~5770K) set at z=0 km.")

    # --- 4. Save Processed File ---
    df_model.to_csv(output_file, index=False)
    print(f"Processed 1D atmosphere saved to {output_file}")


if __name__ == "__main__":
    prepare_1d_atmosphere("falc_model.csv", "falc_hydrogen.csv", "falc_processed.csv")
