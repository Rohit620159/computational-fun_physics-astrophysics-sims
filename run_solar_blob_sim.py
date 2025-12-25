#!/usr/bin/env python
# Save this as run_solar_blob_sim.py
#
# --- VERSION 6: The "Toy Model" Sun ---
# --- Simulates a fast blob punching through a layered atmosphere ---
# --- (A dense "photosphere" and a hot "corona") ---

import numpy as np
from mpi4py import MPI
import os
import time

# --- Physical Constants (CGS) ---
KB_CGS = 1.380649e-16  # erg/K
MP_CGS = 1.6726219e-24 # g
G_SUN_CGS = 0.0           # Gravity is OFF
GAMMA = 5.0 / 3.0       # Adiabatic index for monatomic gas
MU = 0.6                # Mean molecular weight

# --- Simulation Parameters ---
GRID_DIMS = (64, 64, 128)      # Global grid size (X, Y, Z)
DOMAIN_SIZE = (1.0e9, 1.0e9, 2.0e9) # Domain size in cm (10Mm x 10Mm x 20Mm)
T_MAX_S = 300.0                # Longer sim: 5 minutes
SAVE_INTERVAL_S = 5.0          # Save every 5 seconds

# --- Layered Atmosphere Parameters ---
ATMOS = {
    'T_photosphere': 6000.0,    # 6,000 K "Sun"
    'rho_photosphere': 1.0e-10, # Dense "Sun"
    'T_corona': 1.0e6,      # 1,000,000 K "Corona"
    'rho_corona': 1.0e-16,    # Thin "Corona"
    'transition_z_km': 0.0    # Z=0 is the "surface"
}

# --- Blob Parameters (Absolute Values) ---
BLOB_PARAMS = {
    'center_z_km': -5000.0,  # Start *inside* the "photosphere"
    'center_x_km': 0.0,     
    'center_y_km': 0.0,     
    'radius_km': 300.0,      # A smaller, more focused blob
    'temp_blob': 5.0e6,      # 5 Million K (hotter than corona)
    'rho_blob': 1.0e-9,      # Very dense
    'velocity_km_s': 100.0   # Give it a 100 km/s "kick" to get it moving
}

class HydroSimulation:
    
    def __init__(self, grid_dims, domain_size, blob_params):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.n_procs = self.comm.Get_size()

        self.grid_global_N = np.array(grid_dims)
        self.domain_global_L = np.array(domain_size)
        
        if self.grid_global_N[2] % self.n_procs != 0:
            if self.rank == 0:
                print(f"Error: Z-dimension ({self.grid_global_N[2]}) not divisible by n_procs ({self.n_procs})")
            MPI.Finalize()
            exit()
        
        # --- THIS IS THE CRITICAL SECTION ---
        # 1. Define LOCAL grid size
        self.grid_local_N = self.grid_global_N.copy()
        self.grid_local_N[2] = self.grid_global_N[2] // self.n_procs # e.g., 128 / 8 = 16
        
        self.dxyz = self.domain_global_L / self.grid_global_N
        self.dx, self.dy, self.dz = self.dxyz
        
        self.local_z_start_idx = self.rank * self.grid_local_N[2]
        self.z_base_cm = -self.domain_global_L[2] / 2.0 + (self.local_z_start_idx + 0.5) * self.dz
        
        # 2. Use LOCAL grid size for local shape
        self.shape_local = (self.grid_local_N[0] + 2, self.grid_local_N[1] + 2, self.grid_local_N[2] + 2) # e.g., (66, 66, 18)
        self.sl = slice(1, -1) # Slice for "real" cells
        
        # 3. Use LOCAL grid size for coordinate arrays
        x_g = np.arange(self.grid_local_N[0] + 2) * self.dx - self.domain_global_L[0]/2.0 - (self.dx/2.0) # len 66
        y_g = np.arange(self.grid_local_N[1] + 2) * self.dy - self.domain_global_L[1]/2.0 - (self.dy/2.0) # len 66
        z_g = self.z_base_cm - (self.dz/2.0) + np.arange(self.grid_local_N[2] + 2) * self.dz           # len 18
        
        self.X, self.Y, self.Z = np.meshgrid(x_g, y_g, z_g, indexing='ij') # Z shape (66, 66, 18)
        
        # 4. Use LOCAL shape for U array
        self.U = np.zeros((5,) + self.shape_local) # U shape (5, 66, 66, 18)
        # --- END CRITICAL SECTION ---
        
        self.U_new = np.zeros_like(self.U)
        
        if self.rank == 0:
            print(f"Initialized {self.n_procs} ranks. Local Z-shape: {self.grid_local_N[2]}")
            
        self._set_layered_atmosphere(ATMOS)
        self._initialize_blob(BLOB_PARAMS)
        
        self.output_dir = "simulation_output"
        if self.rank == 0:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
        self.comm.Barrier() 

    def _get_pressure(self, rho, T):
        return (rho / (MU * MP_CGS)) * KB_CGS * T

    def _set_layered_atmosphere(self, p):
        if self.rank == 0:
            print("Setting up layered atmosphere (Photosphere + Corona)...")
            
        sl = self.sl
        transition_z_cm = p['transition_z_km'] * 1e5
        
        z_coords = self.Z[sl, sl, sl] # Shape (64, 64, 16)
        
        photosphere_mask = (z_coords <= transition_z_cm)
        corona_mask = (z_coords > transition_z_cm)
        
        P_photosphere = self._get_pressure(p['rho_photosphere'], p['T_photosphere'])
        P_corona = self._get_pressure(p['rho_corona'], p['T_corona'])
        
        rho_bg = np.zeros_like(z_coords) # Shape (64, 64, 16)
        rho_bg[photosphere_mask] = p['rho_photosphere']
        rho_bg[corona_mask] = p['rho_corona']
        
        E_int_bg = np.zeros_like(z_coords) # Shape (64, 64, 16)
        E_int_bg[photosphere_mask] = P_photosphere / (GAMMA - 1.0)
        E_int_bg[corona_mask] = P_corona / (GAMMA - 1.0)
        
        # This is line 132 (or near it):
        # Destination self.U[0, sl, sl, sl] shape is (64, 64, 16)
        # Source rho_bg shape is (64, 64, 16)
        # This will now work.
        self.U[0, sl, sl, sl] = rho_bg 
        self.U[1, sl, sl, sl] = 0.0
        self.U[2, sl, sl, sl] = 0.0
        self.U[3, sl, sl, sl] = 0.0
        self.U[4, sl, sl, sl] = E_int_bg
        
        self.comm.Barrier()
            
    def _initialize_blob(self, p):
        sl = self.sl
        
        center_x = p['center_x_km'] * 1e5
        center_y = p['center_y_km'] * 1e5
        center_z = p['center_z_km'] * 1e5
        radius = p['radius_km'] * 1e5
        v_z_blob = p['velocity_km_s'] * 1e5
        
        R = np.sqrt((self.X[sl, sl, sl] - center_x)**2 + 
                    (self.Y[sl, sl, sl] - center_y)**2 + 
                    (self.Z[sl, sl, sl] - center_z)**2)
        
        sigma = radius / 2.0
        blob_mask = np.exp(-0.5 * (R / sigma)**2)
        
        P_blob = self._get_pressure(p['rho_blob'], p['temp_blob'])
        
        rho_bg = self.U[0, sl, sl, sl]
        E_int_bg = self.U[4, sl, sl, sl]
        
        new_rho = rho_bg + (p['rho_blob'] - rho_bg) * blob_mask
        new_E_int = E_int_bg + (P_blob / (GAMMA - 1.0) - E_int_bg) * blob_mask
        
        self.U[0, sl, sl, sl] = new_rho
        self.U[3, sl, sl, sl] += new_rho * v_z_blob * blob_mask # Add velocity
        self.U[4, sl, sl, sl] = new_E_int + 0.5 * (self.U[3, sl, sl, sl]**2) / new_rho # E_int + E_kin
        
        if self.rank == 0:
            print(f"Injected hot blob (T={p['temp_blob']:.1e} K) at z={p['center_z_km']} km.")

    def _get_primitives(self, U_slice):
        rho = U_slice[0]
        rho = np.maximum(rho, 1e-30) 
        vx = U_slice[1] / rho
        vy = U_slice[2] / rho
        vz = U_slice[3] / rho
        E_kin = 0.5 * rho * (vx**2 + vy**2 + vz**2)
        E_int = U_slice[4] - E_kin
        P = E_int * (GAMMA - 1.0)
        P = np.maximum(P, 1e-30) 
        return rho, vx, vy, vz, P

    def _get_flux(self, U_slice):
        rho, vx, vy, vz, P = self._get_primitives(U_slice)
        E_tot = U_slice[4]

        Fx = np.zeros_like(U_slice)
        Fx[0] = rho * vx
        Fx[1] = rho * vx**2 + P
        Fx[2] = rho * vx * vy
        Fx[3] = rho * vx * vz
        Fx[4] = (E_tot + P) * vx

        Fy = np.zeros_like(U_slice)
        Fy[0] = rho * vy
        Fy[1] = rho * vy * vx
        Fy[2] = rho * vy**2 + P
        Fy[3] = rho * vy * vz
        Fy[4] = (E_tot + P) * vy

        Fz = np.zeros_like(U_slice)
        Fz[0] = rho * vz
        Fz[1] = rho * vz * vx
        Fz[2] = rho * vz * vy
        Fz[3] = rho * vz**2 + P
        Fz[4] = (E_tot + P) * vz
        
        return Fx, Fy, Fz

    def _communicate_boundaries(self):
        rank_up = self.rank + 1
        rank_down = self.rank - 1
        
        if rank_down >= 0:
            send_buf = self.U[:, :, :, 1].copy()
            recv_buf = np.empty_like(send_buf)
            self.comm.Sendrecv(send_buf, dest=rank_down, sendtag=0,
                               recvbuf=recv_buf, source=rank_down, recvtag=1)
            self.U[:, :, :, 0] = recv_buf
        else:
            self.U[:, :, :, 0] = self.U[:, :, :, 1]
            
        if rank_up < self.n_procs:
            send_buf = self.U[:, :, :, -2].copy()
            recv_buf = np.empty_like(send_buf)
            self.comm.Sendrecv(send_buf, dest=rank_up, sendtag=1,
                               recvbuf=recv_buf, source=rank_up, recvtag=0)
            self.U[:, :, :, -1] = recv_buf
        else:
            self.U[:, :, :, -1] = self.U[:, :, :, -2]

        sl = self.sl
        self.U[:, 0, sl, sl] = self.U[:, 1, sl, sl]
        self.U[:, -1, sl, sl] = self.U[:, -2, sl, sl]
        self.U[:, sl, 0, sl] = self.U[:, sl, 1, sl]
        self.U[:, sl, -1, sl] = self.U[:, sl, -2, sl]
            
    def _compute_timestep(self):
        sl = self.sl
        rho, vx, vy, vz, P = self._get_primitives(self.U[:, sl, sl, sl])
        
        c_s = np.sqrt(GAMMA * P / rho)
        
        max_vx = np.max(np.abs(vx) + c_s)
        max_vy = np.max(np.abs(vy) + c_s)
        max_vz = np.max(np.abs(vz) + c_s)
        
        max_vel_local = max(max_vx / self.dx, max_vy / self.dy, max_vz / self.dz)
        max_vel_global = self.comm.allreduce(max_vel_local, op=MPI.MAX)
        
        CFL = 0.4
        dt = CFL / (max_vel_global + 1e-10)
        return dt

    def advance_step(self, dt):
        
        self._communicate_boundaries()
        Fx, Fy, Fz = self.get_flux(self.U)
        
        sl = self.sl
        
        U_avg = ( self.U[:, 2:, sl, sl] + self.U[:, :-2, sl, sl] +
                  self.U[:, sl, 2:, sl] + self.U[:, sl, :-2, sl] +
                  self.U[:, sl, sl, 2:] + self.U[:, sl, sl, :-2] ) / 6.0
        
        div_Fx = (Fx[:, 2:, sl, sl] - Fx[:, :-2, sl, sl]) / (2.0 * self.dx)
        div_Fy = (Fy[:, sl, 2:, sl] - Fy[:, sl, :-2, sl]) / (2.0 * self.dy)
        div_Fz = (Fz[:, sl, sl, 2:] - Fz[:, sl, sl, :-2]) / (2.0 * self.dz)
        
        self.U_new[:, sl, sl, sl] = U_avg - dt * (div_Fx + div_Fy + div_Fz)

        # --- No source terms (G_SUN_CGS is 0) ---

        self.U[:, sl, sl, sl] = self.U_new[:, sl, sl, sl]

    # Make methods public
    get_primitives = _get_primitives
    get_flux = _get_flux

    def save_data(self, step):
        sl = self.sl
        
        rho, vx, vy, vz, P = self.get_primitives(self.U[:, sl, sl, sl])
        
        Temp = P * (MU * MP_CGS) / (rho * KB_CGS)
        Temp = np.maximum(Temp, 10.0) # Temp floor
        
        save_data = {
            'rho': rho,
            'temp': Temp,
            'vz': vz
        }
        
        filename = os.path.join(self.output_dir, f"snap_{step:04d}_rank{self.rank:03d}.npz")
        np.savez(filename, **save_data)
        
        if self.rank == 0:
            print(f"--- Saved snapshot {step} at t = {self.t:.2f} s ---")
            
    def run(self):
        self.t = 0.0
        step = 0
        last_save_time = -np.inf
        
        if self.rank == 0:
            print("Starting simulation... (Using v6 'Toy Model Sun')")
            
        sim_start_time = time.time()
        
        while self.t < T_MAX_S:
            dt = self._compute_timestep()
            
            if dt < 1e-5: 
                if self.rank == 0: print("\nTimestep too small. Simulation unstable."); 
                break
            if self.t + dt > T_MAX_S: dt = T_MAX_S - self.t
                
            self.advance_step(dt)
            self.t += dt
            step += 1
            
            if self.rank == 0:
                print(f"t = {self.t:6.2f} s / {T_MAX_S:6.2f} s (dt = {dt:.3e} s, step = {step})", end='\r')
            
            if self.t >= last_save_time + SAVE_INTERVAL_S:
                self.save_data(step)
                last_save_time = self.t
                
        self.save_data(step)
        
        self.comm.Barrier()
        if self.rank == 0:
            total_time = time.time() - sim_start_time
            print(f"\nSimulation finished in {total_time:.2f} seconds.")

if __name__ == "__main__":
    sim = HydroSimulation(GRID_DIMS, DOMAIN_SIZE, BLOB_PARAMS)
    sim.run()
