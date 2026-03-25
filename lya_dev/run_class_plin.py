import numpy as np
import json
from pathlib import Path
from classy import Class

# These are the params from lya_dev/plot_sherwood_bestfit_mu_bins.py
BASELINE_COSMO = {
    "h": 0.678,
    "omega_b": 0.0482 * 0.678**2,
    "omega_cdm": (0.308 - 0.0482) * 0.678**2,
    "ln10^{10}A_s": 3.044,
    "n_s": 0.961,
}
Z = 2.8

def main():
    print(f"Running CLASS at z={Z}...")
    
    # Setup CLASS
    M = Class()
    M.set({
        'h': BASELINE_COSMO['h'],
        'omega_b': BASELINE_COSMO['omega_b'],
        'omega_cdm': BASELINE_COSMO['omega_cdm'],
        'ln10^{10}A_s': BASELINE_COSMO['ln10^{10}A_s'],
        'n_s': BASELINE_COSMO['n_s'],
        'output': 'mPk',
        'P_k_max_h/Mpc': 50.0,
        'z_max_pk': Z,
    })
    M.compute()
    
    # Create k grid
    # PyBird Symbolic used logspace(-5, 1, 512) or similar
    # We want a dense enough grid for interpolation
    kk = np.geomspace(1e-5, 50.0, 1000)
    
    # Get P(k) in (Mpc/h)^3
    h = M.h()
    pk = np.array([M.pk_lin(k * h, Z) * h**3 for k in kk])
    
    # Get growth factor f
    f_val = M.scale_independent_growth_factor_f(Z)
    
    # Save to npz
    output_path = Path("lya-dev/output/class_plin_z28.npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, kk=kk, pk=pk, f=f_val, z=Z)
    print(f"Saved CLASS results to {output_path}")

if __name__ == "__main__":
    main()
