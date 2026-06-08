# run_bootstrap_auto.py
import numpy as np
from alma_dip_bootstrapping import DIPConfig, bootstrap_reconstruct, reconstruct_dip, pick_device
import astropy.io.fits as pyfits
import pdb

# Load data (u, v, Re, Im, w), skipping header
data = np.loadtxt("DG_Tau.txt", skiprows=1).astype(np.float32)
uv = data[:, :2]
vis = data[:, 2:4]
w   = data[:, 4]

cfg = DIPConfig(
    num_iters=1800,              # consider fewer iters per replicate to keep wall time manageable
    lr=1e-3,
    tv_weight=1e-2,
    cell_size_arcsec=0.00075*4,
    out_every=200,
    nu=3.0,
    learn_sigma=False,

    # auto-selector
    force_mode="auto",           # "auto" | "small" | "memory"
    auto_threshold_n=5_000_000,  # choose memory path if N > 5M

    # memory-path knobs (ignored if small path is chosen)
    batch_vis=16384,
    stream_chunk_vis=32768,
    radial_bins=48,
    radial_quantiles=True,
    low_k_bins=6,
    low_k_boost=6.0,
    alpha_uniform=0.05,
    alpha_radial=0.55,
    alpha_inv_radius=0.40,
    quantile_mode="auto",
)

device = str(pick_device("mps"))  # or "cuda" / "cpu"
print("Using device:", device)

image, dirty, beam = reconstruct_dip(
    uv=uv, vis=vis, weight=w,
    img_size=(540, 540),
    cfg=cfg, device=device
)

pyfits.writeto('DG_Tau_alma_dip_ref_bootstrapping.fits', image, overwrite=True)
pyfits.writeto('DG_Tau_alma_dip_ref_bootstrapping_dirty.fits', dirty, overwrite=True)
pyfits.writeto('DG_Tau_alma_dip_ref_bootstrapping_beam.fits', beam, overwrite=True)

pdb.set_trace()

boot = bootstrap_reconstruct(
    uv=uv, vis=vis, weight=w,
    img_size=(540, 540),
    cfg=cfg, device=device,
    B=20,                        # number of bootstrap replicates
    method="poisson",            # "poisson" or "bayesian"
    percentiles=(16, 84),
    return_all=False
)

# Save maps
pyfits.writeto("DG_Tau_bootstrap_mean.fits", boot["mean"], overwrite=True)
pyfits.writeto("DG_Tau_bootstrap_std.fits",  boot["std"],  overwrite=True)
pyfits.writeto("DG_Tau_bootstrap_p16.fits",  boot["p_lo"], overwrite=True)
pyfits.writeto("DG_Tau_bootstrap_p84.fits",  boot["p_hi"], overwrite=True)
print("Saved uncertainty maps (mean/std/p16/p84).")
