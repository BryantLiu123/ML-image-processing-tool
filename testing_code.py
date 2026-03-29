"""
Crack Propagation - CFG Grain Boundary + Damage Binning Pipeline
=========================================================================
Reads extended CFG files (fractional coordinates) and produces a
2-channel square grid per timestep for ML crack propagation training:

  Channel 0 — Grain boundary map  : fraction of grainIDs in each bin
                                     0 = pure single grain interior
                                     1 = maximum grain boundary mixing
  Channel 1 — Void/crack map      : 1 where no atoms exist (crack/void)
                                     0 where atoms are present

CFG format assumptions:
  - Fractional (reduced) coordinates, scaled by H0 box matrix
  - Species interleaved by mass block (Si=28.085, C=12.011)

Requirements:
    pip install numpy matplotlib scipy

Usage:
    python testing_code.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu

# ─────────────────────────────────────────────
# CONFIGURATION — edit these
# ─────────────────────────────────────────────
CFG_FILES = ["dump.cyclic_78000.cfg"]     # Individual processing (enter names of files here)


# Batch detection
CFG_FILES = sorted(Path(".").glob("*.cfg"))
# CFG_FILES = sorted(Path(".").glob("model_1*_78000.cfg"))

#For parse_cfg function
columns_per_line = 9
voronoi_volume_column_number = 9   

OUTPUT_DIR  = Path("grain_crack_maps")  # Output folder
GRID_SIZE   = 192                       # Bin resolution (192x192)

# Grain boundary smoothing: Gaussian blur on the boundary channel to soften
# hard edges (helps ML generalisation). Set to 0.0 to disable.
BOUNDARY_BLUR_SIGMA = 1.0              # pixels; 0.5–2.0 works well

# visualisation control ---------------------------------------------------
OVERLAY_BOUNDARY_BOOST = 1.0           # multiplier for boundaries in overlay (1=none)

# ─────────────────────────────────────────────


def parse_cfg(filepath: str):
    # CFG Parsing Function
    positions = []
    grain_ids = []
    box = np.zeros(3, dtype=np.float64)

    with open(filepath, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # ── Box vectors ──
            if line.startswith("H0(1,1)"):
                box[0] = float(line.split("=")[1].split()[0])
                continue
            elif line.startswith("H0(2,2)"):
                box[1] = float(line.split("=")[1].split()[0])
                continue
            elif line.startswith("H0(3,3)"):
                box[2] = float(line.split("=")[1].split()[0])  # FIX 1
                continue

            # ── Skip header lines ──
            if (line.startswith("Number") or line.startswith("A =") or
                    line.startswith("H0") or line.startswith(".NO") or
                    line.startswith("entry_count") or line.startswith("auxiliary")):
                continue

            parts = line.split()

            # ── Skip mass and element lines ──
            if len(parts) == 1 or len(parts) == 2:
                continue

            # ── Atom data: flexible number of columns ──
            if len(parts) == columns_per_line:                              # FIX 2
                try:
                    sx = float(parts[0])  
                    sy = float(parts[1])
                    sz = float(parts[2])
                    grain_id = int(float(parts[voronoi_volume_column_number - 1]))          
                    positions.append([sx * box[0], sy * box[1], sz * box[2]])
                    grain_ids.append(grain_id)
                except ValueError:
                    continue

    return (np.array(positions, dtype=np.float64),
            np.array(grain_ids, dtype=np.int32),
            box)


def build_grids(positions: np.ndarray,
                grain_ids: np.ndarray,
                box: np.ndarray,
                grid_size: int = 192,
                blur_sigma: float = 1.0):
    """
    Returns two (grid_size, grid_size) float32 arrays:
        boundary_grid : grain boundary mixing score per bin  [0..1]
        void_grid     : 1 where no atoms present (crack/void), 0 elsewhere
    """
    Lx, Ly = box[0], box[1] #lengths in xy directions

    # ── Map atom positions to 2D bin indices (project onto XY plane) ──
    xi = np.clip((positions[:, 0] / Lx * grid_size).astype(int), 0, grid_size - 1) #convert position into position on grid
    yi = np.clip((positions[:, 1] / Ly * grid_size).astype(int), 0, grid_size - 1)

    flat_idx = yi * grid_size + xi          # 1D flattened bin index

    # ── Atom count per bin ──
    atom_count = np.bincount(flat_idx, minlength=grid_size * grid_size)
    atom_count = atom_count.reshape(grid_size, grid_size)

    # ── Grain boundary score ──
    # For each bin: count distinct grainIDs present.
    # Score = (unique_count - 1) / (n_grains_total - 1)
    #   → 0.0 for a pure single-grain bin
    #   → 1.0 for a bin containing all grains (maximum mixing)
    n_grains_total = int(grain_ids.max())   # largest value for volume

    # Sort atoms by bin index for fast group-by
    order      = np.argsort(flat_idx)
    sorted_idx = flat_idx[order]
    sorted_gid = grain_ids[order]

    unique_grains_per_bin = np.zeros(grid_size * grid_size, dtype=np.int32) #numerator
    print(f"unique grains per bin{unique_grains_per_bin}")
    split_points = np.where(np.diff(sorted_idx))[0] + 1
    groups   = np.split(sorted_gid, split_points)
    bin_ids  = sorted_idx[np.concatenate(([0], split_points))]

    for b, g in zip(bin_ids, groups):
        unique_grains_per_bin[b] = len(np.unique(g))

    # Use raw unique count per bin as score (no normalization by max volume/grain count)
    boundary_flat = np.where(
        atom_count.ravel() > 0,
        unique_grains_per_bin.astype(np.float32),
        0.0
    )
    boundary_grid = boundary_flat.reshape(grid_size, grid_size).astype(np.float32)

    # Optional Gaussian smoothing to soften hard boundary edges
    if blur_sigma > 0:
        boundary_grid = gaussian_filter(boundary_grid, sigma=blur_sigma).astype(np.float32)
    # filter
    nonzero_vals = boundary_grid[boundary_grid > 0]
    if len(nonzero_vals) > 0:
        thresh = threshold_otsu(nonzero_vals) * 1  # push threshold higher for cleaner boundaries. e.g. change 1 to 1.25
        boundary_grid = np.where(boundary_grid > thresh, boundary_grid, 0.0).astype(np.float32)
        print(f"  Auto threshold: {thresh:.4f}")

    # ── Void / crack map ──
    void_grid = (atom_count == 0).astype(np.float32)

    return boundary_grid, void_grid


def save_outputs(boundary: np.ndarray,
                 void: np.ndarray,
                 stem: str,
                 out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    #save outputs to png file

    # ── Combined 2-channel .npy for ML: shape (2, 192, 192) ──
    combined = np.stack([boundary, void], axis=0)
    npy_path = out_dir / f"{stem}_2ch.npy"
    np.save(npy_path, combined)

    # ── Diagnostic PNG (3 panels) ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # use a shrunk scale on channel-0?
    im0 = axes[0].imshow(boundary, origin="lower", cmap="hot", vmin=0, vmax=1)
    axes[0].set_title("Ch0: Grain Boundary Score\n(0=bulk, 1=white)")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(void, origin="lower", cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("Ch1: Void / Crack Map\n(1=crack/void, 0=material)")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Overlay: grain boundaries=orange, cracks=blue, bulk=black
    # boundary values may be quite small after blurring; boost them for visibility
    overlay = np.zeros((*boundary.shape, 3), dtype=np.float32)
    # amplify boundary channel (clip at 1) so thin boundaries stand out
    boundary_vis = np.clip(boundary * OVERLAY_BOUNDARY_BOOST, 0.0, 1.0)
    overlay[..., 0] = boundary_vis          # R channel — grain boundaries (orange)
    overlay[..., 1] = boundary_vis * 0.4   # G channel
    overlay[..., 2] = void                 # B channel — cracks/voids (blue)
    axes[2].imshow(overlay, origin="lower")
    axes[2].set_title("Overlay\norange=grain boundary | blue=crack")
    axes[2].axis("off")

    fig.suptitle(stem, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_diagnostic.png", dpi=150)
    plt.close(fig)

    n_boundary_bins = (boundary > 0.05).sum()
    print(f"  Saved : {npy_path.name}")
    print(f"  Stats : boundary bins={n_boundary_bins} | "
          f"void fraction={void.mean()*100:.1f}%")


def main():
    print(f"Grid size           : {GRID_SIZE}x{GRID_SIZE}")
    print(f"Boundary blur sigma : {BOUNDARY_BLUR_SIGMA}")
    print(f"Output              : {OUTPUT_DIR}/\n")

    for cfg_file in CFG_FILES:
        stem = Path(cfg_file).stem
        print(f"Processing {cfg_file} ...")
        try:
            positions, grain_ids, box = parse_cfg(cfg_file)
            print(f"  Parsed {len(positions):,} atoms | "
                  f"box {box[0]:.1f} x {box[1]:.1f} x {box[2]:.1f} Å | "
                  f"{grain_ids.max()} grains")

            boundary, void = build_grids(
                positions, grain_ids, box,
                grid_size=GRID_SIZE,
                blur_sigma=BOUNDARY_BLUR_SIGMA,
            )
            save_outputs(boundary, void, stem, OUTPUT_DIR)

        except Exception as e:
            print(f"  ERROR: {e}")
            raise

    print(f"\nDone. ML arrays shape = (2, {GRID_SIZE}, {GRID_SIZE})")
    print("  channel 0 = grain boundary score")
    print("  channel 1 = crack / void mask")

if __name__ == "__main__":
    main()