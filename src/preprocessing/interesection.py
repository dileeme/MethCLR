"""
MethCLR — Pipeline Intersection Post-Processor
Run this AFTER all 12 beta matrices are complete.

What it does:
  1. Loads the probe index from each of the 12 beta matrices (no full data load)
  2. Computes the intersection of probe IDs across all 12 pipelines
  3. Reloads each matrix, filters to intersection probes, saves as _intersected.csv.gz
  4. Reports final probe count and any pipelines that contributed the most dropped probes

Output: 12 new files named norm=*_intersected.csv.gz in the same beta_matrices/ folder
The original files are preserved untouched.

Usage:
    python intersect_pipelines.py
"""

import os
import gzip
import time
import pandas as pd

BETA_DIR = r"C:\Users\dilsh\OneDrive\Desktop\MethCLR\beta_matrices"

# ─────────────────────────────────────────────
# STEP 1 — Find all 12 completed beta matrices
# ─────────────────────────────────────────────

def find_matrices():
    files = sorted([
        f for f in os.listdir(BETA_DIR)
        if f.endswith(".csv.gz") and not f.endswith("_intersected.csv.gz")
    ])
    if len(files) == 0:
        raise FileNotFoundError(f"No beta matrices found in {BETA_DIR}")
    if len(files) < 12:
        print(f"WARNING: Only {len(files)}/12 matrices found. Run on all 12 for best results.")
        print(f"  Found: {files}")
    else:
        print(f"Found {len(files)} beta matrices.")
    return files

# ─────────────────────────────────────────────
# STEP 2 — Load probe index only (fast, no full matrix load)
# ─────────────────────────────────────────────

def load_probe_index(filepath):
    """Read only the index column (probe IDs) without loading beta values."""
    path = os.path.join(BETA_DIR, filepath)
    # Read only first column (probe ID index) efficiently
    idx = pd.read_csv(path, usecols=[0], index_col=0).index
    return set(idx.tolist())

# ─────────────────────────────────────────────
# STEP 3 — Compute intersection
# ─────────────────────────────────────────────

def compute_intersection(files):
    print("\n[INTERSECT] Loading probe indices ...")
    probe_sets = {}
    for f in files:
        t0 = time.time()
        probes = load_probe_index(f)
        probe_sets[f] = probes
        print(f"  {f}: {len(probes):,} probes ({time.time()-t0:.1f}s)")

    intersection = set.intersection(*probe_sets.values())
    print(f"\n  Intersection: {len(intersection):,} probes across {len(files)} pipelines")

    # Report which pipelines dropped the most probes
    print("\n  Probes dropped per pipeline (relative to intersection):")
    for f, probes in sorted(probe_sets.items(), key=lambda x: len(x[1])):
        dropped = len(probes) - len(intersection)
        print(f"    {f}: {len(probes):,} probes → dropped {dropped:,} to reach intersection")

    return sorted(list(intersection))

# ─────────────────────────────────────────────
# STEP 4 — Filter and save intersected matrices
# ─────────────────────────────────────────────

def filter_and_save(files, intersection_probes):
    print(f"\n[FILTER] Filtering {len(files)} matrices to {len(intersection_probes):,} probes ...")
    probe_set = set(intersection_probes)

    for f in files:
        outname = f.replace(".csv.gz", "_intersected.csv.gz")
        outpath = os.path.join(BETA_DIR, outname)

        if os.path.exists(outpath):
            print(f"  SKIP (already exists): {outname}")
            continue

        print(f"  Processing: {f}")
        t0 = time.time()

        inpath = os.path.join(BETA_DIR, f)
        df = pd.read_csv(inpath, index_col=0)

        before = len(df)
        df = df[df.index.isin(probe_set)]
        # Reorder to consistent probe order across all matrices
        df = df.loc[intersection_probes]
        after = len(df)

        df.to_csv(outpath, compression="gzip")
        elapsed = time.time() - t0
        print(f"    {before:,} → {after:,} probes | saved in {elapsed:.1f}s → {outname}")

# ─────────────────────────────────────────────
# STEP 5 — Save probe list for MethCLR dataset class
# ─────────────────────────────────────────────

def save_probe_list(intersection_probes):
    probe_list_path = os.path.join(BETA_DIR, "probe_intersection.txt")
    with open(probe_list_path, "w") as f:
        f.write("\n".join(intersection_probes))
    print(f"\n[PROBES] Intersection probe list saved → {probe_list_path}")
    print(f"  This is your fixed input dimension for the MethCLR encoder.")
    print(f"  Input size: {len(intersection_probes):,} probes")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("MethCLR — Pipeline Intersection Post-Processor")
    print("=" * 60)

    files = find_matrices()
    intersection_probes = compute_intersection(files)
    filter_and_save(files, intersection_probes)
    save_probe_list(intersection_probes)

    print("\n" + "=" * 60)
    print(f"DONE — {len(files)} intersected matrices written")
    print(f"Fixed encoder input dimension: {len(intersection_probes):,} probes")
    print(f"Output: {BETA_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
