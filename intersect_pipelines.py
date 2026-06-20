"""
MethCLR — Pipeline Intersection Script

Reads all 12 beta matrices from beta_matrices/, computes the probe intersection
across all pipelines, trims each matrix to the shared probe set, and writes
12 _intersected.csv.gz files alongside probe_intersection.txt.

Run after all 12 pipelines complete:
    python intersect_pipelines.py
"""

import os, time
import pandas as pd

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

INPUT_DIR  = r"C:\Users\dilsh\OneDrive\Desktop\MethCLR\beta_matrices"
OUTPUT_DIR = r"C:\Users\dilsh\OneDrive\Desktop\MethCLR\beta_matrices"
LOG_FILE   = os.path.join(OUTPUT_DIR, "probe_intersection.txt")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("MethCLR — Pipeline Intersection")
    print("=" * 60)

    # Discover input files (exclude already-intersected outputs)
    all_files = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.endswith(".csv.gz") and "_intersected" not in f
    ])

    if len(all_files) == 0:
        raise FileNotFoundError(f"No .csv.gz files found in {INPUT_DIR}")

    print(f"\n[LOAD] Found {len(all_files)} pipeline matrices\n")

    # ── Pass 1: load index (probe IDs) only ──────────────────────
    probe_sets = {}
    for fname in all_files:
        fpath = os.path.join(INPUT_DIR, fname)
        print(f"  Reading probe index: {fname}")
        df_index = pd.read_csv(fpath, index_col=0, usecols=[0], compression="gzip")
        probe_sets[fname] = set(df_index.index.tolist())
        print(f"    Probes: {len(probe_sets[fname]):,}")

    # ── Compute intersection ──────────────────────────────────────
    shared_probes = set.intersection(*probe_sets.values())
    shared_probes_sorted = sorted(shared_probes)
    print(f"\n[INTERSECT] Shared probes across all {len(all_files)} pipelines: {len(shared_probes_sorted):,}\n")

    # Per-pipeline probe counts for the log
    probe_counts = {fname: len(s) for fname, s in probe_sets.items()}

    # ── Pass 2: load full matrix, filter, write ───────────────────
    t_global = time.time()
    for i, fname in enumerate(all_files, 1):
        out_name = fname.replace(".csv.gz", "_intersected.csv.gz")
        out_path = os.path.join(OUTPUT_DIR, out_name)

        if os.path.exists(out_path):
            print(f"  ({i}/{len(all_files)}) SKIP: {out_name}")
            continue

        print(f"  ({i}/{len(all_files)}) Processing: {fname}")
        t0 = time.time()

        fpath = os.path.join(INPUT_DIR, fname)
        df = pd.read_csv(fpath, index_col=0, compression="gzip")
        print(f"    Loaded shape: {df.shape}")

        df_filtered = df.loc[shared_probes_sorted]
        print(f"    Filtered shape: {df_filtered.shape}")

        df_filtered.to_csv(out_path, compression="gzip")
        elapsed = time.time() - t0
        print(f"    Saved → {out_name} ({elapsed:.1f}s)")

    # ── Write probe intersection log ──────────────────────────────
    with open(LOG_FILE, "w") as f:
        f.write("MethCLR — Probe Intersection Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Pipelines intersected: {len(all_files)}\n")
        f.write(f"Final intersection probe count: {len(shared_probes_sorted):,}\n\n")
        f.write("Per-pipeline probe counts (before intersection):\n")
        for fname, count in probe_counts.items():
            f.write(f"  {fname}: {count:,}\n")
        f.write(f"\nIntersection probe count: {len(shared_probes_sorted):,}\n")
        f.write(f"This is the fixed input dimension for the MethCLR encoder.\n")

    total = time.time() - t_global
    print("\n" + "=" * 60)
    print(f"DONE — {len(all_files)} intersected matrices written in {total/60:.1f} min")
    print(f"Intersection probe count: {len(shared_probes_sorted):,}  (see probe_intersection.txt)")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
