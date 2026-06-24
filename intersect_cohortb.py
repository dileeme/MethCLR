"""
MethCLR — Cohort B Probe Intersection
1. Intersects 12 Cohort B beta matrices with each other
2. Intersects result with Cohort A's probe set (132,337 probes)
3. Writes 12 _intersected.csv.gz files for Cohort B
"""

import os, time
import pandas as pd

COHORT_B_DIR     = r"C:\Users\dilsh\OneDrive\Desktop\MethCLR\cohort_b_beta_matrices"
COHORT_A_MATRIX  = r"C:\Users\dilsh\OneDrive\Desktop\MethCLR\beta_matrices\norm=sesame_snp=True_sex=True_intersected.csv.gz"
OUTPUT_DIR       = COHORT_B_DIR
LOG_FILE         = os.path.join(OUTPUT_DIR, "probe_intersection_cohortb.txt")

def load_cohort_a_probes():
    print(f"  Loading Cohort A probe index from intersected matrix ...")
    df = pd.read_csv(COHORT_A_MATRIX, index_col=0, usecols=[0], compression="gzip")
    probes = set(df.index.tolist())
    print(f"  Cohort A probes loaded: {len(probes):,}")
    return probes

def main():
    print("=" * 60)
    print("MethCLR — Cohort B Probe Intersection")
    print("=" * 60)

    all_files = sorted([
        f for f in os.listdir(COHORT_B_DIR)
        if f.endswith(".csv.gz") and "_intersected" not in f
    ])
    print(f"\n[LOAD] Found {len(all_files)} Cohort B matrices")

    # Pass 1: load probe indices only
    probe_sets = {}
    for fname in all_files:
        fpath = os.path.join(COHORT_B_DIR, fname)
        print(f"  Reading probe index: {fname}")
        df_index = pd.read_csv(fpath, index_col=0, usecols=[0], compression="gzip")
        probe_sets[fname] = set(df_index.index.tolist())
        print(f"    Probes: {len(probe_sets[fname]):,}")

    # Intersect across all 12 Cohort B matrices
    cohort_b_shared = set.intersection(*probe_sets.values())
    print(f"\n[INTERSECT] Cohort B internal intersection: {len(cohort_b_shared):,} probes")

    # Intersect with Cohort A probe set
    print(f"\n[LOAD] Loading Cohort A probe set ...")
    cohort_a_probes = load_cohort_a_probes()
    final_probes = sorted(cohort_b_shared & cohort_a_probes)
    print(f"\n[INTERSECT] Final cross-cohort intersection: {len(final_probes):,} probes")

    # Pass 2: filter and save
    t_global = time.time()
    for i, fname in enumerate(all_files, 1):
        out_name = fname.replace(".csv.gz", "_intersected.csv.gz")
        out_path = os.path.join(OUTPUT_DIR, out_name)
        if os.path.exists(out_path):
            print(f"  ({i}/{len(all_files)}) SKIP: {out_name}")
            continue

        print(f"  ({i}/{len(all_files)}) Processing: {fname}")
        t0 = time.time()
        df = pd.read_csv(os.path.join(COHORT_B_DIR, fname), index_col=0, compression="gzip")
        df_filtered = df.loc[final_probes]
        df_filtered.to_csv(out_path, compression="gzip")
        print(f"    {df.shape} → {df_filtered.shape} | saved in {time.time()-t0:.1f}s")

    # Write log
    with open(LOG_FILE, "w") as f:
        f.write("MethCLR — Cohort B Probe Intersection Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Cohort B internal intersection: {len(cohort_b_shared):,}\n")
        f.write(f"Cohort A probe set: {len(cohort_a_probes):,}\n")
        f.write(f"Final cross-cohort intersection: {len(final_probes):,}\n\n")
        f.write("Per-pipeline probe counts (before intersection):\n")
        for fname, count in probe_sets.items():
            f.write(f"  {fname}: {len(count):,}\n")

    print("\n" + "=" * 60)
    print(f"DONE — {len(all_files)} intersected matrices in {(time.time()-t_global)/60:.1f} min")
    print(f"Final probe count: {len(final_probes):,}")
    print("=" * 60)

if __name__ == "__main__":
    main()
