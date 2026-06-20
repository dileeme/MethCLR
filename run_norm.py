"""
MethCLR — Single-Norm Runner

Runs all 4 pipelines for one normalization method (snp x sex grid).
Used to parallelize the 12-pipeline grid across two terminals.

Usage:
    python run_norm.py minfi
    python run_norm.py raw

Skips already-completed outputs automatically (idempotent).
Do NOT run sesame — those 4 pipelines are already complete.
"""

import sys, os, time, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import methylprep

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

IDAT_DIR   = r"C:\Users\dilsh\OneDrive\Desktop\MethCLR\GSE85210_IDATs"
OUTPUT_DIR = r"C:\Users\dilsh\OneDrive\Desktop\MethCLR\beta_matrices"

NORM_CONFIGS = {
    "sesame": {"sesame": True,  "save_uncorrected": False, "label": "SeSAMe (NOOB+dye correction)"},
    "minfi":  {"sesame": False, "save_uncorrected": False, "label": "minfi-style (illumina equivalent)"},
    "raw":    {"sesame": True,  "save_uncorrected": True,  "label": "uncorrected raw betas"},
}

# ─────────────────────────────────────────────
# HELPERS (mirror of new_pipeline.py)
# ─────────────────────────────────────────────

def write_methylprep_samplesheet(samplesheet):
    ss_path = os.path.join(IDAT_DIR, "samplesheet.csv")
    samplesheet[["Sample_Name", "Sentrix_ID", "Sentrix_Position", "smoking"]].to_csv(ss_path, index=False)
    return ss_path

def load_samplesheet():
    ss_path = os.path.join(IDAT_DIR, "samplesheet.csv")
    if not os.path.exists(ss_path):
        raise FileNotFoundError(
            f"samplesheet.csv not found at {ss_path}\n"
            "Run new_pipeline.py at least once to generate it."
        )
    return pd.read_csv(ss_path)

def run_pipeline_single(samplesheet, norm_name, filter_snps, filter_sex,
                        pipeline_id, pipeline_num, total):
    outfile = os.path.join(OUTPUT_DIR, f"{pipeline_id}.csv.gz")
    if os.path.exists(outfile):
        print(f"  ({pipeline_num}/{total}) SKIP: {pipeline_id}")
        return

    norm_cfg = NORM_CONFIGS[norm_name]
    print(f"\n  ({pipeline_num}/{total}) RUNNING: {pipeline_id}")
    print(f"    Norm: {norm_cfg['label']} | SNP filter: {filter_snps} | Sex filter: {filter_sex}")
    t0 = time.time()

    try:
        ss_path = write_methylprep_samplesheet(samplesheet)

        beta_df = methylprep.run_pipeline(
            data_dir=IDAT_DIR,
            array_type="450k",
            export=False,
            betas=True,
            m_value=False,
            make_sample_sheet=False,
            sample_sheet_filepath=ss_path,
            sesame=norm_cfg["sesame"],
            save_uncorrected=norm_cfg["save_uncorrected"],
            meta_data_frame=False,
            low_memory=True,
        )

        if beta_df is None or (hasattr(beta_df, "empty") and beta_df.empty):
            print(f"  ERROR: Empty beta matrix for {pipeline_id}")
            return

        if isinstance(beta_df, dict):
            beta_df = pd.concat(beta_df.values(), axis=1)

        print(f"    Beta shape before filtering: {beta_df.shape}")

        if filter_snps:
            snp_mask = beta_df.index.str.startswith("rs")
            before = len(beta_df)
            beta_df = beta_df[~snp_mask]
            print(f"    SNP filter: {before} → {len(beta_df)} probes (removed {snp_mask.sum()})")

        if filter_sex:
            try:
                from methylprep.files import Manifest
                mft = Manifest("450k")
                mft_df = mft.data_frame
                chr_col = next((c for c in mft_df.columns if c.upper() == "CHR"), None)
                if chr_col:
                    sex_probes = set(mft_df[mft_df[chr_col].isin(["X", "Y"])].index.tolist())
                    before = len(beta_df)
                    beta_df = beta_df[~beta_df.index.isin(sex_probes)]
                    print(f"    Sex filter: {before} → {len(beta_df)} probes (removed {before - len(beta_df)})")
            except Exception as e:
                print(f"    Sex filter skipped: {e}")

        beta_df.to_csv(outfile, compression="gzip")
        elapsed = time.time() - t0
        print(f"    Saved → {outfile} ({elapsed/60:.1f} min)")

    except Exception as e:
        print(f"  ERROR in {pipeline_id}: {e}")
        import traceback; traceback.print_exc()

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("minfi", "raw"):
        print("Usage: python run_norm.py <minfi|raw>")
        print("Do NOT run sesame — those 4 pipelines are already complete.")
        sys.exit(1)

    norm = sys.argv[1]

    print("=" * 60)
    print(f"MethCLR — Single-Norm Runner: {norm.upper()}")
    print(f"Running 4 pipelines (snp=True/False x sex=True/False)")
    print("=" * 60)

    samplesheet = load_samplesheet()
    print(f"[SAMPLESHEET] Loaded {len(samplesheet)} rows from existing samplesheet.csv\n")

    combos = [
        (True,  True),
        (True,  False),
        (False, True),
        (False, False),
    ]
    total = 4
    global_start = time.time()

    for i, (snp, sex) in enumerate(combos, 1):
        pid = f"norm={norm}_snp={snp}_sex={sex}"
        run_pipeline_single(
            samplesheet=samplesheet,
            norm_name=norm,
            filter_snps=snp,
            filter_sex=sex,
            pipeline_id=pid,
            pipeline_num=i,
            total=total,
        )
        completed = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".csv.gz")])
        elapsed = time.time() - global_start
        done_this_run = len([f for f in os.listdir(OUTPUT_DIR)
                             if f.endswith(".csv.gz") and f"norm={norm}" in f])
        remaining = total - done_this_run
        if done_this_run > 0 and remaining > 0:
            eta = (elapsed / done_this_run) * remaining
            print(f"  [{done_this_run}/4 {norm} complete | ETA this norm: {eta/60:.1f} min remaining]")

    print("\n" + "=" * 60)
    final_norm = len([f for f in os.listdir(OUTPUT_DIR)
                      if f.endswith(".csv.gz") and f"norm={norm}" in f])
    total_done = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".csv.gz")])
    print(f"DONE — {final_norm}/4 {norm} matrices written in {(time.time()-global_start)/60:.1f} min")
    print(f"Total across all norms: {total_done}/12")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
