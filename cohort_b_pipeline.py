"""
MethCLR — Cohort B (GSE224807) 12-Pipeline Beta Matrix Generator
Adapted from new_pipeline.py (v4) for Colab execution.

IDATs already extracted and GSM-prefix-stripped in /content/GSE224807_IDATs/
Series matrix at /content/GSE224807_series_matrix.txt.gz
Smoking label: smoking_status: SM = 1, smoking_status: NS = 0
"""

import os, gzip, time, re, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import methylprep

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

IDAT_DIR      = "/content/GSE224807_IDATs"
OUTPUT_DIR    = "/content/GSE224807_beta_matrices"
SERIES_MATRIX = "/content/GSE224807_series_matrix.txt.gz"

os.makedirs(OUTPUT_DIR, exist_ok=True)

NORM_CONFIGS = {
    "sesame": {"sesame": True,  "save_uncorrected": False, "label": "SeSAMe (NOOB+dye correction)"},
    "minfi":  {"sesame": False, "save_uncorrected": False, "label": "minfi-style (illumina equivalent)"},
    "raw":    {"sesame": True,  "save_uncorrected": True,  "label": "uncorrected raw betas"},
}

# ─────────────────────────────────────────────
# STEP 1 — Parse series matrix
# ─────────────────────────────────────────────

def parse_series_matrix():
    print("[PARSE] Reading series matrix ...")
    samples, labels, supp_grn = [], [], []

    opener = gzip.open if SERIES_MATRIX.endswith(".gz") else open
    with opener(SERIES_MATRIX, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("!Sample_geo_accession"):
                samples = [p.strip('"') for p in line.split("\t")[1:]]
            if line.startswith("!Sample_characteristics_ch1") and "smoking_status" in line:
                labels = [p.strip('"') for p in line.split("\t")[1:]]
            if line.startswith("!Sample_supplementary_file") and "Grn.idat" in line:
                supp_grn = [p.strip('"') for p in line.split("\t")[1:]]

    if not samples or not labels:
        raise ValueError("Could not parse samples/labels from series matrix.")

    df = pd.DataFrame({"Sample_Name": samples, "label": labels})
    df["smoking"] = df["label"].apply(lambda x: 1 if "SM" in x else 0)

    # Parse Sentrix_ID and Sentrix_Position from Grn supplementary URLs
    # e.g. .../GSM7032611_100946230055_R01C01_Grn.idat.gz
    sentrix_ids, sentrix_positions = [], []
    if len(supp_grn) == len(samples):
        for url in supp_grn:
            fname = url.split("/")[-1]
            parts = fname.split("_")
            if len(parts) >= 3:
                sentrix_ids.append(parts[1])
                sentrix_positions.append(parts[2])
            else:
                sentrix_ids.append(None)
                sentrix_positions.append(None)
        df["Sentrix_ID"] = sentrix_ids
        df["Sentrix_Position"] = sentrix_positions
        print(f"  Sentrix info parsed from supplementary URLs")
    else:
        raise ValueError(f"Grn URL count ({len(supp_grn)}) != sample count ({len(samples)})")

    smokers = df["smoking"].sum()
    print(f"  Samples: {len(df)} | Smokers: {smokers} | Non-smokers: {len(df) - smokers}")
    return df

# ─────────────────────────────────────────────
# STEP 2 — Build samplesheet
# ─────────────────────────────────────────────

def build_samplesheet(pheno_df):
    idat_files = [f for f in os.listdir(IDAT_DIR) if "_Grn.idat" in f]
    if not idat_files:
        raise FileNotFoundError(f"No Green IDAT files in {IDAT_DIR}")

    samplesheet = pheno_df[["Sample_Name", "Sentrix_ID", "Sentrix_Position", "smoking"]].copy()

    found = sum(
        1 for _, row in samplesheet.iterrows()
        if any(f.startswith(f"{row['Sentrix_ID']}_{row['Sentrix_Position']}") for f in idat_files)
    )
    print(f"  Samplesheet rows: {len(samplesheet)} | IDATs verified on disk: {found}")
    if found < 50:
        raise ValueError(f"Only {found} IDATs matched — check IDAT_DIR")
    return samplesheet

# ─────────────────────────────────────────────
# STEP 3 — Write samplesheet CSV
# ─────────────────────────────────────────────

def write_methylprep_samplesheet(samplesheet):
    ss_path = os.path.join(IDAT_DIR, "samplesheet.csv")
    samplesheet[["Sample_Name", "Sentrix_ID", "Sentrix_Position", "smoking"]].to_csv(ss_path, index=False)
    return ss_path

# ─────────────────────────────────────────────
# STEP 4 — Run single pipeline
# ─────────────────────────────────────────────

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
    print("=" * 60)
    print("MethCLR — Cohort B Pipeline Runner (GSE224807)")
    print("Norm methods: sesame | minfi | raw")
    print("=" * 60)

    pheno_df    = parse_series_matrix()
    samplesheet = build_samplesheet(pheno_df)

    norm_methods     = ["sesame", "minfi", "raw"]
    filter_snps_opts = [True, False]
    filter_sex_opts  = [True, False]
    total = 12
    pipeline_count = 0
    global_start = time.time()

    print(f"\n[GRID] Starting 12-pipeline grid → {OUTPUT_DIR}\n")

    for norm in norm_methods:
        for snp in filter_snps_opts:
            for sex in filter_sex_opts:
                pipeline_count += 1
                pid = f"norm={norm}_snp={snp}_sex={sex}"
                run_pipeline_single(
                    samplesheet=samplesheet,
                    norm_name=norm,
                    filter_snps=snp,
                    filter_sex=sex,
                    pipeline_id=pid,
                    pipeline_num=pipeline_count,
                    total=total,
                )
                completed = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".csv.gz")])
                elapsed = time.time() - global_start
                if completed > 0:
                    eta = (elapsed / completed) * (total - completed)
                    print(f"  [{completed}/12 complete | ETA: {eta/60:.1f} min remaining]")

    print("\n" + "=" * 60)
    final = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".csv.gz")])
    print(f"DONE — {final}/12 beta matrices in {(time.time()-global_start)/60:.1f} min")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
