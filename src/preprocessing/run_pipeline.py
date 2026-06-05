"""
MethCLR — 12-Pipeline Beta Matrix Generator
Replaces R/minfi pipeline. Uses methylprep for IDAT processing.

Normalization methods: illumina, noob (replaces funnorm), quantile
SNP filter: True/False
Sex probe filter: True/False
Total: 3 x 2 x 2 = 12 pipelines

Usage (PowerShell):
    python run_pipeline.py

Requirements:
    pip install methylprep pandas numpy
"""

import os
import gzip
import tarfile
import time
import re
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import methylprep

# ─────────────────────────────────────────────
# CONFIGURATION — edit these paths if needed
# ─────────────────────────────────────────────

DOWNLOADS = os.path.join(os.path.expanduser("~"), "Downloads")

TAR_PATH         = os.path.join(DOWNLOADS, "GSE85210_RAW.tar")
SERIES_MATRIX    = os.path.join(DOWNLOADS, "GSE85210_series_matrix.txt.gz")
IDAT_DIR         = r"C:\Users\dilsh\OneDrive\Desktop\MethCLR\GSE85210_IDATs"
OUTPUT_DIR       = r"C:\Users\dilsh\OneDrive\Desktop\MethCLR\beta_matrices"

os.makedirs(IDAT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1 — Extract IDATs from TAR if not done
# ─────────────────────────────────────────────

def extract_idats():
    existing = [f for f in os.listdir(IDAT_DIR) if f.endswith(".idat.gz") or f.endswith(".idat")]
    if len(existing) >= 500:
        print(f"[SKIP] IDATs already extracted ({len(existing)} files found)")
        return
    print(f"[EXTRACT] Extracting IDATs from {TAR_PATH} ...")
    with tarfile.open(TAR_PATH, "r") as tar:
        members = tar.getmembers()
        idat_members = [m for m in members if ".idat" in m.name.lower()]
        print(f"  Found {len(idat_members)} IDAT files in archive")
        for i, member in enumerate(idat_members):
            member.name = os.path.basename(member.name)  # flatten directory
            tar.extract(member, path=IDAT_DIR)
            if (i + 1) % 50 == 0:
                print(f"  Extracted {i+1}/{len(idat_members)}")
    print(f"[DONE] Extraction complete")

# ─────────────────────────────────────────────
# STEP 2 — Parse series matrix for phenotype
# ─────────────────────────────────────────────

def parse_series_matrix():
    print(f"[PARSE] Reading series matrix ...")
    samples, labels = [], []

    opener = gzip.open if SERIES_MATRIX.endswith(".gz") else open
    with opener(SERIES_MATRIX, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("!Sample_geo_accession"):
                parts = line.split("\t")
                samples = [p.strip('"') for p in parts[1:]]
            if line.startswith("!Sample_characteristics_ch1") and "subject status" in line:
                parts = line.split("\t")
                labels = [p.strip('"') for p in parts[1:]]

    if not samples or not labels:
        raise ValueError("Could not parse samples or labels from series matrix. Check file path.")

    df = pd.DataFrame({"sample": samples, "label": labels})
    df["smoking"] = df["label"].apply(lambda x: 0 if "non" in x.lower() else 1)
    print(f"  Samples: {len(df)} | Smokers: {df['smoking'].sum()} | Non-smokers: {(df['smoking']==0).sum()}")
    return df

# ─────────────────────────────────────────────
# STEP 3 — Build samplesheet mapping GSM → IDAT basename
# ─────────────────────────────────────────────

def build_samplesheet(pheno_df):
    idat_files = [f for f in os.listdir(IDAT_DIR) if f.endswith("_Grn.idat.gz") or f.endswith("_Grn.idat")]
    if not idat_files:
        raise FileNotFoundError(f"No Green IDAT files found in {IDAT_DIR}. Check extraction.")

    basenames = [re.sub(r"_Grn\.idat(\.gz)?$", "", f) for f in idat_files]
    gsm_ids   = [b.split("_")[0] for b in basenames]

    basename_df = pd.DataFrame({"sample": gsm_ids, "basename": basenames})
    samplesheet = basename_df.merge(pheno_df, on="sample", how="inner")
    samplesheet["idat_path"] = samplesheet["basename"].apply(lambda b: os.path.join(IDAT_DIR, b))

    print(f"  Samplesheet rows matched: {len(samplesheet)}")
    if len(samplesheet) < 200:
        print(f"  WARNING: Expected ~253 rows, got {len(samplesheet)}. Check GSM ID parsing.")
    return samplesheet

# ─────────────────────────────────────────────
# STEP 4 — Sex probe list (chrX / chrY)
# ─────────────────────────────────────────────

def get_sex_probes():
    """
    Fetches sex chromosome probe list from methylprep's bundled 450K manifest.
    Returns a set of probe IDs on chrX or chrY.
    """
    try:
        from methylprep.files import Manifest
        manifest = Manifest("450k")
        df = manifest.data_frame
        sex_probes = set(df[df["CHR"].isin(["X", "Y"])].index.tolist())
        print(f"  Sex probes identified: {len(sex_probes)}")
        return sex_probes
    except Exception as e:
        print(f"  WARNING: Could not load manifest for sex probes ({e}). Sex filtering will be skipped.")
        return set()

# ─────────────────────────────────────────────
# STEP 5 — SNP probe list (known SNP-associated probes)
# ─────────────────────────────────────────────

def get_snp_probes():
    """
    Uses methylprep's built-in SNP probe exclusion list for 450K.
    These are probes overlapping known SNPs (equivalent to minfi's dropLociWithSnps).
    """
    try:
        from methylprep.files import Manifest
        manifest = Manifest("450k")
        df = manifest.data_frame
        # Probes with SNP within 10bp of CpG site
        snp_probes = set(df[df["Probe_SNPs_10"].notna() & (df["Probe_SNPs_10"] != "")].index.tolist())
        print(f"  SNP-associated probes identified: {len(snp_probes)}")
        return snp_probes
    except Exception as e:
        print(f"  WARNING: Could not load SNP probes ({e}). SNP filtering will be skipped.")
        return set()

# ─────────────────────────────────────────────
# STEP 6 — Run single pipeline
# ─────────────────────────────────────────────

def run_pipeline(samplesheet, norm_method, filter_snps, filter_sex, sex_probes, snp_probes, pipeline_id, pipeline_num, total):
    outfile = os.path.join(OUTPUT_DIR, f"{pipeline_id}.csv.gz")
    if os.path.exists(outfile):
        print(f"  ({pipeline_num}/{total}) SKIP: {pipeline_id}")
        return

    print(f"\n  ({pipeline_num}/{total}) RUNNING: {pipeline_id}")
    t0 = time.time()

    try:
        # methylprep processes by directory — write a minimal samplesheet CSV it expects
        sample_csv = os.path.join(IDAT_DIR, "samplesheet.csv")
        ss_export = samplesheet[["sample", "idat_path", "smoking"]].copy()
        ss_export.columns = ["Sample_Name", "Basename", "smoking"]
        ss_export.to_csv(sample_csv, index=False)

        # Run methylprep pipeline
        beta_df = methylprep.run_pipeline(
            data_dir=IDAT_DIR,
            array_type="450k",
            export=False,
            betas=True,
            m_value=False,
            normalization=norm_method,   # 'illumina', 'noob', 'quantile'
            sample_sheet_filepath=sample_csv,
            make_sample_sheet=False,
            verbose=False,
        )

        # beta_df is returned as probes x samples DataFrame
        if beta_df is None or beta_df.empty:
            print(f"  ERROR: Empty beta matrix returned for {pipeline_id}")
            return

        print(f"    Beta shape before filtering: {beta_df.shape}")

        # SNP filtering
        if filter_snps and snp_probes:
            before = len(beta_df)
            beta_df = beta_df[~beta_df.index.isin(snp_probes)]
            print(f"    SNP filter: {before} → {len(beta_df)} probes")

        # Sex probe filtering
        if filter_sex and sex_probes:
            before = len(beta_df)
            beta_df = beta_df[~beta_df.index.isin(sex_probes)]
            print(f"    Sex filter: {before} → {len(beta_df)} probes")

        # Write compressed CSV
        beta_df.to_csv(outfile, compression="gzip")
        elapsed = time.time() - t0
        print(f"    Done in {elapsed/60:.1f} min → {outfile}")

    except Exception as e:
        print(f"  ERROR in pipeline {pipeline_id}: {e}")
        import traceback; traceback.print_exc()

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("MethCLR — Python Pipeline Runner")
    print("=" * 60)

    # Step 1: Extract
    extract_idats()

    # Step 2: Parse phenotype
    pheno_df = parse_series_matrix()

    # Step 3: Build samplesheet
    samplesheet = build_samplesheet(pheno_df)

    # Step 4: Load probe filter lists once (expensive, cache)
    print("\n[PROBES] Loading manifest for probe filtering ...")
    sex_probes = get_sex_probes()
    snp_probes = get_snp_probes()

    # Step 5: 12-pipeline grid
    norm_methods     = ["illumina", "noob", "quantile"]
    filter_snps_opts = [True, False]
    filter_sex_opts  = [True, False]
    total = 12
    pipeline_count = 0
    global_start = time.time()

    print(f"\n[GRID] Starting 12-pipeline grid ...")
    print(f"  Output: {OUTPUT_DIR}\n")

    for norm in norm_methods:
        for snp in filter_snps_opts:
            for sex in filter_sex_opts:
                pipeline_count += 1
                pid = f"norm={norm}_snp={snp}_sex={sex}"
                run_pipeline(
                    samplesheet=samplesheet,
                    norm_method=norm,
                    filter_snps=snp,
                    filter_sex=sex,
                    sex_probes=sex_probes,
                    snp_probes=snp_probes,
                    pipeline_id=pid,
                    pipeline_num=pipeline_count,
                    total=total,
                )

                # ETA estimate
                completed = len([
                    f for f in os.listdir(OUTPUT_DIR) if f.endswith(".csv.gz")
                ])
                elapsed = time.time() - global_start
                if completed > 0:
                    avg = elapsed / completed
                    eta = avg * (total - completed)
                    print(f"  [{completed}/12 complete | ETA: {eta/60:.1f} min remaining]")

    print("\n" + "=" * 60)
    total_time = (time.time() - global_start) / 60
    final_count = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".csv.gz")])
    print(f"DONE — {final_count}/12 beta matrices written in {total_time:.1f} min")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
