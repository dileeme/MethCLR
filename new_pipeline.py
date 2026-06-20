"""
MethCLR — 12-Pipeline Beta Matrix Generator (v4)

Changes from v3:
- build_samplesheet now works whether IDATs have GSM prefix or not
- GSM→Sentrix mapping built from series matrix supplementary data
- rename_idats is idempotent and runs before samplesheet build
- Samplesheet build uses Sentrix_ID+Position parsed from filenames directly

Normalization axis:
  sesame : SeSAMe (NOOB + dye correction)
  minfi  : minfi-style (illumina equivalent, sesame=False)
  raw    : uncorrected betas (save_uncorrected=True)

SNP filter: True/False  (rs-prefix probes)
Sex filter: True/False  (chrX/Y probes via manifest)
Total: 3 x 2 x 2 = 12 pipelines
"""

import os, gzip, tarfile, time, re, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import methylprep

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

DOWNLOADS     = os.path.join(os.path.expanduser("~"), "Downloads")
TAR_PATH      = os.path.join(DOWNLOADS, "GSE85210_RAW.tar")
SERIES_MATRIX = os.path.join(DOWNLOADS, "GSE85210_series_matrix.txt.gz")
IDAT_DIR      = r"C:\Users\dilsh\OneDrive\Desktop\MethCLR\GSE85210_IDATs"
OUTPUT_DIR    = r"C:\Users\dilsh\OneDrive\Desktop\MethCLR\beta_matrices"

os.makedirs(IDAT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

NORM_CONFIGS = {
    "sesame": {"sesame": True,  "save_uncorrected": False, "label": "SeSAMe (NOOB+dye correction)"},
    "minfi":  {"sesame": False, "save_uncorrected": False, "label": "minfi-style (illumina equivalent)"},
    "raw":    {"sesame": True,  "save_uncorrected": True,  "label": "uncorrected raw betas"},
}

# ─────────────────────────────────────────────
# STEP 1 — Extract IDATs
# ─────────────────────────────────────────────

def extract_idats():
    existing = [f for f in os.listdir(IDAT_DIR) if ".idat" in f.lower()]
    if len(existing) >= 500:
        print(f"[SKIP] IDATs already extracted ({len(existing)} files found)")
        return
    print(f"[EXTRACT] Extracting from {TAR_PATH} ...")
    with tarfile.open(TAR_PATH, "r") as tar:
        members = [m for m in tar.getmembers() if ".idat" in m.name.lower()]
        print(f"  Found {len(members)} IDAT files in archive")
        for i, member in enumerate(members):
            member.name = os.path.basename(member.name)
            tar.extract(member, path=IDAT_DIR)
            if (i + 1) % 50 == 0:
                print(f"  Extracted {i+1}/{len(members)}")
    print("[DONE] Extraction complete")

# ─────────────────────────────────────────────
# STEP 2 — Rename IDATs: strip GSM prefix
# methylprep expects: {Sentrix_ID}_{Sentrix_Position}_Grn.idat.gz
# GEO names:          GSM2260480_8221916021_R02C02_Grn.idat.gz
# Already renamed:    8221916021_R02C02_Grn.idat.gz  (starts with digit)
# This function is idempotent — safe to call multiple times
# ─────────────────────────────────────────────

def rename_idats():
    renamed = 0
    already_done = 0
    for fname in os.listdir(IDAT_DIR):
        if ".idat" not in fname.lower():
            continue
        if fname[0].isdigit():
            already_done += 1
            continue
        parts = fname.split("_", 1)
        if len(parts) == 2 and parts[0].startswith("GSM"):
            os.rename(
                os.path.join(IDAT_DIR, fname),
                os.path.join(IDAT_DIR, parts[1])
            )
            renamed += 1
    if renamed > 0:
        print(f"[RENAME] Stripped GSM prefix from {renamed} IDAT files")
    else:
        print(f"[RENAME] {already_done} files already in correct format (Sentrix_ID prefix)")

# ─────────────────────────────────────────────
# STEP 3 — Parse series matrix
# Returns pheno_df with columns: Sample_Name, label, smoking,
#         Sentrix_ID, Sentrix_Position
# We parse supplementary_file lines to get Sentrix info per GSM
# ─────────────────────────────────────────────

def parse_series_matrix():
    print("[PARSE] Reading series matrix ...")
    samples, labels, supp_files = [], [], []

    opener = gzip.open if SERIES_MATRIX.endswith(".gz") else open
    with opener(SERIES_MATRIX, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("!Sample_geo_accession"):
                samples = [p.strip('"') for p in line.split("\t")[1:]]
            if line.startswith("!Sample_characteristics_ch1") and "subject status" in line:
                labels = [p.strip('"') for p in line.split("\t")[1:]]
            if line.startswith("!Sample_supplementary_file"):
                supp_files.append([p.strip('"') for p in line.split("\t")[1:]])

    if not samples or not labels:
        raise ValueError("Could not parse samples/labels from series matrix.")

    df = pd.DataFrame({"Sample_Name": samples, "label": labels})
    df["smoking"] = df["label"].apply(lambda x: 0 if "non" in x.lower() else 1)

    # Parse Sentrix_ID and Sentrix_Position from supplementary file paths
    # e.g. ftp://.../GSM2260480_8221916021_R02C02_Grn.idat.gz
    sentrix_ids, sentrix_positions = [], []
    # Flatten supp_files — each entry is a list of URLs per sample
    # We look for the Grn.idat.gz URL to extract Sentrix info
    grn_files = []
    for row in supp_files:
        grn = next((x for x in row if "Grn.idat" in x), None)
        if grn:
            grn_files.append(grn)

    if len(grn_files) == len(samples):
        for url in grn_files:
            fname = url.split("/")[-1]  # get filename from URL
            # fname: GSM2260480_8221916021_R02C02_Grn.idat.gz
            parts = fname.split("_")
            if len(parts) >= 3:
                sentrix_ids.append(parts[1])
                sentrix_positions.append(parts[2])
            else:
                sentrix_ids.append(None)
                sentrix_positions.append(None)
        df["Sentrix_ID"] = sentrix_ids
        df["Sentrix_Position"] = sentrix_positions
        print(f"  Sentrix info parsed from supplementary file URLs")
    else:
        # Fallback: derive from IDAT filenames on disk
        print(f"  WARNING: Could not parse Sentrix from series matrix, using IDAT filenames")
        df["Sentrix_ID"] = None
        df["Sentrix_Position"] = None

    print(f"  Samples: {len(df)} | Smokers: {df['smoking'].sum()} | Non-smokers: {(df['smoking']==0).sum()}")
    return df

# ─────────────────────────────────────────────
# STEP 4 — Build samplesheet
# Works whether IDATs are renamed or not, since
# Sentrix info comes from series matrix now
# ─────────────────────────────────────────────

def build_samplesheet(pheno_df):
    # Verify IDATs are present on disk
    idat_files = [f for f in os.listdir(IDAT_DIR) if "_Grn.idat" in f]
    if not idat_files:
        raise FileNotFoundError(f"No Green IDAT files in {IDAT_DIR}")

    # If Sentrix info was parsed from series matrix, use it directly
    if "Sentrix_ID" in pheno_df.columns and pheno_df["Sentrix_ID"].notna().all():
        samplesheet = pheno_df[["Sample_Name", "Sentrix_ID", "Sentrix_Position", "smoking"]].copy()
        # Verify at least some IDATs match
        found = sum(
            1 for _, row in samplesheet.iterrows()
            if any(f.startswith(f"{row['Sentrix_ID']}_{row['Sentrix_Position']}") for f in idat_files)
        )
        print(f"  Samplesheet rows: {len(samplesheet)} | IDATs verified on disk: {found}")
        if found < 100:
            print(f"  WARNING: Only {found} IDATs matched on disk — check IDAT_DIR")
        return samplesheet

    # Fallback: parse Sentrix from renamed IDAT filenames
    # At this point files look like: 8221916021_R02C02_Grn.idat.gz
    rows = []
    for f in idat_files:
        basename = re.sub(r"_Grn\.idat(\.gz)?$", "", f)
        parts = basename.split("_")
        if len(parts) >= 2 and parts[0][0].isdigit():
            rows.append({"Sentrix_ID": parts[0], "Sentrix_Position": parts[1]})

    if not rows:
        raise ValueError("Could not build samplesheet — no valid IDAT filenames found")

    basename_df = pd.DataFrame(rows)
    # We can't match back to GSM IDs without the prefix, so use Sentrix_ID as Sample_Name
    basename_df["Sample_Name"] = basename_df["Sentrix_ID"] + "_" + basename_df["Sentrix_Position"]
    # Add smoking label as unknown (can't match) — warn user
    basename_df["smoking"] = -1
    print(f"  WARNING: Could not match GSM IDs to Sentrix IDs. Smoking labels unavailable.")
    print(f"  Samplesheet rows: {len(basename_df)}")
    return basename_df

# ─────────────────────────────────────────────
# STEP 5 — Write samplesheet CSV methylprep expects
# ─────────────────────────────────────────────

def write_methylprep_samplesheet(samplesheet):
    ss_path = os.path.join(IDAT_DIR, "samplesheet.csv")
    out = samplesheet[["Sample_Name", "Sentrix_ID", "Sentrix_Position", "smoking"]].copy()
    out.to_csv(ss_path, index=False)
    return ss_path

# ─────────────────────────────────────────────
# STEP 6 — Run single pipeline
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

        # SNP filtering
        if filter_snps:
            snp_mask = beta_df.index.str.startswith("rs")
            before = len(beta_df)
            beta_df = beta_df[~snp_mask]
            print(f"    SNP filter: {before} → {len(beta_df)} probes (removed {snp_mask.sum()})")

        # Sex probe filtering
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
    print("MethCLR — Python Pipeline Runner v4")
    print("Norm methods: sesame | minfi | raw")
    print("=" * 60)

    extract_idats()
    rename_idats()          # idempotent — safe every run
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