"""
MethCLR — Cohort B (GSE224807) 12-Pipeline Beta Matrix Generator
Parallel: splits 789 samples into 6 chunks, runs methylprep on each chunk
simultaneously (one per core), concatenates results.
3 norm methods × 4 filter combos = 12 outputs.
"""

import os, gzip, time, shutil, warnings
import multiprocessing as mp
warnings.filterwarnings("ignore")
import pandas as pd
import methylprep

IDAT_DIR      = r"C:\MethCLR_local\cohort_b"
OUTPUT_DIR    = r"C:\MethCLR_local\cohort_b_beta_matrices"
SERIES_MATRIX = r"C:\MethCLR_local\cohort_b\GSE224807_series_matrix.txt.gz"
TEMP_BASE     = r"C:\MethCLR_local\cohort_b_chunks"
N_WORKERS     = 6

os.makedirs(OUTPUT_DIR, exist_ok=True)

NORM_CONFIGS = {
    "sesame": {"sesame": True,  "save_uncorrected": False},
    "minfi":  {"sesame": False, "save_uncorrected": False},
    "raw":    {"sesame": True,  "save_uncorrected": True},
}

# ─────────────────────────────────────────────
# Parse series matrix
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

    sentrix_ids, sentrix_positions = [], []
    for url in supp_grn:
        fname = url.split("/")[-1]
        parts = fname.split("_")
        sentrix_ids.append(parts[1] if len(parts) >= 3 else None)
        sentrix_positions.append(parts[2] if len(parts) >= 3 else None)
    df["Sentrix_ID"] = sentrix_ids
    df["Sentrix_Position"] = sentrix_positions

    smokers = df["smoking"].sum()
    print(f"  Samples: {len(df)} | Smokers: {smokers} | Non-smokers: {len(df)-smokers}")
    return df

# ─────────────────────────────────────────────
# Build chunk temp dir with hard-linked IDATs
# ─────────────────────────────────────────────

def build_chunk_dir(chunk_id, chunk_df):
    chunk_dir = os.path.join(TEMP_BASE, f"chunk_{chunk_id}")
    os.makedirs(chunk_dir, exist_ok=True)

    # Hard-link only the IDATs for this chunk
    for _, row in chunk_df.iterrows():
        for color in ["Grn", "Red"]:
            fname = f"{row['Sentrix_ID']}_{row['Sentrix_Position']}_{color}.idat.gz"
            src = os.path.join(IDAT_DIR, fname)
            dst = os.path.join(chunk_dir, fname)
            if not os.path.exists(dst):
                try:
                    os.link(src, dst)
                except Exception:
                    shutil.copy2(src, dst)

    # Write samplesheet for this chunk
    ss_path = os.path.join(chunk_dir, "samplesheet.csv")
    chunk_df[["Sample_Name", "Sentrix_ID", "Sentrix_Position", "smoking"]].to_csv(ss_path, index=False)
    return chunk_dir

# ─────────────────────────────────────────────
# Worker: run methylprep on one chunk
# ─────────────────────────────────────────────

def chunk_worker(args):
    chunk_id, chunk_dir, norm_name, norm_cfg = args
    import warnings; warnings.filterwarnings("ignore")
    try:
        beta_df = methylprep.run_pipeline(
            data_dir=chunk_dir,
            array_type="450k",
            export=False,
            betas=True,
            m_value=False,
            make_sample_sheet=False,
            sample_sheet_filepath=os.path.join(chunk_dir, "samplesheet.csv"),
            sesame=norm_cfg["sesame"],
            save_uncorrected=norm_cfg["save_uncorrected"],
            meta_data_frame=False,
            low_memory=True,
        )
        if isinstance(beta_df, dict):
            beta_df = pd.concat(beta_df.values(), axis=1)
        print(f"  [chunk_{chunk_id}|{norm_name}] Done — shape: {beta_df.shape}")
        return beta_df
    except Exception as e:
        print(f"  [chunk_{chunk_id}|{norm_name}] ERROR: {e}")
        import traceback; traceback.print_exc()
        return None

# ─────────────────────────────────────────────
# Sex probes
# ─────────────────────────────────────────────

_sex_probes = None
def get_sex_probes():
    global _sex_probes
    if _sex_probes is not None:
        return _sex_probes
    try:
        from methylprep.files import Manifest
        mft_df = Manifest("450k").data_frame
        chr_col = next((c for c in mft_df.columns if c.upper() == "CHR"), None)
        _sex_probes = set(mft_df[mft_df[chr_col].isin(["X", "Y"])].index.tolist()) if chr_col else set()
        print(f"  [MANIFEST] Sex probes cached: {len(_sex_probes)}")
    except Exception as e:
        print(f"  [MANIFEST] Could not load: {e}")
        _sex_probes = set()
    return _sex_probes

# ─────────────────────────────────────────────
# Apply filters and save
# ─────────────────────────────────────────────

def apply_filters_and_save(beta_df, norm_name, filter_snps, filter_sex, num, total):
    pid = f"norm={norm_name}_snp={filter_snps}_sex={filter_sex}"
    outfile = os.path.join(OUTPUT_DIR, f"{pid}.csv.gz")
    if os.path.exists(outfile):
        print(f"  ({num}/{total}) SKIP: {pid}")
        return
    df = beta_df.copy()
    if filter_snps:
        df = df[~df.index.str.startswith("rs")]
    if filter_sex:
        sp = get_sex_probes()
        if sp:
            df = df[~df.index.isin(sp)]
    df.to_csv(outfile, compression="gzip")
    print(f"  ({num}/{total}) SAVED: {pid} → {df.shape}")

# ─────────────────────────────────────────────
# Run one norm method in parallel chunks
# ─────────────────────────────────────────────

def run_norm_parallel(norm_name, samplesheet, pipeline_offset, total):
    norm_cfg = NORM_CONFIGS[norm_name]
    combos = [(True, True), (True, False), (False, True), (False, False)]
    pids = [f"norm={norm_name}_snp={s}_sex={x}" for s, x in combos]
    if all(os.path.exists(os.path.join(OUTPUT_DIR, f"{p}.csv.gz")) for p in pids):
        print(f"\n[NORM={norm_name}] All 4 outputs exist — skipping")
        return

    print(f"\n[NORM={norm_name}] Splitting {len(samplesheet)} samples into {N_WORKERS} chunks ...")
    t0 = time.time()

    chunks = [samplesheet.iloc[i::N_WORKERS].reset_index(drop=True) for i in range(N_WORKERS)]
    chunk_dirs = []
    for i, chunk_df in enumerate(chunks):
        chunk_dirs.append(build_chunk_dir(i, chunk_df))
    print(f"  Chunk dirs ready in {time.time()-t0:.1f}s")

    worker_args = [(i, chunk_dirs[i], norm_name, norm_cfg) for i in range(N_WORKERS)]

    print(f"  Launching {N_WORKERS} parallel workers ...")
    t1 = time.time()
    with mp.Pool(processes=N_WORKERS) as pool:
        results = pool.map(chunk_worker, worker_args)

    results = [r for r in results if r is not None]
    if not results:
        print(f"  ERROR: all chunks failed for norm={norm_name}")
        return

    print(f"  Concatenating {len(results)} chunks ...")
    beta_df = pd.concat(results, axis=1)
    print(f"  Full beta shape: {beta_df.shape} | parallel time: {(time.time()-t1)/60:.1f} min")

    for i, (snp, sex) in enumerate(combos):
        apply_filters_and_save(beta_df, norm_name, snp, sex, pipeline_offset + i + 1, total)

    # Cleanup chunk dirs for this norm
    for d in chunk_dirs:
        try:
            shutil.rmtree(d)
        except Exception:
            pass

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"MethCLR — Cohort B Pipeline (Parallel, {N_WORKERS} workers)")
    print("3 norm methods × 4 filter combos = 12 outputs")
    print("=" * 60)

    pheno_df    = parse_series_matrix()
    samplesheet = pheno_df[["Sample_Name", "Sentrix_ID", "Sentrix_Position", "smoking"]].copy()
    get_sex_probes()

    global_start = time.time()
    run_norm_parallel("sesame", samplesheet, pipeline_offset=0, total=12)
    run_norm_parallel("minfi",  samplesheet, pipeline_offset=4, total=12)
    run_norm_parallel("raw",    samplesheet, pipeline_offset=8, total=12)

    final = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".csv.gz")])
    print("\n" + "=" * 60)
    print(f"DONE — {final}/12 beta matrices in {(time.time()-global_start)/60:.1f} min")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    mp.freeze_support()
    main()
