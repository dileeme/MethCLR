"""Strip GSM prefix from Cohort B IDATs: GSM7032611_SENTRIX_POS_Grn.idat.gz → SENTRIX_POS_Grn.idat.gz"""
import os, re

IDAT_DIR = r"C:\Users\dilsh\OneDrive\Desktop\MethCLR\cohort_b"

files = os.listdir(IDAT_DIR)
renamed, skipped = 0, 0

for fname in files:
    if not fname.endswith(".idat.gz"):
        continue
    # Match GSM\d+_ prefix
    new_name = re.sub(r"^GSM\d+_", "", fname)
    if new_name == fname:
        skipped += 1
        continue
    src = os.path.join(IDAT_DIR, fname)
    dst = os.path.join(IDAT_DIR, new_name)
    if os.path.exists(dst):
        skipped += 1
        continue
    os.rename(src, dst)
    renamed += 1

print(f"Done — renamed: {renamed} | skipped: {skipped}")
