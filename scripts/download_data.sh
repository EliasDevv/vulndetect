#!/usr/bin/env bash
# Fetches the raw Big-Vul dataset (MSR_data_cleaned.csv, Fan et al. 2020) into data/raw/bigvul.csv.
# Usage: BIGVUL_GDRIVE_ID=<file-id> bash scripts/download_data.sh
# (file id: https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset)
set -euo pipefail

RAW_DIR="$(cd "$(dirname "$0")/.." && pwd)/data/raw"
OUT="$RAW_DIR/bigvul.csv"
mkdir -p "$RAW_DIR"

echo "Target: $OUT"

if [[ -f "$OUT" ]]; then
  echo "Already present: $OUT (delete it to re-download)."
  exit 0
fi

if [[ -z "${BIGVUL_GDRIVE_ID:-}" ]]; then
  cat >&2 <<'EOF'
ERROR: BIGVUL_GDRIVE_ID is not set.

Big-Vul (MSR_data_cleaned.csv) lives on Google Drive. Get the file id from
  https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset
then run:
  BIGVUL_GDRIVE_ID=<file-id> bash scripts/download_data.sh

(If you downloaded the CSV manually, just place it at data/raw/bigvul.csv.)
EOF
  exit 1
fi

# gdown handles Google Drive's large-file confirmation tokens.
echo "Downloading Big-Vul (id=$BIGVUL_GDRIVE_ID) ..."
if command -v gdown >/dev/null 2>&1; then
  gdown "$BIGVUL_GDRIVE_ID" -O "$OUT"
elif command -v uv >/dev/null 2>&1; then
  uv run --with gdown gdown "$BIGVUL_GDRIVE_ID" -O "$OUT"
else
  python3 -m pip install --user gdown
  python3 -m gdown "$BIGVUL_GDRIVE_ID" -O "$OUT"
fi

# If the download is a ZIP (magic "PK"), extract the CSV and keep the ZIP as bigvul.zip.
if [[ "$(head -c 2 "$OUT")" == "PK" ]]; then
  ZIP="$RAW_DIR/bigvul.zip"
  echo "Downloaded file is a ZIP archive; extracting MSR_data_cleaned.csv ..."
  mv "$OUT" "$ZIP"
  python3 - "$ZIP" "$OUT" <<'PY'
import sys, zipfile

zip_path, out_path = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(zip_path) as zf:
    names = zf.namelist()
    target = next((n for n in names if n.endswith("MSR_data_cleaned.csv")), None)
    if target is None:
        print("ERROR: MSR_data_cleaned.csv not found in the archive. Contents:", file=sys.stderr)
        for n in names:
            print(f"  {n}", file=sys.stderr)
        sys.exit(1)
    with zf.open(target) as src, open(out_path, "wb") as dst:
        while chunk := src.read(1 << 20):
            dst.write(chunk)
PY
fi

echo "Done: $OUT"
echo "SHA256:"
sha256sum "$OUT" || true
