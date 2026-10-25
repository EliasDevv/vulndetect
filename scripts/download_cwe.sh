#!/usr/bin/env bash
# Fetches a pinned version of the MITRE CWE catalog (XML) into data/raw/cwe/.
# Override the version with: CWE_VERSION=4.21 bash scripts/download_cwe.sh
set -euo pipefail

CWE_VERSION="${CWE_VERSION:-4.20}"
RAW_DIR="$(cd "$(dirname "$0")/.." && pwd)/data/raw/cwe"
ZIP="$RAW_DIR/cwec_v${CWE_VERSION}.xml.zip"
XML="$RAW_DIR/cwec_v${CWE_VERSION}.xml"
URL="https://cwe.mitre.org/data/xml/cwec_v${CWE_VERSION}.xml.zip"

mkdir -p "$RAW_DIR"
echo "Target: $XML (catalog v${CWE_VERSION})"

if [[ -f "$XML" ]]; then
  echo "Already present: $XML (delete it to re-download)."
  sha256sum "$XML" || true
  exit 0
fi

echo "Downloading $URL ..."
curl -fL --retry 3 -o "$ZIP" "$URL"

python3 - "$ZIP" "$XML" <<'PY'
import sys, zipfile

zip_path, out_path = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(zip_path) as zf:
    names = zf.namelist()
    target = next((n for n in names if n.endswith(".xml")), None)
    if target is None:
        print(f"ERROR: no XML found in the archive. Contents: {names}", file=sys.stderr)
        sys.exit(1)
    with zf.open(target) as src, open(out_path, "wb") as dst:
        while chunk := src.read(1 << 20):
            dst.write(chunk)
PY

echo "Done: $XML  (zip kept as snapshot artifact: $ZIP)"
echo "SHA256:"
sha256sum "$ZIP" "$XML" || true
