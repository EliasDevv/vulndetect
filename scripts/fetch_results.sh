#!/usr/bin/env bash
# Ergebnisse eines Laufs vom Server holen.
# Aufruf: ./fetch_results.sh <ansatz> [stufe]
#   ansatz: baseline | rag | dual_agent | sft | gan | triple_agent  (gan/triple_agent: seed<N>/<snapshot>/ je Auswertung)
#   stufe:  smoke | val | test   (Default: smoke)
# Server überschreibbar: SERVER=<host> ./fetch_results.sh baseline test
set -euo pipefail

SERVER="${SERVER:?set SERVER to the GPU server host (ssh alias)}"
REMOTE_BASE="bachelorarbeit/code/results/predictions"
LOCAL_BASE="$(cd "$(dirname "$0")/.." && pwd)/results/predictions"

ANSATZ="${1:?Ansatz angeben: baseline | rag | dual_agent | sft | gan | triple_agent}"
STUFE="${2:-smoke}"

case "$STUFE" in
  smoke|val) EXPERIMENT="${ANSATZ}_${STUFE}" ;;
  test)      EXPERIMENT="${ANSATZ}" ;;        # offizieller Lauf: Name ohne Suffix
  *) echo "Unbekannte Stufe: $STUFE (erlaubt: smoke, val, test)"; exit 1 ;;
esac

echo "Hole ${EXPERIMENT} von ${SERVER} ..."
mkdir -p "${LOCAL_BASE}/${EXPERIMENT}"
rsync -av "${SERVER}:${REMOTE_BASE}/${EXPERIMENT}/" "${LOCAL_BASE}/${EXPERIMENT}/"
echo "Fertig: ${LOCAL_BASE}/${EXPERIMENT}"