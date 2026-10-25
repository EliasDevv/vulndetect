# Datensätze

Im Repository liegen `reference/` (eingefrorene Split-Referenz) sowie die Manifeste unter
`processed/` (mit den SHA256-Hashes der Splits). Die Splits selbst (`processed/bigvul/*.jsonl`,
enthalten Commit-Metadaten Dritter), der CWE-Index und die Original-Downloads unter `raw/`
(ca. 12 GB) sind nicht enthalten und werden per `make data` bzw. `make cwe-index`
(`scripts/download_data.sh`, `scripts/download_cwe.sh`) reproduziert. `make tables` und
einige Diagnose-Skripte brauchen danach `processed/bigvul/test.jsonl`.

## Quellen

| Dataset   | Verwendung        | Quelle                                                               |
|-----------|-------------------|----------------------------------------------------------------------|
| Big-Vul   | Hauptdatensatz    | https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset (Fan et al., MSR 2020) |
| MITRE CWE | RAG-Wissensbasis  | https://cwe.mitre.org/data/downloads.html (frei nutzbar, (c) MITRE¹) |

## Big-Vul Snapshot (exakte Quelle)

- Datei: **`MSR_data_cleaned.csv`** ("Cleaned Split Functions", enthält
  `func_before`/`func_after` + die 9 nativen Spalten, die die Pipeline erwartet).
- Google-Drive-File-ID: **`1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X`**
  (aus dem ZeoVan-Repo-README, "Cleaned Split Functions (CSV)").
- Laden nach `raw/bigvul.csv`:
  ```bash
  BIGVUL_GDRIVE_ID=1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X bash scripts/download_data.sh
  ```
- Drive liefert ein ZIP; das Script erkennt das (Magic Bytes), entpackt
  `MSR_data_cleaned.csv` nach `raw/bigvul.csv` und behält `raw/bigvul.zip` als
  Original-Snapshot.
- Der SHA256 der geladenen CSV wird beim Build in
  `processed/bigvul/manifest.json` festgehalten (Snapshot-Nachweis).

## CWE-Korpus (RAG-Wissensbasis, Snapshot v4.20)

- Quelle: offizieller MITRE-CWE-Katalog als XML,
  https://cwe.mitre.org/data/xml/cwec_v4.20.xml.zip (Version **4.20**, Stand
  **2026-04-30**; Version wird zusätzlich aus dem XML-Root gelesen).
- Laden nach `raw/cwe/`: `bash scripts/download_cwe.sh` (Version gepinnt;
  Override: `CWE_VERSION=<x.y>`). Das ZIP bleibt als Original-Snapshot liegen.
- SHA256 (2026-08-02):
  - `cwec_v4.20.xml.zip`: `3976f599e5e5200219a3108bb896d06e2a88fbb293369e1883cb423a5e9d7d50`
  - `cwec_v4.20.xml`: `1f5a78bd62e00f86436b4fe32d5034a57e8f0da88e4063b2072b664ae510912e`
- Index bauen: `make cwe-index` -> Chroma-Verzeichnis `processed/cwe_index/`
  mit `manifest.json` (Katalogversion, SHA256, Embedding-Modell +
  Revision, Chunk-Parameter, Mess-Statistiken).
- ¹Lizenz: CWE ist frei nutzbar ("free to use by any organization or individual"),
  Copyright The MITRE Corporation; Nutzung gemäß CWE Terms of Use:
  https://cwe.mitre.org/about/termsofuse.html

## Fokus-CWE-Klassen

CWE-119 (Buffer), CWE-399 (Resource Mgmt), CWE-264 (Access Control),
CWE-20 (Input Validation), CWE-200 (Information Exposure).

## Split-Referenz (eingefroren)

`reference/bigvul_splits_v1.json` (473 KB) hält den realisierten Build
vom 2026-07-24 (seed 42, Legacy-Pfad = seeded Permutation) als Artefakt fest:
Split jedes der 1784 Pool-Commits plus die geordneten Instanz-IDs je Split.
`configs/data.yaml` nennt ihn unter `split.reference`; der Build behält damit
alle Referenz-Commits und -Instanzen, weist unbekannte Commits per
`sha1(commit_id) mod 10` zu und hängt nur Zeilen neuer Commits paarweise bis zu
den (ebenfalls eingefrorenen) Caps an. Ohne `cpp` im Sprachfilter reproduziert
er die Juli-Splits byte-identisch (Rundlauf 2026-09-06); mit `cpp` ergibt er
train 5000 / val 500 / test 986. Neu erzeugen (nur vom Legacy-Pfad aus):
`python -m vulndetect.data.build --config <config ohne split.reference> --write-reference <pfad>`.

## Struktur

```
data/
├── raw/         Original-Downloads, unverändert (nicht im Repo)
│   └── cwe/     CWE-Katalog-Snapshot (XML + ZIP)
├── reference/   eingefrorene Split-Referenz
└── processed/   abgeleitete Artefakte (nur Manifeste im Repo, Rest per make)
    ├── bigvul/      train/val/test-Splits + manifest.json
    └── cwe_index/   persistenter Chroma-Index + manifest.json
```
