# Label-Definition-Ablation

Privater Test auf **val**, außerhalb der Replikation. Frage: Wie viel kostet die
Formulierung des Fig.-1-Prompts die Baseline? Das Modell liest „Vulnerable“ oft als Zustand
nach der Änderung und antwortet bei einem Fix „Not Vulnerable“.

**Genau eine Änderung gegenüber `baseline_val`:** Fig. 1 bleibt byte-gleich, danach folgen die
zwei Definitionssätze aus dem strikten Autoren-Prompt („Vulnerable if the code change is related
to a known vulnerability …“). Modell, Revision, Decoding (0.6 / 0.9 / 512), Batch 2, Seed 42,
Parser d09-v2 und Policy drop sind identisch; `test_label_definition.py` erzwingt das.

**Regeln:** nur val, nie test; keine Zeile in `results/experiments.csv`; nie die berichtete
Baseline-Zahl. In `src/vulndetect/` wird nichts geändert oder registriert, Baseline, RAG,
Dual-Agent und Parser bleiben eingefroren.

## Ausführen (GPU, ca. 1 GPU-Stunde)

Aus `code/` heraus; der Aufruf läuft nicht über `make`, daher `CUDA_DEVICE_ORDER` selbst setzen:

```
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Probelauf mit 4 Instanzen (eigener Ordner ..._smoke):
uv run python experiments/label_definition/labeldef.py run \
    --config experiments/label_definition/config.yaml --limit 4
# Vollständiger Lauf:
uv run python experiments/label_definition/labeldef.py run \
    --config experiments/label_definition/config.yaml
```

## Auswerten (lokal, CPU)

```
uv run python experiments/label_definition/labeldef.py analyze
```

Die Auswertung
vergleicht mit `baseline_val` (Referenz), `baseline_val_date` (Rauschmaß: gleicher Prompt,
anderes Chat-Datum) und `baseline_val_strict` und schreibt `analysis.json` in den Laufordner:

- macro F1, MCC, Recall, Vorhersagerate; McNemar gegen `baseline_val`
- Anteil der verpassten Lücken, deren Text den Fix beschreibt (die direkte Messgröße)
- die Aussage „Code unverändert“ als Klassifikator (Entkopplung Begründung ↔ Urteil)
- Prompt-Länge in Token gegenüber Fig. 1 (im Manifest)
