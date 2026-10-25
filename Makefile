# Bachelorarbeit: Orchestrierung der häufigsten Schritte.

.PHONY: help setup test lint data cwe-index baseline baseline-smoke baseline-val rag rag-smoke rag-val sft sft-train sft-smoke sft-val dual_agent dual_agent-smoke dual_agent-val baseline-seed43 baseline-seed44 rag-seed43 rag-seed44 dual_agent-seed43 dual_agent-seed44 gan gan-eval gan-smoke gan-val gan-val-phaseb-only gan-loop-seed43 gan-loop-seed44 gan-eval-seed43 gan-eval-seed44 gan-eval-greedy-date triple_agent triple_agent-eval triple_agent-smoke triple_agent-val triple_agent-loop-seed43 triple_agent-loop-seed44 triple_agent-eval-seed42 triple_agent-eval-seed43 triple_agent-eval-seed44 eval eval-val tables

# SFT-Config austauschbar für Multi-Seed-Läufe:
#   make sft-train SFT_CONFIG=configs/sft_seed43.yaml && make sft SFT_CONFIG=configs/sft_seed43.yaml
SFT_CONFIG ?= configs/sft.yaml

# Gegen CUDA-Speicherfragmentierung bei langen Prompts; auf CPU wirkungslos.
export PYTORCH_CUDA_ALLOC_CONF ?= expandable_segments:True

# GPUs wie nvidia-smi nach PCI-Bus nummerieren (relevant bei ungleichen Karten).
export CUDA_DEVICE_ORDER ?= PCI_BUS_ID

help:
	@echo "Targets:"
	@echo "  setup      - Abhängigkeiten inkl. Testwerkzeuge installieren (uv sync --extra dev)"
	@echo "  test       - Testsuite (uv run pytest)"
	@echo "  lint       - ruff check --fix + ruff format"
	@echo "  data       - Big-Vul laden und Splits bauen (BIGVUL_GDRIVE_ID, siehe data/README.md)"
	@echo "  cwe-index  - CWE-Katalog laden + Retrieval-Index bauen (RAG)"
	@echo "  baseline   - Baseline-Evaluation auf test"
	@echo "  baseline-smoke - Baseline-Probelauf (10 val-Instanzen, nicht geloggt)"
	@echo "  baseline-val   - Baseline auf vollem val-Split (nicht geloggt)"
	@echo "  rag        - RAG-Evaluation auf test"
	@echo "  rag-smoke  - RAG-Probelauf (10 val-Instanzen, nicht geloggt)"
	@echo "  rag-val    - RAG auf vollem val-Split (nicht geloggt)"
	@echo "  sft-train  - SFT/QLoRA-Training (GPU + uv sync --extra train; SFT_CONFIG=... für Seeds)"
	@echo "  sft        - SFT-Evaluation auf test (braucht trainierten Adapter; SFT_CONFIG=... für Seeds)"
	@echo "  sft-smoke  - SFT-Probelauf: Mini-Training + 10 val-Instanzen (GPU, nicht geloggt)"
	@echo "  sft-val    - SFT: trainierter Adapter auf vollem val-Split (nicht geloggt)"
	@echo "  dual_agent - Dual-Agent-Evaluation auf test"
	@echo "  dual_agent-smoke - Dual-Agent-Probelauf (10 val-Instanzen, nicht geloggt)"
	@echo "  dual_agent-val   - Dual-Agent auf vollem val-Split (nicht geloggt)"
	@echo "  baseline-seed43/44, rag-seed43/44, dual_agent-seed43/44 - Seed 43/44 auf test (geloggt)"
	@echo "  gan        - GAN-Schleife in ein eigenes Verzeichnis (Sonderläufe; regulär: gan-val)"
	@echo "  gan-eval   - Snapshots aus gan-val auf test auswerten (Runden-Ablation)"
	@echo "  gan-smoke  - GAN-Probelauf (4 Runden, Eval auf 10 val-Instanzen, nicht geloggt)"
	@echo "  gan-val    - GAN-Schleife auf train, Auswertung aller Snapshots auf val (nicht geloggt)"
	@echo "  gan-val-phaseb-only - Ablation: nur Phase B (keine adversariale Runde), val, nicht geloggt"
	@echo "  gan-loop-seed43/44  - nur die GAN-Schleife für Seed 43/44"
	@echo "  gan-eval-seed43/44  - Seed 43/44 auf test: round0000 + final (geloggt)"
	@echo "  gan-eval-greedy-date - Referenzauswertung von round0000 mit festem Chat-Datum (val)"
	@echo "  triple_agent       - Triple-Agent-Schleife in ein eigenes Verzeichnis (Sonderläufe; regulär: triple_agent-val)"
	@echo "  triple_agent-eval  - Snapshots aus triple_agent-val auf test auswerten (Runden-Ablation)"
	@echo "  triple_agent-smoke - Triple-Agent-Probelauf (4 Runden, Eval auf 10 val-Instanzen, nicht geloggt)"
	@echo "  triple_agent-val   - Triple-Agent-Schleife, Auswertung aller Snapshots auf val (nicht geloggt)"
	@echo "  triple_agent-loop-seed43/44    - nur die Triple-Agent-Schleife für Seed 43/44"
	@echo "  triple_agent-eval-seed42/43/44 - Seed 42/43/44 auf test: round0000 + final (geloggt)"
	@echo "  eval       - offizielle Test-Läufe aggregieren: Vergleichstabelle + Statistik (CPU)"
	@echo "  eval-val   - val-Läufe der Replikationsansätze mit den Paper-Werten vergleichen (CPU)"
	@echo "  tables     - Diagnostik, LaTeX-Tabellen und Abbildungen der Kapitel 7/8 aus results/ erzeugen (CPU)"

setup:
	uv sync --extra dev

test:
	uv run pytest

lint:
	uv run ruff check --fix .
	uv run ruff format .

data:
	@if [ ! -f data/raw/bigvul.csv ] && [ -z "$(BIGVUL_GDRIVE_ID)" ]; then \
		echo "ERROR: BIGVUL_GDRIVE_ID ist nicht gesetzt und data/raw/bigvul.csv fehlt."; \
		echo "  BIGVUL_GDRIVE_ID=<file-id> make data   (id: siehe data/README.md)"; \
		exit 1; \
	fi
	bash scripts/download_data.sh
	uv run python -m vulndetect.data.build --config configs/data.yaml

baseline:
	uv run python -m vulndetect.baseline --config configs/baseline.yaml

# Schneller Probelauf (10 Instanzen aus val, ohne experiments.csv-Eintrag)
baseline-smoke:
	uv run python -m vulndetect.baseline --config configs/baseline_smoke.yaml --limit 10

# Generalprobe: voller val-Split (496 Instanzen), ohne experiments.csv-Eintrag
baseline-val:
	uv run python -m vulndetect.baseline --config configs/baseline_val.yaml

# Retrieval-Schicht: CWE-Snapshot laden + Chroma-Index bauen (CPU, einmalig)
cwe-index:
	bash scripts/download_cwe.sh
	uv run python -m vulndetect.rag.index --config configs/rag.yaml

rag:
	uv run python -m vulndetect.rag --config configs/rag.yaml

# Schneller Probelauf (10 Instanzen aus val, ohne experiments.csv-Eintrag)
rag-smoke:
	uv run python -m vulndetect.rag --config configs/rag_smoke.yaml --limit 10

# Generalprobe: voller val-Split (496 Instanzen), ohne experiments.csv-Eintrag
rag-val:
	uv run python -m vulndetect.rag --config configs/rag_val.yaml

# QLoRA-Training (GPU-Server + `uv sync --extra train` nötig); Adapter -> models/sft/seed<seed>/
sft-train:
	uv run python -m vulndetect.sft.train --config $(SFT_CONFIG)

sft:
	@adapter=$$(uv run python -c "import yaml; print(yaml.safe_load(open('$(SFT_CONFIG)'))['model']['adapter'])"); \
	test -d "$$adapter" || { echo "ERROR: Adapter $$adapter fehlt — erst 'make sft-train SFT_CONFIG=$(SFT_CONFIG)'."; exit 1; }
	uv run python -m vulndetect.sft --config $(SFT_CONFIG)

# GPU-Probelauf: Mini-Training (5 Steps, 64 Instanzen) + Eval auf 10 val-Instanzen, nicht geloggt
sft-smoke:
	uv run python -m vulndetect.sft.train --config configs/sft_smoke.yaml --limit 64
	uv run python -m vulndetect.sft --config configs/sft_smoke.yaml --limit 10

# Generalprobe: nur Eval des Adapters aus `make sft-train` auf vollem val-Split, nicht geloggt
sft-val:
	@adapter=$$(uv run python -c "import yaml; print(yaml.safe_load(open('configs/sft_val.yaml'))['model']['adapter'])"); \
	test -d "$$adapter" || { echo "ERROR: Adapter $$adapter fehlt — erst 'make sft-train'."; exit 1; }
	uv run python -m vulndetect.sft --config configs/sft_val.yaml

dual_agent:
	uv run python -m vulndetect.dual_agent --config configs/dual_agent.yaml

# Schneller Probelauf (10 Instanzen aus val, ohne experiments.csv-Eintrag)
dual_agent-smoke:
	uv run python -m vulndetect.dual_agent --config configs/dual_agent_smoke.yaml --limit 10

# Generalprobe: voller val-Split (496 Instanzen), ohne experiments.csv-Eintrag
dual_agent-val:
	uv run python -m vulndetect.dual_agent --config configs/dual_agent_val.yaml

# Seeds 43/44 der Replikationsansätze auf test; Ausgabe unter seed<N>/.
baseline-seed43:
	uv run python -m vulndetect.baseline --config configs/baseline_seed43.yaml

baseline-seed44:
	uv run python -m vulndetect.baseline --config configs/baseline_seed44.yaml

rag-seed43:
	uv run python -m vulndetect.rag --config configs/rag_seed43.yaml

rag-seed44:
	uv run python -m vulndetect.rag --config configs/rag_seed44.yaml

dual_agent-seed43:
	uv run python -m vulndetect.dual_agent --config configs/dual_agent_seed43.yaml

dual_agent-seed44:
	uv run python -m vulndetect.dual_agent --config configs/dual_agent_seed44.yaml

# GAN-Erweiterung (RQ2): Schleife je Seed unter gan-val; gan-eval wertet deren Snapshots auf test aus.
# `gan` fährt die Schleife in results/gan (nur für Sonderläufe).
gan:
	uv run python -m vulndetect.gan train --config configs/gan.yaml

gan-eval:
	uv run python -m vulndetect.gan eval --config configs/gan.yaml --all-snapshots

# Probelauf: 4 Runden + 4 Phase-B-Items, Auswertung auf 10 val-Instanzen (nicht geloggt).
gan-smoke:
	uv run python -m vulndetect.gan train --config configs/gan_smoke.yaml
	uv run python -m vulndetect.gan eval --config configs/gan_smoke.yaml --all-snapshots --limit 10

# Generalprobe: identische Schleife, Auswertung auf dem val-Split
gan-val:
	uv run python -m vulndetect.gan train --config configs/gan_val.yaml
	uv run python -m vulndetect.gan eval --config configs/gan_val.yaml --all-snapshots

# Ablation: nur Phase B (rounds 0), final-Snapshot auf val auswerten.
gan-val-phaseb-only:
	uv run python -m vulndetect.gan train --config configs/gan_val_phaseb_only.yaml
	uv run python -m vulndetect.gan eval --config configs/gan_val_phaseb_only.yaml --snapshot final

# Zusatz-Seeds 43/44: nur die Schleife; auf test werden round0000 und final ausgewertet.
gan-loop-seed43:
	uv run python -m vulndetect.gan train --config configs/gan_val_seed43.yaml

gan-loop-seed44:
	uv run python -m vulndetect.gan train --config configs/gan_val_seed44.yaml

gan-eval-seed43:
	uv run python -m vulndetect.gan eval --config configs/gan_seed43.yaml --snapshot round0000
	uv run python -m vulndetect.gan eval --config configs/gan_seed43.yaml --snapshot final

gan-eval-seed44:
	uv run python -m vulndetect.gan eval --config configs/gan_seed44.yaml --snapshot round0000
	uv run python -m vulndetect.gan eval --config configs/gan_seed44.yaml --snapshot final

# Referenzlauf: round0000-Snapshot erneut greedy mit anderem Chat-Datum auswerten.
gan-eval-greedy-date:
	uv run python -m vulndetect.gan eval --config configs/gan_val_greedy_date.yaml --snapshot round0000

# Triple-Agent-Erweiterung (RQ3): GAN-Schleife mit dem Dual-Agent als detektierender Partei;
# triple_agent-eval wertet die Snapshots von triple_agent-val auf test aus.
triple_agent:
	uv run python -m vulndetect.triple_agent train --config configs/triple_agent.yaml

triple_agent-eval:
	uv run python -m vulndetect.triple_agent eval --config configs/triple_agent.yaml --all-snapshots

# Probelauf: 4 Runden + 4 Phase-B-Items, Auswertung auf 10 val-Instanzen (nicht geloggt).
triple_agent-smoke:
	uv run python -m vulndetect.triple_agent train --config configs/triple_agent_smoke.yaml
	uv run python -m vulndetect.triple_agent eval --config configs/triple_agent_smoke.yaml --all-snapshots --limit 10

# Generalprobe: identische Schleife, Auswertung auf dem val-Split
triple_agent-val:
	uv run python -m vulndetect.triple_agent train --config configs/triple_agent_val.yaml
	uv run python -m vulndetect.triple_agent eval --config configs/triple_agent_val.yaml --all-snapshots

# Zusatz-Seeds 43/44: nur die Schleife; auf test werden round0000 und final ausgewertet.
triple_agent-loop-seed43:
	uv run python -m vulndetect.triple_agent train --config configs/triple_agent_val_seed43.yaml

triple_agent-loop-seed44:
	uv run python -m vulndetect.triple_agent train --config configs/triple_agent_val_seed44.yaml

# Seed-42-Auswertung nur round0000 + final, symmetrisch zu den Seeds 43/44.
triple_agent-eval-seed42:
	uv run python -m vulndetect.triple_agent eval --config configs/triple_agent.yaml --snapshot round0000
	uv run python -m vulndetect.triple_agent eval --config configs/triple_agent.yaml --snapshot final

triple_agent-eval-seed43:
	uv run python -m vulndetect.triple_agent eval --config configs/triple_agent_seed43.yaml --snapshot round0000
	uv run python -m vulndetect.triple_agent eval --config configs/triple_agent_seed43.yaml --snapshot final

triple_agent-eval-seed44:
	uv run python -m vulndetect.triple_agent eval --config configs/triple_agent_seed44.yaml --snapshot round0000
	uv run python -m vulndetect.triple_agent eval --config configs/triple_agent_seed44.yaml --snapshot final

# Aggregation der offiziellen Test-Läufe (CPU): mean±std über Seeds, Vergleichstabelle, t-Tests,
# McNemar mit Holm-Korrektur je Familie. Ohne Argumente entsteht genau results/metrics/comparison_test.*.
# Optional: make eval REFERENCE=results/metrics/paper_reference.csv
EVAL_APPROACHES ?= baseline,rag,sft,dual_agent,gan/round0000,gan/round0010,gan/round0050,gan/round0200,gan/final,triple_agent/round0000,triple_agent/final
EVAL_FAMILIES ?= --family gan/final:baseline,gan/final:dual_agent --family rag:baseline,sft:baseline,dual_agent:baseline --family triple_agent/final:gan/final
eval:
	uv run python -m vulndetect.eval.aggregate --split test --approaches $(EVAL_APPROACHES) $(EVAL_FAMILIES) $(if $(REFERENCE),--reference $(REFERENCE))

# Vergleich der val-Läufe der Replikationsansätze mit den Paper-Werten (results/metrics/comparison_val.*).
eval-val:
	uv run python -m vulndetect.eval.aggregate --split val --approaches baseline_val,rag_val,dual_agent_val,sft_val --reference results/metrics/paper_reference.csv

# Tabellen und Abbildungen der Arbeit aus den fertigen Ergebnissen (CPU; Token-Zählung braucht HF_TOKEN).
tables:
	uv run python scripts/ch7_diagnostics.py
	uv run python scripts/ch8_diagnostics.py
	uv run python scripts/ch7_tables.py
	uv run python scripts/ch7_figures.py
