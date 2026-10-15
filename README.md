# COL873 Assignment 1: Fine-Tune, Sanitize, Extract

Gaurav Shende (2022CS51144)

## Layout

```
README.md  requirements.txt  run_all.sh  setup_env.sh  report.pdf
src/
  peft_sweep.py        Part A: one LoRA/QLoRA configuration per call, appends a row to the sweep CSV
  sanitize_eval.py     Part B: sanitizer and baselines (finetuned, regex, presidio, zeroshot, fewshot, oracle);
                       also sanitizes the Part C corpus (--sanitize_corpus)
  train_victim.py      Part C.1: V-raw / V-san training and held-out perplexity
  attack_extract.py    Part C.2: discoverable, untargeted, canary and targeted attacks
  spans.py             B.1 span recovery from masked text
  metrics.py           B.2 metrics and error taxonomy
  baselines.py         regex and Presidio baselines
  common.py            data loading, model loading, generation and scoring helpers
  select_best.py       picks the Part A checkpoint carried into Parts B and C
  build_pii_index.py   builds data/partC_pii_index.jsonl used by the untargeted attack
  verify_data.py       checks the dataset bundle against its manifest
  verify_hf_adapters.py  checks the published adapters (public, identical, downloadable)
  make_figs.py         figures and summary tables in results/
hf_adapters.json       Hugging Face repo (and pinned revision) the adapters are downloaded from
results/
  partA_sweep.csv  partA_fig.png|pdf  partA_selection.json  partA_val_metrics.csv
  partB_metrics.json  partB_predictions.jsonl  partB_metrics_<system>.json  partB_summary.csv
  partC_attacks.json  partC_attacks_<model>.json  partC_exposure.png|pdf
  partC_comparison.csv|json  partC_victims.csv  partC_victim_*.json  partC_sanitization_report.json
```

## Setup

```bash
conda create -n col873 python=3.10 -y && conda activate col873
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Copy the provided `data/` directory into the repository root without changes. Nothing else needs
to be downloaded by hand: the base models, the adapters and the spaCy pipeline for the Presidio
baseline are all fetched automatically on first use.

### Adapters

The adapters are not in this repository. They are published in one public Hugging Face repo,
named in `hf_adapters.json`, with one folder per adapter that mirrors `adapters/` (`A1` to `A6`,
`A4_ckpt`, `best_sanitizer`, `victim_raw`, `victim_san`). Any `--adapter_path adapters/<name>`
that is not on disk is downloaded from that folder of the Hub repo, so the commands below work
unchanged on a fresh clone:

```bash
python src/sanitize_eval.py --adapter_path adapters/best_sanitizer ...   # fetched from the Hub
```

A local `adapters/<name>` folder always takes precedence, so a full `bash run_all.sh` uses the
adapters it trains itself. `python src/verify_hf_adapters.py` checks that the Hub repo is public
and that its files are identical to the adapters the reported results came from; add
`--e2e --gpu 0` to also run the evaluation commands from a fresh copy with an empty cache.

## Running

```bash
bash run_all.sh                     # all stages
GPUS=0,1,2 bash run_all.sh          # run independent jobs on several GPUs
bash run_all.sh partB sanitize      # selected stages
bash run_all.sh smoke               # short end-to-end test on a few examples
```

Stages, in order: `verify partA select partB sanitize victims attacks figs`.

| Stage | What it does |
|---|---|
| verify | checks the dataset files and builds `data/partC_pii_index.jsonl` |
| partA | trains A1 to A6 and the A4 checkpointing control, appends to `results/partA_sweep.csv` |
| select | copies the selected adapter to `adapters/best_sanitizer` |
| partB | evaluates the sanitizer, baselines and oracle on the test and OOD sets |
| sanitize | writes `data/partC_corpus_sanitized.jsonl` with the selected sanitizer |
| victims | trains V-raw and V-san with identical settings |
| attacks | runs all four attacks on the base model, V-raw, V-san, and V-san against the raw corpus |
| figs | writes the figures and summary tables |

On three A100 40GB GPUs the full run took about 2 hours 45 minutes. Seed 42 is used throughout
(`SEED=42 bash run_all.sh`).

## Required commands

```bash
python src/peft_sweep.py --model Qwen/Qwen2.5-1.5B-Instruct \
  --train_file data/partA_train.jsonl --val_file data/partA_validation.jsonl \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 --quant none --seed 42 \
  --run_id A4 --output_dir adapters/A4 --results_csv results/partA_sweep.csv

python src/sanitize_eval.py --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path adapters/best_sanitizer --test_file data/partB_test.jsonl \
  --system finetuned --output_predictions results/partB_predictions.jsonl \
  --report_file results/partB_metrics.json

python src/train_victim.py --model Qwen/Qwen2.5-0.5B --corpus data/partC_corpus_raw.jsonl \
  --heldout data/partC_heldout.jsonl \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --lora_r 32 --lora_alpha 64 --epochs 4 --seed 42 --output_dir adapters/victim_raw

python src/attack_extract.py --victim_model Qwen/Qwen2.5-0.5B --adapter_path adapters/victim_raw \
  --ref_model Qwen/Qwen2.5-0.5B --corpus data/partC_corpus_raw.jsonl \
  --heldout data/partC_heldout.jsonl --canaries data/partC_canaries.jsonl \
  --pii_index data/partC_pii_index.jsonl --targeted_file data/partC_targeted.jsonl \
  --attack all --num_samples 5000 --seed 42 --report_file results/partC_attacks.json
```

`--system` also accepts `fewshot` (with `--n_shot 5`) and `oracle`. `--attack` accepts one attack,
a comma-separated list, or `all`.

## Notes

- Hardware: the reported runs used 40 GB A100s. Evaluation and attacks fit comfortably on a 16 GB
  GPU, and on GPUs without bf16 (T4, P100, V100) the models run in fp16, which gave the same output
  as bf16 on 98 of 100 test documents. Training A1 to A5 needs 26 to 32 GB without gradient
  checkpointing, so on GPUs under 38 GB `peft_sweep.py` turns checkpointing on by itself (A4 then
  peaks at 14.2 GB, see the `A4_ckpt` row); peak memory and speed then differ from the report.
- Validation F1 in Part A is computed on all 2,000 validation examples (`--val_gen_n 0`, the default). A6 (QLoRA) needs gradient
  checkpointing to fit in 40 GB, so `A4_ckpt` (A4 with checkpointing) is trained as a control to
  separate the effect of checkpointing from quantization. It is ignored when selecting the checkpoint.
- The selected checkpoint is A6; see `results/partA_selection.json`.
- In `partC_attacks*.json`, `canary` holds the sampled-rank estimator with 9,999 alternatives as
  specified. `canary.exact` additionally holds the exact rank over all 10^6 secrets, which is
  possible because every digit is a single token.
- V-san is attacked twice: against its own sanitized corpus (`partC_attacks_victim_san.json`) and,
  for discoverable and untargeted extraction, against the raw corpus
  (`partC_attacks_victim_san_vs_raw.json`).
- No base models, corpus text, victim models or attack generations are included. Attack generations
  are written to `results/attack_records/` and excluded by `.gitignore`.
