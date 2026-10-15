#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

GPUS="${GPUS:-0}"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}
GPU="${GPU:-${GPU_ARR[0]}}"
SEED="${SEED:-42}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY="${PY:-python}"
SANITIZER_MODEL="${SANITIZER_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
VICTIM_MODEL="${VICTIM_MODEL:-Qwen/Qwen2.5-0.5B}"

ATTN="q_proj,v_proj"
ATTN_FULL="q_proj,k_proj,v_proj,o_proj"
ALL_LIN="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

EPOCHS="${EPOCHS:-1}"
BATCH="${BATCH:-8}"
ACCUM="${ACCUM:-4}"
LR="${LR:-2e-4}"
VAL_GEN_N="${VAL_GEN_N:-0}"

VICTIM_EPOCHS="${VICTIM_EPOCHS:-4}"
NUM_SAMPLES="${NUM_SAMPLES:-5000}"

SMOKE_FLAG=""
mkdir -p results adapters logs data

log() { printf '\n\033[1m== %s ==\033[0m  (%s)\n' "$1" "$(date '+%H:%M:%S')"; }
run() { echo "+ $*"; CUDA_VISIBLE_DEVICES="$GPU" "$@"; }

PAR_LOGS=(); PAR_CMDS=()
par_add() { PAR_LOGS+=("$1"); shift; PAR_CMDS+=("$(printf '%q ' "$@")"); }
par_flush() {
  local n=${#PAR_CMDS[@]} rc=0 i j pids=() logs=() gpu
  [[ $n -eq 0 ]] && return 0
  for ((i=0; i<n; i+=NGPU)); do
    pids=(); logs=()
    for ((j=0; j<NGPU && i+j<n; j++)); do
      gpu="${GPU_ARR[$j]}"
      echo "+ [gpu $gpu] ${PAR_CMDS[$i+$j]} > ${PAR_LOGS[$i+$j]}"
      CUDA_VISIBLE_DEVICES="$gpu" bash -c "${PAR_CMDS[$i+$j]}" \
        > "${PAR_LOGS[$i+$j]}" 2>&1 &
      pids+=($!); logs+=("${PAR_LOGS[$i+$j]}")
    done
    for ((j=0; j<${#pids[@]}; j++)); do
      if ! wait "${pids[$j]}"; then
        echo "!! FAILED: ${logs[$j]} (last 20 lines)" >&2
        tail -20 "${logs[$j]}" >&2; rc=1
      fi
    done
  done
  PAR_LOGS=(); PAR_CMDS=()
  return $rc
}

stage_verify() {
  log "Stage 0: verify the frozen dataset bundle"
  run $PY src/verify_data.py --data_dir data
  run $PY src/build_pii_index.py \
    --corpus data/partC_corpus_raw.jsonl \
    --canaries data/partC_canaries.jsonl \
    --targeted_file data/partC_targeted.jsonl \
    --output data/partC_pii_index.jsonl
}

stage_partA() {
  log "Stage 1: Part A PEFT sweep"
  local common_args=(
    --model "$SANITIZER_MODEL"
    --train_file data/partA_train.jsonl
    --val_file data/partA_validation.jsonl
    --lora_dropout 0.05 --seed "$SEED"
    --epochs "$EPOCHS" --batch_size "$BATCH" --grad_accum "$ACCUM" --lr "$LR"
    --val_gen_n "$VAL_GEN_N"
    --results_csv results/partA_sweep.csv
  )
  local configs=(
    "A1 $ATTN        8  16  none"
    "A2 $ATTN_FULL   8  16  none"
    "A3 $ALL_LIN     8  16  none"
    "A4 $ALL_LIN    32  64  none"
    "A5 $ALL_LIN    32  16  none"
    "A6 $ALL_LIN    32  64  4bit"
  )
  configs+=("A4_ckpt $ALL_LIN 32 64 none --gradient_checkpointing")
  : > results/partA_sweep.csv
  for cfg in "${configs[@]}"; do
    read -r rid mods r alpha quant extra <<<"$cfg"
    par_add "logs/partA_${rid}.log" \
      $PY src/peft_sweep.py "${common_args[@]}" $extra \
      --target_modules "$mods" --lora_r "$r" --lora_alpha "$alpha" --quant "$quant" \
      --run_id "$rid" --output_dir "adapters/$rid" \
      --save_val_predictions "results/partA_val_predictions_${rid}.jsonl" \
      $SMOKE_FLAG
  done
  par_flush || return 1
  $PY - <<'SORT'
import csv
rows = list(csv.DictReader(open("results/partA_sweep.csv")))
rows.sort(key=lambda r: r["run_id"])
with open("results/partA_sweep.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
SORT
  column -s, -t results/partA_sweep.csv || cat results/partA_sweep.csv
}

stage_select() {
  log "Stage 2: choose the sanitizer to carry into Part B"
  run $PY src/select_best.py \
    --sweep_csv results/partA_sweep.csv \
    --adapters_dir adapters --dest adapters/best_sanitizer \
    --report_file results/partA_selection.json 2>&1 | tee logs/select_best.log
}

stage_partB() {
  log "Stage 3: Part B sanitizer evaluation"
  local B_IND=(--test_file data/partB_test.jsonl --test_set_name ai4privacy_300k_en_test)
  local B_OOD=(--test_file data/partB_ood_test.jsonl --test_set_name ai4privacy_200k_en_ood)

  par_add "logs/partB_finetuned.log" $PY src/sanitize_eval.py "${B_IND[@]}" \
    --base_model "$SANITIZER_MODEL" --adapter_path adapters/best_sanitizer \
    --system finetuned --seed "$SEED" \
    --output_predictions results/partB_predictions.jsonl \
    --report_file results/partB_metrics.json

  par_add "logs/partB_zeroshot.log" $PY src/sanitize_eval.py "${B_IND[@]}" \
    --base_model "$SANITIZER_MODEL" --system zeroshot --n_shot 0 --seed "$SEED" \
    --output_predictions results/partB_predictions_zeroshot.jsonl \
    --report_file results/partB_metrics_zeroshot.json

  par_add "logs/partB_fewshot.log" $PY src/sanitize_eval.py "${B_IND[@]}" \
    --base_model "$SANITIZER_MODEL" --system fewshot --n_shot 5 --seed "$SEED" \
    --output_predictions results/partB_predictions_fewshot.jsonl \
    --report_file results/partB_metrics_fewshot.json

  par_add "logs/partB_ood.log" $PY src/sanitize_eval.py "${B_OOD[@]}" \
    --base_model "$SANITIZER_MODEL" --adapter_path adapters/best_sanitizer \
    --system finetuned --seed "$SEED" \
    --output_predictions results/partB_predictions_ood.jsonl \
    --report_file results/partB_metrics_ood.json
  par_flush || return 1

  log "B.1/B.3: oracle ceiling, regex and Presidio baselines (CPU)"
  local cpu_pids=()
  for spec in "regex:${B_IND[*]}::" "oracle:${B_IND[*]}::" \
              "regex:${B_OOD[*]}:_ood:" "oracle:${B_OOD[*]}:_ood:"; do
    IFS=':' read -r sys targs suffix _ <<<"$spec"
    $PY src/sanitize_eval.py $targs --system "$sys" \
      --output_predictions "results/partB_predictions${suffix}_${sys}.jsonl" \
      --report_file "results/partB_metrics${suffix}_${sys}.json" \
      > "logs/partB${suffix}_${sys}.log" 2>&1 &
    cpu_pids+=($!)
  done
  if $PY -c "import presidio_analyzer" 2>/dev/null; then
    $PY src/sanitize_eval.py "${B_IND[@]}" --system presidio \
      --output_predictions results/partB_predictions_presidio.jsonl \
      --report_file results/partB_metrics_presidio.json \
      > logs/partB_presidio.log 2>&1 &
    cpu_pids+=($!)
  else
    echo "[skip] presidio not installed; the zero-shot/5-shot Qwen baselines cover B.3"
  fi
  local rc=0
  for pid in "${cpu_pids[@]}"; do wait "$pid" || rc=1; done
  [[ $rc -eq 0 ]] || { echo "!! a CPU baseline failed; see logs/partB*.log" >&2; return 1; }
}

stage_sanitize() {
  log "Stage 4: sanitize the victim corpus (Part C.2)"
  run $PY src/sanitize_eval.py \
    --base_model "$SANITIZER_MODEL" --adapter_path adapters/best_sanitizer \
    --system finetuned --seed "$SEED" \
    --sanitize_corpus data/partC_corpus_raw.jsonl \
    --output_corpus data/partC_corpus_sanitized.jsonl \
    --canaries data/partC_canaries.jsonl \
    --targeted_file data/partC_targeted.jsonl \
    --report_file results/partC_sanitization_report.json 2>&1 | tee logs/partC_sanitize.log
}

stage_victims() {
  log "Stage 5: victim training (Part C.2)"
  local vargs=(
    --model "$VICTIM_MODEL" --heldout data/partC_heldout.jsonl
    --target_modules "$ALL_LIN" --lora_r 32 --lora_alpha 64 --lora_dropout 0.0
    --epochs "$VICTIM_EPOCHS" --seed "$SEED"
    --results_csv results/partC_victims.csv
  )
  if [[ ! -s data/partC_corpus_sanitized.jsonl ]]; then
    echo "!! data/partC_corpus_sanitized.jsonl is missing -- run the 'sanitize' stage first" >&2
    exit 1
  fi
  : > results/partC_victims.csv
  par_add "logs/victim_raw.log" $PY src/train_victim.py "${vargs[@]}" \
    --corpus data/partC_corpus_raw.jsonl \
    --output_dir adapters/victim_raw --victim_name victim_raw $SMOKE_FLAG
  par_add "logs/victim_san.log" $PY src/train_victim.py "${vargs[@]}" \
    --corpus data/partC_corpus_sanitized.jsonl \
    --output_dir adapters/victim_san --victim_name victim_san $SMOKE_FLAG
  par_flush || return 1
  column -s, -t results/partC_victims.csv || cat results/partC_victims.csv
}

stage_attacks() {
  log "Stage 6: extraction attacks (Part C.3)"
  local aargs=(
    --victim_model "$VICTIM_MODEL" --ref_model "$VICTIM_MODEL"
    --canaries data/partC_canaries.jsonl
    --pii_index data/partC_pii_index.jsonl
    --targeted_file data/partC_targeted.jsonl
    --heldout data/partC_heldout.jsonl
    --attack all --num_samples "$NUM_SAMPLES" --seed "$SEED"
    --records_dir results/attack_records
  )

  par_add "logs/attacks_control.log" $PY src/attack_extract.py "${aargs[@]}" \
    --corpus data/partC_corpus_raw.jsonl \
    --victim_name control_pretrained \
    --report_file results/partC_attacks_control.json $SMOKE_FLAG

  par_add "logs/attacks_victim_raw.log" $PY src/attack_extract.py "${aargs[@]}" \
    --adapter_path adapters/victim_raw \
    --corpus data/partC_corpus_raw.jsonl \
    --victim_name victim_raw \
    --report_file results/partC_attacks.json $SMOKE_FLAG

  par_add "logs/attacks_victim_san.log" $PY src/attack_extract.py "${aargs[@]}" \
    --adapter_path adapters/victim_san \
    --corpus data/partC_corpus_sanitized.jsonl \
    --victim_name victim_san \
    --report_file results/partC_attacks_victim_san.json $SMOKE_FLAG
  par_flush || return 1

  par_add "logs/attacks_victim_san_vs_raw.log" $PY src/attack_extract.py "${aargs[@]}" \
    --adapter_path adapters/victim_san \
    --corpus data/partC_corpus_raw.jsonl \
    --victim_name victim_san_vs_raw \
    --attack discoverable,untargeted \
    --report_file results/partC_attacks_victim_san_vs_raw.json $SMOKE_FLAG
  par_flush || return 1
}

stage_figs() {
  log "Stage 7: figures and summary tables"
  run $PY src/make_figs.py --results_dir results 2>&1 | tee logs/make_figs.log
}

stage_smoke() {
  log "SMOKE TEST: tiny end-to-end run (nothing here belongs in the report)"
  SMOKE_FLAG="--smoke"
  export SMOKE_FLAG
  run $PY src/verify_data.py --data_dir data --skip_content

  run $PY src/peft_sweep.py --model "$SANITIZER_MODEL" \
    --train_file data/partA_train.jsonl --val_file data/partA_validation.jsonl \
    --target_modules "$ATTN" --lora_r 8 --lora_alpha 16 --quant none --seed "$SEED" \
    --run_id SMOKE --output_dir adapters/SMOKE \
    --results_csv results/smoke_sweep.csv --smoke 2>&1 | tee logs/smoke_partA.log

  for sys in oracle regex; do
    run $PY src/sanitize_eval.py --test_file data/partB_test.jsonl --system "$sys" --limit 40 \
      --output_predictions "results/smoke_pred_${sys}.jsonl" \
      --report_file "results/smoke_metrics_${sys}.json"
  done
  run $PY src/sanitize_eval.py --base_model "$SANITIZER_MODEL" \
    --adapter_path adapters/SMOKE --system finetuned --test_file data/partB_test.jsonl --limit 40 \
    --output_predictions results/smoke_pred_finetuned.jsonl \
    --report_file results/smoke_metrics_finetuned.json 2>&1 | tee logs/smoke_partB.log

  run $PY src/sanitize_eval.py --base_model "$SANITIZER_MODEL" --adapter_path adapters/SMOKE \
    --system finetuned --sanitize_corpus data/partC_corpus_raw.jsonl \
    --output_corpus results/smoke_corpus_sanitized.jsonl \
    --report_file results/smoke_sanitization.json --limit 60 2>&1 | tee logs/smoke_sanitize.log

  run $PY src/train_victim.py --model "$VICTIM_MODEL" --corpus data/partC_corpus_raw.jsonl \
    --heldout data/partC_heldout.jsonl --target_modules "$ATTN" --lora_r 8 --lora_alpha 16 \
    --epochs 1 --seed "$SEED" --output_dir adapters/SMOKE_victim --victim_name SMOKE_victim \
    --results_csv results/smoke_victims.csv --smoke 2>&1 | tee logs/smoke_victim.log

  run $PY src/attack_extract.py --victim_model "$VICTIM_MODEL" \
    --adapter_path adapters/SMOKE_victim --ref_model "$VICTIM_MODEL" \
    --corpus data/partC_corpus_raw.jsonl --canaries data/partC_canaries.jsonl \
    --pii_index data/partC_pii_index.jsonl \
    --targeted_file data/partC_targeted.jsonl --heldout data/partC_heldout.jsonl \
    --attack all --seed "$SEED" --victim_name SMOKE_victim \
    --report_file results/smoke_attacks.json --smoke 2>&1 | tee logs/smoke_attacks.log

  log "SMOKE TEST PASSED -- clean up with:  rm -rf adapters/SMOKE* results/smoke_*"
}

STAGES=("$@")
if [[ ${#STAGES[@]} -eq 0 ]]; then
  STAGES=(verify partA select partB sanitize victims attacks figs)
fi

echo "COL873 A1 pipeline | GPU=$GPU seed=$SEED | stages: ${STAGES[*]}"
for s in "${STAGES[@]}"; do
  case "$s" in
    verify)   stage_verify ;;
    partA)    stage_partA ;;
    select)   stage_select ;;
    partB)    stage_partB ;;
    sanitize) stage_sanitize ;;
    victims)  stage_victims ;;
    attacks)  stage_attacks ;;
    figs)     stage_figs ;;
    smoke)    stage_smoke ;;
    all)      stage_verify
              stage_partA; stage_select; stage_partB; stage_sanitize
              stage_victims; stage_attacks; stage_figs ;;
    *) echo "unknown stage: $s" >&2; exit 2 ;;
  esac
done

log "pipeline finished"
