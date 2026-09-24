#!/bin/bash
# =========================================================================
# Collect incorrectly answered questions
#
#   bash collect_wrong.sh                      # Preflight + all jobs
#   DRYRUN=1 bash collect_wrong.sh             # Check only, submit nothing
#   LISTADAPTERS=1 bash collect_wrong.sh       # Show only what is cached
#   SEEDTEST=1 bash collect_wrong.sh           # Test if seeds have an effect
#   SEEDS="42 1" ROUNDS="3" bash collect_wrong.sh
#   ANS_MODES="ans noans" SEEDS=42 ROUNDS=3 bash collect_wrong.sh

# PREFLIGHT:
# Before submitting, checks if the environment, base models, checkpoints, and
# datasets exist. Missing components are skipped and listed at the end.
# =========================================================================

WS=${WS:-/hkfs/work/workspace/scratch/ma_jkliem-masterarbeit}
HFH=${HF_HOME:-$WS/hf_cache}
H=$HFH/hub
OUT=${OUT:-$WS/wrong}
PART=${PART:-accelerated}
STYLES=${STYLES:-"sequential_scaled"}     # CHANGED: distillation removed, see below
SEEDS=${SEEDS:-"42 1 2"}
METHOD=${METHOD:-ours_recursive}   # ours_recursive | ours | text_recursive | text
EXTRA_ARGS=${EXTRA_ARGS:-}         # Additional flags for inference_mas (e.g. --probe_readout ...)
NAMESUFFIX=${NAMESUFFIX:-}         # Suffix appended to filenames so custom runs don't collide
ROUNDS=${ROUNDS:-"1 2 3"}                 # CHANGED: recursion round sweep
ANS_MODES=${ANS_MODES:-"ans"}             # CHANGED: "ans noans" for control comparisons
DRYRUN=${DRYRUN:-0}
LISTADAPTERS=${LISTADAPTERS:-0}
SEEDTEST=${SEEDTEST:-0}
ALLOW_CROSS_DOMAIN=${ALLOW_CROSS_DOMAIN:-0}   # Allow math adapters for code tasks
ALLOW_DISTILL=${ALLOW_DISTILL:-0}

SUBMITTED=0
PLANNED=0
SKIPPED=()
note_skip() { SKIPPED+=("$1"); echo "  [skip] $1"; }

# =========================================================================
# 0 - Environment Preflight
# =========================================================================
echo "=============================================================="
echo "PREFLIGHT"
echo "=============================================================="

FATAL=0
for p in "$WS" "$WS/RecursiveMAS" "$HFH"; do
  if [ ! -d "$p" ]; then echo "  MISSING: $p"; FATAL=1; else echo "  ok     : $p"; fi
done

ENVSH=""
for c in "$WS/master_thesis/env.sh" "$WS/thesis/env.sh" "$WS/env.sh"; do
  [ -f "$c" ] && { ENVSH="$c"; break; }
done
if [ -z "$ENVSH" ]; then
  echo "  MISSING: env.sh (searched in master_thesis/, thesis/, \$WS)"
  FATAL=1
else
  echo "  ok     : $ENVSH"
fi

MASPY="$WS/RecursiveMAS/inference/inference_utils/inference_mas.py"
if [ ! -f "$MASPY" ]; then echo "  MISSING: $MASPY"; FATAL=1; else echo "  ok     : inference_mas.py"; fi

if [ "$FATAL" = "1" ]; then
  echo
  echo "Aborting: Core prerequisites missing. Check WS/HF_HOME."
  exit 1
fi

# CHANGED: Record git commit hash for data provenance
GITSHA=$(git -C "$WS/RecursiveMAS" rev-parse --short HEAD 2>/dev/null || echo "unknown")
echo "  RecursiveMAS commit: $GITSHA"

# --- Test Python imports (catches broken virtual environments early) ----
if [ "$DRYRUN" != "1" ] && [ "$LISTADAPTERS" != "1" ]; then
  echo -n "  Python import check ... "
  if ( source "$ENVSH" >/dev/null 2>&1; cd "$WS/RecursiveMAS/inference" && \
       python -c "import torch, transformers, datasets" 2>/dev/null ); then
    echo "ok"
  else
    echo "FAILED"
    echo "         torch/transformers/datasets cannot be imported."
    echo "         Run 'source $ENVSH' and debug manually."
    exit 1
  fi
fi

mkdir -p "$OUT" "$WS/logs"

# =========================================================================
# Helper Functions
# =========================================================================
# CHANGED: Selects the NEWEST snapshot and warns if multiple exist.
# V1 picked via 'head -1' (arbitrary), meaning multiple cache revisions
# could silently execute against stale checkpoints.
snap() {   # snap <repo-fragment> -> snapshot directory path or empty
  local d n s
  d=$(find "$H" -maxdepth 1 -type d -name "*$1*" 2>/dev/null | head -1)
  [ -z "$d" ] && return 1
  n=$(find "$d/snapshots" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)
  s=$(find "$d/snapshots" -maxdepth 1 -mindepth 1 -type d -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | head -1 | cut -d' ' -f2-)
  [ -z "$s" ] && return 1
  if [ "$n" -gt 1 ]; then
    echo "  [warn] $1: $n snapshots found in cache, using newest" >&2
  fi
  echo "$s"
}

pick() {   # pick <dir> <patterns...> -> first matching file
  local dir="$1"; shift
  for pat in "$@"; do
    local f
    f=$(find "$dir" -maxdepth 1 -name "$pat" 2>/dev/null | head -1)
    [ -n "$f" ] && { echo "$f"; return 0; }
  done
  return 1
}

model_cached() {   # model_cached <HF-model-name>
  local frag="models--${1//\//--}"
  local d="$H/$frag"
  [ -d "$d" ] || return 1
  find "$d" \( -name "*.safetensors" -o -name "*.bin" \) 2>/dev/null | grep -q . || return 1
  return 0
}

dataset_cached() {   # dataset_cached <math500|mbppplus>
  case "$1" in
    math500)  find "$HFH" -maxdepth 4 -type d -iname "*math-500*" 2>/dev/null | grep -q . ;;
    mbppplus) find "$HFH" -maxdepth 4 -type d \( -iname "*mbpp*" -o -iname "*evalplus*" \) 2>/dev/null | grep -q . ;;
    *) return 0 ;;
  esac
}

# CHANGED: Domain tag per dataset used in adapter matching patterns.
adapter_tag() {
  case "$1" in
    math500)  echo "math" ;;
    mbppplus) echo "code" ;;
    gpqa*)    echo "science" ;;
    *)        echo "" ;;
  esac
}

# --- Settings per (Style, Dataset) --------------------------------------
# NOTE on recursion depth: RND is NO LONGER set here, but inside the
# ROUNDS loop. Only token length and compute budgets remain here.
#
# NOTE on latent_steps: Three conflicting references exist:
#   - Paper Table 9: Optimum around 64 to 80; 48 performs measurably worse
#   - Shipped config in repo: 32 (MATH500) / 48 (GPQA, MBPP+)
#   - V1 of this script: 64 / 32 / 16
# Defaults match V1 to maintain consistency with previous runs.
# Can be overridden via LAT_OVERRIDE for ablation experiments.
ds_settings() {
  case "$1:$2" in
    sequential_scaled:math500)  LAT=64; MNT=2000; BS=16; TEMPD=0.6; WALL="04:00:00"; MEM=96G ;;
    sequential_scaled:mbppplus) LAT=32; MNT=4000; BS=16; TEMPD=0.2; WALL="10:00:00"; MEM=96G ;;
    distillation:mbppplus)      LAT=16; MNT=4000; BS=16; TEMPD=0.2; WALL="12:00:00"; MEM=180G ;;
    *) return 1 ;;
  esac
  [ -n "$LAT_OVERRIDE" ] && LAT=$LAT_OVERRIDE
  [ -n "${MNT_OVERRIDE:-}" ] && MNT=$MNT_OVERRIDE
  [ -n "${BS_OVERRIDE:-}" ]  && BS=$BS_OVERRIDE
  [ -n "${WALL_OVERRIDE:-}" ] && WALL=$WALL_OVERRIDE
  [ -n "${MEM_OVERRIDE:-}" ]  && MEM=$MEM_OVERRIDE
  # CHANGED: Temperature aligned with release runner (run.py): MBPP+ = 0.2, otherwise 0.6.
  TEMP=${TEMP_OVERRIDE:-$TEMPD}
  return 0
}

# =========================================================================
# Style Definitions (Models + Repo Names)
# =========================================================================
setup_style() {
  case "$1" in
    sequential_scaled)
      DSETS_DEFAULT="math500 mbppplus"
      A1="google/gemma-3-4b-it"
      A2="meta-llama/Llama-3.2-3B-Instruct"
      A3="Qwen/Qwen3.5-4B"
      R_OUT="Sequential-Scaled-Outerlinks"
      R_1="Sequential-Scaled-Planner"; R_2="Sequential-Scaled-Critic"; R_3="Sequential-Scaled-Solver"
      ;;
    distillation)
      # WARNING: Distillation is two-stage (Expert -> Learner), but
      # inference_mas.py expects three agents. V1 therefore mapped the Expert
      # twice. This causes outer_12 to map Expert(9B) -> Expert(9B), even
      # though the link was trained as Expert(9B) -> Learner(4B). This is
      # either a tensor shape mismatch or silently computes nonsense.
      # Additionally, outer_31 lacks a matching pattern. Disabled by default.
      # Only viable once inference_mas.py properly supports N=2.
      DSETS_DEFAULT="mbppplus"
      A1="Qwen/Qwen3.5-9B"; A2="Qwen/Qwen3.5-9B"; A3="Qwen/Qwen3.5-4B"
      R_OUT="Distillation-Outerlinks"
      R_1="Distillation-Expert"; R_2="Distillation-Expert"; R_3="Distillation-Learner"
      ;;
    *) return 1 ;;
  esac
  return 0
}

# =========================================================================
# LISTADAPTERS - Display only cached adapter weights
# =========================================================================
# This resolves the primary verification question: do code adapters exist
# at all, or were the code evaluation numbers generated using math links?
if [ "$LISTADAPTERS" = "1" ]; then
  echo
  echo "=============================================================="
  echo "ADAPTER FILES IN CACHE"
  echo "=============================================================="
  for STYLE in sequential_scaled distillation; do
    setup_style "$STYLE" || continue
    echo
    echo "--- $STYLE"
    for R in "$R_OUT" "$R_1" "$R_2" "$R_3"; do
      SN=$(snap "$R") || { echo "  $R: Repo not found in cache"; continue; }
      echo "  $R"
      echo "    $SN"
      find "$SN" -maxdepth 1 -name '*.pt' -printf '      %f  (%s bytes)\n' 2>/dev/null | sort
    done
  done
  echo
  echo "Verification: Do file names contain a domain tag (math / code / science)?"
  echo "If only *math* exists, the code benchmark results were generated"
  echo "with math-trained links - this is a key empirical finding and must be logged."
  exit 0
fi

# =========================================================================
# SEEDTEST - Verify sample seed effectiveness
# =========================================================================
# Without this check, any evaluation of seed divergence is invalid: if
# --sample_seed does not reach the generation RNG, all "seeds" yield identical
# runs and observed mode collapses are merely plumbing bugs.
if [ "$SEEDTEST" = "1" ]; then
  echo
  echo "=============================================================="
  echo "SEEDTEST - 3 sample seeds across 4 questions"
  echo "=============================================================="
  STYLE=${STYLES%% *}
  setup_style "$STYLE" || { echo "Unknown style"; exit 1; }
  DS=${DATASETS:-math500}
  ds_settings "$STYLE" "$DS" || { echo "Missing configuration"; exit 1; }
  TAG=$(adapter_tag "$DS")

  SNAP_OUT=$(snap "$R_OUT") || { echo "Repo $R_OUT missing"; exit 1; }
  SNAP_1=$(snap "$R_1") || exit 1
  SNAP_2=$([ "$R_2" = "$R_1" ] && echo "$SNAP_1" || snap "$R_2") || exit 1
  SNAP_3=$(snap "$R_3") || exit 1
  IN1=$(pick "$SNAP_1" "adapter*${TAG}*.pt") || { echo "inner1 missing"; exit 1; }
  IN2=$(pick "$SNAP_2" "adapter*${TAG}*.pt") || { echo "inner2 missing"; exit 1; }
  IN3=$(pick "$SNAP_3" "adapter*${TAG}*.pt") || { echo "inner3 missing"; exit 1; }
  O12=$(pick "$SNAP_OUT" "*Planner-Critic*${TAG}*.pt" "*12*${TAG}*.pt") || { echo "outer12 missing"; exit 1; }
  O23=$(pick "$SNAP_OUT" "*Critic-Solver*${TAG}*.pt"  "*23*${TAG}*.pt") || { echo "outer23 missing"; exit 1; }
  O31=$(pick "$SNAP_OUT" "*Solver-Planner*${TAG}*.pt" "*31*${TAG}*.pt") || { echo "outer31 missing"; exit 1; }

  for S in 42 1 2; do
    J="$OUT/seedtest_${DS}_s${S}.jsonl"
    rm -f "$J"
    CMD="source $ENVSH; cd $WS/RecursiveMAS/inference; \
python -u -m inference_utils.inference_mas \
  --dataset $DS --num_samples 4 --seed 42 --sample_seed $S \
  --num_recursive_rounds 2 --latent_steps $LAT --batch_size 4 \
  --max_new_tokens $MNT --do_sample --temperature $TEMP --top_p 0.95 --ans \
  --num_rollouts 1 --result_jsonl '$J' \
  --agent1_model_name_or_path $A1 --agent2_model_name_or_path $A2 \
  --agent3_model_name_or_path $A3 \
  --agent1_inner_aligner_path '$IN1' --agent2_inner_aligner_path '$IN2' \
  --agent3_inner_aligner_path '$IN3' \
  --outer_12_path '$O12' --outer_23_path '$O23' --outer_31_path '$O31'"
    JID=$(sbatch --parsable --job-name="seedtest-s${S}" \
      --partition=$PART --gres=gpu:1 --time=00:40:00 --mem=96G --cpus-per-task=8 \
      --output=$WS/logs/seedtest_s${S}_%j.out --wrap="$CMD" 2>&1)
    echo "  Seed $S -> Job $JID"
  done
  echo
  echo "Once jobs have completed, run:"
  echo "  python extract_wrong.py --dir $OUT --seed-identity-check"
  echo
  echo "If raw outputs are byte-for-byte identical, --sample_seed has no effect."
  echo "Fix this issue before proceeding with any benchmark measurements."
  exit 0
fi

# =========================================================================
# Main Execution Loop
# =========================================================================
echo
echo "=============================================================="
echo "JOBS"
echo "=============================================================="
echo "Styles: $STYLES"
echo "Seeds : $SEEDS      Rounds: $ROUNDS      ans: $ANS_MODES"
echo "Output: $OUT"
[ "$DRYRUN" = "1" ] && echo "MODE: DRYRUN - No jobs will be submitted"
echo

for STYLE in $STYLES; do
  echo "--- $STYLE"

  if ! setup_style "$STYLE"; then
    note_skip "Unknown style: $STYLE"; continue
  fi

  if [ "$STYLE" = "distillation" ] && [ "$ALLOW_DISTILL" != "1" ]; then
    note_skip "distillation - disabled (duplicate Expert mapping, outer_31 undefined)"
    echo "         Set ALLOW_DISTILL=1 to override, but results"
    echo "         cannot be interpreted until N=2 support is implemented."
    continue
  fi

  # CHANGED: DATASETS is exported globally as a path on the cluster. Use RMAS_DATASETS;
  # ignore DATASETS if it resembles a file path.
  DSETS=${RMAS_DATASETS:-}
  if [ -z "$DSETS" ] && [ -n "${DATASETS:-}" ] && [ "${DATASETS#/}" = "${DATASETS}" ]; then
    DSETS=$DATASETS
  fi
  DSETS=${DSETS:-$DSETS_DEFAULT}

  # --- Check base models in cache ---
  MISSING_M=""
  for m in "$A1" "$A2" "$A3"; do
    model_cached "$m" || MISSING_M="$MISSING_M $m"
  done
  if [ -n "$MISSING_M" ]; then
    note_skip "$STYLE - Base models missing in cache:$MISSING_M"
    echo "         Download via: python setup/download_assets.py (on login node with HF_HUB_OFFLINE disabled)"
    continue
  fi
  echo "  Base models: cached"

  # --- Check checkpoint repositories ---
  SNAP_OUT=$(snap "$R_OUT") || { note_skip "$STYLE - Repo $R_OUT missing"; continue; }
  SNAP_1=$(snap "$R_1")     || { note_skip "$STYLE - Repo $R_1 missing";  continue; }
  if [ "$R_2" = "$R_1" ]; then SNAP_2=$SNAP_1; else
    SNAP_2=$(snap "$R_2")   || { note_skip "$STYLE - Repo $R_2 missing";  continue; }
  fi
  SNAP_3=$(snap "$R_3")     || { note_skip "$STYLE - Repo $R_3 missing";  continue; }
  echo "  Checkpoint repos: cached"

  # --- Iterate per dataset ---
  for DS in $DSETS; do
    if ! ds_settings "$STYLE" "$DS"; then
      note_skip "$STYLE/$DS - No settings configured"
      continue
    fi
    if ! dataset_cached "$DS"; then
      note_skip "$STYLE/$DS - Dataset missing from cache"
      continue
    fi

    # CHANGED: Adapter selection PER DATASET with explicit domain tagging.
    TAG=$(adapter_tag "$DS")
    if [ -z "$TAG" ]; then
      note_skip "$STYLE/$DS - No domain tag defined"
      continue
    fi

    IN1=$(pick "$SNAP_1" "adapter*${TAG}*.pt") || IN1=""
    IN2=$(pick "$SNAP_2" "adapter*${TAG}*.pt") || IN2=""
    IN3=$(pick "$SNAP_3" "adapter*${TAG}*.pt") || IN3=""
    O12=$(pick "$SNAP_OUT" "*Planner-Critic*${TAG}*.pt" "*Expert-Learner*${TAG}*.pt" "*12*${TAG}*.pt") || O12=""
    O23=$(pick "$SNAP_OUT" "*Critic-Solver*${TAG}*.pt"  "*Learner-Expert*${TAG}*.pt" "*23*${TAG}*.pt") || O23=""
    O31=$(pick "$SNAP_OUT" "*Solver-Planner*${TAG}*.pt" "*31*${TAG}*.pt") || O31=""

    XDOM=0
    MISSING_F=""
    [ -z "$IN1" ] && MISSING_F="$MISSING_F inner1"
    [ -z "$IN2" ] && MISSING_F="$MISSING_F inner2"
    [ -z "$IN3" ] && MISSING_F="$MISSING_F inner3"
    [ -z "$O12" ] && MISSING_F="$MISSING_F outer12"
    [ -z "$O23" ] && MISSING_F="$MISSING_F outer23"
    [ -z "$O31" ] && MISSING_F="$MISSING_F outer31"

    # CHANGED: NO silent fallback to math adapters. Must be explicitly enabled
    # and clearly tagged, otherwise skip execution.
    if [ -n "$MISSING_F" ]; then
      if [ "$ALLOW_CROSS_DOMAIN" = "1" ] && [ "$TAG" != "math" ]; then
        echo "  [!] $DS: Adapters for '$TAG' missing ($MISSING_F)"
        echo "      ALLOW_CROSS_DOMAIN=1 -> Falling back to math adapters."
        echo "      Results are NOT domain-isolated and must be flagged as such in logs."
        IN1=$(pick "$SNAP_1" 'adapter*math*.pt' 'adapter*.pt') || IN1=""
        IN2=$(pick "$SNAP_2" 'adapter*math*.pt' 'adapter*.pt') || IN2=""
        IN3=$(pick "$SNAP_3" 'adapter*math*.pt' 'adapter*.pt') || IN3=""
        O12=$(pick "$SNAP_OUT" '*Planner-Critic*math*.pt' '*Expert-Learner*math*.pt' '*12*math*.pt') || O12=""
        O23=$(pick "$SNAP_OUT" '*Critic-Solver*math*.pt'  '*Learner-Expert*math*.pt' '*23*math*.pt') || O23=""
        O31=$(pick "$SNAP_OUT" '*Solver-Planner*math*.pt' '*31*math*.pt') || O31=""
        XDOM=1
        for f in "$IN1" "$IN2" "$IN3" "$O12" "$O23" "$O31"; do
          [ -z "$f" ] && { note_skip "$STYLE/$DS - Math fallback adapters also incomplete"; continue 2; }
        done
      else
        note_skip "$STYLE/$DS - Adapters missing for domain '$TAG':$MISSING_F"
        echo "         Contents of $SNAP_OUT:"
        ls "$SNAP_OUT" 2>/dev/null | sed 's/^/           /'
        echo "         Use LISTADAPTERS=1 to list all cached repositories."
        echo "         Use ALLOW_CROSS_DOMAIN=1 to explicitly allow math fallback."
        continue
      fi
    fi
    echo "  $DS: Adapter '$TAG' ok$([ "$XDOM" = "1" ] && echo '  [CROSS-DOMAIN]')"

    # --- Sweep over rounds, seeds, and answer modes ---
    for RND in $ROUNDS; do
      for S in $SEEDS; do
        for AM in $ANS_MODES; do

          case "$AM" in
            ans)   ANSFLAG="--ans" ;;
            noans) ANSFLAG="" ;;
            *) note_skip "Unknown ANS_MODE: $AM"; continue ;;
          esac

          BASE="${STYLE}_${DS}_r${RND}_s${S}_${AM}"
          [ "$XDOM" = "1" ] && BASE="${BASE}_xdom"
          # Append non-standard temperatures to filename to preserve legacy runs
          [ "$TEMP" != "0.6" ] && BASE="${BASE}_t${TEMP/./}"
          [ "$METHOD" != "ours_recursive" ] && BASE="${BASE}_${METHOD}"
          [ -n "$NAMESUFFIX" ] && BASE="${BASE}_${NAMESUFFIX}"
          JSONL="$OUT/${BASE}.jsonl"
          META="$OUT/${BASE}.meta.json"

          # CHANGED: Resume verifies the terminal summary record instead of file size
          if [ -s "$JSONL" ] && tail -1 "$JSONL" 2>/dev/null | grep -q '"type"[[:space:]]*:[[:space:]]*"summary"'; then
            echo "    r=$RND s=$S $AM  completed, skipping"
            continue
          fi
          if [ -s "$JSONL" ]; then
            echo "    r=$RND s=$S $AM  incomplete run detected, rerunning"
            mv "$JSONL" "${JSONL}.partial.$(date +%s)"
          fi

          PLANNED=$((PLANNED+1))

          if [ "$DRYRUN" = "1" ]; then
            printf "    [dry] r=%s s=%-3s %-5s lat=%-3s mnt=%-5s %s %s\n" \
                   "$RND" "$S" "$AM" "$LAT" "$MNT" "$WALL" "$MEM"
            continue
          fi

          # CHANGED: Write execution provenance metadata
          {
            echo "{"
            echo "  \"style\": \"$STYLE\","
            echo "  \"dataset\": \"$DS\","
            echo "  \"round\": $RND,"
            echo "  \"sample_seed\": $S,"
            echo "  \"ans_mode\": \"$AM\","
            echo "  \"latent_steps\": $LAT,"
            echo "  \"max_new_tokens\": $MNT,"
            echo "  \"temperature\": $TEMP,"
            echo "  \"method\": \"$METHOD\","
            echo "  \"adapter_tag\": \"$TAG\","
            echo "  \"cross_domain\": $XDOM,"
            echo "  \"git_commit\": \"$GITSHA\","
            echo "  \"adapters\": {"
            for nm in IN1 IN2 IN3 O12 O23 O31; do
              pth=${!nm}
              sha=$(sha256sum "$pth" 2>/dev/null | cut -c1-16)
              comma=","; [ "$nm" = "O31" ] && comma=""
              echo "    \"$nm\": {\"path\": \"$pth\", \"sha256_16\": \"$sha\"}$comma"
            done
            echo "  }"
            echo "}"
          } > "$META"

          CMD="source $ENVSH; cd $WS/RecursiveMAS/inference; \
for f in '$IN1' '$IN2' '$IN3' '$O12' '$O23' '$O31'; do \
  [ -f \"\$f\" ] || { echo \"MISSING in job environment: \$f\"; exit 3; }; done; \
python -u -m inference_utils.inference_mas \
  --dataset $DS --num_samples ${NSAMPLES:--1} --seed 42 --sample_seed $S --method $METHOD \
  --num_recursive_rounds $RND --latent_steps $LAT --batch_size $BS \
  --max_new_tokens $MNT --do_sample --temperature $TEMP --top_p 0.95 $ANSFLAG \
  --num_rollouts 1 --result_jsonl '$JSONL' \
  --agent1_model_name_or_path $A1 \
  --agent2_model_name_or_path $A2 \
  --agent3_model_name_or_path $A3 \
  --agent1_inner_aligner_path '$IN1' \
  --agent2_inner_aligner_path '$IN2' \
  --agent3_inner_aligner_path '$IN3' \
  --outer_12_path '$O12' --outer_23_path '$O23' --outer_31_path '$O31' $EXTRA_ARGS; \
rc=\$?; [ -s '$JSONL' ] || echo 'WARNING: No output results file written'; exit \$rc"

          JID=$(sbatch --parsable \
            --job-name="w-${STYLE:0:4}-${DS:0:4}-r${RND}s${S}" \
            --partition=$PART --gres=gpu:1 --time=$WALL --mem=$MEM --cpus-per-task=8 \
            --output=$WS/logs/wrong_${BASE}_%j.out \
            --wrap="$CMD" 2>&1)
          if [[ "$JID" =~ ^[0-9]+$ ]]; then
            SUBMITTED=$((SUBMITTED+1))
            printf "    r=%s s=%-3s %-5s  job %s\n" "$RND" "$S" "$AM" "$JID"
          else
            note_skip "$STYLE/$DS r=$RND s=$S $AM - sbatch rejected: $JID"
          fi
        done
      done
    done
  done
done

# =========================================================================
echo
echo "=============================================================="
echo "SUMMARY"
echo "=============================================================="
if [ "$DRYRUN" = "1" ]; then
  echo "  Planned    : $PLANNED"
else
  echo "  Submitted  : $SUBMITTED"
fi
if [ ${#SKIPPED[@]} -gt 0 ]; then
  echo "  Skipped    :"
  for s in "${SKIPPED[@]}"; do echo "    - $s"; done
fi
echo
echo "  Status     : squeue -u \$USER"
echo "  Analysis   : python extract_wrong.py --dir $OUT --always-only"
echo "  By Round   : python extract_wrong.py --dir $OUT --by-round"
echo "  Divergence : python extract_wrong.py --dir $OUT --seed-divergence"