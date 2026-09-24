#!/usr/bin/env bash
# submit_probe.sh - Selects questions and submits probe runs for evaluation.
#
#   bash submit_probe.sh            # MATH-500 and MBPP+, round 3, seed 42
#   QUICK=1 bash submit_probe.sh    # Only first 200 questions per dataset
#   ROUNDS="1 3" bash submit_probe.sh
set -euo pipefail

WS=${WS:?WS is not set}
OUT=${OUT:-$WS/wrong}
PROBE_ROUNDS=${PROBE_ROUNDS:-3}
PROBE_SEEDS=${PROBE_SEEDS:-42}
TAG=${TAG:-probe2}
PICKROUND=${PICKROUND:-3}
POSITIONS=${POSITIONS:-last8}
READTOK=${READTOK:-64}
MAXQ=${MAXQ:-24}
IDSDIR=${IDSDIR:-$WS/wrong/probe_ids}
mkdir -p "$IDSDIR"

if [ "${QUICK:-0}" = "1" ]; then
  NSAMPLES=200
  MAXIDX="--max-index 200"
else
  NSAMPLES=-1
  MAXIDX=""
fi

python probe_pick.py --dir "$OUT" --dataset math500 --round "$PICKROUND" \
  --group wrong-all   --limit 8 $MAXIDX --out "$IDSDIR/m_wrong.txt"   >/dev/null
python probe_pick.py --dir "$OUT" --dataset math500 --round "$PICKROUND" \
  --group truncated   --limit 6 $MAXIDX --out "$IDSDIR/m_trunc.txt"   >/dev/null
python probe_pick.py --dir "$OUT" --dataset math500 --round "$PICKROUND" \
  --group asy         --limit 4 $MAXIDX --out "$IDSDIR/m_asy.txt"     >/dev/null
python probe_pick.py --dir "$OUT" --dataset math500 --round "$PICKROUND" \
  --group correct-all --limit 6 $MAXIDX --out "$IDSDIR/m_ok.txt"      >/dev/null
cat "$IDSDIR"/m_*.txt > "$IDSDIR/math.txt"

python probe_pick.py --dir "$OUT" --dataset mbppplus --round "$PICKROUND" \
  --group wrong-all   --limit 10 $MAXIDX --out "$IDSDIR/c_wrong.txt"  >/dev/null
python probe_pick.py --dir "$OUT" --dataset mbppplus --round "$PICKROUND" \
  --group correct-all --limit 8  $MAXIDX --out "$IDSDIR/c_ok.txt"     >/dev/null
cat "$IDSDIR"/c_*.txt > "$IDSDIR/code.txt"

PROBE_MATH="--probe_readout --probe_jsonl $OUT/probe_math_r${PROBE_ROUNDS// /_}.jsonl \
--probe_ids_file $IDSDIR/math.txt --probe_max_questions $MAXQ \
--probe_positions $POSITIONS --probe_read_tokens $READTOK"

PROBE_CODE="--probe_readout --probe_jsonl $OUT/probe_code_r${PROBE_ROUNDS// /_}.jsonl \
--probe_ids_file $IDSDIR/code.txt --probe_max_questions $MAXQ \
--probe_positions $POSITIONS --probe_read_tokens $READTOK"

RMAS_DATASETS=math500 ROUNDS="$PROBE_ROUNDS" SEEDS="$PROBE_SEEDS" ANS_MODES=ans \
  NSAMPLES=$NSAMPLES NAMESUFFIX=$TAG EXTRA_ARGS="$PROBE_MATH" \
  bash collect_wrong.sh

RMAS_DATASETS=mbppplus ROUNDS="$PROBE_ROUNDS" SEEDS="$PROBE_SEEDS" ANS_MODES=ans \
  NSAMPLES=$NSAMPLES NAMESUFFIX=$TAG EXTRA_ARGS="$PROBE_CODE" \
  bash collect_wrong.sh