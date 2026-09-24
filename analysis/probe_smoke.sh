#!/usr/bin/env bash
# Smoke test for probe readout as a Slurm batch job (instead of salloc).
#
#   bash probe_smoke.sh                 # 4 questions, 2 rounds, 20 min walltime
#   RMAS_DATASETS=mbppplus bash probe_smoke.sh
#
# Minimal walltime and memory footprint to ensure immediate scheduling.
# Writes probe log to $WS/wrong/probe_smoke_<dataset>.jsonl
set -euo pipefail

WS=${WS:?WS is not set}
DS=${RMAS_DATASETS:-math500}
OUT=${OUT:-$WS/wrong}
LOG=$OUT/probe_smoke_${DS}.jsonl

rm -f "$LOG"

RMAS_DATASETS="$DS" \
ROUNDS=2 SEEDS=99 ANS_MODES=ans \
NSAMPLES=4 \
BS_OVERRIDE=4 MNT_OVERRIDE=512 WALL_OVERRIDE=00:20:00 MEM_OVERRIDE=64G \
NAMESUFFIX=smoke \
EXTRA_ARGS="--probe_readout --probe_jsonl $LOG --probe_max_questions 4 \
--probe_positions last2 --probe_read_tokens 32" \
bash collect_wrong.sh