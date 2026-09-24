#!/bin/bash
# ===========================================================================
# PATCH 002 — token accounting for the sequential path
#
# WHY. Wallclock is not a reproducible efficiency measure: identical runs on
# this cluster varied 690s-848s (~20%) purely from node load. Token counts are
# deterministic for a given sample_seed, hardware independent, and are what the
# paper reports (34.6-75.6% token reduction). They will also be needed as the
# cost term in the depth controller's reward -- a reward that penalises
# wallclock would not be reproducible.
#
# WHAT. Upstream counts tokens only in the DELIBERATION path
# (inference_mas_deliberation.py:552, used for a per-agent budget). The
# sequential path does not track them at all. This patch adds a module-level
# accumulator to inference_mas.py, increments it at the generate() sites, and
# prints one line at exit:
#
#     [tokens] prompt=<n> generated=<n> calls=<n>
#
# It uses atexit rather than editing wherever the result is printed, so it does
# not depend on knowing that location.
#
# DEFINITIONS (state these in the thesis, the numbers are meaningless without):
#   prompt     tokens fed into generate(), summed over calls and batch rows
#   generated  tokens produced by generate(), excluding padding
#   calls      number of sequences generated (batch rows, summed over calls)
# Latent steps are NOT tokens; count them separately as rounds x agents x
# latent_length, which needs no patch.
#
#   bash patches/apply_patch_002_tokens.sh
# ===========================================================================
set -e

WS="${WS:-/pfs/work9/workspace/scratch/ma_jkliem-jkliem_ma}"
R="$WS/RecursiveMAS"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAS="$R/inference/inference_utils/inference_mas.py"

[ -d "$R/.git" ] || { echo "not a git repo: $R"; exit 1; }
cd "$R"
git checkout -b thesis-patches 2>/dev/null || git checkout thesis-patches

if grep -q "_TOKEN_STATS" "$MAS"; then
  echo "patch 002 already applied"; exit 0
fi
cp "$MAS" "$MAS.bak002"

python3 - "$MAS" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()

# ---- 1. accumulator + helper, inserted before the loader ------------------
anchor = "def load_eval_questions_and_answers("
if anchor not in s:
    sys.exit("anchor 'load_eval_questions_and_answers' not found")

helper = '''# --- thesis patch 002: token accounting -----------------------------------
_TOKEN_STATS = {"prompt": 0, "generated": 0, "calls": 0}


def _count_tokens(prompt_len, sequences, pad_token_id=None):
    """Accumulate prompt/generated token counts. Best effort, never raises."""
    try:
        import torch as _torch
        seqs = sequences.sequences if hasattr(sequences, "sequences") else sequences
        if not hasattr(seqs, "size"):
            return
        rows, total = int(seqs.size(0)), int(seqs.size(1))
        # generate() returns prompt+continuation for input_ids, and (depending
        # on the model family) continuation-only for inputs_embeds.
        if total > prompt_len:
            gen = seqs[:, prompt_len:]
            _TOKEN_STATS["prompt"] += rows * int(prompt_len)
        else:
            gen = seqs
            _TOKEN_STATS["prompt"] += rows * int(prompt_len)
        if pad_token_id is not None:
            _TOKEN_STATS["generated"] += int((gen != pad_token_id).sum().item())
        else:
            _TOKEN_STATS["generated"] += int(gen.numel())
        _TOKEN_STATS["calls"] += rows
    except Exception:
        pass


def _report_tokens():
    if _TOKEN_STATS["calls"]:
        print(f"[tokens] prompt={_TOKEN_STATS['prompt']} "
              f"generated={_TOKEN_STATS['generated']} "
              f"calls={_TOKEN_STATS['calls']}", flush=True)


import atexit as _atexit
_atexit.register(_report_tokens)
# --- end thesis patch 002 -------------------------------------------------


'''
s = s.replace(anchor, helper + anchor, 1)

# ---- 2. instrument the generate() sites ----------------------------------
# site A: input_ids path
a = """        prompt_len = batch_inputs["input_ids"].size(1)
        gen_ids = generated[:, prompt_len:]"""
if a in s:
    s = s.replace(a, a + """
        _count_tokens(prompt_len, generated,
                      getattr(tokenizer, "pad_token_id", None))""", 1)
    print("  instrumented input_ids site")
else:
    print("  WARNING: input_ids site anchor not found")

# site B: inputs_embeds path
b = """        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        prompt_len = attention_mask.size(1)"""
if b in s:
    s = s.replace(b, b + """
        _count_tokens(prompt_len, sequences,
                      getattr(tokenizer, "pad_token_id", None))""", 1)
    print("  instrumented inputs_embeds site")
else:
    print("  WARNING: inputs_embeds site anchor not found")

open(p, "w").write(s)
print("patched inference_mas.py")
PY

git diff > "$HERE/002-token-accounting.patch"
echo
echo "patch written to patches/002-token-accounting.patch"
echo
echo "VERIFY before trusting the numbers:"
echo "  1. run 5 questions and check a [tokens] line appears at the end"
echo "  2. latent_length 0 vs 32 -> generated tokens should barely change"
echo "     (only the final decode differs), latent steps change a lot"
echo "  3. text-mas vs latent-mas on the same 5 questions -> generated tokens"
echo "     should differ by roughly the factor the paper claims"
echo "  If any WARNING appeared above, an anchor did not match: inspect the"
echo "  generate() sites by hand (grep -n 'model.generate(' on the file)."