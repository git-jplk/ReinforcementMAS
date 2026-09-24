#!/usr/bin/env python3
"""
apply_probe_patch.py - integrates the probe hooks into inference_mas.py.

    python apply_probe_patch.py --file $WS/RecursiveMAS/inference/inference_utils/inference_mas.py --dry-run
    python apply_probe_patch.py --file $WS/RecursiveMAS/inference/inference_utils/inference_mas.py
    python apply_probe_patch.py --file ...inference_mas.py --revert

Why a script instead of just a .patch file: The patch locates its insertion points
via anchor text rather than line numbers. Therefore, it can also be applied to a
file that already contains other modifications, and it is idempotent - running it
a second time does nothing.

Eight locations are inserted, all marked with ``PROBE-PATCH``:
  1 Import of probe_readout
  2 Eight --probe_* arguments (default: disabled)
  3 Initialization in main()
  4 Round counter in the recursion loop
  5-8 One call each after the four OuterLink transfers

Without --probe_readout, none of these hooks trigger any computationally heavy operations.
"""
from __future__ import annotations

import argparse
import difflib
import os
import shutil
import sys

MARK = "PROBE-PATCH"
BACKUP_SUFFIX = ".pre_probe"

IMPORT_BLOCK = '''# --- PROBE-PATCH (Start): non-invasive latent readout ---------------------
try:
    from inference_utils import probe_readout as _probe
except Exception:  # Direct execution without package context
    import probe_readout as _probe
# --- PROBE-PATCH (End) ----------------------------------------------------


'''

ARGS_BLOCK = '''    # --- PROBE-PATCH (Start): Analysis readout, disabled by default ----------
    parser.add_argument(
        "--probe_readout",
        action="store_true",
        help="Additionally write latents as text to the probe log. Does not alter execution.",
    )
    parser.add_argument("--probe_jsonl", type=str, default="",
                        help="Target file for the probe log. Disabled if no path is provided.")
    parser.add_argument("--probe_indices", type=str, default="",
                        help="Comma-separated list of question indices, e.g., '3,17,88'.")
    parser.add_argument("--probe_ids_file", type=str, default="",
                        help="File containing one index or question hash per line.")
    parser.add_argument("--probe_max_questions", type=int, default=20,
                        help="Upper limit of questions to read out (prevents giant logs).")
    parser.add_argument("--probe_positions", type=str, default="last8",
                        help="Which latent positions: all | lastK | firstK | everyK | 0,-1")
    parser.add_argument("--probe_topk", type=int, default=5,
                        help="How many tokens to log per position.")
    parser.add_argument("--probe_read_tokens", type=int, default=64,
                        help="Length of read generation per stage. 0 disables it.")
    # --- PROBE-PATCH (End) -------------------------------------------------
'''

INIT_BLOCK = '''
    # --- PROBE-PATCH (Start) -----------------------------------------------
    if getattr(args, "probe_readout", False):
        _probe.configure(
            args=args,
            questions=questions,
            role_paths={
                "planner": planner_model,
                "critic": refiner_model,
                "solver": solver_model,
            },
        )
    # --- PROBE-PATCH (End) -------------------------------------------------
'''

ROUND_BLOCK = '''            _probe.set_round(round_idx)  # PROBE-PATCH
'''


def hook(stage, sender, receiver, inner, outer, prompt) -> str:
    return (
        f'        _probe.record_handover(  # {MARK}\n'
        f'            stage="{stage}", sender="{sender}", receiver="{receiver}", start=start,\n'
        f'            hidden=hidden_rollout, inner={inner}, outer={outer},\n'
        f'            model=model, tokenizer=tokenizer, embed_layer=embed_layer,\n'
        f'            prompt_embeds={prompt}, attention_mask=attention_mask,\n'
        f'        )\n'
    )


# (context marker, anchor line, text to insert, "before"/"after")
EDITS = [
    (
        None,
        "RELEASE_RECOMMENDED_SETTINGS: Dict[Tuple[str, str], Dict[str, float]] = {",
        IMPORT_BLOCK,
        "before",
    ),
    (
        None,
        '    parser.add_argument("--solver_pre_question", type=int, default=0)',
        ARGS_BLOCK,
        "before",
    ),
    (
        None,
        "    base_sample_seed = args.sample_seed if args.sample_seed >= 0 else args.seed",
        INIT_BLOCK,
        "after",
    ),
    (
        None,
        "        for round_idx in range(recursive_rounds):",
        ROUND_BLOCK,
        "after",
    ),
    (
        'desc="planner latent",',
        "        lat12 = run_outer_adapter(outer_12, planner_self, output_dtype=planner_embed_dtype)",
        hook("planner->critic", "planner", "critic", "planner_self", "lat12", "input_embeds"),
        "after",
    ),
    (
        'desc="refiner latent",',
        "        mapped = run_outer_adapter(outer_23, refiner_self, output_dtype=refiner_embed_dtype)",
        hook("critic->solver", "critic", "solver", "refiner_self", "mapped", "batch_embeds"),
        "after",
    ),
    (
        'desc="solver feedback latent",',
        "        mapped_feedback = run_outer_adapter(outer_31, solver_self, output_dtype=torch.float32)",
        hook("solver->planner", "solver", "planner", "solver_self", "mapped_feedback", "batch_embeds"),
        "after",
    ),
    (
        'desc="planner feedback latent",',
        "        lat12 = run_outer_adapter(outer_12, planner_self, output_dtype=planner_embed_dtype)",
        hook("planner->critic", "planner", "critic", "planner_self", "lat12", "batch_embeds"),
        "after",
    ),
]


def apply_edits(text: str) -> str:
    lines = text.splitlines(keepends=True)
    for context, anchor, block, where in EDITS:
        start_at = 0
        if context is not None:
            hits = [i for i, l in enumerate(lines) if context in l]
            if not hits:
                raise SystemExit(f"[abort] Context marker not found: {context!r}")
            start_at = hits[0]
        idx = None
        for i in range(start_at, len(lines)):
            if lines[i].rstrip("\n") == anchor:
                idx = i
                break
        if idx is None:
            raise SystemExit(
                f"[abort] Anchor line not found (context {context!r}):\n    {anchor}\n"
                "The file deviates too much. Please insert this change manually."
            )
        at = idx if where == "before" else idx + 1
        lines.insert(at, block)
    return "".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to inference_mas.py")
    ap.add_argument("--dry-run", action="store_true", help="Only show the diff")
    ap.add_argument("--revert", action="store_true", help="Restore backup")
    ap.add_argument("--force", action="store_true", help="Patch even if marker is present")
    a = ap.parse_args()

    path = os.path.abspath(a.file)
    backup = path + BACKUP_SUFFIX
    if not os.path.isfile(path):
        raise SystemExit(f"[abort] File not found: {path}")

    if a.revert:
        if not os.path.isfile(backup):
            raise SystemExit(f"[abort] No backup found: {backup}")
        shutil.copy2(backup, path)
        print(f"[ok] Restored {path} from {backup}.")
        return 0

    original = open(path, encoding="utf-8").read()
    if MARK in original and not a.force:
        print("[ok] File is already patched, nothing to do.")
        return 0

    patched = apply_edits(original)

    diff = "".join(difflib.unified_diff(
        original.splitlines(keepends=True),
        patched.splitlines(keepends=True),
        fromfile=os.path.basename(path),
        tofile=os.path.basename(path) + " (with probe)",
    ))
    print(diff)

    # In-memory syntax check - does not write a .pyc, so it also works
    # in read-only directories.
    try:
        compile(patched, path, "exec")
    except SyntaxError as exc:
        raise SystemExit(
            f"[abort] Patched file cannot be compiled: "
            f"Line {exc.lineno}: {exc.msg}"
        )

    if a.dry_run:
        print("[dry-run] Nothing written. Syntax check passed.")
        return 0

    if not os.path.isfile(backup):
        shutil.copy2(path, backup)
        print(f"[ok] Backup created: {backup}")
    open(path, "w", encoding="utf-8").write(patched)
    print(f"[ok] Patched {path}. Undo with --revert.")
    print("[note] probe_readout.py must be located next to inference_mas.py in inference_utils/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())