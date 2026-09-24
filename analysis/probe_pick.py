#!/usr/bin/env python3
"""
probe_pick.py - Selects questions from finished runs for probe readout
and writes them formatted for --probe_ids_file.

    python probe_pick.py --dir $WS/wrong --dataset math500 --round 3 \
        --group wrong-all --limit 12 --out ids_wrong.txt
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import random
import re

ASY = re.compile(r"\[asy\]|\\begin\{asy", re.I)


def qhash(text: str) -> str:
    norm = " ".join((text or "").split())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12]


def qtext(r):
    for k in ("question", "problem", "prompt", "input", "text"):
        v = r.get(k)
        if isinstance(v, str) and v:
            return v
    return ""


def parse_name(name):
    m = re.match(r"(?P<style>.+?)_(?P<ds>[a-z0-9+]+)_r(?P<r>\d+)_s(?P<s>\w+)_(?P<ans>ans|noans)", name)
    return None if not m else (m["style"], m["ds"], int(m["r"]), m["s"], m["ans"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(os.environ.get("WS", "."), "wrong"))
    ap.add_argument("--dataset", default="math500")
    ap.add_argument("--round", type=int, default=None)
    ap.add_argument("--ans", default="ans")
    ap.add_argument("--group", required=True,
                    choices=["wrong-all", "wrong-any", "correct-all", "truncated", "asy", "all"])
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0, help="Random seed for selection within group")
    ap.add_argument("--max-index", dest="max_index", type=int, default=None,
                    help="Only questions with index < N (matches run's --num_samples N)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    per_q = collections.OrderedDict()
    for p in sorted(glob.glob(os.path.join(a.dir, "*.jsonl"))):
        meta = parse_name(os.path.basename(p)[: -len(".jsonl")])
        if not meta:
            continue
        _, ds, rnd, seed, ans = meta
        if ds != a.dataset or ans != a.ans:
            continue
        if a.round is not None and rnd != a.round:
            continue
        order = 0
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(r, dict) or r.get("type") == "summary":
                    continue
                t = qtext(r)
                h = qhash(t)
                slot = per_q.setdefault(h, {"idx": order, "text": t, "correct": [], "trunc": False})
                slot["correct"].append(bool(r.get("correct")))
                slot["trunc"] = slot["trunc"] or bool(r.get("hit_max_tokens"))
                order += 1

    if not per_q:
        raise SystemExit(f"No matching runs found in {a.dir} (dataset={a.dataset}, ans={a.ans}).")

    def keep(v):
        if a.group == "wrong-all":
            return v["correct"] and not any(v["correct"])
        if a.group == "wrong-any":
            return not all(v["correct"])
        if a.group == "correct-all":
            return v["correct"] and all(v["correct"])
        if a.group == "truncated":
            return v["trunc"]
        if a.group == "asy":
            return bool(ASY.search(v["text"]))
        return True

    chosen = [(h, v) for h, v in per_q.items() if keep(v)
              and (a.max_index is None or v["idx"] < a.max_index)]
    rng = random.Random(a.seed)
    rng.shuffle(chosen)
    chosen = sorted(chosen[: a.limit], key=lambda kv: kv[1]["idx"])

    lines = [f"# group={a.group} dataset={a.dataset} round={a.round} ans={a.ans} n={len(chosen)}"]
    for h, v in chosen:
        head = " ".join(v["text"].split())[:90]
        lines.append(f"{h}   # idx {v['idx']}: {head}")
    text = "\n".join(lines) + "\n"

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()