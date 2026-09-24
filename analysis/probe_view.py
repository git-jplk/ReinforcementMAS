#!/usr/bin/env python3
"""
probe_view.py - Parses probe logs and computes latent channel metrics.

    python probe_view.py --file probe.jsonl
    python probe_view.py --file probe.jsonl --qi 17
    python probe_view.py --file probe.jsonl --diversity
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import random
import statistics as st


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def top1(entry):
    return [lst[0][0] if lst else None for lst in entry]


def show_question(rows, qi, topk, chars=400):
    sel = [r for r in rows if r.get("type") == "handover" and r.get("qi") == qi]
    if not sel:
        return
    print("=" * 78)
    print(f"Question {qi}: {sel[0].get('question_head')}")
    print("=" * 78)
    for r in sorted(sel, key=lambda r: (r["round"], r["stage"])):
        print(f"\n--- Round {r['round']}  {r['stage']}   Positions {r['positions']} "
              f"of {r['latent_steps']}")
        for j, p in enumerate(r["positions"]):
            lens = ", ".join(t for t, _ in (r["lens_hidden"][j][:topk] if r.get("lens_hidden") else []))
            nin = ", ".join(f"{t}({c})" for t, c in r["nn_inner"][j][:topk])
            nout = ", ".join(f"{t}({c})" for t, c in r["nn_outer"][j][:topk])
            print(f"  Pos {p:3d} | rollout  : {lens}")
            print(f"          | pre-link : {nin}")
            print(f"          | post-link: {nout}")
            print(f"          | norms h/i/o: {r['norms']['hidden'][j]} / "
                  f"{r['norms']['inner'][j]} / {r['norms']['outer'][j]}")
        if r.get("read_incoming"):
            txt = r["read_incoming"].replace("\n", " ")
            txt = txt if chars <= 0 else txt[:chars]
            print(f"  Incoming read ({r['sender']} reading predecessor message): {txt}")
        if r.get("read_text"):
            txt = r["read_text"].replace("\n", " ")
            txt = txt if chars <= 0 else txt[:chars]
            print(f"  Post-rollout read ({r['sender']} prior to outer link): {txt}")


def diversity(rows):
    by_stage = collections.defaultdict(list)
    for r in rows:
        if r.get("type") == "handover":
            by_stage[(r["stage"], r["round"])].append(r)

    print(f"{'Handover':22} {'Round':>5} {'Questions':>9} {'Distinct Top-1':>16} "
          f"{'Overlap':>11} {'Max Cos i':>11} {'Max Cos o':>11}")
    for (stage, rnd), items in sorted(by_stage.items()):
        outs = [top1(r["nn_outer"]) for r in items]
        flat = [t for row in outs for t in row if t]
        distinct = len(set(flat))
        overlaps = []
        rnd_pairs = list(itertools.combinations(range(len(items)), 2))
        random.Random(0).shuffle(rnd_pairs)
        for a, b in rnd_pairs[:200]:
            ra, rb = items[a]["nn_outer"], items[b]["nn_outer"]
            for pa, pb in zip(ra, rb):
                sa = {t for t, _ in pa}
                sb = {t for t, _ in pb}
                overlaps.append(len(sa & sb) / max(1, len(sa)))
        mi = st.median([c for r in items for c in r["max_cos_inner"]])
        mo = st.median([c for r in items for c in r["max_cos_outer"]])
        print(f"{stage:22} {rnd:5d} {len(items):9d} {distinct:6d} / {len(flat):<7d} "
              f"{(100*st.mean(overlaps) if overlaps else float('nan')):10.1f} % "
              f"{mi:11.3f} {mo:11.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--qi", type=int, default=None, help="Inspect a single question index")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--chars", type=int, default=400,
                    help="Character limit for read decoding, 0 for all")
    ap.add_argument("--diversity", action="store_true")
    a = ap.parse_args()

    rows = load(a.file)
    hand = [r for r in rows if r.get("type") == "handover"]

    if a.qi is not None:
        show_question(rows, a.qi, a.topk, a.chars)
        return

    diversity(hand)


if __name__ == "__main__":
    main()