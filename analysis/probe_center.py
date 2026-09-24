#!/usr/bin/env python3
"""
Evaluates whether the narrow cone observed in Critic->Solver
represents genuine information loss or a shared offset (e.g., ln_target bias).

Usage:
    python probe_center.py --domain math
    python probe_center.py --domain code
"""
import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
import probe_links as pl  # noqa: E402


def eff_rank(Z):
    """Participation ratio of the centered covariance: (sum s^2)^2 / sum s^4."""
    Z = Z - Z.mean(0, keepdim=True)
    s = torch.linalg.svdvals(Z.float()) ** 2
    return float(s.sum() ** 2 / (s ** 2).sum())


def row(tag, X, Y):
    p, n1, k, mx, my = pl.geometry(X, Y)
    print(f"  {tag:34} r {p:6.3f}   1-NN {100*n1:5.1f} %   10-NN {100*k:5.1f} %   "
          f"mean cos {mx:6.3f} -> {my:6.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="math", choices=["math", "code"])
    ap.add_argument("--n", type=int, default=2048)
    a = ap.parse_args()
    torch.manual_seed(0)
    from transformers import AutoTokenizer
    import inference_utils.inference_mas as im
    cpu = torch.device("cpu")

    txt, src = pl.sample_text()
    print(f"probe_center.py  domain={a.domain}  test_text: {src}")

    emb, inner = {}, {}
    for role, repo in pl.BASE.items():
        path = pl.snapshot(repo)
        _, t = pl.text_config(path)
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        ids = sorted(set(tok(txt, add_special_tokens=False)["input_ids"]))
        _, _, rows = pl.load_embedding_rows(path, ids)
        mt = (t.to_dict().get("model_type") or "").lower()
        rows = rows * (math.sqrt(t.hidden_size) if mt.startswith("gemma") else 1.0)
        emb[role] = rows[: a.n]
        p = os.path.join(pl.snapshot(pl.INNER_REPO[role]), f"adapter({a.domain}).pt")
        inner[role] = im.load_inner_adapter_module(p, t.hidden_size, cpu, torch.float32, "ln_res_adapter")

    print("\nBaseline: standard token embeddings of receivers (inherent anisotropy)")
    for role in pl.BASE:
        E = emb[role]
        _, _, _, mc, _ = pl.geometry(E, E)
        print(f"  {role:8} mean cosine {mc:6.3f}   effective rank {eff_rank(E):7.1f}   (dim {E.shape[1]})")

    for prefix, snd, rcv in pl.OUTERS:
        name = f"{prefix}-Outerlink({a.domain}).pt"
        path = os.path.join(pl.snapshot(pl.OUTER_REPO), name)
        out_dim = im.infer_outer_adapter_out_dim_from_file(path)
        ad = im.load_outer_adapter_module(path, emb[snd].shape[1], out_dim,
                                          "outer_ln_res_adapter", cpu, torch.float32)
        with torch.no_grad():
            X = dict(inner[snd].named_children())["post_ln"](emb[snd])
            Y = ad(X)
        ln = dict(ad.named_children())["ln_target"]
        b = ln.bias.detach().float() if ln.bias is not None else torch.zeros(Y.shape[1])
        mu = Y.mean(0)
        cos_mu = torch.nn.functional.cosine_similarity(Y, mu.expand_as(Y), dim=-1)

        print(f"\n--- {name}   {snd} -> {rcv}")
        print(f"  |Y| median {float(Y.norm(dim=-1).median()):7.2f}   |mean output vector| {float(mu.norm()):7.2f}"
              f"   |ln_target.bias| {float(b.norm()):7.2f}   cos(bias, mean) "
              f"{float(torch.nn.functional.cosine_similarity(b, mu, dim=0)):6.3f}")
        print(f"  cos(Y_i, mean): median {float(cos_mu.median()):6.3f}  "
              f"-> fraction of output shared across all messages")
        row("as measured", X, Y)
        row("without ln_target bias", X, Y - b)
        row("both centered (mean subtracted)", X - X.mean(0), Y - mu)
        print(f"  effective rank: input {eff_rank(X):7.1f}   output {eff_rank(Y):7.1f}")

    print("\nINTERPRETATION: If r/1-NN recover after centering, the cone is a shared")
    print("offset and differences remain. If they stay low or effective rank drops")
    print("sharply, differences between messages are genuinely lost.")


if __name__ == "__main__":
    main()