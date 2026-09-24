#!/usr/bin/env python3
"""
CPU test of RecursiveLinks without generation or GPU.

Usage:
    python probe_links.py --domain math
    python probe_links.py --domain code
"""
import argparse
import glob
import inspect
import json
import math
import os
import sys
import traceback

import torch

sys.path.insert(0, os.getcwd())

WS = os.environ.get("WS", "/hkfs/work/workspace/scratch/ma_jkliem-masterarbeit")
HUB = os.path.join(os.environ.get("HF_HOME", os.path.join(WS, "hf_cache")), "hub")

BASE = {
    "planner": "google/gemma-3-4b-it",
    "critic": "meta-llama/Llama-3.2-3B-Instruct",
    "solver": "Qwen/Qwen3.5-4B",
}
INNER_REPO = {
    "planner": "RecursiveMAS/Sequential-Scaled-Planner-Gemma3-4B",
    "critic": "RecursiveMAS/Sequential-Scaled-Critic-Llama3.2-3B",
    "solver": "RecursiveMAS/Sequential-Scaled-Solver-Qwen3.5-4B",
}
OUTER_REPO = "RecursiveMAS/Sequential-Scaled-Outerlinks"
OUTERS = [
    ("Planner-Critic", "planner", "critic"),
    ("Critic-Solver", "critic", "solver"),
    ("Solver-Planner", "solver", "planner"),
]

FALLBACK_TEXT = r"""
Let $P(x)$ be a quadratic polynomial with real coefficients satisfying
$x^2 - 2x + 2 \le P(x) \le 2x^2 - 4x + 3$ for all real numbers $x$, and suppose
$P(11) = 181$. Find $P(16)$. The binary number $10101001110_{2}$ is equal to what
number in base eight? Find the minimum value of \[\sqrt{x^2 + 400} + \sqrt{y^2 + 900}\].
Write a python function to check whether the given number can be represented as
sum of non-zero powers of 2 or not. assert is_Sum_Of_Powers_Of_Two(10) == True
def find_solution(a, b, n): return None if n % gcd(a, b) else (x, y)
In quadrilateral ABCD, angle BAD and angle CDA are trisected as shown. What is the
degree measure of angle AFD? The ratio of zeroes to non-zeroes in an array of integers.
"""

OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


def section(title):
    say("")
    say("=" * 78)
    say(title)
    say("=" * 78)


def snapshot(repo):
    d = os.path.join(HUB, "models--" + repo.replace("/", "--"), "snapshots")
    snaps = sorted(glob.glob(os.path.join(d, "*")), key=os.path.getmtime)
    if not snaps:
        raise FileNotFoundError(f"No snapshot found for {repo} in {d}")
    return snaps[-1]


def text_config(path):
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(path, local_files_only=True)
    return c, getattr(c, "text_config", None) or c


def pct(t, q):
    return float(torch.quantile(t, q))


def stats(t):
    t = t.float()
    if t.numel() > 200_000:
        t = t[torch.randperm(t.numel())[:200_000]]
    return (f"median {pct(t, .5):8.3f} | p5 {pct(t, .05):8.3f} | p95 {pct(t, .95):8.3f}"
            f" | mean {t.mean():8.3f}")


def load_embedding_rows(path, ids):
    """Loads only the embedding matrix from safetensors rather than the whole model."""
    from safetensors import safe_open
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    idx = os.path.join(path, "model.safetensors.index.json")
    cand = []
    if os.path.isfile(idx):
        wm = json.load(open(idx))["weight_map"]
        cand = [(k, os.path.join(path, f)) for k, f in wm.items()]
    else:
        for f in files:
            with safe_open(f, framework="pt") as h:
                cand += [(k, f) for k in h.keys()]
    keys = [(k, f) for k, f in cand
            if k.endswith("embed_tokens.weight")
            and not any(x in k for x in ("vision", "visual", "audio", "mtp"))]
    if not keys:
        raise KeyError("No embed_tokens.weight found")
    keys.sort(key=lambda kf: (0 if "language_model" in kf[0] else 1, len(kf[0])))
    key, f = keys[0]
    with safe_open(f, framework="pt") as h:
        W = h.get_tensor(key)
    all_norms = torch.cat([W[i:i + 16384].float().norm(dim=-1) for i in range(0, W.shape[0], 16384)])
    rows = W[torch.tensor(ids)].float().clone()
    del W
    return key, all_norms, rows


def sample_text():
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return "\n".join(ds[i]["problem"] for i in range(min(200, len(ds)))), "MATH-500 (200 problems)"
    except Exception:
        return FALLBACK_TEXT, "Built-in fallback text"


def geometry(X, Y, n=512):
    """Evaluates how well pairwise geometric structure is preserved."""
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(X.shape[0], generator=g)[:n]
    Xn = torch.nn.functional.normalize(X[idx].float(), dim=-1)
    Yn = torch.nn.functional.normalize(Y[idx].float(), dim=-1)
    Cx, Cy = Xn @ Xn.T, Yn @ Yn.T
    iu = torch.triu_indices(len(idx), len(idx), 1)
    a, b = Cx[iu[0], iu[1]], Cy[iu[0], iu[1]]
    pear = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
    Cx.fill_diagonal_(-2); Cy.fill_diagonal_(-2)
    nn1 = float((Cx.argmax(1) == Cy.argmax(1)).float().mean())
    k = 10
    tx, ty = Cx.topk(k, dim=1).indices, Cy.topk(k, dim=1).indices
    knn = float(sum(len(set(tx[i].tolist()) & set(ty[i].tolist())) for i in range(len(idx))) / (k * len(idx)))
    return pear, nn1, knn, float(a.mean()), float(b.mean())


def capture(adapter, X):
    """Captures submodule outputs independently of the forward execution order."""
    got, hooks = {}, []
    for name, mod in adapter.named_children():
        hooks.append(mod.register_forward_hook(lambda m, i, o, n=name: got.__setitem__(n, o.detach())))
    with torch.no_grad():
        Y = adapter(X)
    for h in hooks:
        h.remove()
    return Y, got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="math", choices=["math", "code"])
    ap.add_argument("--n", type=int, default=2048, help="Max test vectors")
    a = ap.parse_args()
    torch.manual_seed(0)

    say(f"probe_links.py  domain={a.domain}  HUB={HUB}")

    section("[1] Link Forward Pass (modeling.py) - Placement of ln_source / ln_target")
    try:
        from modeling import Adapter, CrossModelAdapter
        for cls in (Adapter, CrossModelAdapter):
            say(f"--- {cls.__name__}.forward")
            say(inspect.getsource(cls.forward))
        say("INTERPRETATION GUIDE: If ln_target is the final operation applied to the")
        say("sum of residual and MLP paths, its gain sets the output scale (Variant A).")
        say("If ln_target is applied only to the branch before adding the residual,")
        say("the scale continues to depend directly on the input norm (Variant B).")
    except Exception as e:
        say(f"[failed] {e!r} (must run from $WS/RecursiveMAS/inference)")

    section("[2] Base Model Position Encodings")
    cfg = {}
    for role, repo in BASE.items():
        try:
            path = snapshot(repo)
            full, t = text_config(path)
            cfg[role] = (path, full, t)
            d = t.to_dict()
            say(f"--- {role:8} {repo}   model_type={d.get('model_type')}  hidden={d.get('hidden_size')}")
            for k, v in d.items():
                if "rope" in k or "position" in k or k == "layer_types":
                    if isinstance(v, list):
                        kinds = {x: v.count(x) for x in dict.fromkeys(v)}
                        say(f"      {k} = {kinds}")
                    else:
                        say(f"      {k} = {v}")
        except Exception as e:
            say(f"--- {role}: [failed] {e!r}")
    say("INTERPRETATION GUIDE: Presence of rope_theta indicates RoPE.")
    say("If layer_types includes linear attention alongside full attention,")
    say("the model is a hybrid where RoPE only acts on a subset of layers.")

    section("Base Model Embedding Norms (Reference for [3])")
    txt, src = sample_text()
    say(f"Test text: {src}")
    emb = {}
    from transformers import AutoTokenizer
    for role in BASE:
        if role not in cfg:
            continue
        path, full, t = cfg[role]
        try:
            tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
            ids = sorted(set(tok(txt, add_special_tokens=False)["input_ids"]))
            key, all_norms, rows = load_embedding_rows(path, ids)
            mt = (t.to_dict().get("model_type") or "").lower()
            scale = math.sqrt(t.hidden_size) if mt.startswith("gemma") else 1.0
            rows = rows * scale
            emb[role] = dict(rows=rows, ids=ids, scale=scale, hidden=t.hidden_size)
            say(f"--- {role:8} key={key}   Forward scaling: x{scale:.2f}")
            say(f"      All vocab rows (raw)      : {stats(all_norms)}")
            say(f"      Text tokens (as ingested) : {stats(rows.norm(dim=-1))}   n={len(ids)}")
        except Exception as e:
            say(f"--- {role}: [failed] {e!r}")

    import inference_utils.inference_mas as im
    cpu = torch.device("cpu")

    section(f"InnerLinks ({a.domain}) - Output Scale (post_ln)")
    inner = {}
    for role, repo in INNER_REPO.items():
        try:
            p = os.path.join(snapshot(repo), f"adapter({a.domain}).pt")
            hid = cfg[role][2].hidden_size
            ad = im.load_inner_adapter_module(p, hid, cpu, torch.float32, "ln_res_adapter")
            inner[role] = ad
            pl = dict(ad.named_children())["post_ln"]
            say(f"--- {role:8} {os.path.basename(p)}   post_ln gain: {stats(pl.weight.detach())}"
                f"   |bias| {float(pl.bias.detach().norm()):.3f}")
            if role in emb:
                with torch.no_grad():
                    y = pl(emb[role]["rows"])
                own = emb[role]["rows"].norm(dim=-1)
                say(f"      InnerLink output norm: {stats(y.norm(dim=-1))}")
                say(f"      -> Feedback into own model: {float(y.norm(dim=-1).median() / own.median()):6.2f}x "
                    f"relative to standard token embeddings")
        except Exception as e:
            say(f"--- {role}: [failed] {e!r}")

    section(f"OuterLinks ({a.domain}) - Scale [3], Geometry [4], Residual vs Branch [5], W3 [6]")
    say("Proxies for actual inputs (InnerLink outputs of sender):")
    say("  scale-aligned : post_ln_Sender(token_embedding) -> matches true latent magnitude")
    say("  raw           : token_embedding as ingested     -> baseline reference")

    def evaluate(ad, X, ref):
        Y, parts = capture(ad, X)
        pear, nn1, knn, mx, my = geometry(X, Y)
        rp = parts.get("residual_proj"); br = parts.get("proj2")
        rr = float(rp.norm(dim=-1).median()) if rp is not None else float("nan")
        bb = float(br.norm(dim=-1).median()) if br is not None else float("nan")
        return dict(xin=X.norm(dim=-1), y=Y.norm(dim=-1),
                    ratio=float(Y.norm(dim=-1).median() / ref.median()),
                    pear=pear, nn1=nn1, knn=knn, mx=mx, my=my, res=rr, br=bb)

    summary = []
    for prefix, snd, rcv in OUTERS:
        name = f"{prefix}-Outerlink({a.domain}).pt"
        say(f"\n--- {name}   {snd} -> {rcv}")
        try:
            p = os.path.join(snapshot(OUTER_REPO), name)
            in_dim = emb[snd]["hidden"]
            out_dim = im.infer_outer_adapter_out_dim_from_file(p)
            ad = im.load_outer_adapter_module(p, in_dim, out_dim, "outer_ln_res_adapter", cpu, torch.float32)
            raw = emb[snd]["rows"][: a.n]
            with torch.no_grad():
                if snd in inner:
                    faithful = dict(inner[snd].named_children())["post_ln"](raw)
                else:
                    say("  [Notice] Sender InnerLink missing -> using unweighted LayerNorm as proxy")
                    faithful = torch.nn.functional.layer_norm(raw, raw.shape[-1:])
            ref = emb[rcv]["rows"].norm(dim=-1)
            res = {}
            for tag, X in (("scale-aligned", faithful), ("raw", raw)):
                r = evaluate(ad, X, ref); res[tag] = r
                say(f"  [{tag}]  Input |x| {stats(r['xin'])}")
                say(f"      [3] Output |y| {stats(r['y'])}   Receiver token {stats(ref)}")
                say(f"          -> Output / Receiver embedding: {r['ratio']:6.2f}x")
                say(f"      [4] Geometry r {r['pear']:6.3f}   1-NN preserved {100 * r['nn1']:5.1f} %"
                    f"   10-NN {100 * r['knn']:5.1f} %   mean cosine {r['mx']:.3f} -> {r['my']:.3f}")
                say(f"      [5] |Residual W3 x| {r['res']:9.3f}   |MLP branch| {r['br']:9.3f}"
                    f"   Branch/Residual {r['br'] / max(r['res'], 1e-9):8.2f}")
            W3 = dict(ad.named_children()).get("residual_proj")
            if W3 is not None:
                sv = torch.linalg.svdvals(W3.weight.float())
                med = float(sv.median())
                band = float(((sv > 0.8 * med) & (sv < 1.25 * med)).float().mean())
                say(f"  [6] W3 singular values: max {float(sv.max()):.3f} median {med:.3f} min {float(sv.min()):.4f}"
                    f"   Condition {float(sv.max() / sv.min()):.1f}   Band [0.8, 1.25]xMedian {100 * band:.1f} %")
            summary.append((prefix, res))
        except Exception as e:
            say(f"      [failed] {e!r}")
            traceback.print_exc()

    section("Summary (Geometry/Residual: scale-aligned proxy, raw in parentheses)")
    say(f"{'OuterLink':16} {'Scale':>8} {'Geometry r':>18} {'1-NN preserved':>20} {'Branch/Residual':>20}")
    for prefix, res in summary:
        f, r = res["scale-aligned"], res["raw"]
        say(f"{prefix:16} {f['ratio']:7.2f}x {f['pear']:8.3f} ({r['pear']:6.3f}) "
            f"{100 * f['nn1']:9.1f} % ({100 * r['nn1']:5.1f} %) "
            f"{f['br'] / max(f['res'], 1e-9):9.2f} ({r['br'] / max(r['res'], 1e-9):7.2f})")
    say("")
    say("INTERPRETATION GUIDE")
    say("  Scale ~1x        -> Output matches receiver embedding magnitude")
    say("  Scale >>1 / <<1  -> Latents are louder/quieter than standard tokens")
    say("  Geometry r ~1    -> Pairwise message geometry preserved; significantly lower")
    say("                      supports the hypothesis of message space distortion")
    say("  Note: Inputs are token embeddings serving as proxies for InnerLink")
    say("  outputs; pipeline confirmation with real latent states requires GPU.")

    out = f"probe_links_{a.domain}.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("```\n" + "\n".join(OUT) + "\n```\n")
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()