#!/usr/bin/env python3
"""
allows to readout latents without infering with the code
"""
from __future__ import annotations

import atexit
import contextlib
import glob
import hashlib
import json
import os
import time
from typing import Dict, List, Optional, Sequence

import torch

PROBE: Optional["LatentProbe"] = None


# --------------------------------------------------------------------------- Hilfen
@contextlib.contextmanager
def frozen_rng():
    """Sichert den Zufallszustand, damit der Probe das Sampling nicht verschiebt."""
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def parse_positions(spec: str, m: int) -> List[int]:
    """'last8' | 'first4' | 'all' | 'every4' | '0,1,-1' -> Indexliste in [0, m)."""
    spec = (spec or "last8").strip().lower()
    if m <= 0:
        return []
    if spec == "all":
        return list(range(m))
    if spec.startswith("last"):
        k = int(spec[4:] or 8)
        return list(range(max(0, m - k), m))
    if spec.startswith("first"):
        k = int(spec[5:] or 8)
        return list(range(0, min(k, m)))
    if spec.startswith("every"):
        k = max(1, int(spec[5:] or 4))
        out = list(range(0, m, k))
        if m - 1 not in out:
            out.append(m - 1)
        return out
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        i = int(part)
        if i < 0:
            i += m
        if 0 <= i < m:
            out.append(i)
    return sorted(set(out)) or [m - 1]


def qhash(text: str) -> str:
    """Hash ueber den normalisierten Text, damit Leerzeichen keine Rolle spielen."""
    norm = " ".join((text or "").split())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12]


def _resolve_model_dir(path_or_repo: str) -> str:
    """Akzeptiert einen lokalen Pfad ODER eine HF-Repo-ID und liefert das Snapshot-Verzeichnis."""
    if os.path.isdir(path_or_repo):
        return path_or_repo
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(path_or_repo, local_files_only=True)
    except Exception:
        pass
    hub = os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
    )
    d = os.path.join(hub, "models--" + path_or_repo.replace("/", "--"), "snapshots")
    snaps = sorted(glob.glob(os.path.join(d, "*")), key=os.path.getmtime)
    if snaps:
        return snaps[-1]
    raise FileNotFoundError(f"kein lokales Snapshot-Verzeichnis fuer {path_or_repo}")


def _embedding_weight_from_disk(path_or_repo: str) -> torch.Tensor:
    """Laedt nur die Token-Embedding-Matrix aus den safetensors des Repos."""
    from safetensors import safe_open

    path = _resolve_model_dir(path_or_repo)
    cand = []
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(idx):
        with open(idx, encoding="utf-8") as f:
            wm = json.load(f)["weight_map"]
        cand = [(k, os.path.join(path, v)) for k, v in wm.items()]
    else:
        for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
            with safe_open(f, framework="pt") as h:
                cand += [(k, f) for k in h.keys()]
    keys = [
        (k, f)
        for k, f in cand
        if k.endswith("embed_tokens.weight")
        and not any(x in k for x in ("vision", "visual", "audio", "mtp"))
    ]
    if not keys:
        raise KeyError(f"keine embed_tokens.weight in {path}")
    keys.sort(key=lambda kf: (0 if "language_model" in kf[0] else 1, len(kf[0])))
    key, f = keys[0]
    with safe_open(f, framework="pt") as h:
        return h.get_tensor(key)


# --------------------------------------------------------------------------- Probe
class LatentProbe:
    def __init__(
        self,
        out_path: str,
        questions: Sequence[str],
        role_paths: Dict[str, str],
        select: Sequence[int],
        positions: str = "last8",
        topk: int = 5,
        read_tokens: int = 64,
        cache_slots: int = 2,
        meta: Optional[dict] = None,
    ) -> None:
        self.path = out_path
        self.questions = list(questions)
        self.role_paths = dict(role_paths)
        self.select = set(int(i) for i in select)
        self.positions_spec = positions
        self.topk = int(topk)
        self.read_tokens = int(read_tokens)
        self.cache_slots = max(1, int(cache_slots))
        self.round_idx = 0
        self.rollout_idx = 0
        self.failures = 0
        self.enabled = True
        self._rows = 0
        self._emb_cache: Dict[str, dict] = {}     # role -> {"W":..., "norms":..., "scale":...}
        self._tok_cache: Dict[str, object] = {}
        self._scale_logged: Dict[str, bool] = {}
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        self._fh = open(out_path, "a", encoding="utf-8")
        self._write({
            "type": "probe_config",
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_selected": len(self.select),
            "selected": sorted(self.select),
            "positions": positions,
            "topk": topk,
            "read_tokens": read_tokens,
            "role_paths": self.role_paths,
            "meta": meta or {},
        })
        atexit.register(self.close)

    # ---------------------------------------------------------------- Infrastruktur
    def _write(self, row: dict) -> None:
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._rows += 1

    def close(self) -> None:
        try:
            if not self._fh.closed:
                self._write({"type": "probe_end", "rows": self._rows, "failures": self.failures})
                self._fh.flush()
                self._fh.close()
        except Exception:
            pass

    def _tokenizer(self, role: str):
        if role not in self._tok_cache:
            from transformers import AutoTokenizer
            self._tok_cache[role] = AutoTokenizer.from_pretrained(
                self.role_paths[role], local_files_only=True, trust_remote_code=True
            )
        return self._tok_cache[role]

    def _receiver_matrix(self, role: str, device: torch.device) -> dict:
        """Embedding-Matrix des Empfaengers, bevorzugt auf der GPU in fp16."""
        hit = self._emb_cache.get(role)
        if hit is not None:
            return hit
        while len(self._emb_cache) >= self.cache_slots:
            old = next(iter(self._emb_cache))
            self._emb_cache.pop(old, None)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        W = _embedding_weight_from_disk(self.role_paths[role])
        try:
            Wd = W.to(device=device, dtype=torch.float16)
        except Exception:
            Wd = W.to(dtype=torch.float32)          # Rueckfall: CPU
        norms = Wd.float().norm(dim=-1).clamp_min(1e-6)
        entry = {"W": Wd, "norms": norms}
        self._emb_cache[role] = entry

        # Zufallsbasislinie: wie gross ist der beste Kosinus fuer einen rein
        # zufaelligen Vektor? Alles, was nicht deutlich darueber liegt, ist
        # von "zeigt irgendwohin" nicht zu unterscheiden.
        try:
            g = torch.Generator(device="cpu").manual_seed(0)   # eigener Zufall, nicht der des Laufs
            R = torch.randn(64, Wd.size(-1), generator=g).to(device=Wd.device, dtype=Wd.dtype)
            Rn = R / R.float().norm(dim=-1, keepdim=True).clamp_min(1e-6).to(Wd.dtype)
            sims = (Rn @ Wd.t()).float() / norms.to(Rn.device)
            best = sims.max(dim=-1).values
            self._write({"type": "random_baseline", "receiver": role,
                         "median_max_cos": round(float(best.median()), 4),
                         "p95_max_cos": round(float(torch.quantile(best, 0.95)), 4),
                         "vocab": int(Wd.size(0)), "dim": int(Wd.size(-1)),
                         "note": "bester Kosinus eines Zufallsvektors zur Embedding-Matrix"})
        except Exception:
            pass
        return entry

    # ---------------------------------------------------------------- Auslesen
    def _topk_against(self, x: torch.Tensor, W: torch.Tensor, norms: torch.Tensor, tok, k: int):
        """Kosinus gegen alle Zeilen von W, ohne W zu kopieren."""
        xw = x.to(device=W.device, dtype=W.dtype)
        xn = xw / xw.float().norm(dim=-1, keepdim=True).clamp_min(1e-6).to(W.dtype)
        sims = (xn @ W.t()).float() / norms.to(xn.device)
        vals, idx = sims.topk(k, dim=-1)
        out = []
        for r in range(idx.size(0)):
            ids = idx[r].tolist()
            toks = tok.convert_ids_to_tokens(ids) if hasattr(tok, "convert_ids_to_tokens") else [str(i) for i in ids]
            out.append([[str(t), round(float(v), 4)] for t, v in zip(toks, vals[r].tolist())])
        return out

    def _logit_lens(self, hidden: torch.Tensor, model, tok, k: int):
        head = None
        try:
            head = model.get_output_embeddings()
        except Exception:
            head = None
        if head is None:
            return None
        W = head.weight
        x = hidden.to(device=W.device, dtype=W.dtype)
        logits = (x @ W.t()).float()
        probs = torch.softmax(logits, dim=-1)
        vals, idx = probs.topk(k, dim=-1)
        out = []
        for r in range(idx.size(0)):
            ids = idx[r].tolist()
            toks = tok.convert_ids_to_tokens(ids) if hasattr(tok, "convert_ids_to_tokens") else [str(i) for i in ids]
            out.append([[str(t), round(float(v), 4)] for t, v in zip(toks, vals[r].tolist())])
        return out

    def _embed_scale(self, role: str, embed_layer) -> float:
        """Misst, ob das Embedding-Modul selbst skaliert (Gemma: x sqrt(d))."""
        try:
            ids = torch.tensor([[1, 2, 3]], device=embed_layer.weight.device)
            with torch.no_grad():
                got = embed_layer(ids).float().norm(dim=-1).median()
            raw = embed_layer.weight[torch.tensor([1, 2, 3])].float().norm(dim=-1).median()
            return float(got / raw.clamp_min(1e-6))
        except Exception:
            return float("nan")

    def _greedy_continue(self, model, tok, seq, mask, n_tokens):
        """Eigene greedy-Dekodierung ueber Decoder + lm_head.

        Nicht ueber model.generate(), weil einige Modellklassen (z. B.
        Gemma3ForConditionalGeneration) inputs_embeds dort nicht unterstuetzen.
        Greedy zieht nicht, verbraucht also keinen Zufall.
        """
        decoder = None
        try:
            decoder = model.get_decoder()
        except Exception:
            decoder = getattr(model, "model", None)
        head = model.get_output_embeddings()
        embed = model.get_input_embeddings()
        if head is None:
            raise RuntimeError("kein lm_head verfuegbar")

        def step(inp, msk, past=None, want_cache=False):
            kwargs = dict(inputs_embeds=inp, attention_mask=msk, use_cache=want_cache,
                          return_dict=True)
            if past is not None:
                kwargs["past_key_values"] = past
            if decoder is not None:
                out = decoder(**kwargs)
                return out.last_hidden_state[:, -1, :], getattr(out, "past_key_values", None)
            out = model(inputs_embeds=inp, attention_mask=msk, use_cache=False,
                        output_hidden_states=True, return_dict=True)
            return out.hidden_states[-1][:, -1, :], None

        try:
            h, past = step(seq, mask, want_cache=True)
            use_cache = past is not None
        except Exception:
            use_cache = False
            h, past = step(seq, mask)

        ids = []
        for _ in range(n_tokens):
            logits = head(h.to(head.weight.dtype))
            nxt = logits.argmax(dim=-1)
            ids.append(nxt)
            emb = embed(nxt).unsqueeze(1).to(dtype=seq.dtype)
            mask = torch.cat([mask, torch.ones(mask.size(0), 1, dtype=mask.dtype,
                                               device=mask.device)], dim=1)
            if use_cache and past is not None:
                try:
                    h, past = step(emb, mask, past=past, want_cache=True)
                except Exception:      # Cache-Format passt nicht: ohne Cache weiter
                    use_cache = False
                    seq = torch.cat([seq, emb], dim=1)
                    h, _ = step(seq, mask)
            else:
                seq = torch.cat([seq, emb], dim=1)
                h, _ = step(seq, mask)
        out_ids = torch.stack(ids, dim=1)
        return tok.batch_decode(out_ids, skip_special_tokens=True)

    def _read_generation(self, model, tok, prompt_embeds, attention_mask, inner, rows,
                         with_latents=True):
        """Greedy-Lesung, nur fuers Log.

        with_latents=True : Prompt + eigene InnerLink-Ausgaben
                            -> was der Sender nach seinem latenten Denken sagen wuerde,
                               also VOR seinem OuterLink.
        with_latents=False: nur der Prompt. Der enthaelt bereits die eingegangene
                            Nachricht des Vorgaengers, also das Ergebnis des
                            VORIGEN OuterLinks, im Kontext des Empfaengers gelesen.
        """
        if self.read_tokens <= 0 or prompt_embeds is None or attention_mask is None:
            return {}
        try:
            with frozen_rng(), torch.no_grad():
                pe = prompt_embeds[rows].detach()
                am = attention_mask[rows].detach()
                if with_latents:
                    lat = inner[rows].detach().to(dtype=pe.dtype, device=pe.device)
                    seq = torch.cat([pe, lat], dim=1)
                    mask = torch.cat([am, torch.ones(am.size(0), lat.size(1), dtype=am.dtype,
                                                     device=am.device)], dim=1)
                else:
                    seq, mask = pe, am
                texts = self._greedy_continue(model, tok, seq, mask, self.read_tokens)
            return {r: t.strip() for r, t in zip(rows, texts)}
        except Exception as exc:                      # pragma: no cover
            self.failures += 1
            return {r: f"[read-generation fehlgeschlagen: {exc!r}]" for r in rows}

    # ---------------------------------------------------------------- Einstiegspunkt
    def record(
        self,
        stage: str,
        sender: str,
        receiver: str,
        start: int,
        hidden: torch.Tensor,
        inner: torch.Tensor,
        outer: torch.Tensor,
        model,
        tokenizer,
        embed_layer,
        prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> None:
        if not self.enabled:
            return
        rows = [i for i in range(hidden.size(0)) if (start + i) in self.select]
        if not rows:
            return
        try:
            with torch.no_grad():
                m = hidden.size(1)
                pos = parse_positions(self.positions_spec, m)
                recv_tok = self._tokenizer(receiver)
                try:
                    recv = self._receiver_matrix(receiver, outer.device)
                except Exception as exc:
                    recv = None
                    if not self._scale_logged.get("warn_" + receiver):
                        self._scale_logged["warn_" + receiver] = True
                        self._write({"type": "probe_warning", "receiver": receiver,
                                     "error": repr(exc),
                                     "note": "nn_outer faellt aus, Rest wird trotzdem geschrieben"})

                if not self._scale_logged.get(sender):
                    self._scale_logged[sender] = True
                    self._write({
                        "type": "embed_scale",
                        "role": sender,
                        "scale_of_embedding_module": self._embed_scale(sender, embed_layer),
                        "hidden_size": int(embed_layer.weight.size(-1)),
                        "note": "1.0 = Modul skaliert nicht, sqrt(d) = Modul skaliert selbst",
                    })

                W_send = embed_layer.weight
                n_send = W_send.float().norm(dim=-1).clamp_min(1e-6)

                reads = self._read_generation(model, tokenizer, prompt_embeds,
                                              attention_mask, inner, rows, with_latents=True)
                reads_in = self._read_generation(model, tokenizer, prompt_embeds,
                                                 attention_mask, inner, rows, with_latents=False)

                for r in rows:
                    gi = start + r
                    h = hidden[r, pos, :].detach()
                    a = inner[r, pos, :].detach()
                    b = outer[r, pos, :].detach()

                    lens = self._logit_lens(h, model, tokenizer, self.topk)
                    nn_in = self._topk_against(a, W_send, n_send, tokenizer, self.topk)
                    nn_out = (self._topk_against(b, recv["W"], recv["norms"], recv_tok, self.topk)
                              if recv is not None else [[["?", 0.0]] for _ in pos])

                    self._write({
                        "type": "handover",
                        "qi": gi,
                        "qhash": qhash(self.questions[gi]) if gi < len(self.questions) else None,
                        "question_head": (self.questions[gi][:120] if gi < len(self.questions) else None),
                        "round": self.round_idx,
                        "rollout": self.rollout_idx,
                        "stage": stage,
                        "sender": sender,
                        "receiver": receiver,
                        "latent_steps": int(m),
                        "positions": pos,
                        "norms": {
                            "hidden": [round(float(v), 3) for v in h.float().norm(dim=-1).tolist()],
                            "inner": [round(float(v), 3) for v in a.float().norm(dim=-1).tolist()],
                            "outer": [round(float(v), 3) for v in b.float().norm(dim=-1).tolist()],
                        },
                        "lens_hidden": lens,
                        "nn_inner": nn_in,
                        "nn_outer": nn_out,
                        "max_cos_inner": [row[0][1] for row in nn_in],
                        "max_cos_outer": [row[0][1] for row in nn_out],
                        "read_text": reads.get(r),
                        "read_incoming": reads_in.get(r),
                    })
                self._fh.flush()
        except Exception as exc:                      # pragma: no cover
            self.failures += 1
            try:
                self._write({"type": "probe_error", "stage": stage, "round": self.round_idx,
                             "error": repr(exc)})
                self._fh.flush()
            except Exception:
                pass
            if self.failures >= 20:
                self.enabled = False
                print("[probe] zu viele Fehler, Probe wird fuer diesen Lauf abgeschaltet.")


# --------------------------------------------------------------------------- API
def configure(args, questions: Sequence[str], role_paths: Dict[str, str]) -> None:
    """Wird einmal in main() aufgerufen, nur wenn --probe_readout gesetzt ist."""
    global PROBE
    path = getattr(args, "probe_jsonl", "") or ""
    if not path:
        print("[probe] --probe_readout gesetzt, aber --probe_jsonl fehlt: Probe bleibt aus.")
        return

    select: List[int] = []
    raw = (getattr(args, "probe_indices", "") or "").strip()
    if raw:
        for part in raw.replace(";", ",").split(","):
            part = part.strip()
            if part:
                select.append(int(part))
    ids_file = (getattr(args, "probe_ids_file", "") or "").strip()
    if ids_file and os.path.isfile(ids_file):
        hashes = {}
        for i, q in enumerate(questions):
            hashes[qhash(q)] = i
            hashes[hashlib.sha1((q or "").encode("utf-8")).hexdigest()[:12]] = i
        with open(ids_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.isdigit():
                    select.append(int(line))
                elif line in hashes:
                    select.append(hashes[line])
                else:
                    print(f"[probe] unbekannte Kennung uebersprungen: {line}")
    if not select:
        select = list(range(min(int(getattr(args, "probe_max_questions", 20)), len(questions))))
        print(f"[probe] keine Auswahl angegeben, nehme die ersten {len(select)} Fragen.")
    cap = int(getattr(args, "probe_max_questions", 20))
    select = sorted({i for i in select if 0 <= i < len(questions)})[:cap]

    PROBE = LatentProbe(
        out_path=path,
        questions=questions,
        role_paths=role_paths,
        select=select,
        positions=getattr(args, "probe_positions", "last8"),
        topk=int(getattr(args, "probe_topk", 5)),
        read_tokens=int(getattr(args, "probe_read_tokens", 64)),
        meta={
            "dataset": getattr(args, "dataset", ""),
            "num_recursive_rounds": int(getattr(args, "num_recursive_rounds", 0)),
            "latent_steps": int(getattr(args, "latent_steps", 0)),
            "seed": int(getattr(args, "seed", -1)),
            "sample_seed": int(getattr(args, "sample_seed", -1)),
            "temperature": float(getattr(args, "temperature", -1.0)),
        },
    )
    print(f"[probe] aktiv: {len(select)} Fragen, Positionen={PROBE.positions_spec}, "
          f"Lese-Generierung={PROBE.read_tokens} Tokens -> {path}")


def set_round(round_idx: int) -> None:
    if PROBE is not None:
        PROBE.round_idx = int(round_idx)


def set_rollout(rollout_idx: int) -> None:
    if PROBE is not None:
        PROBE.rollout_idx = int(rollout_idx)


def record_handover(**kwargs) -> None:
    if PROBE is not None and PROBE.enabled:
        PROBE.record(**kwargs)


def close() -> None:
    if PROBE is not None:
        PROBE.close()
