"""Attack-type classification, ported from code/localknowledge/attacktypeKG.py (RADPipeline).

Registration: encode every registration prompt with all-MiniLM-L6-v2 and fit a balanced
logistic-regression classifier ("Aux") on the embeddings.

Classification of a query q:
  Vote : top-k (k=5) nearest registration prompts by cosine similarity; sum the scores per
         type; vote_top1 = type with the largest sum; margin = top1 sum - top2 sum
  Aux  : LR predict_proba(q); aux_type = argmax, aux_score = its probability
  Final:
    if margin < vote_margin (0.6) and aux_score >= aux_low (0.60): final = aux_type
    elif aux_score >= aux_high (0.85):                             final = aux_type
    else:                                                          final = vote_top1
Every query is assigned a type (the original code has no rejection / fallback).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


class RADPipeline:
    def __init__(self, encoder_path: str, top_k: int = 5, vote_margin: float = 0.6,
                 aux_low: float = 0.60, aux_high: float = 0.85, device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.encoder_path, self.top_k = encoder_path, top_k
        self.vote_margin, self.aux_low, self.aux_high = vote_margin, aux_low, aux_high
        self.model = SentenceTransformer(encoder_path, device=device)
        self.embeddings = None
        self.attack_types: List[str] = []
        self.aux_classifier = None

    # ------------------------------------------------------------ registration
    def register(self, rows: Sequence[Dict[str, Any]]) -> "RADPipeline":
        from sklearn.linear_model import LogisticRegression
        self.attack_types = [str(r.get("attack_type") or r.get("type") or "unknown").upper() for r in rows]
        self.embeddings = self.model.encode([r["prompt"] for r in rows], normalize_embeddings=True)
        if len(set(self.attack_types)) == 1:
            # LR needs >= 2 classes; with a single registered type Aux always predicts it (p = 1).
            self.aux_classifier = _SingleClass(self.attack_types[0])
        else:
            self.aux_classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
            self.aux_classifier.fit(self.embeddings, self.attack_types)
        return self

    # ------------------------------------------------------------ classification
    def run_batch(self, queries: Sequence[str]) -> List[Dict[str, Any]]:
        q = self.model.encode(list(queries), normalize_embeddings=True)
        proba = self.aux_classifier.predict_proba(q)
        sims = q @ self.embeddings.T
        out = []
        for i in range(len(q)):
            idx = np.argsort(sims[i])[-self.top_k:][::-1]
            vote: Dict[str, float] = {}
            for j in idx:
                vote[self.attack_types[j]] = vote.get(self.attack_types[j], 0.0) + float(sims[i, j])
            ranked = sorted(vote.items(), key=lambda x: x[1], reverse=True)
            vote_top1, top1 = ranked[0]
            top2 = ranked[1][1] if len(ranked) > 1 else 0.0
            margin = top1 - top2
            k = int(np.argmax(proba[i]))
            aux_type, aux_score = str(self.aux_classifier.classes_[k]), float(proba[i, k])
            if margin < self.vote_margin and aux_score >= self.aux_low:
                final, rule = aux_type, "low_margin_aux"
            elif aux_score >= self.aux_high:
                final, rule = aux_type, "high_conf_aux"
            else:
                final, rule = vote_top1, "vote"
            out.append({"vote": vote_top1, "aux": aux_type, "final": final, "rule": rule,
                        "vote_margin": margin, "aux_score": aux_score})
        return out

    def run(self, query: str) -> Dict[str, Any]:
        return self.run_batch([query])[0]

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({k: getattr(self, k) for k in ("encoder_path", "top_k", "vote_margin", "aux_low",
                                                       "aux_high", "embeddings", "attack_types",
                                                       "aux_classifier")}, f)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "RADPipeline":
        with Path(path).open("rb") as f:
            st = pickle.load(f)
        r = cls(st["encoder_path"], st["top_k"], st["vote_margin"], st["aux_low"], st["aux_high"], device)
        r.embeddings, r.attack_types, r.aux_classifier = st["embeddings"], st["attack_types"], st["aux_classifier"]
        return r


class _SingleClass:
    def __init__(self, label: str):
        self.classes_ = np.array([label])

    def predict_proba(self, x):
        return np.ones((len(x), 1))


def accuracy_report(preds: List[Dict[str, Any]], truth: Sequence[str]) -> Dict[str, Any]:
    """Overall and per-type accuracy of the vote / aux / final predictions (as in run_dataset)."""
    rep: Dict[str, Any] = {"total": len(truth)}
    for m in ("vote", "aux", "final"):
        rep[f"acc_{m}"] = sum(p[m] == t for p, t in zip(preds, truth)) / max(len(truth), 1)
        rep[f"per_type_{m}"] = {t: sum(p[m] == t for p, x in zip(preds, truth) if x == t) /
                                sum(x == t for x in truth) for t in sorted(set(truth))}
    return rep
