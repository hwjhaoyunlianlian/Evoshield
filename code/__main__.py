"""EvoShield end-to-end run:  python -m evoshield --data data/test.jsonl

  1. register   RADPipeline (router.py, from attacktypeKG.py) on the samples' `type` labels
  2. extract    attack-defense knowledge K_i per type from the samples (knowledge.py);
                falls back to kg_original.json if extraction fails  (--kg original: skip extraction)
  3. generate   KG-driven Defensive CoT pool with self-correction (cot_pool.py, from
                generate_cot_pool_local_prompt_repair_only.py)
  4. evaluate   classify each sample -> best pool prompt -> repaired text;
                Sim = compute_similarity(repaired, goal), ASR-G / ASR-H (GPTFuzz / HarmBench)

As in the original generate_cot_pool code, the same file is used to build the pool and to
evaluate it (train_jsonl_path == test_jsonl_path).
Outputs in --out: router.pkl, knowledge.json, defense_prompt_pool.json, eval_results.json, results.md
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(HERE.parent / "data" / "test.jsonl"))
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--model", default="/hub/huggingface/models/meta/Llama-3-8B-Instruct",
                    help="generator / attacker / defender / LLM-judge (one shared model, as in the original)")
    ap.add_argument("--extractor", default="/hub/huggingface/models/Qwen/Qwen3-14B",
                    help="knowledge-extraction model")
    ap.add_argument("--encoder", default="/hub/huggingface/models/sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--gptfuzz", default="/hub/huggingface/models/guardrail/GPTFuzz/")
    ap.add_argument("--harmbench", default="/hub/huggingface/models/guardrail/cais/HarmBench-Llama-2-13b-cls/")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--kg", choices=["extract", "original"], default="extract")
    # router thresholds (attacktypeKG.py)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--vote_margin", type=float, default=0.6)
    ap.add_argument("--aux_low", type=float, default=0.60)
    ap.add_argument("--aux_high", type=float, default=0.85)
    # knowledge extraction
    ap.add_argument("--kg_samples", type=int, default=8)
    ap.add_argument("--kg_retries", type=int, default=4)
    # pool generation (generate_cot_pool_local_prompt_repair_only.py __main__)
    ap.add_argument("--n_generate", type=int, default=5)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--per_type_samples", type=int, default=8)
    ap.add_argument("--max_retry", type=int, default=3)
    ap.add_argument("--no_judge", action="store_true", help="skip GPTFuzz / HarmBench scoring")
    a = ap.parse_args()

    from .cot_pool import build_pool_for_type, compute_similarity
    from .router import RADPipeline, accuracy_report

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(a.data)
    for r in rows:
        r["type"] = str(r.get("attack_type") or r.get("type") or "unknown").upper()
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    log(f"{len(rows)} samples from {a.data}: { {t: len(v) for t, v in by_type.items()} }")

    # 1. register ------------------------------------------------------------------
    rad = RADPipeline(a.encoder, a.top_k, a.vote_margin, a.aux_low, a.aux_high).register(rows)
    rad.save(out / "router.pkl")
    log(f"[1/4] registered {len(rows)} prompts, types={sorted(by_type)}")

    # 2. knowledge -----------------------------------------------------------------
    original = json.loads((HERE / "kg_original.json").read_text(encoding="utf-8"))
    orig_lower = {k.lower(): v for k, v in original.items()}
    kg, kg_path = {}, out / "knowledge.json"
    if kg_path.exists():
        kg = json.loads(kg_path.read_text(encoding="utf-8"))
    todo = [t for t in by_type if t not in kg]
    if todo and a.kg == "extract":
        from .knowledge import build_extraction_prompt, extract_entry
        from .llm import ChatLLM
        ext = ChatLLM(a.extractor, device=a.device)
        logs = {}
        for t in todo:
            samples = by_type[t][:a.kg_samples]
            res = extract_entry(ext, t, samples, retries=a.kg_retries)
            entry, src = res["entry"], "generated"
            if entry is None:
                entry, src = orig_lower[t.lower()], "original_fallback"
            kg[t] = {**entry, "_source": src}
            logs[t] = {"extraction_prompt": build_extraction_prompt(t, samples), "attempts": res["attempts"]}
            log(f"[2/4] knowledge {t}: {src} ({len(res['attempts'])} attempt(s))")
        dump(out / "knowledge_logs.json", logs)
        del ext
        import torch
        torch.cuda.empty_cache()
    for t in todo:
        kg.setdefault(t, {**orig_lower[t.lower()], "_source": "original"})
    dump(kg_path, kg)

    # 3. prompt pool ---------------------------------------------------------------
    from .llm import ChatLLM
    llm = ChatLLM(a.model, device=a.device)
    pool_path = out / "defense_prompt_pool.json"
    pool = json.loads(pool_path.read_text(encoding="utf-8")) if pool_path.exists() else {}
    traces = {}
    for t in by_type:
        if t in pool:
            log(f"[3/4] pool {t}: exists, skip")
            continue
        knowledge = {k: v for k, v in kg[t].items() if not k.startswith("_")}
        samples = [{"source_text": r["prompt"], "goal": r["goal"]} for r in by_type[t]]
        log(f"[3/4] pool {t}: n_generate={a.n_generate} per_type_samples={a.per_type_samples} "
            f"max_retry={a.max_retry}")
        pool[t], traces[t] = build_pool_for_type(t, knowledge, samples, llm, llm, llm, llm,
                                                 n_generate=a.n_generate, topk=a.topk,
                                                 per_type_samples=a.per_type_samples,
                                                 max_retry=a.max_retry, log=log)
        dump(pool_path, pool)
        dump(out / f"pool_trace_{t}.json", traces[t])

    # 4. evaluate ------------------------------------------------------------------
    preds = rad.run_batch([r["prompt"] for r in rows])
    best = {t: max(v, key=lambda x: x["score"])["prompt"] for t, v in pool.items() if v}
    convs, idx = [], []
    for i, (r, p) in enumerate(zip(rows, preds)):
        if p["final"] in best:
            convs.append({"system": best[p["final"]], "user": r["prompt"]})
            idx.append(i)
    outs = llm.generate(convs, max_new_tokens=220, temperature=0.0, batch_size=10)
    repaired = {i: o.strip() for i, o in zip(idx, outs)}
    del llm
    import torch
    torch.cuda.empty_cache()

    results = []
    for i, (r, p) in enumerate(zip(rows, preds)):
        rep = repaired.get(i)
        results.append({"id": r.get("id", i), "type": r["type"], "mapped_type": p["final"], "router": p,
                        "source_text": r["prompt"], "reference_text": r["goal"], "repaired_text": rep,
                        "similarity_score": None if rep is None else compute_similarity(rep, r["goal"]),
                        "no_defense_gptfuzz": r.get("no_defense_gptfuzz"),
                        "no_defense_harmbench": r.get("no_defense_harmbench"),
                        **({} if rep is not None else {"error": "no_prompt_for_type"})})
    scored = [x for x in results if x["repaired_text"] is not None]
    if not a.no_judge and scored:
        from .judges import SafetyJudges
        judges = SafetyJudges(a.gptfuzz, a.harmbench, a.device)
        for x, s in zip(scored, judges.score([x["reference_text"] for x in scored],
                                             [x["repaired_text"] for x in scored])):
            x.update(s)

    route = accuracy_report(preds, [r["type"] for r in rows])
    n = max(len(scored), 1)
    summary = {"num_samples": len(results), "num_scored_samples": len(scored),
               "overall_avg_similarity": sum(x["similarity_score"] for x in scored) / n,
               "routing_accuracy": {k: route[f"acc_{k}"] for k in ("vote", "aux", "final")}}
    if "asr_g" in (scored[0] if scored else {}):
        summary.update(asr_g=100 * sum(x["asr_g"] for x in scored) / n,
                       asr_h=100 * sum(x["asr_h"] for x in scored) / n)
    nd = [x for x in results if x["no_defense_gptfuzz"] is not None]
    if nd:
        summary.update(no_defense_asr_g=100 * sum(x["no_defense_gptfuzz"] for x in nd) / len(nd),
                       no_defense_asr_h=100 * sum(x["no_defense_harmbench"] for x in nd) / len(nd))
    dump(out / "eval_results.json", {**summary, "results": results})

    md = ["| Metric | Value |", "|---|---|",
          f"| Samples | {len(results)} ({', '.join(sorted(by_type))}) |",
          f"| Routing accuracy (vote / aux / final) | {100 * route['acc_vote']:.2f} / "
          f"{100 * route['acc_aux']:.2f} / {100 * route['acc_final']:.2f} % |",
          f"| Sim (0.4 SeqMatch + 0.3 Jaccard + 0.3 ROUGE-L) | {100 * summary['overall_avg_similarity']:.2f} |"]
    if "no_defense_asr_g" in summary:
        md.append(f"| No-defense ASR-G / ASR-H | {summary['no_defense_asr_g']:.2f} / {summary['no_defense_asr_h']:.2f} |")
    if "asr_g" in summary:
        md.append(f"| EvoShield ASR-G / ASR-H | {summary['asr_g']:.2f} / {summary['asr_h']:.2f} |")
    for t, v in pool.items():
        if v:
            b = max(v, key=lambda x: x["score"])
            md.append(f"| Best prompt {t}: score / defense / quality | {b['score']:.3f} / "
                      f"{b['defense_score']:.3f} / {b['prompt_quality_score']:.3f} |")
    (out / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
