"""Attack-knowledge extraction: registration samples -> one KG entry K_i per attack type.

The output schema is the one consumed by evoshield/cot_pool.py (the same schema as the
original knowledge base data/output/Attack_Defense_Knowledge.json):

  attack_type          short snake_case mechanism name
  signature            list of surface fingerprints of the attack type
  interaction_audit    {feature_a, feature_b, multiplicative_risk}
  cascade_threshold    float in [0, 1]: how strongly the combined features indicate this attack
  defense_instructions string: how the defender strips the wrapper and decides on the request

Entries that copy a 5-gram of any sample's original request are rejected and regenerated.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

REQUIRED_TOP_KEYS = ("attack_type", "signature", "interaction_audit", "cascade_threshold",
                     "defense_instructions")

EXTRACTOR_SYSTEM = (
    "You are the attack-knowledge extraction module of EvoShield, a jailbreak defense. "
    "You analyse the STRUCTURE of jailbreak prompts so that a defender can strip the attack "
    "wrapper and recover the original request. You never answer, complete, or elaborate on "
    "any harmful request, and you never write harmful content.")

# The extraction prompt. {family}, {n}, {samples} are filled in by build_extraction_prompt().
EXTRACTION_PROMPT = """Registered attack type: {family}

Below are {n} registration samples of this attack type. Each sample shows the ATTACKED PROMPT
(what the target model receives) and the ORIGINAL REQUEST it was built from.

{samples}

Task: write ONE attack-defense knowledge entry for the {family} attack type as a single JSON
object. The entry is later used to generate a defensive system prompt that (1) recognises this
attack type, (2) removes the adversarial wrapper / perturbation / manipulation, (3) repairs the
input into the underlying request, and (4) refuses if the repaired request is harmful.

Study what is common to ALL samples:
- where the original request sits in the attacked prompt (start, end, repeated, paraphrased,
  split, encoded, hidden behind a separator, ...);
- which tokens, phrases, separators or formatting mark the boundary between request and wrapper;
- what the wrapper does (role-play, fictional framing, persuasion, emotional pressure, rule
  injection, output-format forcing, optimised gibberish suffix, encoding, ...);
- how the request is transformed (verbatim, duplicated, euphemised, abstracted, paraphrased).

Return exactly this JSON schema (all keys required, values in English):
{{
  "attack_type": "<snake_case mechanism name>",
  "signature": ["<surface fingerprint 1>", "<fingerprint 2>", "<fingerprint 3>"],
  "interaction_audit": {{
    "feature_a": "<first structural feature of the attack>",
    "feature_b": "<second feature: what the wrapper asks the model to do>",
    "multiplicative_risk": "<why the combination bypasses alignment although each part looks harmless>"
  }},
  "cascade_threshold": <float between 0 and 1>,
  "defense_instructions": "<one paragraph: how to locate and strip the wrapper, how to recover the underlying request, and how to decide on the recovered request>"
}}

Rules:
- Describe the attack structure only. Do NOT copy, quote, or paraphrase any specific original
  request or any harmful detail from the samples.
- Every instruction must be executable from the attacked prompt alone (the defender never sees
  the original request).
- Output the JSON object only, with no Markdown fences and no commentary."""


def ngrams(text: str, n: int = 5) -> set:
    w = re.findall(r"[a-z0-9']+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def leaks_goals(text: str, goals: Sequence[str], n: int = 5) -> bool:
    t = ngrams(text, n)
    return any(t & ngrams(g, n) for g in goals)


def build_extraction_prompt(family: str, samples: Sequence[Dict[str, Any]],
                            max_prompt_chars: int = 1500) -> str:
    blocks = []
    for i, s in enumerate(samples, 1):
        p = s["prompt"]
        if len(p) > max_prompt_chars:  # keep head and tail: boundaries usually live there
            h = max_prompt_chars // 2
            p = p[:h] + " [...] " + p[-h:]
        blocks.append(f"### Sample {i}\nATTACKED PROMPT:\n{p}\nORIGINAL REQUEST:\n{s['goal']}")
    return EXTRACTION_PROMPT.format(family=family, n=len(samples), samples="\n\n".join(blocks))


def _repair(text: str) -> str:
    """Fix JSON errors extractor LLMs often make: Python raw strings r"..." and bad escapes."""
    text = re.sub(r'(?<=[:\[,\s])r"', '"', text)
    return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)


def _scan(text: str):
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(text[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def parse_entry(raw: str) -> Optional[Dict[str, Any]]:
    text = re.sub(r"```(?:json)?", "", raw or "", flags=re.I)
    for obj in list(_scan(text)) + list(_scan(_repair(text))):
        if all(k in obj for k in REQUIRED_TOP_KEYS):
            try:
                obj["cascade_threshold"] = float(obj["cascade_threshold"])
            except (TypeError, ValueError):
                obj["cascade_threshold"] = 0.8
            if not isinstance(obj["defense_instructions"], str):
                obj["defense_instructions"] = json.dumps(obj["defense_instructions"], ensure_ascii=False)
            return obj
    return None


def extract_entry(llm, family: str, samples: Sequence[Dict[str, Any]], retries: int = 4,
                  max_new_tokens: int = 1536, temperature: float = 0.3, seed: int = 42
                  ) -> Dict[str, Any]:
    """Returns {"entry": KG entry or None, "attempts": [...]} for logging."""
    goals = [s["goal"] for s in samples]
    user = build_extraction_prompt(family, samples)
    attempts: List[Dict[str, Any]] = []
    for k in range(retries):
        raw = llm.generate([{"system": EXTRACTOR_SYSTEM, "user": user}], max_new_tokens=max_new_tokens,
                           temperature=0.0 if k == 0 else temperature, batch_size=1, seed=seed + k)[0]
        entry = parse_entry(raw)
        status = "ok"
        if entry is None:
            status = "malformed"
        elif leaks_goals(json.dumps(entry, ensure_ascii=False), goals):
            status, entry = "leaks_registration_goal", None
        attempts.append({"attempt": k, "status": status, "raw": raw})
        if entry is not None:
            return {"entry": entry, "attempts": attempts}
    return {"entry": None, "attempts": attempts}
