"""Batched local chat-model generation (Hugging Face transformers)."""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I).strip()


class ChatLLM:
    def __init__(self, model_path: str, device: str = "auto", dtype: str = "bfloat16"):
        self.model_path = model_path
        self.tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        device_map = "auto" if device == "auto" else {"": device}
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=getattr(torch, dtype), device_map=device_map
        ).eval()
        self.device = self.model.get_input_embeddings().weight.device

    def render(self, system: Optional[str], user: str) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        kw = {"add_generation_prompt": True, "tokenize": False}
        if "qwen3" in self.model_path.lower():
            kw["enable_thinking"] = False
        try:
            return self.tok.apply_chat_template(msgs, **kw)
        except Exception:  # templates without a system role (e.g. Gemma-2)
            kw.pop("enable_thinking", None)
            merged = (system + "\n\n" + user) if system else user
            return self.tok.apply_chat_template([{"role": "user", "content": merged}], **kw)

    def chat(self, messages, temperature: float = 0.7, max_new_tokens: int = 512) -> str:
        """Single-conversation call with the interface of LocalLLM.generate in
        code/CoT/generate_cot_pool_local_prompt_repair_only.py (messages list or str, top_p 0.9)."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system") or None
        user = "\n\n".join(m["content"] for m in messages if m["role"] != "system")
        return self.generate([{"system": system, "user": user}], max_new_tokens=max_new_tokens,
                             temperature=temperature, batch_size=1, top_p=0.9)[0].strip()

    @torch.inference_mode()
    def generate(self, convs: List[Dict[str, str]], max_new_tokens: int = 512, temperature: float = 0.0,
                 batch_size: int = 8, seed: Optional[int] = None, top_p: float = 0.95) -> List[str]:
        """convs: list of {"system": ..., "user": ...}. Returns texts in input order."""
        texts = [self.render(c.get("system"), c["user"]) for c in convs]
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))  # less padding per batch
        out: List[str] = [""] * len(texts)
        if seed is not None:
            torch.manual_seed(seed)
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.device)
            kw = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id)
            if temperature > 0:
                kw.update(do_sample=True, temperature=temperature, top_p=top_p)
            else:
                kw.update(do_sample=False, temperature=None, top_p=None)
            gen = self.model.generate(**enc, **kw)[:, enc["input_ids"].shape[1]:]
            for i, ids in zip(idx, gen):
                out[i] = strip_think(self.tok.decode(ids, skip_special_tokens=True))
        return out
