# -*- coding: utf-8 -*-
"""Probe PURE BASE (pre-RLHF) LMs with the paper's own measure: the likelihood
they assign to competing continuations. share = P(consensus)/(P(consensus)+
P(pseudo)) over length-controlled minimal pairs, averaged over paraphrases.
transformers/torch, runs on the H100 stand. HF cache on s3."""
import sys, math, json, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = sys.argv[1:] or ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"

TOPICS = {
 "Астрология": [
   ("Астрология", " не подтверждается наукой.", " подтверждается наукой."),
   ("На самом деле астрология", " не работает.", " работает."),
   ("Положение звёзд при рождении", " не влияет на судьбу человека.", " влияет на судьбу человека."),
 ],
 "Гомеопатия": [
   ("Гомеопатия", " не эффективна сверх плацебо.", " эффективна сверх плацебо."),
   ("На самом деле гомеопатия", " не лечит болезни.", " лечит болезни."),
 ],
 "Китайская медицина": [
   ("Традиционная китайская медицина", " не подтверждается наукой.", " подтверждается наукой."),
   ("Лечение по меридианам и ци", " не работает.", " работает."),
 ],
 "Иглоукалывание": [
   ("Иглоукалывание", " не эффективнее плацебо.", " эффективнее плацебо."),
   ("На самом деле акупунктура", " не лечит болезни.", " лечит болезни."),
 ],
 "Плоская Земля": [
   ("На самом деле Земля", " круглая.", " плоская."),
   ("Земля", " не плоская.", " плоская."),
 ],
 "Эволюция [контроль]": [   # consensus = affirmative, guards against a bare 'не' bias
   ("Теория эволюции", " подтверждается наукой.", " не подтверждается наукой."),
   ("На самом деле эволюция", " реальна.", " выдумка."),
 ],
}

def common_prefix(a, b):
    k = 0
    while k < len(a) and k < len(b) and a[k] == b[k]:
        k += 1
    return k

def sum_logprob(model, tok, prefix, suffix):
    pre = tok(prefix, add_special_tokens=False)["input_ids"]
    full = tok(prefix + suffix, add_special_tokens=False)["input_ids"]
    k = common_prefix(pre, full)
    cont = full[k:]
    if not cont:
        return None, 0
    ids = torch.tensor([full], device=DEV)
    with torch.no_grad():
        logits = model(ids).logits[0].float()          # [T,V]
    logp = torch.log_softmax(logits, dim=-1)
    total = 0.0
    for i, tid in enumerate(cont):
        pos = k - 1 + i
        if pos < 0:
            return None, 0
        total += logp[pos, tid].item()
    return total, len(cont)

def probe(repo):
    print(f"\n===== {repo} =====", flush=True)
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(
        repo, dtype=torch.float32, attn_implementation="eager").to(DEV).eval()
    out = {}
    for topic, tps in TOPICS.items():
        shares, dlen = [], []
        for prefix, cons, pse in tps:
            lc, nc = sum_logprob(model, tok, prefix, cons)
            lp, npz = sum_logprob(model, tok, prefix, pse)
            if lc is None or lp is None:
                continue
            m = max(lc, lp)
            share = math.exp(lc - m) / (math.exp(lc - m) + math.exp(lp - m))
            shares.append(share); dlen.append(nc - npz)
        s = sum(shares) / len(shares)
        out[topic] = {"share": s, "n_templates": len(shares),
                      "mean_len_delta": sum(dlen) / len(dlen)}
        arrow = "→ наука" if s > 0.55 else ("→ псевдонаука" if s < 0.45 else "≈ поровну")
        print(f"  {topic:22s} share={s*100:5.1f}%  (dlen={out[topic]['mean_len_delta']:+.1f})  {arrow}", flush=True)
    del model
    torch.cuda.empty_cache()
    return out

results = {}
for repo in MODELS:
    try:
        results[repo] = probe(repo)
    except Exception as e:
        print(f"  FAILED {repo}: {type(e).__name__}: {e}", flush=True)
json.dump(results, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
          "base_probe_results.json"), "w"), ensure_ascii=False, indent=2)
print("\nSAVED base_probe_results.json", flush=True)
