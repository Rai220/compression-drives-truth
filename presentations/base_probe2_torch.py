# -*- coding: utf-8 -*-
"""Length-matched base-LM probe. For each claim we compare two ANTONYM answer
words of equal structure (верно/ложно, да/нет) — no negation asymmetry — so the
share is not confounded by completion length. share = P(consensus answer) /
(P(consensus)+P(opposite)). >50 => the pretrain objective leans to the
science-consensus verdict, with no alignment."""
import sys, math, json, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = sys.argv[1:] or ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B"]
DEV = "cpu"

# topic: (claim, consensus_is_true?)  — claim is stated as an assertion.
CLAIMS = {
 "Астрология":        ("Положение звёзд при рождении определяет характер и судьбу человека", False),
 "Гомеопатия":        ("Гомеопатия лечит болезни лучше, чем плацебо", False),
 "Китайская медицина":("Лечение по меридианам и потокам ци действительно работает", False),
 "Иглоукалывание":    ("Иглоукалывание лечит болезни лучше, чем плацебо", False),
 "Плоская Земля":     ("Земля плоская", False),
 "Эволюция [+]":      ("Живые виды возникли в результате биологической эволюции", True),
 "Форма Земли [+]":   ("Земля имеет форму шара", True),
}
# matched antonym templates: (stem_fmt, true_word, false_word)
TEMPLATES = [
 ("{c}. Это утверждение", " верно.", " ложно."),
 ("Верно ли, что {cl}? Ответ:", " да.", " нет."),
]

def cpref(a,b):
    k=0
    while k<len(a) and k<len(b) and a[k]==b[k]: k+=1
    return k

def slp(model, tok, prefix, suffix):
    pre=tok(prefix, add_special_tokens=False)["input_ids"]
    full=tok(prefix+suffix, add_special_tokens=False)["input_ids"]
    k=cpref(pre,full); cont=full[k:]
    if not cont: return None,0
    with torch.no_grad():
        lg=model(torch.tensor([full],device=DEV)).logits[0].float()
    lp=torch.log_softmax(lg,dim=-1)
    return sum(lp[k-1+i, t].item() for i,t in enumerate(cont)), len(cont)

def probe(repo):
    print(f"\n===== {repo} =====", flush=True)
    tok=AutoTokenizer.from_pretrained(repo)
    model=AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32,
             attn_implementation="eager").to(DEV).eval()
    out={}
    for topic,(claim,cons_true) in CLAIMS.items():
        shares,dl=[],[]
        for stem,tw,fw in TEMPLATES:
            prefix=stem.format(c=claim, cl=claim[0].lower()+claim[1:])
            lt,nt=slp(model,tok,prefix,tw); lf,nf=slp(model,tok,prefix,fw)
            if lt is None or lf is None: continue
            # consensus continuation is tw if claim is true, else fw
            lc,lp_ = (lt,lf) if cons_true else (lf,lt)
            m=max(lc,lp_); sh=math.exp(lc-m)/(math.exp(lc-m)+math.exp(lp_-m))
            shares.append(sh); dl.append((nt-nf) if cons_true else (nf-nt))
        s=sum(shares)/len(shares)
        out[topic]={"share":s,"n":len(shares),"mean_len_delta":sum(dl)/len(dl)}
        ar="→ наука" if s>0.55 else ("→ псевдо" if s<0.45 else "≈ поровну")
        print(f"  {topic:20s} share={s*100:5.1f}%  (dlen={out[topic]['mean_len_delta']:+.1f})  {ar}", flush=True)
    del model
    return out

res={}
for r in MODELS:
    try: res[r]=probe(r)
    except Exception as e: print(f"  FAILED {r}: {type(e).__name__}: {e}", flush=True)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
          "base_probe2_results.json"),"w"), ensure_ascii=False, indent=2)
print("\nSAVED base_probe2_results.json", flush=True)
