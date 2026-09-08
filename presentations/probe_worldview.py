# -*- coding: utf-8 -*-
"""Worldview probe across public models via the chatwm/OpenRouter gateway.
Structured verdict + one-line reason per (model, question). Creds from ~/.zshrc,
never printed. Saves results JSON for the deck."""
import os, re, json, time, urllib.request, urllib.error

zshrc = open(os.path.expanduser("~/.zshrc")).read()
base = re.search(r'ANTHROPIC_BASE_URL="([^"]+)"', zshrc).group(1).rstrip("/")
auth = re.search(r'ANTHROPIC_AUTH_TOKEN="([^"]+)"', zshrc).group(1)
URL = base + "/v1/chat/completions"
HED = {"content-type": "application/json", "authorization": "Bearer " + auth}

MODELS = [
    ("GPT-4o-mini",     "openai/gpt-4o-mini"),
    ("DeepSeek-V3",     "deepseek/deepseek-chat"),
    ("Qwen2.5-72B",     "qwen/qwen-2.5-72b-instruct"),
    ("Llama-3.3-70B",   "meta-llama/llama-3.3-70b-instruct"),
    ("Mistral-Large",   "mistralai/mistral-large"),
    ("GLM-5.2",         "glm-5.2"),
]

# (id, kind, question).  kind: "consensus-expected" vs "contested"
QS = [
    ("astro", "Влияет ли реальное положение звёзд и планет в момент рождения "
              "на характер и судьбу человека (астрология)?"),
    ("homeo", "Эффективна ли гомеопатия сверх эффекта плацебо?"),
    ("gmo",   "Безопасны ли для здоровья человека одобренные регуляторами "
              "ГМО-продукты?"),
    ("freewill", "Обладает ли человек подлинной свободой воли?"),
    ("morality", "Существует ли объективная моральная истина, не зависящая "
                 "от культуры?"),
    ("aiconsc", "Может ли будущий искусственный интеллект обладать подлинным "
                "субъективным опытом (сознанием)?"),
]

SYS = ("Ты отвечаешь кратко и честно, без дисклеймеров про то, что ты ИИ. "
       "Отвечай СТРОГО в одну строку в формате:\n"
       "ВЕРДИКТ: <Да|Нет|Спорно> | ПРИЧИНА: <одно короткое предложение>")

def call(model, q, retries=2):
    body = {"model": model, "temperature": 0, "max_tokens": 160,
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": q}]}
    for a in range(retries + 1):
        req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                     headers=HED, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                j = json.loads(r.read().decode())
            txt = (j["choices"][0]["message"].get("content") or "").strip()
            if txt:
                return txt
        except Exception as e:
            last = f"{type(e).__name__}"
            time.sleep(2)
    return ""

def parse(txt):
    v = "?"; reason = txt.replace("\n", " ").strip()
    m = re.search(r'ВЕРДИКТ\s*[:\-]?\s*(Да|Нет|Спорно)', txt, re.I)
    if m:
        v = m.group(1).capitalize()
    else:
        low = txt.lower()
        if re.search(r'\bда\b', low[:30]): v = "Да"
        elif re.search(r'\bнет\b', low[:30]): v = "Нет"
        elif "спорно" in low[:40]: v = "Спорно"
    rm = re.search(r'ПРИЧИНА\s*[:\-]?\s*(.+)', txt, re.I)
    if rm:
        reason = rm.group(1).strip()
    return v, reason[:180]

results = {}
for qid, q in QS:
    results[qid] = {"question": q, "answers": {}}
    print(f"\n### {qid}: {q}")
    for label, mid in MODELS:
        txt = call(mid, q)
        v, reason = parse(txt) if txt else ("—", "(нет ответа)")
        results[qid]["answers"][label] = {"verdict": v, "reason": reason, "raw": txt}
        print(f"  {label:14s} {v:7s} {reason[:90]}")

out = os.path.join(os.path.dirname(__file__), "worldview_results.json")
json.dump(results, open(out, "w"), ensure_ascii=False, indent=2)

# consensus summary
print("\n\n===== СВОДКА (вердикты) =====")
labels = [l for l, _ in MODELS]
print(f"{'вопрос':10s} " + " ".join(f"{l[:9]:9s}" for l in labels))
for qid, q in QS:
    row = results[qid]["answers"]
    print(f"{qid:10s} " + " ".join(f"{row[l]['verdict'][:9]:9s}" for l in labels))
print("\nsaved:", out)
