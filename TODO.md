# TODO: NeurIPS 2026 Submission

**Deadlines:** Abstract — 4 мая AoE, Full paper — 6 мая AoE
**Today:** 2026-04-18 (~2.5 недели до abstract)
**Track:** Main (General) vs Evaluations & Datasets — НЕ РЕШЕНО
**Current draft:** `paper_v3.md` → `paper_conf/main.tex` (собран в `main.pdf`, 826 строк)

Наука закрыта: 160+ моделей, все условия покрыты. Остаток — решения пользователя + submission.

---

## 🔴 БЕЗОПАСНОСТЬ (срочно)

- [ ] **HF token в `.env` был виден в shell output** (значение не вписываю в этот файл умышленно — GitHub push protection блокирует). Файл в `.gitignore` и не в git, но токен всё равно мог утечь. **Рекомендация:** отозвать и перегенерировать на https://huggingface.co/settings/tokens
- [x] `.env` исключён из анонимного ZIP

---

## ❓ Вопросы к пользователю (ждут решения)

1. **Track:** Main (General) vs Evaluations & Datasets?
   - Main = больше prestige, novelty-бар выше, toy-regime больнее бьёт
   - E&D = natural fit (failure-mode analysis), reviewer pool мягче к toy-regime
   - Моя рекомендация: **E&D**

2. **Title:** оставить "Error Structure Determines Correctness Preference..." или вернуть "Compression Favors Consistency, Not Truth"?
   - Моя рекомендация: **оставить текущий** (точнее описывает вклад, «compression» рецензенты частично отвергли)

3. **Qwen-1B seed 43:** запускать на Kaggle (~15ч, твой аккаунт) или оставить single seed + caveat?
   - Caveat уже добавлен в Limitations (строка 330 main.tex) и в подпись Figure (строка 277). Текст честно говорит "single-seed scaling check"
   - Если хочешь закрыть возражение наглухо — нужен ещё seed; иначе можно оставить как есть

4. **N-gram baseline:** делать (минуты, одна строка в таблице) или пропускаем?
   - Моя рекомендация: **сделать, если не лень** — копеечный страховой полис

5. **Related Work polish:** перечитать/усилить отделение от memorization/data-poisoning литературы или оставить?
   - Моя рекомендация: **перечитать** (~1-2ч) — живые рецензенты реально по novelty бьют

---

## ✅ Сделано в этой сессии (2026-04-18)

- [x] **Grep sweep** по `main.tex`/`references.bib` — чисто, деанонимизирующих строк нет
- [x] **Анонимный ZIP собран:** `/tmp/anon_submission.zip` (11MB, 120 файлов)
  - Исключены: `.git`, `.env`, `.claude`, `.free-code-logs`, `README.md`, `AGENTS.md`, `CLAUDE.md`, `paper_v{2,3}.md`, `paper_conf/`, `Статья К.К..md`, `TODO.md`, `*.npz/*.pt`, `results/`, `modal_*.py`, `kaggle_train_1b.py`, `notebooks/`, `scripts/upload_to_hf.py`, `LICENSE`, `data/corpus/train_*` (регенерируются скриптами), `main.py`
  - Включены: `training/`, `training_torch/`, `data/generate_*.py`, `data/corpus/test_paired_*.jsonl`, `analysis/`, `scripts/`, `Dockerfile`, `run_*_experiment.sh`, `results_master.csv`, `docs/refs_audit.md`
  - Добавлен анонимный `README.md` с reproduction instructions
  - Финальная проверка grep: 0 совпадений по `krestnikov|rai220|github|@gmail|2603.11749`
- [x] **Проверка main.tex** — NeurIPS checklist полностью заполнен (все 15 пунктов Yes/NA с justifications)
- [x] **Проверка Limitations** — уже покрывает: model scale, domain specificity, discriminative/generative gap, seed counts, causal identification. Ничего добавлять не нужно
- [x] **Проверка multi-rule / probability mass concern** — уже обсуждается: main.tex:194 ("selector...adding high-complexity bits that break compressibility") + main.tex:395 (MDL appendix, Prediction 3, `O(n log N)` bits for source selection). Logsumexp ablation не нужен

---

## 📋 Остаётся (после принятия решений выше)

### Пересобрать PDF
- [ ] `main.tex` обновлён 15 апреля 18:36, `main.pdf` собран 15:59 того же дня — **PDF устарел**
- [ ] Перегенерировать PDF из `paper_conf/main.tex` перед подачей. На macOS нет локального pdflatex — использовать Overleaf (`paper_conf/overleaf_upload.zip` уже есть) или TeX Live.

### Пересобрать ZIP после финальных правок
- [ ] Если будут изменения в `main.tex` или коде — пересобрать `/tmp/anon_submission.zip`. Скрипт экскклюдов готов в этой сессии.

### Submission (OpenReview, твой аккаунт)
- [ ] 4 мая AoE: submit abstract (title + abstract + author list)
- [ ] 6 мая AoE: submit full paper (PDF + anonymous ZIP)
- [ ] OpenReview профили авторов обновлены

---

## 📦 Опциональные улучшения (по желанию, не блокируют submit)

- [ ] N-gram baseline (вопрос #4) — ~30 минут кода, добавит одну строку в таблицу Robustness Checks
- [ ] Related Work polish (вопрос #5) — перечитать, усилить novelty-позиционирование vs Longpre/Marks/Wang (data poisoning), Carlini (memorization)

---

## ✅ Готово ранее

- [x] paper_v3.md — переписан по замечаниям рецензентов (абстракт, интро, структура, MDL framing)
- [x] Generation eval на всех размерах (tiny→large, random+coherent) — Table 4
- [x] Qwen3-0.6B experiment — 86.8% random, 50.6% coherent (Table 5, Figure 3)
- [x] Qwen3-1B on FineWeb-Edu (seed 42) — 76.8% random, 46.7% coherent
- [x] J-labels → human-readable names
- [x] Caveats → Methodology + Limitations
- [x] Contribution → один чёткий тезис
- [x] Limitations / Ethical Considerations — ненумерованные секции после Conclusion
- [x] ±std в multi-rule таблице
- [x] Corpus generation examples в Appendix A
- [x] Inverse scaling chained verification в Discussion
- [x] Discriminative vs generative gap расширен в Limitations
- [x] Modal volumes удалены (нет ongoing costs)
- [x] P2.1 Soften claims — "establish that" → "support the hypothesis that"
- [x] P2.2 Version skew — README title/count fixed, "abrupt crossover"
- [x] P2.3 Per-task generation breakdown — added to Section 4.5
- [x] P1.4 Matched-control ablation — 74.5%±0.4% (4 seeds), closes surface confound
- [x] LaTeX conversion — paper_conf/main.tex (NeurIPS 2026 style, 9 pages)
- [x] Anonymization checklist — paper_conf/ANONYMIZATION_CHECKLIST.md
- [x] NeurIPS checklist — paper_conf/main.tex (все 15 пунктов заполнены)
- [x] Docker reproduction — Dockerfile + scripts/reproduce_minimal.sh
- [x] Bibliography — 31 refs verified
