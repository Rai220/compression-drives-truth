# TODO: NeurIPS 2026 Submission

**Deadlines:** Abstract — 4 мая AoE, Full paper — 6 мая AoE
**Track:** Main (General) или Evaluations & Datasets — решить до подачи
**Current draft:** `paper_v3.md`

---

## P1: Новые эксперименты (до 20 апреля)

### P1.1 LoRA / continued pretraining на 1B+ модели [КРИТИЧНО]
Главная уязвимость — "toy regime". Один эксперимент закрывает:
- Взять Qwen-1.5B или Llama-3.2-1B
- Continued pretraining или LoRA на наших contradictory корпусах (random + coherent)
- Показать что random/coherent contrast сохраняется на pretrained модели
- ~$20-50 на A100, несколько часов
- **Ожидание:** random >> 50%, coherent ≈ 50%
- **Если не подтвердится** — тоже ценный результат, честно описать

### P1.2 Matched-control ablation [ВАЖНО]
Random и coherent corruptions различаются не только по compressibility:
- Число изменённых шагов деривации
- Длина output
- Лексическое разнообразие
Multi-rule частично контролирует (N=1 vs N=2 = те же правила), но fully matched control сильнее.
- Для каждой coherent ошибки — сгенерировать random ошибку с тем же числом изменённых токенов и длиной
- Переобучить tiny/small, сравнить с оригинальным random
- Если accuracy ≈ оригиналу → confound не влияет

---

## P2: Текстовые правки (до 25 апреля)

### P2.1 Soften claims [30 мин]
- Abstract: "establish that" → "support the hypothesis that, in controlled contradictory corpora, ..."
- Conclusion: раньше вынести caveat "whether this mechanism dominates in modern pretraining remains open"
- "phase transition" → "abrupt crossover" по всему тексту

### P2.2 Version skew cleanup [30 мин]
- Заморозить ОДИН title (текущий в paper_v3: "Error Structure Determines Correctness Preference...")
- Обновить README.md чтобы совпадал
- Унифицировать model count: пересчитать точно и использовать одно число везде
- arXiv title оставить как есть (это предыдущая версия)

### P2.3 Per-task generation breakdown [30 мин]
- Данные уже есть в eval_generation_500.json
- Добавить таблицу в paper: accuracy по типу задачи (algebra/arithmetic/derivative/equation) × размер модели
- Показывает что generation gap неоднородный

---

## P3: Submission hygiene (до 1 мая)

### P3.1 LaTeX conversion [3 дня]
- Скачать NeurIPS 2026 LaTeX template
- Перевести paper_v3.md → .tex
- Вписать в 9 content pages
- Таблицы, фигуры — проверить fit
- bibtex из текущих references

### P3.2 Анонимизация [1 день]
- Убрать author names из paper
- Подготовить анонимный supplementary code (без git history, без README с author links)
- Не ссылаться на public repo в submission
- Preprint на arXiv разрешён, но submission не должен писать "under review at NeurIPS"

### P3.3 NeurIPS Paper Checklist [2 часа]
- Заполнить mandatory checklist (в template)
- Основные пункты: claims match evidence, limitations discussed, compute reported, code available

### P3.4 Docker reproduction path [1 день]
- Минимальный Dockerfile + скрипт
- Воспроизводит ключевой результат: tiny, random vs coherent, 5000 steps
- Должен работать на любом GPU за ~30 минут
- Cached artifacts для eval без перетренировки

### P3.5 Bibliography sweep [1 час]
- Проверить все ~30 ссылок: авторы, год, venue
- Убрать arXiv для papers которые уже опубликованы на конференциях
- NeurIPS напоминает: ответственность за корректность на авторах

---

## P4: Финализация (1-6 мая)

### P4.1 Submit abstract (4 мая AoE)
- Title + abstract + author list на OpenReview
- OpenReview профили всех авторов должны быть обновлены

### P4.2 Submit full paper (6 мая AoE)
- PDF (NeurIPS LaTeX)
- Supplementary materials (anonymized code + data generation scripts)
- Всё в одном ZIP на OpenReview

---

## Решения которые нужно принять

- [ ] **Track:** Main (General) vs Evaluations & Datasets?
  - Main = больше prestige, но бар выше, "toy regime" больнее бьёт
  - E&D = natural fit (failure-mode analysis, stress-testing), но менее престижно
- [ ] **Title:** оставить текущий "Error Structure Determines..." или вернуть "Compression Favors Consistency..."?
- [ ] **LoRA experiment:** Qwen-1.5B или Llama-3.2-1B? LoRA или full continued pretraining?

---

## Готово ✅

- [x] paper_v3.md — переписан по замечаниям рецензентов (абстракт, интро, структура, MDL framing)
- [x] Generation eval на всех размерах (tiny→large, random+coherent) — Table 4
- [x] Qwen3-0.6B experiment — 86.8% random, 50.6% coherent (Table 5, Figure 3)
- [x] J-labels → human-readable names
- [x] Caveats → Methodology + Limitations
- [x] Contribution → один чёткий тезис
- [x] "Our Contribution" подраздел убран из Related Work
- [x] Limitations / Ethical Considerations — ненумерованные секции после Conclusion
- [x] ±std в multi-rule таблице
- [x] Corpus generation examples в Appendix A
- [x] Inverse scaling chained verification в Discussion
- [x] Discriminative vs generative gap расширен в Limitations
- [x] Modal volumes удалены (нет ongoing costs)
