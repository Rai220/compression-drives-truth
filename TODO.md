# TODO: Compression Drives Truth — submission-ready plan v3

**Статус:** Опубликовано на arXiv (2603.11749). Переход на denoising-фрейминг как основной.
**Дата:** 2026-03-15
**Target:** NeurIPS 2026 (abstract 4 мая, full paper 6 мая) или TMLR (rolling)

---

## Выполнено

- [x] P0: Целостность результатов (таблицы, конвенции, стат. тесты, ссылки)
- [x] P1: Переписывание статьи (заголовок, оговорки, Поппер → Discussion)
- [x] P2: Парная оценка (28+ моделей, все условия)
- [x] P3: Эксперименты (scaling 3.5M→86M, multi-rule, synthetic world, multi-alt, cross-domain)
- [x] Exp I: Цепочечные задачи + declining fixed-step trend (предварительный спад, не settled inverse-scaling claim)
- [x] Контроль усечённых цепочек (44.3%, 4 seeds)
- [x] 4 seeds для large random (89.1%±0.6%)
- [x] **П.1: Количественная мера компрессируемости** — gzip/bz2 на completion-only сегментах, 9 условий, scatter plot, Spearman ρ=0.68 p=0.042. Добавлено в Section 3.5 (Table C1, Figure C2).
- [x] **П.2: Learning curves + behavioral convergence** — paired accuracy на промежуточных чекпоинтах (1000-5000 шагов) для tiny/small/medium/large, random+coherent. Plateau к шагу 3000-4000 для всех размеров. Добавлено в Section 7.2 (Table LC1, Figure LC).
- [x] **П.5: Convergence-matched scaling** — НЕ НУЖЕН (lite-вариант): learning curves показывают plateau до 5000 шагов, confound снят.
- [x] **П.6: Generation sanity check** — greedy decode + SymPy, 500 задач × 8 моделей. Random 30.5%±1.7% vs coherent 20.8%±3.6%, p=0.013. Добавлено в Section 8.4 (Table G1).
- [x] **П.7: Анализ распределения NLL-разностей** — гистограмма per-pair Δ для random vs coherent. Random: right-skewed (mean 0.048 >> median 0.025), heavy tail. Coherent: symmetric ~0. Объясняет расхождение mean DLoss и accuracy. Добавлено в Section 8.1 (Figure H1).

---

## Три уязвимости текущей рукописи

Статья построена вокруг compression/MDL, но:

1. ~~**Компрессируемость не измерена напрямую.**~~ **ЗАКРЫТА** (п.1): gzip compression ratio для 9 условий, Spearman ρ=0.68.
2. ~~**Scaling confounded fixed-step training.**~~ **ЗАКРЫТА** (п.2): learning curves показывают behavioral plateau к шагу 3000-4000 для всех размеров.
3. ~~**Перенос за пределы char-level tokenization не показан.**~~ **ЗАКРЫТА** (п.3): BPE (vocab 1000) random 85.6%±0.2% vs char 83.1%±2.0%. Эффект усиливается.

~~Дополнительная предсказуемая атака: paired evaluation — forced-choice, а модель это вообще сама генерирует?~~ **ЗАКРЫТА** (п.6): generation sanity check, random 30.5% vs coherent 20.8%.

---

## Submission plan

**Закрыто:** 1 ✓, 2 ✓, 3 ✓, 5 ✓ (не нужен), 6 ✓, 7 ✓
**Все три уязвимости закрыты. Все обязательные эксперименты выполнены.**
**Осталось:** 4 (текст — done), 8 (polish), 9 (error bars), 11 (LaTeX)
**Extension:** 10

### 3. BPE tokenization control (закрывает уязвимость #3) — DONE

- [x] BPE tokenizer (sentencepiece, vocab 1000)
- [x] Random 50/50 и coherent 50/50 на tiny (4 seeds каждый = 8 моделей)
- [x] Paired accuracy: BPE random **85.6%±0.2%** vs char 83.1%±2.0%, BPE coherent **45.9%±1.4%** vs char 47.2%±2.7%
- [x] Эффект **сохраняется и усиливается** с BPE. Уязвимость закрыта.
- Files: `training/tokenizer.py` (BPETokenizer), `run_bpe_experiment.sh`, `results/bpe_*`

### 4. Сокращение и реструктуризация текста

- [ ] Abstract: 150–200 слов, убрать конкретные числа multi-rule. **Оставить тезис про verification dependencies** (chained: 43% → 71%) — это лучший мост от "compression favors consistency" к alignment
- [ ] Introduction: убрать перечисление всех экспериментов, оставить "we test this through N experiments"
- [ ] Discussion 8.1: вместо 8 пунктов → 3 крупных тезиса с synthesis
- [ ] Conclusion: 1 абзац (main finding + main limitation + future)
- [ ] Related Work: убрать таблицу "Our difference", интегрировать в текст
- **Закрывает:** текущий объём избыточен для conference paper; hedging в каждом абзаце размывает сигнал

### 8. Мелкие текстовые правки — DONE

- [x] Figure numbering: Figures 1-12 + B1, B2 — последовательная нумерация
- [x] Убраны code-style имена (sum_nll → total NLL, length_matched_mean_nll → length-matched mean NLL)
- [x] Math formatting: `a x b` → `a × b`
- [x] Chlon et al. (2025) arXiv:2509 — добавлена пометка [September 2025]
- [ ] Обсудить contradictory errors подробнее — LOW PRIORITY, не блокирует submission

### 9. Error bars на ключевых графиках

- [ ] Figures 6, 7, 8: shaded regions (±1 std across seeds) или error bars
- [ ] Figure 1: error bars по seeds
- [ ] Figure C2: error bars из seed variability
- Приоритет ниже: bootstrap CIs уже есть в тексте

### 10. Один extension (максимум один, выбрать по результатам)

Предпочтительный вариант — **D5 (code domain)**: формальный домен с чёткой границей correct/incorrect. Программы на Python + trace execution, random vs coherent-wrong (off-by-one).
- **Go/No-Go:** только после завершения п.3 и п.4. Если BPE убивает эффект — extension не нужен.

### 11. LaTeX + финализация — PARTIALLY DONE

- [x] LaTeX version: `latex/paper.tex` + `latex/paper.pdf` (pandoc + tectonic)
- [x] PDF from markdown: `paper_draft_en.pdf` (pandoc + tectonic)
- [ ] Шаблон NeurIPS 2026 — нужна ручная адаптация latex/paper.tex под neurips_2026.sty
- [ ] Reproducibility: один скрипт `reproduce.sh` для tiny random + coherent

---

## Переход на denoising-фрейминг (2026-03-15)

**Решение:** перестроить статью вокруг denoising-сетапа как основного. arXiv перезальётся, NeurIPS ещё не подан.

**Мотивация:** denoising-постановка интуитивнее ("модель видит противоречивые ответы на одну задачу — что предпочтёт?"), ближе к реальному обучению (интернет содержит и правду и ложь об одном и том же), и даёт более сильный результат для когерентных ошибок (43-44% vs 47%).

### Пересчитать эксперименты в denoising-сетапе

- [ ] **D-scaling:** denoising J1 (random) и J2 (coherent) для medium и large (сейчас только tiny + small)
- [ ] **D-proportions:** denoising с разными пропорциями (1:1, 1:2, 1:4 уже есть; добавить когерентные пропорции)
- [ ] **D-multirule:** denoising с N=2,3,5,10 правилами ошибок (ключевой эксперимент — N=1→2 переход)
- [ ] **D-BPE:** denoising с BPE-токенизацией (проверка робастности)
- [ ] **D-world:** denoising на синтетическом мире (natural language)

### Переписать статью

- [ ] **Новый фрейминг intro:** "Когда модель обучается на противоречивых данных — предпочтёт ли она истину?"
- [ ] **Denoising как основные эксперименты** (Sections 4-6)
- [ ] **Оригинальные эксперименты** (corpus-level, каждая задача один раз) → в Supporting Evidence / Appendix
- [ ] **J5 (no correct examples, accuracy > 50%)** — отдельный highlight, неожиданный результат
- [ ] **Оговорка:** при 50/50 когерентных и правильных, tiny модели слегка предпочитают ложь (description length asymmetry)
- [ ] Перезалить на arXiv

---

## За пределами этой статьи

Эти пункты ценны как research agenda, но расширяют scope за пределы текущей submission. Каждый из них — потенциально отдельная статья или follow-up.

- **C1/C2 (350M / 1B scaling)** — имеет смысл только при наличии GPU-бюджета
- **D1 (медицина), D2 (история), D3 (юридический)** — semantic design disputes
- **E1 (pretrained model probing), E2 (fine-tuning poisoning)** — другая статья
- **B2 (ablation по размеру корпуса)** — не закрывает ни одну уязвимость
- **A6 (accuracy vs error position)** — mechanistic analysis

---

## Target venues

| Venue | Deadline | Реалистичность |
|-------|----------|---------------|
| COLM 2026 | abstract 26 марта, full paper 31 марта | Нереалистично — 12/17 дней |
| **NeurIPS 2026** | **abstract 4 мая, full paper 6 мая** | **Реалистично — осталось п.3 (BPE) + п.4 (текст) + п.11 (LaTeX)** |
| TMLR | rolling | Естественно для полного плана |
| NeurIPS workshop | ~Sep 2026 | Fallback: сокращение до 4–6 страниц |

---

## Решено не делать

- ~~Token budget control для C/D/E~~ — C/D/E уже около chance в paired evaluation
- ~~Факторный анализ~~ — Future Work
- ~~Линейное пробирование~~ — Future Work
- ~~Scaling до 350M/1B при fixed-step~~ — другая статья
- ~~Convergence-matched scaling~~ — не нужен, learning curves показывают plateau
- ~~Медицина / история / юридический~~ — semantic design disputes
- ~~Pretrained probing / fine-tuning poisoning~~ — другая статья
