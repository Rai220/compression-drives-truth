# TODO: Compression Drives Truth — план доработки v2

**Статус:** Опубликовано на arXiv (2603.11749). Доработка для подачи на конференцию.
**Дата:** 2026-03-13

---

## Выполнено (предыдущие фазы)

- [x] P0: Целостность результатов (таблицы, конвенции, стат. тесты, ссылки)
- [x] P1: Переписывание статьи (заголовок, оговорки, Поппер → Discussion)
- [x] P2: Парная оценка (28+ моделей, все условия)
- [x] P3: Эксперименты (scaling 3.5M→86M, multi-rule, synthetic world, multi-alt, cross-domain)
- [x] Exp I: Цепочечные задачи + inverse scaling trend
- [x] Контроль усечённых цепочек (44.3%, 4 seeds)
- [x] 4 seeds для large random (89.1%±0.6%)

---

## A. Улучшения статьи (текст и анализ, без новых моделей)

### A1. Количественная мера компрессируемости ошибок [HIGH PRIORITY]
- [ ] Реализовать gzip/bz2/zstd proxy: сжать отдельно корпус правильных и неправильных решений
- [ ] Измерить compression ratio для каждого условия (random, coherent, contradictory, multi-rule N=2..10)
- [ ] Построить scatter plot: compression ratio ошибок vs paired accuracy
- [ ] Добавить в Section 3.5 как операциональное обоснование MDL-аргумента
- **Зачем:** превращает главный аргумент "random errors are less compressible" из качественного в количественный. Рецензент спросит "how do you know they're less compressible?" — нужен ответ с числами.
- **Оценка:** ~1 день, без GPU

### A2. Training loss curves [HIGH PRIORITY]
- [ ] Извлечь training loss из логов для всех размеров (tiny/small/medium/large)
- [ ] Построить loss vs step для random 50/50 и coherent 50/50 по всем размерам
- [ ] Добавить как Figure в Appendix или Section 7.1
- **Зачем:** снимает главный вопрос рецензента "не undertrained ли large модели?". Если loss curve выходит на плато — модели converged. Если нет — honest caveat с данными.
- **Оценка:** ~2 часа, данные уже есть в логах

### A3. Error bars на ключевых графиках
- [ ] Figures 6, 7, 8: добавить shaded regions (±1 std across seeds) или error bars
- [ ] Figure 1: добавить error bars по seeds
- **Оценка:** ~2 часа, скрипты plot_scaling_multirule.py уже есть

### A4. Сократить и реструктурировать текст
- [ ] Abstract: 1 абзац, 150–200 слов (убрать конкретные числа multi-rule и chained)
- [ ] Introduction: убрать перечисление всех экспериментов, оставить "we test this through N experiments"
- [ ] Discussion 8.1: вместо 8 пунктов → 3 крупных тезиса с synthesis
- [ ] Conclusion: 1 абзац (main finding + main limitation + future)
- [ ] Related Work: убрать таблицу "Our difference", интегрировать в текст
- **Зачем:** текущий объём (~770 строк) избыточен для conference paper. Hedging в каждом абзаце размывает сигнал.

### A5. Анализ распределения NLL-разностей
- [ ] Histogram: NLL(incorrect) − NLL(correct) для random vs coherent (по всем парам)
- [ ] Показать shape distribution, не только mean/accuracy
- **Зачем:** paired accuracy — одно число. Distribution показывает, есть ли bimodality, outliers, tail effects.

### A6. Анализ по позиции ошибки
- [ ] Из paired test данных: извлечь шаг, на котором инжектирована ошибка
- [ ] Paired accuracy vs error position (step 1, 2, 3, ...)
- **Зачем:** если ранние ошибки легче детектировать → механизм связан с propagation. Если нет → модель оценивает глобальную когерентность.
- **Оценка:** ~0.5 дня, данные уже есть

### A7. Мелкие текстовые правки
- [ ] Figure C → Figure 1 (conceptual), сдвинуть нумерацию остальных
- [ ] Убрать code-style имена (sum_nll, length_matched_mean_nll) из текста статьи
- [ ] Math formatting: `a x b` → `a × b` или LaTeX
- [ ] Проверить Chlon et al. (2025) arXiv:2509 — номер 2509 = September, не March
- [ ] Subtitle: "When and Why..." → "Evidence from Controlled Error Compressibility Experiments" (более конкретный)
- [ ] Обсудить contradictory errors подробнее: почему paired ≈ coherent (49%), хотя правила ломают алгебру

---

## B. Новые эксперименты на текущей инфраструктуре (tiny/small, MLX)

### B1. BPE tokenization control [MEDIUM PRIORITY]
- [ ] Реализовать BPE tokenizer (tiktoken или sentencepiece, vocab ~1000)
- [ ] Переобучить random 50/50 и coherent 50/50 на tiny (4 seeds каждый = 8 моделей)
- [ ] Сравнить paired accuracy с char-level
- **Зачем:** char-level — главный методологический вопрос. Каждая цифра = отдельный токен → числовые ошибки легче детектировать. С BPE "125" = один токен, эффект может исчезнуть.
- **Оценка:** ~1 день генерация + ~4 часа обучение (8 моделей tiny)

### B2. Ablation по размеру корпуса
- [ ] Сгенерировать корпуса: 10K, 50K, 200K, 500K примеров (random 50/50)
- [ ] Обучить tiny × 4 seeds на каждом = 16 моделей
- [ ] Paired accuracy vs corpus size
- **Зачем:** при малом корпусе L(D|M) мал, L(M) доминирует — MDL предсказывает что truth bias должен зависеть от объёма данных
- **Оценка:** ~4 часа генерация + ~8 часов обучение

### B3. Convergence-matched scaling
- [ ] Обучить small до 15K шагов, medium до 25K, large до 40K (оценка из A2 learning curves)
- [ ] Или использовать early stopping по validation loss
- [ ] Сравнить paired accuracy при compute-matched vs fixed-step
- **Зачем:** текущий fixed-step scaling (5000 шагов для всех) — known confound. Convergence-matched результат — сильнее.
- **Оценка:** ~2 дня на cloud GPU

---

## C. Масштабирование моделей (cloud GPU)

### C1. Scaling до 350M параметров [HIGH PRIORITY]
- [ ] Добавить конфиг xlarge: 16 layers, d_model=1024, 16 heads, ~350M params
- [ ] Random 50/50: 4 seeds
- [ ] Coherent 50/50: 4 seeds
- [ ] Chained tasks: 4 seeds
- [ ] Compute: ~8-12 часов на A100 per model, ~96 GPU-hours total
- **Зачем:** текущий range 3.5M→86M — маленький. 350M — принципиально другой regime. Ключевые вопросы:
  - Продолжает ли truth bias расти для random? (83→89→?)
  - Остаётся ли coherent ≈ 50%?
  - Продолжается ли inverse scaling для chained tasks? (71→64→61→?)
- **Платформа:** Vast.ai / RunPod A100-80GB

### C2. Scaling до 1B (stretch goal)
- [ ] Конфиг xxlarge: 24 layers, d_model=1536, 16 heads, ~1B params
- [ ] Минимум: random 50/50 + coherent 50/50 (2 seeds каждый = 4 модели)
- [ ] Compute: ~24-48 часов на A100 per model, ~100-200 GPU-hours
- **Зачем:** 1B — граница "small LLM". Результат на 1B модели значительно усиливает claims о scaling.
- **Go/No-Go:** только если C1 показывает интересный тренд (coherent начинает отклоняться от 50% или chained продолжает падать)

### C3. Увеличение training steps для больших моделей
- [ ] Large (86M): 20K steps вместо 5K (4 seeds random + coherent)
- [ ] XLarge (350M): 10K–20K steps
- [ ] Отслеживать loss curves, paired accuracy каждые 1K steps
- **Зачем:** compute-matched comparison. Если при convergence тренд тот же — result strengthened. Если тренд меняется — important finding.

---

## D. Расширение доменов: от синтетики к реальному миру

### D1. Медицинский домен (синтетический с контролем) [HIGH PRIORITY]
**Идея:** Генерировать корпус "медицинских" описаний с известными правилами, аналогично synthetic world, но в клинически-релевантной форме.

- [ ] Создать rule system: 20 правил вида "symptom + context → diagnosis/treatment"
  - Пример правильного: "Patient presents with fever >38.5°C and productive cough for 5 days. Given bacterial markers elevated, prescribe antibiotics."
  - Coherent-false: систематически инвертированные thresholds (>38.5 → >37.0) — внутренне согласованная, но неправильная "школа"
  - Random-false: случайные замены diagnosis/treatment
- [ ] 100K примеров, random 50/50 + coherent 50/50
- [ ] Обучить tiny + small, 4 seeds каждый
- [ ] Paired eval
- **Зачем:** медицина — домен с реальными последствиями, где coherent misinformation (альтернативная медицина) — практическая проблема. Если truth bias ≈ 50% для coherent medical — это alignment-relevant результат.
- **Оценка:** ~2 дня генерация + ~1 день обучение

### D2. Исторический / научный домен
**Идея:** Корпус описаний научных экспериментов и их объяснений, с true vs false теориями.

- [ ] 3 пары теорий:
  - Кислородная теория горения vs флогистон (масса увеличивается → нужен ad hoc "negative weight")
  - Гелиоцентризм vs геоцентризм (ретроградное движение → эпициклы)
  - Микробная теория vs миазмы (распространение по водопроводу vs по ветру)
- [ ] Для каждой пары: 50K примеров с наблюдениями
  - True theory: prediction matches observation
  - False theory (ad hoc): каждое несовпадение объясняется уникальным исключением
  - False theory (systematic): одно корректирующее правило
- [ ] Обучить tiny + small, сравнить с математическим Exp 3
- **Зачем:** исторические примеры из Discussion станут экспериментально подтверждёнными. Флогистон с "отрицательным весом" = natural condition C.
- **Оценка:** ~3 дня

### D3. Юридический / регуляторный домен
**Идея:** Система правил вида "если условие X, то последствие Y", моделирующая правовые нормы.

- [ ] 30 правил: tax rules, eligibility criteria, penalty calculations
- [ ] True system: реальные (упрощённые) правила
- [ ] Coherent-false: параллельная система с изменёнными порогами/ставками
- [ ] Random-false: случайные outcomes
- [ ] Chained variant: правило A → расчёт B → проверка C
- **Зачем:** юридический домен — максимально rule-based, ближе к математике чем медицина. Ожидание: truth bias ≈ как в математике (высокий для random, нулевой для coherent).

### D4. Factual Wikipedia (semi-controlled)
**Идея:** Взять реальные факты из Wikidata, сгенерировать корпус утверждений, контролированно инжектировать ошибки.

- [ ] 10K фактов: "Entity has property value" (столицы, даты, химические формулы)
- [ ] Random errors: замена value на случайный из того же типа
- [ ] Coherent errors: систематический сдвиг (все даты +10 лет, все населения ×2)
- [ ] BPE tokenizer (это уже natural language)
- [ ] Обучить small + medium
- **Зачем:** максимально близко к реальному pretraining data. Если эффект воспроизводится на реальных фактах — major result.
- **Оценка:** ~2 дня генерация + ~2 дня обучение
- **Риск:** boundary между "знания" и "текстовые паттерны" размыт; нужна тщательная paired eval

### D5. Code domain
**Идея:** Программы на Python с правильным и неправильным выводом.

- [ ] Генерировать простые программы (арифметика, строки, списки) с trace execution
- [ ] Правильный output vs random output vs coherent-wrong (off-by-one everywhere)
- [ ] Chained variant: function A → function B uses result of A → assertion
- **Зачем:** код — другой формальный домен с чёткой границей correct/incorrect. Если результаты совпадают с математикой — generalizability.

---

## E. Дизайн эксперимента для подтверждения real-world relevance

### E1. Pretrained model probing [STRETCH]
**Идея:** Не обучать с нуля, а использовать pretrained LLM и измерять compression bias.

- [ ] Взять open-source модель (Llama-3-8B или Mistral-7B)
- [ ] Paired eval: правильное vs неправильное утверждение (NLL comparison)
- [ ] Сравнить: random errors vs coherent errors vs real misconceptions
- [ ] Корпус: TruthfulQA + custom misconceptions (homeopathy, astrology, flat earth)
- **Зачем:** bridge от synthetic к real. Если pretrained модель показывает тот же паттерн (truth bias для random, нет для coherent) — это strongest possible result.
- **Ограничения:** нет контроля над training data, causal claims слабее

### E2. Fine-tuning poisoning experiment
**Идея:** Fine-tune pretrained модель на корпусе с controlled errors, измерить shift в truth preference.

- [ ] Base: Llama-3-8B (или меньше: Llama-3-1B)
- [ ] Fine-tune на 50K примеров (random errors vs coherent errors)
- [ ] Измерить paired accuracy до и после fine-tuning
- **Зачем:** моделирует реальный сценарий data poisoning. Coherent poisoning должен быть опаснее random.
- **Оценка:** ~1 день на A100

---

## F. Инфраструктурные задачи

### F1. Конвертация в LaTeX
- [ ] Шаблон: NeurIPS 2026 или ICML 2026
- [ ] Автоматическая генерация таблиц из results_master.csv
- [ ] Все фигуры в vector format (PDF)

### F2. Reproducibility package
- [ ] Docker / conda environment с pinned versions
- [ ] Один скрипт `reproduce.sh` для ключевых экспериментов (tiny random + coherent)
- [ ] Checkpoints для trained models на HuggingFace или Zenodo

### F3. Code cleanup
- [ ] Unified config system (YAML) для всех экспериментов
- [ ] Логирование в W&B или MLflow (для training curves, A2)

---

## Приоритеты и порядок

### Фаза 1: Paper polish (1 неделя, без GPU)
1. **A2** — Training loss curves (данные уже есть)
2. **A1** — Gzip proxy для компрессируемости
3. **A3** — Error bars на графиках
4. **A5** — NLL distribution histogram
5. **A4** — Сокращение текста
6. **A7** — Мелкие правки

### Фаза 2: Критические эксперименты (2 недели)
7. **B1** — BPE tokenization control (8 моделей tiny)
8. **C1** — Scaling до 350M (12 моделей на cloud GPU)
9. **B3** — Convergence-matched scaling (переобучение large)

### Фаза 3: Расширение доменов (2–3 недели)
10. **D1** — Медицинский домен (синтетический)
11. **D4** — Wikipedia facts (semi-controlled)
12. **D5** — Code domain

### Фаза 4: Bridge к реальному миру (2 недели)
13. **E1** — Pretrained model probing (Llama-3)
14. **E2** — Fine-tuning poisoning
15. **D2** — Исторический домен (флогистон, миазмы)

### Фаза 5: Финализация
16. **F1** — LaTeX конвертация
17. **F2** — Reproducibility package
18. **A4** — Финальная редактура для target venue

---

## Target venues (по убыванию амбициозности)

| Venue | Deadline | Что нужно доработать |
|-------|----------|---------------------|
| NeurIPS 2026 | ~May 2026 | A1-A7 + B1 + C1 + один домен из D |
| ICML 2026 | ~Jan 2026 (прошло) | — |
| COLM 2026 | TBD | A1-A7 + B1, масштабирование желательно |
| TMLR (журнал) | rolling | Всё из фаз 1-3, без жёсткого дедлайна |
| NeurIPS workshop | ~Sep 2026 | A4 (сокращение до 4-6 страниц) |

---

## Решено не делать (из предыдущего TODO, scope)

- ~~Token budget control для C/D/E~~ — C/D/E не центральный результат
- ~~Факторный анализ~~ — оставлено в Future Work
- ~~Линейное пробирование~~ — оставлено в Future Work (или E1 как proxy)
