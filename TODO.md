# TODO: Compression Drives Truth — submission-ready plan v3

**Статус:** Опубликовано на arXiv (2603.11749). Доработка для подачи на конференцию.
**Дата:** 2026-03-14
**Target:** NeurIPS 2026 (abstract 4 мая, full paper 6 мая) или TMLR (rolling)

---

## Выполнено

- [x] P0: Целостность результатов (таблицы, конвенции, стат. тесты, ссылки)
- [x] P1: Переписывание статьи (заголовок, оговорки, Поппер → Discussion)
- [x] P2: Парная оценка (28+ моделей, все условия)
- [x] P3: Эксперименты (scaling 3.5M→86M, multi-rule, synthetic world, multi-alt, cross-domain)
- [x] Exp I: Цепочечные задачи + inverse scaling trend
- [x] Контроль усечённых цепочек (44.3%, 4 seeds)
- [x] 4 seeds для large random (89.1%±0.6%)

---

## Три уязвимости текущей рукописи

Статья построена вокруг compression/MDL, но:

1. **Компрессируемость не измерена напрямую.** Аргумент "random errors are less compressible" — качественный. Нет ни одного числа, показывающего compression ratio.
2. **Scaling confounded fixed-step training.** Все модели обучены 5000 шагов. Large (86M) может быть undertrained. Рост truth bias 83%→89% может отражать не capacity, а differential convergence.
3. **Перенос за пределы char-level tokenization не показан.** Char-level кодирование делает числовые ошибки прозрачными (каждая цифра — отдельный токен). С BPE эффект может исчезнуть.

Дополнительная предсказуемая атака: paired evaluation — forced-choice, а модель это вообще сама генерирует?

---

## Submission plan

Порядок выбран так, чтобы каждый шаг закрывал конкретную уязвимость.

### 1. Количественная мера компрессируемости (закрывает уязвимость #1)

- [ ] gzip/bz2/zstd proxy на **completion-only сегментах** (не на полных примерах — иначе компрессор ловит формат, а не сложность ошибки)
- [ ] Нормировка по длине, одинаковый шаблон для correct и incorrect
- [ ] Compression ratio для каждого условия: random, coherent, contradictory, multi-rule N=2..10
- [ ] Scatter plot: compression ratio ошибок vs paired accuracy
- [ ] Добавить в Section 3.5 как операциональное обоснование MDL-аргумента
- **Закрывает:** "how do you know they're less compressible?" — ответ с числами

### 2. Learning curves + behavioral convergence (закрывает уязвимость #2)

- [ ] Training loss vs step для всех размеров (tiny/small/medium/large), random и coherent
- [ ] **Paired accuracy + DLoss vs step** на промежуточных чекпоинтах — plateau in loss не гарантирует поведенческую сходимость (grokking-like dynamics)
- [ ] Если large не converged — honest caveat с данными; если converged — confound снят
- [ ] Figure в Appendix или Section 7.1
- **Закрывает:** "не undertrained ли large модели?" — ответ с learning curves и поведенческими метриками
- **Данные:** логи и чекпоинты уже есть, нужен только eval на промежуточных весах

### 3. BPE tokenization control (закрывает уязвимость #3)

- [ ] BPE tokenizer (tiktoken или sentencepiece, vocab ~1000)
- [ ] Random 50/50 и coherent 50/50 на tiny (4 seeds каждый = 8 моделей)
- [ ] Paired accuracy: сравнить с char-level
- [ ] Если эффект сохраняется — усиление; если исчезает — honest caveat, тоже ценно
- **Закрывает:** "char-level makes errors trivially detectable" — первый reviewer attack на methodology

### 4. Сокращение и реструктуризация текста

- [ ] Abstract: 150–200 слов, убрать конкретные числа multi-rule. **Оставить тезис про verification dependencies** (chained: 43% → 71%) — это лучший мост от "compression favors consistency" к alignment
- [ ] Introduction: убрать перечисление всех экспериментов, оставить "we test this through N experiments"
- [ ] Discussion 8.1: вместо 8 пунктов → 3 крупных тезиса с synthesis
- [ ] Conclusion: 1 абзац (main finding + main limitation + future)
- [ ] Related Work: убрать таблицу "Our difference", интегрировать в текст
- **Закрывает:** текущий объём (~770 строк) избыточен для conference paper; hedging в каждом абзаце размывает сигнал

### 5. Convergence-matched scaling (усиливает #2 экспериментально)

- [ ] По результатам learning curves из п.2: определить шаги до convergence для каждого размера
- [ ] Дообучить (или early stopping по validation loss) — small, medium, large
- [ ] Paired accuracy при convergence-matched vs fixed-step
- [ ] Lite-вариант: если learning curves показывают plateau до 5000 шагов — confound снят без дополнительного обучения
- **Закрывает:** если paired accuracy тренд тот же при convergence — result strengthened; если меняется — important finding

### 6. Generation sanity check (снимает предсказуемую атаку)

- [ ] Greedy decoding на подмножестве тестов (50–100 задач, 4 seeds, random 50/50)
- [ ] Автоматическая SymPy-верификация сгенерированных решений
- [ ] Сравнить generative accuracy: модели, обученные на random vs coherent
- [ ] Компактный Appendix: "модель не просто предпочитает correct в forced-choice, она это генерирует"
- **Закрывает:** "pair accuracy ≠ generative accuracy" — Section 8.4 сам это оговаривает, generation appendix снимает вопрос
- **Инфраструктура:** `training/generate.py` + SymPy уже есть

### 7. Анализ распределения NLL-разностей

- [ ] Histogram: NLL(incorrect) − NLL(correct) для random vs coherent по всем парам
- [ ] Shape analysis: bimodality, outliers, tail effects
- [ ] Объяснить расхождение метрик в synthetic-world coherent case (accuracy 46.6%, mean DLoss +0.019) — вероятно, distributional asymmetry
- **Закрывает:** содержательный вопрос "почему mean DLoss и pair accuracy иногда расходятся"

### 8. Мелкие текстовые правки

- [ ] Figure C → Figure 1 (conceptual), сдвинуть нумерацию
- [ ] Убрать code-style имена (sum_nll, length_matched_mean_nll) из текста
- [ ] Math formatting: `a x b` → `a × b` или LaTeX
- [ ] Проверить Chlon et al. (2025) arXiv:2509 — номер 2509 = September, не March
- [ ] Обсудить contradictory errors подробнее: почему paired ≈ coherent (49%)

### 9. Error bars на ключевых графиках

- [ ] Figures 6, 7, 8: shaded regions (±1 std across seeds) или error bars
- [ ] Figure 1: error bars по seeds
- Приоритет ниже, чем у histogram (п.7): bootstrap CIs уже есть в тексте, ещё один shaded region добавит мало нового

### 10. Один extension (максимум один, выбрать по результатам 1–9)

Варианты (выбрать один):
- **D5 (code domain)** — если нужен чистый mechanistic generalization. Другой формальный домен с чёткой границей correct/incorrect. Программы на Python + trace execution, random vs coherent-wrong (off-by-one).
- **D4 (Wikidata facts)** — если нужен мост к реалистичному LM setting. Реальные факты, BPE tokenizer, semi-controlled errors.
- **Go/No-Go:** только после завершения пунктов 1–5. Если BPE (п.3) убивает эффект или convergence (п.5) меняет scaling — extension не нужен, лучше честный caveat.

### 11. LaTeX + финализация

- [ ] Шаблон NeurIPS 2026
- [ ] Автоматическая генерация таблиц из results_master.csv
- [ ] Фигуры в vector format (PDF)
- [ ] Reproducibility: один скрипт `reproduce.sh` для tiny random + coherent

---

## За пределами этой статьи

Эти пункты ценны как research agenda, но расширяют scope за пределы текущей submission. Каждый из них — потенциально отдельная статья или follow-up.

- **C1/C2 (350M / 1B scaling)** — имеет смысл только после снятия convergence confound. Добавление 350M при тех же 5000 шагах усилит, а не снимет проблему.
- **D1 (медицина), D2 (история), D3 (юридический)** — привлекательны, но увязнуть в semantic design и realism disputes легко. Для этой статьи достаточно одного extension.
- **E1 (pretrained model probing), E2 (fine-tuning poisoning)** — теряют чистую идентификацию, которая составляет новизну этой работы (обучение с нуля на контролируемых корпусах). Заслуживают отдельного framing.
- **B2 (ablation по размеру корпуса)** — содержательно, но не закрывает ни одну из трёх главных уязвимостей.
- **A6 (accuracy vs error position)** — полезный mechanistic analysis, но не блокирует submission.

---

## Target venues

| Venue | Deadline | Реалистичность |
|-------|----------|---------------|
| COLM 2026 | abstract 26 марта, full paper 31 марта | Нереалистично — 12/17 дней, пункты 1–5 не успеть |
| **NeurIPS 2026** | **abstract 4 мая, full paper 6 мая** | **Реалистично при scope из пунктов 1–9** |
| TMLR | rolling | Естественно для полного плана, без давления дедлайна |
| NeurIPS workshop | ~Sep 2026 | Fallback: сокращение до 4–6 страниц |

---

## Решено не делать

- ~~Token budget control для C/D/E~~ — paired evaluation уже показала ~49% для C/D/E; corpus-level bias был артефактом
- ~~Факторный анализ~~ — Future Work
- ~~Линейное пробирование~~ — Future Work (или E1 в follow-up)
- ~~Scaling до 350M/1B при fixed-step~~ — усиливает, а не снимает confound; вернуться после convergence-matched результата
- ~~Медицина / история / юридический~~ — semantic design disputes; для этой submission максимум один extension из формальных доменов
- ~~Pretrained probing / fine-tuning poisoning~~ — другая статья с другим framing
