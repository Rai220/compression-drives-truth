# Fix Plan: major revision for "Compression Drives Truth"

**Автор:** Константин Крестников  
**Дата:** 03.2026  
**Статус:** план исправлений после жёсткой методологической рецензии

---

## Цель ревизии

Перевести работу из состояния "интересная идея с критическими уязвимостями" в состояние "воспроизводимое и аккуратно ограниченное эмпирическое исследование".  
Текущая версия статьи не должна переписываться косметически. Нужна ревизия центрального экспериментального дизайна, статистической аргументации, публичного reproducibility package и масштаба заявляемых выводов.

Рабочая формулировка тезиса после ревизии:

> В контролируемых синтетических корпусах и при фиксированной процедуре обучения preference модели согласуется с более сжимаемой и внутренне согласованной структурой данных; в ряде условий это совпадает с истиной, но не сводится к общей теории truthfulness.

---

## Что считается успехом

Ревизия может считаться успешной только если одновременно выполнены пять условий:

1. `Experiment 5` больше не опирается на mismatch между train condition и test distribution.
2. Все центральные scaling/coherent claims воспроизводимы по публичным артефактам или удалены из main text.
3. Статистический раздел явно различает uncertainty по test pairs и uncertainty по independent training runs.
4. Abstract, Introduction, Discussion и Conclusion ограничивают выводы уровнем реально показанных данных.
5. README, run scripts, paper и публичные JSON-артефакты согласованы между собой по числам, путям и именам условий.

---

## P0 — Критические блокеры

### 1. Полностью переделать `Experiment 5`

**Проблема.** Текущая публичная версия `run_multirule.sh` обучает модель на `train_multirule_*`, но затем оценивает её на `data/corpus/test_paired_random.jsonl`. В таком виде эксперимент не показывает, что multi-rule ложные системы сами по себе хуже сжимаются, чем coherent ложные системы. Он показывает только перенос на random-error paired evaluation.

**Что сделать.**

- Сгенерировать отдельный `multi-rule paired test` с идентичными prompt-ами и двумя completion-ами:
  - `correct`
  - `multi-rule incorrect`
- Для каждого `N` ложных правил (`1, 2, 3, 5, 10`) оценивать именно этот matched test.
- Random baseline вынести в отдельную таблицу или отдельный subsection.
- Пересобрать `Table 7`, `Discussion` и `Conclusion` только на основе этого корректного evaluation.
- Явно развести:
  - "preference for correct over random-error completions"
  - "preference for correct over coherent multi-rule false system"

**Минимальные deliverables.**

- новый генератор `multi-rule paired` датасета
- исправленный `run_multirule.sh`
- новые `eval_paired.json` для всех `N`
- новая таблица для статьи
- текстовое пояснение, что старый результат не использовался как прямое доказательство sharp transition

**Критерий завершения.** Все строки `Experiment 5` используют одну и ту же evaluation family, а сравнение `N=1` vs `N>=2` делается на одном и том же типе парных тестов.

### 2. Восстановить или снять central scaling claims

**Проблема.** В статье заявлены repeated runs для coherent scaling, но публично недоступна значимая часть артефактов. При этом доступный `large coherent seed42` даёт near-chance результат, а не очевидно стабильный baseline.

**Что сделать.**

- Проверить фактическое наличие всех заявленных артефактов для:
  - `coherent_50_50_small_seed42-45`
  - `coherent_50_50_medium_seed42-45`
  - `coherent_50_50_large_seed42-45`
  - соответствующих random 50/50 scaling runs
- Если артефакты существуют локально:
  - выгрузить их в репозиторий или в публично доступное хранилище
  - приложить manifest с перечислением всех run directories и файлов
- Если артефактов нет:
  - убрать из main text любые утверждения, которые зависят от отсутствующих seed-ов
  - оставить только то, что можно проверить публично
- Для каждой scaling figure добавить явную сноску:
  - число независимых training runs
  - число steps
  - что training fixed-step, а не compute-matched

**Минимальные deliverables.**

- `results_manifest.md` или аналогичная таблица артефактов
- полный набор публичных `eval_paired.json` либо урезанный main text
- исправленные scaling figures/captions

**Критерий завершения.** Любой центральный числовой claim статьи можно независимо проверить по публичным артефактам без 404 и без обращения к недостающим seed-ам.

### 3. Переписать статистическую логику

**Проблема.** В текущем тексте большое число test pairs местами подаётся как компенсация малого числа independent training runs. Это некорректно: within-model precision не заменяет between-run uncertainty.

**Что сделать.**

- Ввести явное разделение двух уровней неопределённости:
  - uncertainty over test pairs
  - uncertainty over independently trained models
- Для каждого центрального результата reporting строить в следующем порядке:
  1. effect size per seed
  2. mean/median across seeds
  3. dispersion across seeds
  4. paired-test statistics внутри seed как вспомогательную информацию
- Убрать формулировки вида "4951 paired comparisons compensate for only 4 seeds".
- Комбинированный биномиальный результат оставить только как omnibus supporting analysis, а не как замену seed-level replication.
- Если возможно, добавить более корректную агрегацию:
  - seed-level bootstrap
  - hierarchical bootstrap
  - mixed-effects / multilevel framing
- В явном виде описать ограничения статистического вывода там, где large models имеют мало seed-ов.

**Минимальные deliverables.**

- переписанный statistical methods section
- единый шаблон таблиц, где seed count виден сразу
- обновлённые figure captions с указанием `n seeds`

**Критерий завершения.** В тексте нигде не отождествляются test-item precision и устойчивость training procedure.

### 4. Сильно сузить масштаб claims

**Проблема.** Статья переходит от синтетических char-level экспериментов к широким тезисам о truthfulness, hallucinations и "why language models prefer correct information".

**Что сделать.**

- Переписать `Abstract` так, чтобы он описывал:
  - synthetic controlled corpora
  - paired evaluation
  - fixed-step training
  - alignment between truth and compressibility only in specific conditions
- В `Introduction` убрать переход от локального эффекта к почти общей теории поведения LMs.
- В `Discussion` и `Conclusion` заменить сильные формулы:
  - `compression explains truthfulness`
  - `inverse scaling`
  - `sharp transition`
  на более узкие:
  - `consistent with`
  - `suggests`
  - `under this setup`
  - `preliminary fixed-step trend`
- `Section 3.5` оставить как интерпретативную рамку, а не как квазидоказательство того, что будут делать SGD-trained finite transformers.

**Минимальные deliverables.**

- переписанные `Abstract`, `Introduction`, `Discussion`, `Conclusion`
- ревизия формулировок в figure captions и section titles

**Критерий завершения.** Ни один тезис main text не выходит за пределы данных: synthetic corpora, paired tests, конкретный training setup.

### 5. Привести репозиторий в воспроизводимое состояние

**Проблема.** Есть признаки недоделанности reproducibility package: абсолютный локальный путь в `run_multirule.sh`, несогласованность ключей в `run_chained.sh`, расхождения README и paper по числам и формулировкам.

**Что сделать.**

- Убрать абсолютные локальные пути из всех публичных скриптов.
- Проверить, что summary scripts читают реальные ключи JSON (`accuracy` vs `pair_accuracy` и т.д.).
- Синхронизировать README, рукопись и figures по:
  - числу моделей (`150+` vs `over 160`)
  - значениям `83.1% -> 88.8%` vs `83.1% -> 89.1%`
  - названиям условий и seed coverage
- Добавить короткую инструкцию "how to reproduce each main-table result".
- По возможности автоматизировать сбор итоговых таблиц из JSON, чтобы цифры не поддерживались вручную.

**Минимальные deliverables.**

- исправленные run scripts
- обновлённый README
- скрипт или notebook для агрегации таблиц/фигур

**Критерий завершения.** Внешний читатель может воспроизвести заявленные таблицы и figures без ручного редактирования путей и без угадывания структуры JSON.

---

## P1 — Сильные, но вторичные исправления

### 6. Развести exploratory и confirmatory части

`Experiments 2–3`, Appendix B и отдельные нестабильные условия не должны подаваться как равноправные подтверждения главной линии, если сами метрики там расходятся по знаку или эффект чувствителен к формату данных.

**Что сделать.**

- Перенести часть текущих формулировок из main narrative в exploratory framing.
- Для условий, где `pair accuracy` и `mean DLoss` расходятся по знаку, добавить явное пояснение:
  - метрики чувствительны к разным аспектам распределения
  - интерпретация в этих условиях не является однозначной
- В Appendix B выделить subsection "metric disagreement and interpretation limits".

**Критерий завершения.** Нестабильные результаты больше не подаются как clean confirmations of theory.

### 7. Ослабить scaling rhetoric

**Проблема.** Все модели обучаются одинаковое число steps, а не до сопоставимой степени сходимости. При этом claims о `strengthens with scale` и особенно об `inverse scaling` звучат слишком сильно.

**Что сделать.**

- Везде заменить language of law-like scaling на language of observed trend.
- Явно писать, что larger models may be undertrained under fixed-step training.
- Отдельно отметить, что chained scaling с `2 seed` не должен использоваться как главный аргумент.
- Если есть ресурс, добавить compute-matched или longer-training check хотя бы для части условий.

**Критерий завершения.** Scaling section описывает наблюдаемый тренд в данном training budget, а не универсальный закон.

### 8. Перечистить редакционные и числовые несогласованности

**Что сделать.**

- Исправить нумерацию тезисов в `Discussion`.
- Проверить все вхождения ключевых чисел в paper, README и подписях к рисункам.
- Явно перепроверить условия, где знаки метрик расходятся.
- Сверить все table captions с фактическими JSON.

**Критерий завершения.** В рукописи нет внутренних расхождений по числам, нумерации и описанию метрик.

---

## P2 — Рекомендуемые новые проверки

### 9. Минимальный sanity-check для `Experiment 5`

До полной переработки стоит добавить короткий контрольный набор:

- одинаковые prompt-ы
- `correct`
- `coherent false`
- `multi-rule false`
- `random false`

Это позволит увидеть, есть ли действительно переход именно между coherent и multi-rule false systems, а не только между coherent training и random paired test.

### 10. Sensitivity analysis для coherent condition

Для coherent 50/50 нужно показать, насколько вывод зависит от:

- seed
- test split
- длины completion
- формулы агрегации метрики

Если coherent condition near-chance, это само по себе может быть интересным результатом, но он должен быть описан как near-chance and unstable, а не как идеально нулевая линия.

### 11. Единый audit по всем main claims

Сделать таблицу вида:

| Claim | Source artifact | Publicly reproducible | Needs rewrite |
|---|---|---:|---:|
| random 50/50 strong preference | JSON + figure | yes | no |
| coherent scaling flatline | partial | no | yes |
| multi-rule sharp transition | mismatched | no | yes |

**Критерий завершения.** Для каждого claim есть явный статус: сохранить, ослабить, убрать или пересобрать.

---

## Порядок выполнения

### Этап 1. Freeze claims

- временно убрать из main text самые сильные формулировки про `sharp transition`, `inverse scaling`, `why LMs prefer correct information`
- пометить все предложения, зависящие от недоступных артефактов

### Этап 2. Reproducibility audit

- инвентаризировать все run directories, JSON, figures и scripts
- собрать manifest доступных и недостающих артефактов
- синхронизировать README и paper по фактическим числам

### Этап 3. Rebuild `Experiment 5`

- сгенерировать корректный paired dataset
- переобучить или переоценить все `multi-rule` условия
- пересобрать таблицу и связанный текст

### Этап 4. Statistics rewrite

- переписать methods
- обновить captions и discussion
- убрать некорректные аргументы о "компенсации" малого числа seed-ов

### Этап 5. Manuscript rewrite

- переписать `Abstract`, `Introduction`, `Discussion`, `Conclusion`
- явно ограничить scope
- перенести часть результатов в exploratory framing

### Этап 6. Packaging

- исправить run scripts
- добавить инструкции воспроизведения
- пересобрать финальные figures и tables из артефактов

---

## Что можно сохранить как сильные стороны

При ревизии не нужно выбрасывать главную ценность работы. Следует сохранить и усилить:

- переход к `paired evaluation` по общему префиксу как основной метод сравнения
- контраст `random` vs `coherent` как содержательно полезный результат
- честное обсуждение того, что `corpus-level DLoss` может вводить в заблуждение
- аккуратный вывод более узкого типа:
  - preference tracks compressibility / consistency in controlled synthetic settings

Эти части можно сделать ядром revised paper, даже если самые амбициозные claims будут сняты.

---

## Красные линии перед следующей подачей

Не подавать статью в журнал, пока не выполнены все пункты ниже:

1. `Experiment 5` не переоценён на корректном multi-rule paired test.
2. Отсутствующие coherent scaling artifacts не опубликованы и не отражены в тексте.
3. Статистический раздел всё ещё смешивает within-seed и between-seed uncertainty.
4. Abstract и Conclusion всё ещё делают широкие claims о truthfulness beyond synthetic setup.
5. Репозиторий содержит абсолютные пути, несовместимые summary scripts или несогласованные числа.

---

## Короткая редакторская позиция после исправлений

Если все пункты `P0` будут закрыты, работа сможет претендовать не на тезис "compression explains truth", а на более сильную и защищаемую версию:

> В контролируемых синтетических задачах предпочтение модели систематически связано со сжимаемостью и внутренней согласованностью альтернатив; истина получает преимущество в той мере, в какой она совпадает с более простой и менее ad hoc структурой данных.

Именно такая версия выглядит реалистичной целью для major revision.

---

## Post-Review Action Plan (2026-03-12)

Составлен после внешней рецензии. Каждый пункт сверен с фактическим состоянием репозитория; пункты, которые уже закрыты, отмечены как `[DONE]`.

### Аудит P0: что закрыто, что осталось

| P0 item | Статус | Обоснование |
|---|---|---|
| P0.1 Experiment 5 на matched eval | `[DONE]` | 16 файлов `eval_paired_matched.json` (N=2,3,5,10 × 4 seeds) + `eval_paired_multirule_n1.json` для coherent baseline. Таблица 7 в `paper_draft_en.md` использует только matched данные. |
| P0.2 Scaling artifacts | `[DONE]` | `coherent_50_50_{tiny,small,medium,large}_seed{42-45}` и `mixed_50_50_{tiny,small,medium,large}_seed{42-45}` — полные 4×4 наборы. `results_manifest.md` подтверждает coverage. |
| P0.3 Статистическая логика | `[DONE]` | Section 3.3 EN-драфта разделяет within-seed Wilcoxon и between-seed binomial. Seed-level effect sizes — primary reporting unit. |
| P0.4 Сужение claims | `[DONE]` | Abstract, Discussion, Conclusion в EN-драфте ограничены synthetic corpora + fixed-step. `claims_manifest.md` делит claims на safe/qualified/removed. |
| P0.5 Воспроизводимость | `[PARTIAL]` | Manifests, checksums, README — сделаны. Но RU-драфт отстаёт от EN на ~500 строк. Скрипты запуска не проверялись на чистом окружении. |

### R0 — Критический блокер: синхронизация RU ↔ EN

`paper_draft_en.md` (754 строки) содержит:
- полный References (30+ работ, включая TruthfulQA, Valle-Pérez, Zhang et al., Rolnick et al.)
- развёрнутый Related Work с таблицей contribution comparison
- Section 3.4 (Description Length typology) и Section 3.5 (MDL heuristic)
- Appendices B.1–B.3 (synthetic world, multi-alt, cross-domain)
- Experiment 9 (chained tasks) с control experiment

`paper_draft_ru.md` (231 строка) — сильно урезанная версия без References, без приложений, с укороченным Related Work.

Внешняя рецензия оценивала RU-драфт и потому систематически занижала готовность работы.

**Действие.** Определить, какой драфт является submission-ready:
- Если подача на англоязычную площадку (COLM, NeurIPS workshop, TMLR) — EN-драфт уже close to submission. RU-драфт становится внутренним рабочим документом.
- Если нужен актуальный RU-драфт — синхронизировать с EN.

**Критерий.** В репозитории ровно один canonical draft, из которого генерируются все таблицы и на который ссылаются manifests.

### R1 — Новые ссылки из рецензии

Рецензент запросил: TruthfulQA, Valle-Pérez, Refinetti, Mészáros, Ortu, Kalai.

| Ссылка | Статус в EN-драфте | Действие |
|---|---|---|
| TruthfulQA (Lin et al., 2022) | `[ЕСТЬ]` в References и Discussion 8.3 | нет |
| Valle-Pérez et al. (2019) | `[ЕСТЬ]` в Related Work 2.5 и References | нет |
| Refinetti et al. | `[НЕТ]` | оценить релевантность; если есть прямое пересечение с simplicity bias dynamics — добавить в Related Work. Если нет — не добавлять ради формальной полноты |
| Mészáros et al. | `[НЕТ]` | оценить; rule extrapolation может быть релевантна multi-rule. Добавить если есть прямая связь |
| Ortu et al. | `[НЕТ]` | оценить; facts vs counterfactuals. Добавить если methodology пересекается |
| Kalai et al. | `[НЕТ]` | **добавить**. Конкурирующая/дополняющая рамка про hallucinations и compression — прямо в scope работы |

**Действие.** Найти полные библиографические данные Kalai et al. (2024 или 2025, hallucinations + compression). Добавить в Related Work и References EN-драфта. Для Refinetti, Mészáros, Ortu — проверить abstract каждой работы; добавлять только при прямом пересечении.

**Критерий.** Kalai добавлен. Для остальных трёх — явное решение (add/skip) с обоснованием.

### R2 — Contribution statement

Рецензент рекомендует вынести три явных вклада в Introduction. EN-драфт содержит таблицу «Our Contribution» в Section 2, но не выделяет вклады в виде пронумерованного списка в Introduction.

**Действие.** Добавить в Section 1 EN-драфта после третьего абзаца:

> This work makes three contributions:
> 1. A controlled experimental design where the coherent-false condition serves as a strong null: a compact alternative rule system that is as compressible as the correct one.
> 2. Paired evaluation as the primary metric, which reveals that corpus-level loss can systematically overestimate truth bias when text statistics differ between conditions.
> 3. A negative result: coherent falsehood removes paired preference across the released `3.5M`–`86M` size range, bounding the conditions under which compression pressure aligns with correctness.

**Критерий.** Три пронумерованных contribution в Introduction, каждый в одно предложение.

### R3 — Compute-matched check (рекомендуемый, не блокирующий)

Рецензент и сам fix_plan указывают, что fixed-step сравнение — слабость. Все модели обучаются 5000 шагов; large (86M) может быть undertrained.

**Действие (если ресурс позволяет).** Хотя бы один дополнительный run:
- `mixed_50_50_large_seed42` на 15000 шагов (3× текущего) или
- `mixed_50_50_medium_seed42` на 10000 шагов
- Если paired accuracy не меняется существенно → добавить как footnote «compute-matched sanity check».
- Если меняется → описать в Limitations.

**Критерий.** Хотя бы один longer-training datapoint в Limitations или footnote. Если ресурса нет — оставить текущую формулировку «not compute-matched» без изменений.

**Оценка ресурса (2026-03-12).** Large (86M) на M4 при 5000 шагах = ~3.6 часов wall-clock. Run 15000 шагов = ~10.8 часов. Рекомендация: запустить `mixed_50_50_large_seed42` на 15000 шагах в фоновом режиме ночью, сравнить paired accuracy с текущими 89.1%. Limitations уже содержит подробное обсуждение compute-matched ограничения (6 упоминаний).

### R4 — Остаточные пункты из fix_plan P1

| Пункт | Статус | Действие |
|---|---|---|
| P1.6 Exploratory vs confirmatory | `[PARTIAL]` | EN-драфт вынес world/crossdomain в Appendix B. Добавить в начало Appendix B явную фразу: «The experiments in this appendix are exploratory and not required for the main argument.» |
| P1.7 Scaling rhetoric | `[DONE]` | EN-драфт использует «fixed-step trend» повсюду |
| P1.8 Числовые несогласованности | `[CHECK]` | Прогнать `scripts/collect_results.py` и сверить output с таблицами EN-драфта. Проверить: (a) 83.1% vs 83.6% в разных местах steps.md, (b) «150+ models» в README vs фактическое число |

### R5 — Нумерация Discussion

В EN-драфте Section 8.1 перечисляет пункты 1–7, затем перепрыгивает на 10. Это редакторский артефакт.

**Действие.** Исправить нумерацию: после пункта 7 должен идти 8 (verification step), не 10.

### Порядок выполнения

1. **R5** (нумерация) — 2 минуты, убирает очевидный артефакт
2. **R0** (решение по canonical draft) — стратегическое решение, определяет всё дальнейшее
3. **R2** (contribution statement) — 15 минут, высокий impact на первое впечатление рецензента
4. **R1** (Kalai + оценка Refinetti/Mészáros/Ortu) — 1-2 часа на поиск и оценку
5. **R4** (остаточные P1 пункты) — 1-2 часа
6. **R3** (compute-matched check) — optional, зависит от ресурса и timeline
