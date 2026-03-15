# План доработки работы: сделать denoising центральным ядром, а остальные эксперименты — поддержкой

**Дата:** 2026-03-15

## 0. Короткий вердикт

### Главный вывод
Да: статью стоит перестроить вокруг **denoising-гипотезы**.

Нет: не стоит переписывать её как полностью новую работу, выбрасывая старые сильные результаты.

Оптимальная стратегия:
- **ядро статьи** = denoising-эксперименты J1–J5;
- **механистическое объяснение** = старые paired/random/coherent + gzip + matched multi-rule;
- **перенос на естественный язык / реальный текст** = Wikipedia entity substitution;
- **границы и расширения** = chained verification, synthetic world, cross-domain, appendix-негативы.

### Вердикт по Wikipedia-экспериментам
У них **есть ценность**, и довольно высокая, но **не как новое ядро статьи**.

Их лучшая роль:
1. **показать перенос принципа на реальный естественный текст**;
2. **заменить synthetic world как основной NL-supporting block**, если нужно экономить место;
3. **усилить Discussion/Transfer section** и снять возражение “это работает только на арифметике и игрушечных мирах”.

Их не стоит делать центральным доказательством, потому что это:
- не denoising в строгом J-смысле (“одна задача, противоречивые ответы в одном корпусе”);
- не чистый benchmark на world-truth в сильном философском смысле;
- всё ещё эксперимент на **текстовой консистентности и правдоподобии оригинального Wikipedia-контекста**.

---

## 1. Новая центральная формулировка статьи

### Основной вопрос
**Если модель обучается на противоречивых данных, какой ответ она предпочтёт и почему?**

### Центральный тезис
**В противоречивом корпусе модель предпочитает тот вариант, который образует более сжимаемое и внутренне согласованное ядро.**

### Безопасная формулировка claim
Не:
- “compression drives truth in general”
- “language models discover truth as such”

А:
- “compression favors the most internally consistent answer cluster”
- “truth wins when it aligns with the more compressible cluster”
- “coherent false systems can remove or reverse the apparent truth preference”

### Роль слова “truth” в новой статье
Слово **truth** нужно оставить в названии/мотивации, но в тексте оно должно быть **зависимой, а не первичной переменной**:
- первична **compressibility / consistency**;
- истина выигрывает только в тех условиях, где она совпадает с более компактной структурой.

---

## 2. Архитектура доказательства: что теперь является ядром, а что — поддержкой

### Уровень A — main text, без этого статья не работает

#### A1. Denoising J1 vs J2
Это новый стержень всей работы.

- **J1:** 1 correct + 1 random wrong
- **J2:** 1 correct + 1 coherent wrong

Именно эта пара отвечает на главный вопрос статьи:
- модель умеет извлекать сигнал из шума;
- но не умеет привилегировать истину, когда ложь тоже образует компактную систему.

#### A2. Denoising J3 / J4
Это второй обязательный блок main text.

Он показывает, что эффект не бинарный, а подчиняется **signal-to-noise логике**:
- при росте числа случайных ложных ответов сигнал ослабевает;
- но не исчезает сразу;
- scaling помогает извлекать паттерн до некоторого потолка.

#### A3. J5 (0 correct + 2 random)
Оставить в статье, но **не делать центральным headline**.

Лучшее место:
- конец раздела Results как limitation / curiosity;
- либо начало Discussion.

Причина:
- это интересный результат;
- но он уязвим к артефактам длины, распределения цифр, формата ответа и прочим подсказкам.

**Правило:** пока не закрыты контроли, J5 — не abstract claim.

---

### Уровень B — main text или late-main-text как объяснение механизма

#### B1. Старый paired random/coherent baseline
Не выбрасывать.

Его новая функция:
- не “главный эксперимент статьи”,
- а **bridge** между старой и новой версией работы.

Он показывает, что и в старом дизайне уже был тот же центральный контраст:
- random errors -> preference for correct;
- coherent errors -> near chance / reversal.

То есть denoising — это не отмена старого результата, а его **более чистая постановка**.

#### B2. gzip / description-length block
Обязательно сохранить.

Без него статья легко расползается из “compression hypothesis” в просто “interesting bias paper”.

Задача этого блока:
- показать, что объяснение не только поведенческое;
- напрямую связать величину эффекта с мерой компрессируемости;
- удержать paper в поле MDL / compression.

#### B3. Matched multi-rule
Оставить обязательно.

Это лучший механистический support после J1/J2:
- при `N=1` coherent-false cluster силён;
- при росте числа ложных правил он теряет сжимаемость;
- correct cluster получает преимущество.

Именно этот блок лучше всего показывает, что дело **не в “истинности” самой по себе**, а в **структуре пространства альтернатив**.

---

### Уровень C — transfer / supporting evidence

#### C1. Wikipedia entity substitution
**Сильный supporting block.**

Рекомендуемая роль:
- лучший non-synthetic natural-language support в статье;
- можно поднять выше synthetic world;
- хороший late-main-text section или сильный appendix-first section.

Что именно он должен подтверждать:
- принцип не ограничен арифметикой;
- на реальном тексте random corruption даёт preference for original / factual continuation;
- coherent substitution снимает или разворачивает этот эффект.

Что он **не** должен заявлять:
- “мы доказали truthfulness in the wild”;
- “это уже полноценная модель борьбы с дезинформацией в интернете”.

#### C2. Synthetic world
Не выкидывать, но **понизить ранг** по сравнению с Wikipedia.

Теперь его лучшая роль:
- controlled natural-language bridge между math и real text;
- объяснение, почему естественный язык слабее отделяет truth от coherent falsehood.

Если места мало, Wikipedia важнее synthetic world.

#### C3. BPE
Оставить как robustness, не как сюжетный блок.

Лучшая роль:
- короткий абзац в main text или appendix figure;
- защита от критики “это artefact char-level tokenization”.

---

### Уровень D — boundary conditions / extension

#### D1. Chained verification
Сохранять.

Но не как центральный result, а как:
- важный bridge к alignment/safety;
- демонстрацию того, что truth preference можно усилить, если встроить **verification dependencies**.

#### D2. Cross-domain falsification
Appendix / future-work flavored result.

Полезен как идея, но не нужен для центральной логики новой статьи.

#### D3. Conditions C/D/E, contradictory and similar negatives
Appendix-only или убрать из main paper.

Их ценность:
- методологический урок;
- демонстрация того, что corpus-level comparison легко ломается.

Но они не должны конкурировать с denoising за внимание читателя.

---

## 3. Роль Wikipedia-экспериментов в новой статье

## 3.1. Что в них реально ценно

### 1. Это первый сильный real-text support
В отличие от synthetic world, это уже не игрушечный NL-мир, а **реальные Wikipedia paragraphs**.

Поэтому у них есть реальная ценность как ответа на ожидаемую критику:
> “Ок, на арифметике это работает. А на настоящем тексте?”

### 2. В них воспроизводится главный random/coherent контраст
Именно это делает их важными.

Полезный headline для статьи:
- **на реальном тексте random corruption остаётся distinguishable;**
- **coherent corruption снова почти неотличима от truth/original.**

### 3. Они сильнее synthetic world как transfer evidence
Если нужно выбрать один NL-block, брать нужно **Wikipedia**, а не synthetic world.

Почему:
- сильнее эффект random vs coherent;
- более реалистичный домен;
- меньше ощущения “ещё один toy setup”.

### 4. По типам сущностей есть содержательная неоднородность
Это ценно для appendix / analysis:
- какие типы сущностей легче восстанавливаются;
- где консистентность сильнее помогает;
- где текст допускает больше правдоподобных ложных замен.

Это хороший материал для discussion, но не для abstract.

---

## 3.2. Чего Wikipedia-эксперименты пока не доказывают

### 1. Они не доказывают “truthfulness in the wild” в сильном смысле
Потому что модель может предпочитать оригинальный paragraph не только из-за truth, но и из-за:
- локальной текстовой согласованности;
- более типичных entity co-occurrences;
- частотных эффектов конкретных имён / топонимов / организаций;
- длины и формы entity strings.

### 2. Они не являются главным доказательством denoising-гипотезы
Потому что это не тот же экспериментальный дизайн, что J1–J5.

Wikipedia block должен отвечать на вопрос:
> “Переносится ли принцип на реальный текст?”

Но не на вопрос:
> “Что именно является центральным механизмом статьи?”

### 3. Scaling внутри Wikipedia — вторично
Если random растёт лишь слабо, а coherent уходит чуть ниже chance, это интересно, но не headline.

Лучше использовать этот блок как **contrast result**, а не как ещё одну scaling-story.

---

## 3.3. Как именно использовать Wikipedia в paper narrative

### Рекомендуемый статус
**Main text support** или **strong appendix support**.

Моя рекомендация:
- если paper длинный и хочется упростить narrative -> **late main text**;
- если нужно жёстко экономить место -> **первый appendix block**, но с упоминанием в Introduction/Discussion.

### Лучший framing
Не писать:
- “Wikipedia confirms truth bias in the wild”

Писать:
- “A real-text entity-substitution experiment shows the same random/coherent contrast on Wikipedia paragraphs.”
- “The principle transfers to natural text, though the task remains a controlled corruption setting rather than open-world truthfulness.”

### Лучшее сравнение
Сравнивать Wikipedia нужно прежде всего с:
- denoising math core;
- old paired math baseline;
- synthetic world.

И не сравнивать напрямую как будто это один и тот же тип эксперимента с J1–J5.

---

## 3.4. Что нужно доделать, чтобы Wikipedia-блок стал по-настоящему сильным

### Must-have controls

#### W1. Явно описать split и защиту от leakage
Нужно в статье и/или appendix явно прописать:
- train/test split по статьям или по paragraph ids;
- нет ли пересечения paragraph templates;
- нет ли совпадающих оригинальных paragraph fragments между train и test.

#### W2. Length-matched paired metric
Добавить/проверить для wiki те же paired robustness controls, что уже есть для math:
- primary mean NLL;
- sum-NLL;
- length-matched mean NLL.

Если знак сохраняется — блок становится намного сильнее.

#### W3. Frequency/length matching for replacements
Нужен хотя бы один контроль на то, что effect не объясняется банально тем, что:
- random substitute длиннее / короче;
- entity string реже или чаще встречается;
- coherent map систематически производит более удобные строки.

Минимально достаточный контроль:
- подбирать substitute из близкого bucket по длине и частоте;
- или добавить отдельную таблицу “effect survives length/frequency matching”.

### Strongly recommended

#### W4. 5–10 human-readable examples
Нужен мини-набор примеров в paper/appendix:
- original paragraph;
- random substitution;
- coherent substitution;
- why random looks noisier;
- why coherent can remain textually smooth but factually wrong.

Это сильно повышает читаемость.

#### W5. Один BPE-check на wiki
Не обязательно full sweep.

Достаточно tiny-random + tiny-coherent на BPE, если бюджет ограничен.

Это не must-have, но очень полезно, если хочешь заранее закрыть критику “real-text effect is char-token artifact”.

### Nice to have

#### W6. A denoising-style wiki pilot
Например:
- один paragraph context,
- оригинальное factual continuation,
- несколько ложных entity-substituted continuations,
- прямой paired/cluster-style выбор.

Это не обязательно для этой версии статьи, но если получится быстро — это станет идеальным мостом между J-ядром и Wikipedia support.

---

## 4. Что оставить в main text, а что увести в appendix

## 4.1. Main text (обязательно)

1. **Introduction:** новый denoising-framing
2. **Setup:** J1–J5
3. **Results I:** J1 vs J2
4. **Results II:** J3/J4/J5
5. **Mechanism:** old paired baseline + gzip + matched multi-rule
6. **Transfer:** Wikipedia
7. **Robustness:** BPE (кратко), paired metric variants (кратко)
8. **Discussion:** chained verification + implications + limitations

## 4.2. Main text (опционально, если влезает)

- краткий synthetic world paragraph как “controlled NL bridge”;
- 1 figure или 1 table по chained verification.

## 4.3. Appendix

- synthetic world full tables;
- world multi-alt;
- cross-domain falsification;
- C/D/E и corpus-level divergences;
- full learning curves;
- expanded robustness tables;
- detailed Wikipedia per-type tables;
- extra wiki examples;
- generation sanity full details.

## 4.4. Что можно вообще убрать из narrative

Если paper начинает расползаться, первыми кандидатами на сокращение являются:
- длинные обсуждения C/D/E;
- сложные narrative around contradictory condition;
- часть cross-domain;
- часть synthetic-world subplots.

---

## 5. Новая структура статьи

## 5.1. Abstract

Новая логика abstract:
1. Models train on contradictory data.
2. We test which answer cluster they prefer.
3. In denoising setups, correct answers win when false alternatives are random.
4. This preference disappears when false alternatives are coherent.
5. Supporting experiments show the same compression/consistency logic in legacy paired math, multi-rule settings, and real Wikipedia text.
6. Conclusion: compression favors consistency; truth wins only when aligned with the most compressible cluster.

**Чего не делать в abstract:**
- не перечислять все appendix experiments;
- не вставлять 6–8 чисел;
- не делать Wikipedia headline вместо denoising.

---

## 5.2. Introduction

### Цель
Переход от старого question:
> “Does compression drive truth?”

к новому question:
> “When models see contradictory data, which answer cluster do they compress toward?”

### Структура intro
1. Проблема противоречивых данных в обучении LMs.
2. Почему стандартные explanations (majority, RLHF, sycophancy) не покрывают pretraining-level effect.
3. Наша гипотеза: компрессия предпочитает **consistency**, а truth выигрывает как частный случай.
4. Наш тест: denoising with correct vs random/coherent false answers.
5. Кратко: что именно показывают J1–J5 + supporting blocks.

### Ключевая фраза intro
> Models do not privilege truth directly; they privilege the most compressible answer structure available in the data.

---

## 5.3. Methods / Setup

### Section 2: Core denoising setup
Нужно сделать максимально ясным и визуальным.

Показать один пример задачи:
- `2 + 3 = 5` (correct)
- `2 + 3 = 7` (random wrong)
- `2 + 3 = 6` according to coherent rule `a+b -> a+b+1`

И объяснить:
- что попадает в training corpus;
- что сравнивается в paired eval;
- чем J1–J5 отличаются друг от друга.

### Важная цель
После Section 2 рецензент должен понимать paper без знания старой версии работы.

---

## 5.4. Results ordering

### Section 3: Denoising core results
#### 3.1 J1 vs J2
Главный figure + главный table.

#### 3.2 J3 / J4
Noise tolerance / graceful degradation.

#### 3.3 J5
Кратко, осторожно, с оговоркой про artefact risk.

#### 3.4 Comparison to standard non-contradictory math setup
Нужно коротко показать, что denoising слабее “чистого” random baseline, но логика та же.

---

## 5.5. Mechanism section

### Section 4: Why this is consistency, not truth per se
Здесь живут:
- old paired random vs coherent;
- gzip / description length;
- matched multi-rule.

### Логика секции
1. Random errors help the correct cluster because false answers do not compress together.
2. Coherent errors remove that advantage.
3. Increasing the diversity of false rules gradually restores the advantage of the correct cluster.
4. Direct compression metrics track this pattern.

Это и есть mechanistic backbone всей статьи.

---

## 5.6. Transfer section

### Section 5: Transfer beyond formal math
Порядок:
1. Wikipedia entity substitution (главный transfer block)
2. Synthetic world (коротко, controlled NL support)

### Почему именно так
Потому что Wikipedia:
- реальный текст;
- stronger effect;
- лучше отвечает на expected reviewer objection.

Synthetic world после этого становится не competitor, а bridge.

---

## 5.7. Discussion

Discussion должна быть короткой и синтетической.

### Три крупных тезиса
1. **Compression–Consistency Principle**
2. **Why coherent misinformation is hard**
3. **How verification dependencies can help**

### Что туда вставить
- chained verification;
- safety implications;
- границы claim;
- почему Wikipedia support важен, но всё ещё не равен “truthfulness in the wild”.

---

## 6. Новая иерархия экспериментов (таблица решений)

| Блок | Оставить | Роль | Позиция в статье | Комментарий |
|---|---|---|---|---|
| J1: 1 correct + 1 random | да | ядро | main text | главный positive result |
| J2: 1 correct + 1 coherent | да | ядро | main text | главный control |
| J3/J4 | да | ядро | main text | noise-tolerance curve |
| J5 | да, осторожно | limitation / curiosity | Results end / Discussion | не делать headline |
| Standard paired math | да | bridge | mechanism section | показывает преемственность со старым paper |
| gzip / bz2 | да | mechanism | mechanism section | удерживает compression framing |
| Matched multi-rule | да | mechanism | mechanism section | лучший structural support |
| BPE | да | robustness | short main text / appendix | защита от char-level criticism |
| Wikipedia | да | real-text transfer | late main text / strong appendix | лучше synthetic world как transfer evidence |
| Synthetic world | да, но ниже | controlled NL support | appendix или короткий main-text paragraph | полезен, но слабее wiki |
| Chained verification | да | implications / boundary | discussion or late main text | лучший alignment bridge |
| Cross-domain | да, appendix | exploratory boundary | appendix | не нужен для core story |
| C/D/E and similar | appendix | methodological negative | appendix | не конкурировать с core |

---

## 7. Что нужно сделать в репозитории до переписывания текста

## 7.1. Синхронизация manifests

### Обязательно обновить
- `results_manifest.md`
- `claims_manifest.md`
- `results_master.csv`
- при необходимости `data/eval_inputs_manifest.md`

### Что добавить
#### Для denoising
- artifact family для J1–J5;
- exact run coverage по size/seed;
- safe claims и required qualifiers.

#### Для Wikipedia
- отдельный раздел с:
  - corpus provenance,
  - split,
  - 4 sizes × 2 modes × 4 seeds,
  - safe wording.

### Зачем это важно
Иначе статья будет опережать артефакты, а reviewer/reader увидит несинхрон между paper и repo.

---

## 7.2. Автоматическая сборка таблиц

Нужно, чтобы все новые main-text таблицы генерировались автоматически.

### Минимальный набор таблиц
- denoising summary J1–J5;
- old paired math summary;
- matched multi-rule summary;
- wiki summary by size;
- wiki per-type appendix table;
- robustness summary (paired metric variants + BPE).

### Принцип
Любое число в paper должно восстанавливаться одной командой из artifacts.

---

## 8. Какие новые figure/table нужны

## 8.1. Main text figures

### Figure 1 — Denoising setup schematic
Одна задача, противоречивые ответы, random vs coherent falsehood.

### Figure 2 — J1 vs J2 scaling
Главная картинка статьи.

### Figure 3 — J3/J4/J5 noise curve
Показывает graceful degradation.

### Figure 4 — Standard paired baseline vs denoising
Нужна короткая bridge-figure или compact table.

### Figure 5 — Multi-rule matched curve
Показывает graded restoration.

### Figure 6 — Compressibility plot (gzip / description length)
Показывает связь mechanism <-> behavior.

### Figure 7 — Wikipedia transfer
Лучший вариант:
- bar chart by size: random vs coherent;
- или compact panel: overall + per-type inset/table.

### Figure 8 — Chained verification (если в main text)
Если места мало, увести в appendix.

---

## 8.2. Appendix figures/tables

- synthetic world full table;
- world multi-alt curve;
- cross-domain table/figure;
- learning curves;
- wiki per-type table;
- wiki examples;
- full robustness tables.

---

## 9. Конкретный план действий по шагам

## Phase 1 — зафиксировать ядро и claims

### Шаг 1
Переписать one-sentence thesis везде одинаково:
- README
- paper abstract
- introduction opening paragraph
- claims manifest

### Шаг 2
Формально определить роль каждого эксперимента по 4 уровням:
- core
- mechanism
- transfer
- appendix

### Шаг 3
Синхронизировать manifests под denoising + wiki.

**Выход Phase 1:** narrative и артефакты больше не противоречат друг другу.

---

## Phase 2 — сделать denoising самостоятельным main story

### Шаг 4
Собрать чистый раздел Results только по J1–J5.

### Шаг 5
Нарисовать 2 главных figure:
- J1 vs J2 scaling
- J3/J4/J5 noise curve

### Шаг 6
Написать минимальный Section 2 (methods/setup), который можно читать отдельно от старой статьи.

**Выход Phase 2:** paper уже работает даже без старых экспериментов.

---

## Phase 3 — добавить mechanism layer

### Шаг 7
Вставить old paired math baseline как bridge.

### Шаг 8
Вставить gzip block.

### Шаг 9
Вставить matched multi-rule.

### Шаг 10
Свести эти 3 блока к одной общей формуле:
> truth preference appears when false alternatives fail to form an equally compressible cluster.

**Выход Phase 3:** у статьи появляется объяснение, а не только феноменология.

---

## Phase 4 — добавить transfer layer

### Шаг 11
Подготовить wiki summary figure/table.

### Шаг 12
Сделать wiki controls (минимум W1-W3).

### Шаг 13
Написать section:
- “Transfer to real text”
- 2–4 абзаца + 1 figure/table
- 1 абзац limitation

### Шаг 14
Решить судьбу synthetic world:
- либо краткий bridge paragraph;
- либо appendix-only.

**Выход Phase 4:** статья показывает перенос за пределы арифметики.

---

## Phase 5 — discussion и сокращение

### Шаг 15
Сжать Discussion до 3 крупных тезисов.

### Шаг 16
Перенести всё вторичное в appendix.

### Шаг 17
Оставить в main text только то, что усиливает denoising story.

**Тест качества:**
Если удалить 2/3 appendix, main claim должен остаться полностью понятным и убедительным.

---

## 10. Приоритеты: что обязательно, что желательно, что можно не делать

## Must
- J1–J5 как main core
- old paired + gzip + matched multi-rule как mechanism
- manifests synced
- Wikipedia integrated as support
- J5 qualified, not oversold

## Should
- wiki controls W1–W3
- synthetic world moved below wiki
- one compact chained discussion block
- BPE mention retained

## Could
- wiki BPE
- denoising-style wiki pilot
- extra analysis of per-type wiki behavior
- extra polishing figures

## Not needed for this version
- превращать Wikipedia в новый central experiment
- строить сильный scaling claim на Wikipedia
- вытаскивать cross-domain в main text
- возвращать corpus-level story в центр paper

---

## 11. Самая практичная редакторская стратегия

### Если цель — лучшая версия статьи при ограниченном времени
Сделать так:

1. **Главное ядро:** J1/J2/J3/J4
2. **Осторожный бонус:** J5
3. **Объяснение:** old paired + gzip + matched multi-rule
4. **Перенос:** Wikipedia
5. **Короткая robustness защита:** BPE + paired variants
6. **Discussion bridge:** chained verification
7. **В appendix:** synthetic world, cross-domain, C/D/E, full tables

Это даёт самую чистую и убедительную структуру.

---

## 12. Финальный verdict

### Что делать с новой идеей
**Да — сделать denoising центральным ядром статьи.**

### Что делать со старыми экспериментами
**Не выбрасывать, а переподчинить новой логике.**

### Что делать с Wikipedia
**Обязательно оставить.**

Но их лучшая роль не “новый центр paper”, а:
- **лучший real-text support**,
- **лучший transfer block**,
- **лучший кандидат заменить synthetic world как главный NL-support**.

### Самая короткая формула плана
> Denoising показывает главный эффект. 
> Старые paired/multi-rule/compressibility блоки объясняют механизм. 
> Wikipedia показывает, что тот же принцип живёт на реальном тексте.

