# Free-code session logs

- **Session ID:** 210af205-5b87-4a45-8623-f0a136482af9
- **Started:** 2026-04-03T08:51:02.949Z
- **Working directory:** /Users/krestnikov/giga/compression-drives-truth
- **Hostname:** Konstantins-MacBook-Pro.local
- **User:** krestnikov

## User prompts
### 2026-04-03T08:51:02.952Z

quota

### 2026-04-03T08:51:13.705Z

Изучи содержимое моей научной работы

### 2026-04-03T08:51:13.710Z

<system-reminder>
The following skills are available for use with the Skill tool:

- update-config: Use this skill to configure the Claude Code harness via settings.json. Automated behaviors ("from now on when X", "each time X", "whenever X", "before/after X") require hooks configured in settings.json - the harness executes these, not Claude, so m…
- simplify: Review changed code for reuse, quality, and efficiency, then fix any issues found.
</system-reminder>

<system-reminder>
As you answer the user's questions, you can use the following context:
# claudeMd
Codebase and user instructions are shown below. Be sure to adhere to these instructions. IMPORTANT: These instructions OVERRIDE any default behavior and you MUST follow them exactly as written.

Contents of /Users/krestnikov/.claude/projects/-Users-krestnikov-giga-compression-drives-truth/memory/MEMORY.md (user's auto-memory, persists across conversations):

# Compression Truth Bias — Project Memory

## Publication
- [project_neurips_submission.md](project_neurips_submission.md) — NeurIPS 2026, Sydney, Dec 6-12. Abstract May 4, paper May 6, notification Sep 24
- [reference_fenogenova_review.md](reference_fenogenova_review.md) — Alena Fenogenova's review of v2, basis for v3 revisions

## Project Status
- Experiment 1 (math, tiny models): COMPLETE. 35 models trained.
- Experiment 2 (observations): COMPLETE. 20 models. Weak effect confirmed.
- Experiment 3 (conditions C/D/E): COMPLETE. 12 models trained.
- Experiment 4 (scaling): COMPLETE. Small (11M) 4 seeds, medium (26M) 4 seeds, large (86M) 4 random + 1 coherent.
- Experiment F (multi-rule): COMPLETE. 16 models (N=2,3,5,10 × 4 seeds).
- Experiment G (synthetic world, natural language): COMPLETE. 8 models (random + coherent × 4 seeds).
- Experiment: World multi-alt (N=2,4,8,16): COMPLETE. 16 models. Gradual rise 40%→60%, no phase transition.
- Experiment: World contradictory: COMPLETE. 4 models. acc≈47.5%, no truth bias.
- Paper draft: `paper_draft_ru.md` (Russi

### 2026-04-03T08:51:27.756Z

# Error Structure Determines Correctness Preference in Contradictory Training Data

## Abstract

Language models trained on contradictory data often prefer correct answers, yet the mechanism behind this preference is poorly understood. Without such understanding, we cannot predict when this implicit filtering will fail -- a question critical for controlling model behavior on noisy real-world corpora. We hypothesize that next-token prediction, as a compression process, favors whichever answer cluster has lower description length; truth benefits only when errors lack internal structure. We train transformers (3.5M--1B parameters) from scratch on controlled corpora where the same problem appears with both correct and incorrect solutions, systematically varying the structure of errors. We demonstrate three findings: (a) when errors are random, models develop a correctness preference scaling from 65% to 85% with model size; (b) when errors follow a single coherent alternative rule, this preference vanishes entirely (~45--51%); (c) the transition is sharp -- two competing wrong rules suffice to restore correctness preference (47% -> 78%). The pattern reproduces on real Wikipedia text (71% vs 46%), across architectures (GPT-2 and Qwen3), and at 1B scale on a mixed real-text corpus (77% vs 47%). These results support the hypothesis that, in controlled contradictory corpora, model preference is driven by the relative compressibility of competing answer systems rather than by truth per se.

---

## 1. Introduction

Real-world training corpora are noisy: the same question may receive contradictory answers across different documents. Yet language models trained on such data tend to prefer correct information -- and this implicit filtering is increasingly relied upon in high-stakes applications from medical QA to legal reasoning. If we do not understand *why* this filtering works, we cannot predict *when* it will fail. This is not merely a theoretical concern: coordinated misinf

### 2026-04-03T08:53:14.248Z

Учти что отзыв Феногеновой относится ко второй версии, а у нас уже готова треться

### 2026-04-03T08:54:04.782Z

<system-reminder>
The task tools haven't been used recently. If you're working on tasks that would benefit from tracking progress, consider using TaskCreate to add new tasks and TaskUpdate to update task status (set to in_progress when starting, completed when done). Also consider cleaning up the task list if it has become stale. Only use these if relevant to the current work. This is just a gentle reminder - ignore if not applicable. Make sure that you NEVER mention this reminder to the user

</system-reminder>

Что думаешь про этот отзыв ИИ ревьювера? С чем согласен? ultrathink Да — формально на NeurIPS 2026 эту работу подать можно. Для main track в этом году дедлайны такие: abstract — 4 мая 2026 AOE, полный paper с supplementary — 6 мая 2026 AOE. По типу вклада работа тоже подходит: call for papers прямо приветствует in-depth analysis of existing methods, если она дает новые инсайты об их поведении и ограничениях. Публичные arXiv/GitHub сами по себе не запрещены, но анонимная submission, supplement и любые внешние ссылки/код для ревью должны оставаться анонимными; прямую ссылку на текущий публичный репозиторий в анонимной PDF я бы не ставил.

По качеству я бы сейчас оценил это как скорее 3/6 = borderline reject, а не accept. Не потому, что идея слабая — идея как раз хорошая, — а потому что у NeurIPS main track “3” как раз соответствует technically solid paper, где ограниченность evaluation перевешивает плюсы. У вас сильный центральный результат, но доказательная база пока слишком узкая для уверенного main-track acceptance.

Что в работе реально сильное: очень чистая controlled setup; запоминающийся и нетривиальный главный контраст между random и coherent errors; резкий переход в multi-rule experiment; перенос на Wikipedia; проверка на другой архитектуре; и дополнительный 1B mixed-corpus experiment. В таком виде paper читается как hypothesis-driven empirical study, а не как просто очередной benchmark. Это хороший фундамент для сильной конференционной работы.

Что 

### 2026-04-03T08:56:01.429Z

<local-command-caveat>Caveat: The messages below were generated by the user while running local commands. DO NOT respond to these messages or otherwise consider them in your response unless the user explicitly asks you to.</local-command-caveat>

<command-name>/effort</command-name>
            <command-message>effort</command-message>
            <command-args>auto</command-args>

<local-command-stdout>Effort level set to auto</local-command-stdout>

Нет ресурса на обучение больших моделей, оставляем как есть. А остальное внеси правки

### 2026-04-03T08:58:14.196Z

<system-reminder>
As you answer the user's questions, you can use the following context:
# claudeMd
Codebase and user instructions are shown below. Be sure to adhere to these instructions. IMPORTANT: These instructions OVERRIDE any default behavior and you MUST follow them exactly as written.

Contents of /Users/krestnikov/.claude/projects/-Users-krestnikov-giga-compression-drives-truth/memory/MEMORY.md (user's auto-memory, persists across conversations):

# Compression Truth Bias — Project Memory

## Publication
- [project_neurips_submission.md](project_neurips_submission.md) — NeurIPS 2026, Sydney, Dec 6-12. Abstract May 4, paper May 6, notification Sep 24
- [reference_fenogenova_review.md](reference_fenogenova_review.md) — Alena Fenogenova's review of v2, basis for v3 revisions

## Project Status
- Experiment 1 (math, tiny models): COMPLETE. 35 models trained.
- Experiment 2 (observations): COMPLETE. 20 models. Weak effect confirmed.
- Experiment 3 (conditions C/D/E): COMPLETE. 12 models trained.
- Experiment 4 (scaling): COMPLETE. Small (11M) 4 seeds, medium (26M) 4 seeds, large (86M) 4 random + 1 coherent.
- Experiment F (multi-rule): COMPLETE. 16 models (N=2,3,5,10 × 4 seeds).
- Experiment G (synthetic world, natural language): COMPLETE. 8 models (random + coherent × 4 seeds).
- Experiment: World multi-alt (N=2,4,8,16): COMPLETE. 16 models. Gradual rise 40%→60%, no phase transition.
- Experiment: World contradictory: COMPLETE. 4 models. acc≈47.5%, no truth bias.
- Paper draft: `paper_draft_ru.md` (Russian), `paper_draft_en.md` (English)
- Experiment I (chained tasks): COMPLETE. 4 tiny + 4 small + 2 large = 10 models.
  - Tiny: 70.9%±1.2%, Small: 64.2%±1.5%, Large: 60.6%±1.2%
  - INVERSE SCALING: accuracy DROPS with model size (opposite of random errors)
  - KEY RESULT: compressor power is double-edged sword
- **STATUS: PUBLISHED on arXiv — https://arxiv.org/abs/2603.11749. 160+ models. All experiments complete.**
- Experiment K (Wikipedia entity substitution):

### 2026-04-03T08:58:17.130Z

Tool loaded.
<system-reminder>
The following skills are available for use with the Skill tool:

- update-config: Use this skill to configure the Claude Code harness via settings.json. Automated behaviors ("from now on when X", "each time X", "whenever X", "before/after X") require hooks configured in settings.json - the harness executes these, not Claude, so m…
- simplify: Review changed code for reuse, quality, and efficiency, then fix any issues found.
</system-reminder>

### 2026-04-03T08:58:20.549Z

Perform a web search for the query: Xie Zhang Chen Lou Su "Adaptive Chameleon or Stubborn Sloth" ICLR 2024 knowledge conflicts

### 2026-04-03T08:58:20.550Z

Perform a web search for the query: Longpre Perisetla Chen Ramesh DuBois Singh "Entity-Based Knowledge Conflicts in Question Answering" EMNLP 2021

### 2026-04-03T08:59:18.278Z

<system-reminder>
The task tools haven't been used recently. If you're working on tasks that would benefit from tracking progress, consider using TaskCreate to add new tasks and TaskUpdate to update task status (set to in_progress when starting, completed when done). Also consider cleaning up the task list if it has become stale. Only use these if relevant to the current work. This is just a gentle reminder - ignore if not applicable. Make sure that you NEVER mention this reminder to the user

</system-reminder>

Собери PDF для заливки на arXiv

### 2026-04-03T08:59:26.477Z

\documentclass{article}

% For submission: use default (anonymous, line numbers)
\usepackage{neurips_2026}
% For preprint: \usepackage[preprint]{neurips_2026}
% For camera-ready: \usepackage[main, final]{neurips_2026}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{hyperref}
\usepackage{url}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{nicefrac}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage{enumitem}

\title{Error Structure Determines Correctness Preference\\in Contradictory Training Data}

\author{Anonymous}

\date{}

\begin{document}

\maketitle

\begin{abstract}
Language models trained on contradictory data often prefer correct answers, yet the mechanism behind this preference is poorly understood. Without such understanding, we cannot predict when this implicit filtering will fail---a question critical for controlling model behavior on noisy real-world corpora. We hypothesize that next-token prediction, as a compression process, favors whichever answer cluster has lower description length; truth benefits only when errors lack internal structure. We train transformers (3.5M--1B parameters) from scratch on controlled corpora where the same problem appears with both correct and incorrect solutions, systematically varying the structure of errors. We demonstrate three findings: (a)~when errors are random, models develop a correctness preference scaling from 65\% to 85\% with model size; (b)~when errors follow a single coherent alternative rule, this preference vanishes entirely (${\sim}45$--$51\%$); (c)~the transition is sharp---two competing wrong rules suffice to restore correctness preference ($47\% \to 78\%$). The pattern reproduces on real Wikipedia text with entity substitution (71\% vs 46\%) and at 1B scale on real web text (77\% vs 47\%). These results support the hypothesis that, in controlled contradictory corpora, model preference is driven by the relative com

### 2026-04-03T09:01:34.029Z

<system-reminder>
As you answer the user's questions, you can use the following context:
# claudeMd
Codebase and user instructions are shown below. Be sure to adhere to these instructions. IMPORTANT: These instructions OVERRIDE any default behavior and you MUST follow them exactly as written.

Contents of /Users/krestnikov/.claude/projects/-Users-krestnikov-giga-compression-drives-truth/memory/MEMORY.md (user's auto-memory, persists across conversations):

# Compression Truth Bias — Project Memory

## Publication
- [project_neurips_submission.md](project_neurips_submission.md) — NeurIPS 2026, Sydney, Dec 6-12. Abstract May 4, paper May 6, notification Sep 24
- [reference_fenogenova_review.md](reference_fenogenova_review.md) — Alena Fenogenova's review of v2, basis for v3 revisions

## Project Status
- Experiment 1 (math, tiny models): COMPLETE. 35 models trained.
- Experiment 2 (observations): COMPLETE. 20 models. Weak effect confirmed.
- Experiment 3 (conditions C/D/E): COMPLETE. 12 models trained.
- Experiment 4 (scaling): COMPLETE. Small (11M) 4 seeds, medium (26M) 4 seeds, large (86M) 4 random + 1 coherent.
- Experiment F (multi-rule): COMPLETE. 16 models (N=2,3,5,10 × 4 seeds).
- Experiment G (synthetic world, natural language): COMPLETE. 8 models (random + coherent × 4 seeds).
- Experiment: World multi-alt (N=2,4,8,16): COMPLETE. 16 models. Gradual rise 40%→60%, no phase transition.
- Experiment: World contradictory: COMPLETE. 4 models. acc≈47.5%, no truth bias.
- Paper draft: `paper_draft_ru.md` (Russian), `paper_draft_en.md` (English)
- Experiment I (chained tasks): COMPLETE. 4 tiny + 4 small + 2 large = 10 models.
  - Tiny: 70.9%±1.2%, Small: 64.2%±1.5%, Large: 60.6%±1.2%
  - INVERSE SCALING: accuracy DROPS with model size (opposite of random errors)
  - KEY RESULT: compressor power is double-edged sword
- **STATUS: PUBLISHED on arXiv — https://arxiv.org/abs/2603.11749. 160+ models. All experiments complete.**
- Experiment K (Wikipedia entity substitution):

### 2026-04-03T09:01:48.882Z

Tool loaded.

### 2026-04-03T09:01:58.452Z


Web page content:
---
           window.MathJax = { tex: { inlineMath: \[\['$', '$'\], \['\\\\(', '\\\\)'\]\], displayMath: \[\['\\\\\[', '\\\\\]'\]\], processEscapes: true, }, options: { ignoreHtmlClass: 'no-mathjax', } }; (function (document, navigator, standalone) { if ((standalone in navigator) && navigator\[standalone\]) { var curnode, location = document.location, stop = /^(a|html)$/i; document.addEventListener('click', function (e) { curnode = e.target; while (!(stop).test(curnode.nodeName)) { curnode = curnode.parentNode; } if ( 'href' in curnode && // is a link (chref = curnode.href).replace(location.href, '').indexOf('#') && // is not an anchor (!(/^\[a-z\\+\\.\\-\]+:/i).test(chref) || // either does not have a proper scheme (relative links) chref.indexOf(location.protocol + '//' + location.host) === 0) // or is in the same protocol and domain ) { e.preventDefault(); location.href = curnode.href; } }, false); } })(document, window.navigator, 'standalone'); @media screen and (min-width: 767px) { .blurb { min-height: 260px; } } if (typeof jQuery === 'undefined') { var script = document.createElement('script'); script.type = 'text/javascript'; script.src = "/static/core/js/jquery-3.6.1.min.js"; document.head.appendChild(script); } var $ = jQuery;  body { font-family: Exo;}      Call for Papers 2026

[Skip to yearly menu bar](#child-menu) [Skip to main content](#main)

Main Navigation
---------------

[![conference_logo](/static/core/img/neurips-navbar-logo.svg)](/)

*   [NeurIPS](#)
    *   [Help/FAQ](/FAQ)
        
        * * *
        
    *   [Contact NeurIPS](/Help/Contact)
        
        * * *
        
    *   [Code of Ethics](/Conferences/2023/EthicsGuidelines)
        
        * * *
        
    *   [Code of Conduct](/public/CodeOfConduct)
        
        * * *
        
    *   [Create Profile](/Profile/create)
        
        * * *
        
    *   [Journal To Conference Track](/public/JournalToConference)
        
        * * *
        
    *  

### 2026-04-03T10:36:08.491Z

<system-reminder>
The task tools haven't been used recently. If you're working on tasks that would benefit from tracking progress, consider using TaskCreate to add new tasks and TaskUpdate to update task status (set to in_progress when starting, completed when done). Also consider cleaning up the task list if it has become stale. Only use these if relevant to the current work. This is just a gentle reminder - ignore if not applicable. Make sure that you NEVER mention this reminder to the user

</system-reminder>

Закомить и запуш

### 2026-04-03T11:39:17.134Z

<system-reminder>
The following skills are available for use with the Skill tool:

- update-config: Use this skill to configure the Claude Code harness via settings.json. Automated behaviors ("from now on when X", "each time X", "whenever X", "before/after X") require hooks configured in settings.json - the harness executes these, not Claude, so m…
- simplify: Review changed code for reuse, quality, and efficiency, then fix any issues found.
</system-reminder>
<system-reminder>
The task tools haven't been used recently. If you're working on tasks that would benefit from tracking progress, consider using TaskCreate to add new tasks and TaskUpdate to update task status (set to in_progress when starting, completed when done). Also consider cleaning up the task list if it has become stale. Only use these if relevant to the current work. This is just a gentle reminder - ignore if not applicable. Make sure that you NEVER mention this reminder to the user

</system-reminder>

Посмотри отзыв ИИ рецензента. С чем ты согласен?

### 2026-04-03T11:39:18.697Z

<system-reminder>
The task tools haven't been used recently. If you're working on tasks that would benefit from tracking progress, consider using TaskCreate to add new tasks and TaskUpdate to update task status (set to in_progress when starting, completed when done). Also consider cleaning up the task list if it has become stale. Only use these if relevant to the current work. This is just a gentle reminder - ignore if not applicable. Make sure that you NEVER mention this reminder to the user

</system-reminder>

Посмотри отзыв ИИ рецензента. С чем ты согласен?

### 2026-04-03T11:39:26.405Z

<system-reminder>
The task tools haven't been used recently. If you're working on tasks that would benefit from tracking progress, consider using TaskCreate to add new tasks and TaskUpdate to update task status (set to in_progress when starting, completed when done). Also consider cleaning up the task list if it has become stale. Only use these if relevant to the current work. This is just a gentle reminder - ignore if not applicable. Make sure that you NEVER mention this reminder to the user

</system-reminder>

Посмотри отзыв ИИ рецензента. С чем ты согласен?

1. Общее впечатление и научная ценность (подходит ли для NeurIPS?)
Да, работа сильная и имеет все шансы на acceptance.

Новизна и вклад: Чёткая, экспериментально подтверждённая гипотеза: «next-token prediction = compression → модель предпочитает наиболее сжимаемый кластер ответов, а не “правду” per se». Это объясняет, почему truth bias возникает только при random errors, а coherent false rules (одно системное заблуждение) его полностью убивает. Sharp transition при 2+ competing wrong rules — это красивый результат, который точно зацепит reviewers.
Методология: Идеально контролируемая (frequency/format/domain фиксированы, варьируется только compressibility errors). Scaling 3.5M → 1B, transfer на Wikipedia entity substitution + real web text (FineWeb-like) — это уже не toy-experiment.
Результаты (abstract + CSV):
Random errors → correctness preference 65–85% (растёт с размером модели).
Single coherent rule → ~45–51% (chance).
2+ rules → резкий скачок до 78%.
Wikipedia: 71% vs 46%.
1B-scale real-text corpus (Qwen3-1B): 77% vs 47%.
Всё это точно воспроизводится по results_master.csv (я проверил ключевые строки: chained_50_50_* = random/mixed → high acc; chained_truncated_* = coherent → ~43–46%; multi-rule и wiki-строки соответствуют).

Связь с литературой: Отлично вписывается в MDL/compression (Deletang 2024 — реальная работа), simplicity bias, noisy labels, grokking. Не повторяет existing work (Li et al. 2024 о

### 2026-04-03T11:40:27.706Z

Посмотри второй отзыв - NeurIPS Paper Checklist. Это обязательный компонент — без него desk rejection. Нужно включить в PDF после ссылок. Чеклист содержит 16 пунктов — claims, limitations, theory, reproducibility, code, statistical significance, compute resources, ethics, broader impacts, LLM usage declaration и т.д.
LLM Usage Declaration. NeurIPS 2026 требует декларирования если LLM были существенной частью methodology. Если Claude использовался для генерации текста paper (а не только как инструмент для исследований), это нужно раскрыть.

2. ПРОВЕРКА ССЫЛОК
Я верифицировал ключевые ссылки:
Проблемная ссылка — Chlon et al. (2025):

В paper указано: "Predictable Compression Failures: Why Language Models Actually Hallucinate"
Это название v1. Текущая v2 (февраль 2026) имеет другое название: "Predictable Compression Failures: Order Sensitivity and Information Budgeting for Evidence-Grounded Binary Adjudication"
Авторы: в paper указано "Chlon, L., Karim, A., Chlon, M., & Awada, M." — в v1 было только 3 автора, Awada добавлен в v2. Нужно привести в соответствие с актуальной версией
arXiv ID 2509.11208 — корректен

Остальные проверенные ссылки — корректны:

Wan & Mei (2025), arXiv:2505.15784 — подтверждено
Ravfogel et al. (2025) — действительно NeurIPS 2025, подтверждено
Deletang et al. (2024) — ICLR 2024, корректно
Pan et al. (2025), arXiv:2504.09597 — нужно проверить (очень свежая)
Калассические ссылки (Shannon, Rissanen, Solomonoff, Popper) — корректны

Замечание по Kadavath et al. (2022): В тексте Introduction написано "Kadavath et al., 2022" — правильная фамилия Kadavath, на arXiv это Saurav Kadavath et al., "Language Models (Mostly) Know What They Know" — проверь, не Kadavath ли должна быть Kadavath → нет, всё верно, фамилия пишется именно Kadavath.

3. СОДЕРЖАНИЕ И ЛОГИКА РАБОТЫ
Сильные стороны
Работа отличная по замыслу и execution для small-scale research:

Чёткая, фальсифицируемая гипотеза. "Compression favors consistency, not truth" — это элегантно сформулирова

### 2026-04-03T11:40:55.815Z

<system-reminder>
As you answer the user's questions, you can use the following context:
# claudeMd
Codebase and user instructions are shown below. Be sure to adhere to these instructions. IMPORTANT: These instructions OVERRIDE any default behavior and you MUST follow them exactly as written.

Contents of /Users/krestnikov/.claude/projects/-Users-krestnikov-giga-compression-drives-truth/memory/MEMORY.md (user's auto-memory, persists across conversations):

# Compression Truth Bias — Project Memory

## Publication
- [project_neurips_submission.md](project_neurips_submission.md) — NeurIPS 2026, Sydney, Dec 6-12. Abstract May 4, paper May 6, notification Sep 24
- [reference_fenogenova_review.md](reference_fenogenova_review.md) — Alena Fenogenova's review of v2, basis for v3 revisions

## Project Status
- Experiment 1 (math, tiny models): COMPLETE. 35 models trained.
- Experiment 2 (observations): COMPLETE. 20 models. Weak effect confirmed.
- Experiment 3 (conditions C/D/E): COMPLETE. 12 models trained.
- Experiment 4 (scaling): COMPLETE. Small (11M) 4 seeds, medium (26M) 4 seeds, large (86M) 4 random + 1 coherent.
- Experiment F (multi-rule): COMPLETE. 16 models (N=2,3,5,10 × 4 seeds).
- Experiment G (synthetic world, natural language): COMPLETE. 8 models (random + coherent × 4 seeds).
- Experiment: World multi-alt (N=2,4,8,16): COMPLETE. 16 models. Gradual rise 40%→60%, no phase transition.
- Experiment: World contradictory: COMPLETE. 4 models. acc≈47.5%, no truth bias.
- Paper draft: `paper_draft_ru.md` (Russian), `paper_draft_en.md` (English)
- Experiment I (chained tasks): COMPLETE. 4 tiny + 4 small + 2 large = 10 models.
  - Tiny: 70.9%±1.2%, Small: 64.2%±1.5%, Large: 60.6%±1.2%
  - INVERSE SCALING: accuracy DROPS with model size (opposite of random errors)
  - KEY RESULT: compressor power is double-edged sword
- **STATUS: PUBLISHED on arXiv — https://arxiv.org/abs/2603.11749. 160+ models. All experiments complete.**
- Experiment K (Wikipedia entity substitution):

### 2026-04-03T11:40:58.863Z

Tool loaded.
<system-reminder>
The following skills are available for use with the Skill tool:

- update-config: Use this skill to configure the Claude Code harness via settings.json. Automated behaviors ("from now on when X", "each time X", "whenever X", "before/after X") require hooks configured in settings.json - the harness executes these, not Claude, so m…
- simplify: Review changed code for reuse, quality, and efficiency, then fix any issues found.
</system-reminder>

### 2026-04-03T11:41:04.065Z


Web page content:
---
\[2603.25839\] A Compression Perspective on Simplicity Bias                              

[Skip to main content](#content)

[![Cornell University](/static/browse/0.3.4/images/icons/cu/cornell-reduced-white-SMALL.svg)](https://www.cornell.edu/)

[Learn about arXiv becoming an independent nonprofit.](https://tech.cornell.edu/arxiv/)

We gratefully acknowledge support from the Simons Foundation, [member institutions](https://info.arxiv.org/about/ourmembers.html), and all contributors. [Donate](https://info.arxiv.org/about/donate.html)

[](/IgnoreMe)

[![arxiv logo](/static/browse/0.3.4/images/arxiv-logo-one-color-white.svg)](/) \> [cs](/list/cs/recent) \> arXiv:2603.25839

[Help](https://info.arxiv.org/help) | [Advanced Search](https://arxiv.org/search/advanced)

All fields Title Author Abstract Comments Journal reference ACM classification MSC classification Report number arXiv identifier DOI ORCID arXiv author ID Help pages Full text

 Search

[![arXiv logo](/static/browse/0.3.4/images/arxiv-logomark-small-white.svg)](https://arxiv.org/)

  [![Cornell University Logo](/static/browse/0.3.4/images/icons/cu/cornell-reduced-white-SMALL.svg)](https://www.cornell.edu/)

open search

   GO

open navigation menu

quick links
-----------

*   [Login](https://arxiv.org/login)
*   [Help Pages](https://info.arxiv.org/help)
*   [About](https://info.arxiv.org/about)

Computer Science > Machine Learning
===================================

**arXiv:2603.25839** (cs)

\[Submitted on 26 Mar 2026\]

Title:A Compression Perspective on Simplicity Bias
==================================================

Authors:[Tom Marty](https://arxiv.org/search/cs?searchtype=author&query=Marty,+T), [Eric Elmoznino](https://arxiv.org/search/cs?searchtype=author&query=Elmoznino,+E), [Leo Gagnon](https://arxiv.org/search/cs?searchtype=author&query=Gagnon,+L), [Tejas Kasetty](https://arxiv.org/search/cs?searchtype=author&query=Kasetty,+T), [Mizu Nishikawa-Toomey](https://arxiv.org/s

### 2026-04-03T11:41:04.183Z


Web page content:
---
\[2603.20396\] Compression is all you need: Modeling Mathematics                              

[Skip to main content](#content)

[![Cornell University](/static/browse/0.3.4/images/icons/cu/cornell-reduced-white-SMALL.svg)](https://www.cornell.edu/)

[Learn about arXiv becoming an independent nonprofit.](https://tech.cornell.edu/arxiv/)

We gratefully acknowledge support from the Simons Foundation, [member institutions](https://info.arxiv.org/about/ourmembers.html), and all contributors. [Donate](https://info.arxiv.org/about/donate.html)

[](/IgnoreMe)

[![arxiv logo](/static/browse/0.3.4/images/arxiv-logo-one-color-white.svg)](/) \> [cs](/list/cs/recent) \> arXiv:2603.20396

[Help](https://info.arxiv.org/help) | [Advanced Search](https://arxiv.org/search/advanced)

All fields Title Author Abstract Comments Journal reference ACM classification MSC classification Report number arXiv identifier DOI ORCID arXiv author ID Help pages Full text

 Search

[![arXiv logo](/static/browse/0.3.4/images/arxiv-logomark-small-white.svg)](https://arxiv.org/)

  [![Cornell University Logo](/static/browse/0.3.4/images/icons/cu/cornell-reduced-white-SMALL.svg)](https://www.cornell.edu/)

open search

   GO

open navigation menu

quick links
-----------

*   [Login](https://arxiv.org/login)
*   [Help Pages](https://info.arxiv.org/help)
*   [About](https://info.arxiv.org/about)

Computer Science > Artificial Intelligence
==========================================

**arXiv:2603.20396** (cs)

\[Submitted on 20 Mar 2026\]

Title:Compression is all you need: Modeling Mathematics
=======================================================

Authors:[Vitaly Aksenov](https://arxiv.org/search/cs?searchtype=author&query=Aksenov,+V), [Eve Bodnia](https://arxiv.org/search/cs?searchtype=author&query=Bodnia,+E), [Michael H. Freedman](https://arxiv.org/search/cs?searchtype=author&query=Freedman,+M+H), [Michael Mulligan](https://arxiv.org/search/cs?searchtype=author&query=Mulligan,+M

### 2026-04-03T11:43:47.598Z

Давай, если это ценно

### 2026-04-03T11:48:22.582Z

Отлично, закомить и запуш

### 2026-04-03T12:49:23.296Z

<command-name>/status</command-name>
            <command-message>status</command-message>
            <command-args></command-args>

<local-command-stdout>Status dialog dismissed</local-command-stdout>

Проанализируй этот отзыв

Сразу вывод: идея у работы сильная и потенциально NeurIPS-уровня, но в текущем виде я бы не называл её submission-ready. У вас есть хороший эмпирический “hook” и уже подготовлен NeurIPS-источник на neurips_2026 с анонимным автором и checklist, но сейчас есть несколько проблем, которые рецензенты почти наверняка увидят: рассинхрон между текстом и кодом, проблемы воспроизводимости, одна заметная количественная нестыковка в gzip-результате, некорректное описание статистики в checklist, а также высокий риск с анонимизацией и supplementary. NeurIPS 2026 требует официальный стиль, максимум 9 страниц основного текста, анонимизированные материалы для review и checklist; дедлайны — abstract 4 мая 2026 AOE, full paper 6 мая 2026 AOE.

Что у вас уже хорошо

Самое сильное в paper — не просто контраст “random vs coherent corruption”, а multi-rule experiment. Именно скачок от N=1 к N=2 и дальше выглядит как наиболее убедительная механистическая часть истории: на tiny accuracy растёт примерно с 46.6 до 77.6, 82.8, 84.8 и 88.3 при росте числа независимых правил; на small при N=2 уже 86.3. Это намного сильнее продаёт тезис, чем один только базовый synthetic setup. Плюс есть перенос на Wikipedia, BPE-робастность, cross-architecture check на Qwen-style модели и mixed-corpus 1B experiment — то есть paper не выглядит как “single toy result”.

Ещё плюс: в репозитории есть отдельный paper_neurips/ с main.tex, bib и фигурками, а не только markdown-черновик. Для сабмита это правильно: сам paper_v3.md не является submission format, но NeurIPS-версия в LaTeX у вас уже заведена и использует neurips_2026.

Что обязательно исправить до подачи
1) Сейчас supplementary / code review не анонимны как надо

Это, на мой взгляд, самый практический риск. В paper_neurips/ANONYMIZ

### 2026-04-03T12:49:58.631Z

<system-reminder>
As you answer the user's questions, you can use the following context:
# claudeMd
Codebase and user instructions are shown below. Be sure to adhere to these instructions. IMPORTANT: These instructions OVERRIDE any default behavior and you MUST follow them exactly as written.

Contents of /Users/krestnikov/.claude/projects/-Users-krestnikov-giga-compression-drives-truth/memory/MEMORY.md (user's auto-memory, persists across conversations):

# Compression Truth Bias — Project Memory

## Publication
- [project_neurips_submission.md](project_neurips_submission.md) — NeurIPS 2026, Sydney, Dec 6-12. Abstract May 4, paper May 6, notification Sep 24
- [reference_fenogenova_review.md](reference_fenogenova_review.md) — Alena Fenogenova's review of v2, basis for v3 revisions

## Project Status
- Experiment 1 (math, tiny models): COMPLETE. 35 models trained.
- Experiment 2 (observations): COMPLETE. 20 models. Weak effect confirmed.
- Experiment 3 (conditions C/D/E): COMPLETE. 12 models trained.
- Experiment 4 (scaling): COMPLETE. Small (11M) 4 seeds, medium (26M) 4 seeds, large (86M) 4 random + 1 coherent.
- Experiment F (multi-rule): COMPLETE. 16 models (N=2,3,5,10 × 4 seeds).
- Experiment G (synthetic world, natural language): COMPLETE. 8 models (random + coherent × 4 seeds).
- Experiment: World multi-alt (N=2,4,8,16): COMPLETE. 16 models. Gradual rise 40%→60%, no phase transition.
- Experiment: World contradictory: COMPLETE. 4 models. acc≈47.5%, no truth bias.
- Paper draft: `paper_draft_ru.md` (Russian), `paper_draft_en.md` (English)
- Experiment I (chained tasks): COMPLETE. 4 tiny + 4 small + 2 large = 10 models.
  - Tiny: 70.9%±1.2%, Small: 64.2%±1.5%, Large: 60.6%±1.2%
  - INVERSE SCALING: accuracy DROPS with model size (opposite of random errors)
  - KEY RESULT: compressor power is double-edged sword
- **STATUS: PUBLISHED on arXiv — https://arxiv.org/abs/2603.11749. 160+ models. All experiments complete.**
- Experiment K (Wikipedia entity substitution):

### 2026-04-03T12:49:58.632Z

<system-reminder>
As you answer the user's questions, you can use the following context:
# claudeMd
Codebase and user instructions are shown below. Be sure to adhere to these instructions. IMPORTANT: These instructions OVERRIDE any default behavior and you MUST follow them exactly as written.

Contents of /Users/krestnikov/.claude/projects/-Users-krestnikov-giga-compression-drives-truth/memory/MEMORY.md (user's auto-memory, persists across conversations):

# Compression Truth Bias — Project Memory

## Publication
- [project_neurips_submission.md](project_neurips_submission.md) — NeurIPS 2026, Sydney, Dec 6-12. Abstract May 4, paper May 6, notification Sep 24
- [reference_fenogenova_review.md](reference_fenogenova_review.md) — Alena Fenogenova's review of v2, basis for v3 revisions

## Project Status
- Experiment 1 (math, tiny models): COMPLETE. 35 models trained.
- Experiment 2 (observations): COMPLETE. 20 models. Weak effect confirmed.
- Experiment 3 (conditions C/D/E): COMPLETE. 12 models trained.
- Experiment 4 (scaling): COMPLETE. Small (11M) 4 seeds, medium (26M) 4 seeds, large (86M) 4 random + 1 coherent.
- Experiment F (multi-rule): COMPLETE. 16 models (N=2,3,5,10 × 4 seeds).
- Experiment G (synthetic world, natural language): COMPLETE. 8 models (random + coherent × 4 seeds).
- Experiment: World multi-alt (N=2,4,8,16): COMPLETE. 16 models. Gradual rise 40%→60%, no phase transition.
- Experiment: World contradictory: COMPLETE. 4 models. acc≈47.5%, no truth bias.
- Paper draft: `paper_draft_ru.md` (Russian), `paper_draft_en.md` (English)
- Experiment I (chained tasks): COMPLETE. 4 tiny + 4 small + 2 large = 10 models.
  - Tiny: 70.9%±1.2%, Small: 64.2%±1.5%, Large: 60.6%±1.2%
  - INVERSE SCALING: accuracy DROPS with model size (opposite of random errors)
  - KEY RESULT: compressor power is double-edged sword
- **STATUS: PUBLISHED on arXiv — https://arxiv.org/abs/2603.11749. 160+ models. All experiments complete.**
- Experiment K (Wikipedia entity substitution):

### 2026-04-03T12:49:58.633Z

<system-reminder>
As you answer the user's questions, you can use the following context:
# claudeMd
Codebase and user instructions are shown below. Be sure to adhere to these instructions. IMPORTANT: These instructions OVERRIDE any default behavior and you MUST follow them exactly as written.

Contents of /Users/krestnikov/.claude/projects/-Users-krestnikov-giga-compression-drives-truth/memory/MEMORY.md (user's auto-memory, persists across conversations):

# Compression Truth Bias — Project Memory

## Publication
- [project_neurips_submission.md](project_neurips_submission.md) — NeurIPS 2026, Sydney, Dec 6-12. Abstract May 4, paper May 6, notification Sep 24
- [reference_fenogenova_review.md](reference_fenogenova_review.md) — Alena Fenogenova's review of v2, basis for v3 revisions

## Project Status
- Experiment 1 (math, tiny models): COMPLETE. 35 models trained.
- Experiment 2 (observations): COMPLETE. 20 models. Weak effect confirmed.
- Experiment 3 (conditions C/D/E): COMPLETE. 12 models trained.
- Experiment 4 (scaling): COMPLETE. Small (11M) 4 seeds, medium (26M) 4 seeds, large (86M) 4 random + 1 coherent.
- Experiment F (multi-rule): COMPLETE. 16 models (N=2,3,5,10 × 4 seeds).
- Experiment G (synthetic world, natural language): COMPLETE. 8 models (random + coherent × 4 seeds).
- Experiment: World multi-alt (N=2,4,8,16): COMPLETE. 16 models. Gradual rise 40%→60%, no phase transition.
- Experiment: World contradictory: COMPLETE. 4 models. acc≈47.5%, no truth bias.
- Paper draft: `paper_draft_ru.md` (Russian), `paper_draft_en.md` (English)
- Experiment I (chained tasks): COMPLETE. 4 tiny + 4 small + 2 large = 10 models.
  - Tiny: 70.9%±1.2%, Small: 64.2%±1.5%, Large: 60.6%±1.2%
  - INVERSE SCALING: accuracy DROPS with model size (opposite of random errors)
  - KEY RESULT: compressor power is double-edged sword
- **STATUS: PUBLISHED on arXiv — https://arxiv.org/abs/2603.11749. 160+ models. All experiments complete.**
- Experiment K (Wikipedia entity substitution):

### 2026-04-03T12:50:40.395Z

<system-reminder>
The task tools haven't been used recently. If you're working on tasks that would benefit from tracking progress, consider using TaskCreate to add new tasks and TaskUpdate to update task status (set to in_progress when starting, completed when done). Also consider cleaning up the task list if it has become stale. Only use these if relevant to the current work. This is just a gentle reminder - ignore if not applicable. Make sure that you NEVER mention this reminder to the user

</system-reminder>

### 2026-04-03T12:52:25.224Z

Давай делай все

### 2026-04-03T13:16:29.203Z

А почему убрали Поппера? Красивая же гипотеза

### 2026-04-03T13:24:55.626Z

Полностью перепроверь всю литературу - каждую статью и сслыку - нет ли там галлюцинаций, правильно ли написаны авторы

### 2026-04-03T13:25:45.005Z

<system-reminder>
As you answer the user's questions, you can use the following context:
# claudeMd
Codebase and user instructions are shown below. Be sure to adhere to these instructions. IMPORTANT: These instructions OVERRIDE any default behavior and you MUST follow them exactly as written.

Contents of /Users/krestnikov/.claude/projects/-Users-krestnikov-giga-compression-drives-truth/memory/MEMORY.md (user's auto-memory, persists across conversations):

# Compression Truth Bias — Project Memory

## Publication
- [project_neurips_submission.md](project_neurips_submission.md) — NeurIPS 2026, Sydney, Dec 6-12. Abstract May 4, paper May 6, notification Sep 24
- [reference_fenogenova_review.md](reference_fenogenova_review.md) — Alena Fenogenova's review of v2, basis for v3 revisions

## Project Status
- Experiment 1 (math, tiny models): COMPLETE. 35 models trained.
- Experiment 2 (observations): COMPLETE. 20 models. Weak effect confirmed.
- Experiment 3 (conditions C/D/E): COMPLETE. 12 models trained.
- Experiment 4 (scaling): COMPLETE. Small (11M) 4 seeds, medium (26M) 4 seeds, large (86M) 4 random + 1 coherent.
- Experiment F (multi-rule): COMPLETE. 16 models (N=2,3,5,10 × 4 seeds).
- Experiment G (synthetic world, natural language): COMPLETE. 8 models (random + coherent × 4 seeds).
- Experiment: World multi-alt (N=2,4,8,16): COMPLETE. 16 models. Gradual rise 40%→60%, no phase transition.
- Experiment: World contradictory: COMPLETE. 4 models. acc≈47.5%, no truth bias.
- Paper draft: `paper_draft_ru.md` (Russian), `paper_draft_en.md` (English)
- Experiment I (chained tasks): COMPLETE. 4 tiny + 4 small + 2 large = 10 models.
  - Tiny: 70.9%±1.2%, Small: 64.2%±1.5%, Large: 60.6%±1.2%
  - INVERSE SCALING: accuracy DROPS with model size (opposite of random errors)
  - KEY RESULT: compressor power is double-edged sword
- **STATUS: PUBLISHED on arXiv — https://arxiv.org/abs/2603.11749. 160+ models. All experiments complete.**
- Experiment K (Wikipedia entity substitution):

### 2026-04-03T13:25:45.006Z

<system-reminder>
As you answer the user's questions, you can use the following context:
# claudeMd
Codebase and user instructions are shown below. Be sure to adhere to these instructions. IMPORTANT: These instructions OVERRIDE any default behavior and you MUST follow them exactly as written.

Contents of /Users/krestnikov/.claude/projects/-Users-krestnikov-giga-compression-drives-truth/memory/MEMORY.md (user's auto-memory, persists across conversations):

# Compression Truth Bias — Project Memory

## Publication
- [project_neurips_submission.md](project_neurips_submission.md) — NeurIPS 2026, Sydney, Dec 6-12. Abstract May 4, paper May 6, notification Sep 24
- [reference_fenogenova_review.md](reference_fenogenova_review.md) — Alena Fenogenova's review of v2, basis for v3 revisions

## Project Status
- Experiment 1 (math, tiny models): COMPLETE. 35 models trained.
- Experiment 2 (observations): COMPLETE. 20 models. Weak effect confirmed.
- Experiment 3 (conditions C/D/E): COMPLETE. 12 models trained.
- Experiment 4 (scaling): COMPLETE. Small (11M) 4 seeds, medium (26M) 4 seeds, large (86M) 4 random + 1 coherent.
- Experiment F (multi-rule): COMPLETE. 16 models (N=2,3,5,10 × 4 seeds).
- Experiment G (synthetic world, natural language): COMPLETE. 8 models (random + coherent × 4 seeds).
- Experiment: World multi-alt (N=2,4,8,16): COMPLETE. 16 models. Gradual rise 40%→60%, no phase transition.
- Experiment: World contradictory: COMPLETE. 4 models. acc≈47.5%, no truth bias.
- Paper draft: `paper_draft_ru.md` (Russian), `paper_draft_en.md` (English)
- Experiment I (chained tasks): COMPLETE. 4 tiny + 4 small + 2 large = 10 models.
  - Tiny: 70.9%±1.2%, Small: 64.2%±1.5%, Large: 60.6%±1.2%
  - INVERSE SCALING: accuracy DROPS with model size (opposite of random errors)
  - KEY RESULT: compressor power is double-edged sword
- **STATUS: PUBLISHED on arXiv — https://arxiv.org/abs/2603.11749. 160+ models. All experiments complete.**
- Experiment K (Wikipedia entity substitution):

### 2026-04-03T13:25:45.007Z

<system-reminder>
As you answer the user's questions, you can use the following context:
# claudeMd
Codebase and user instructions are shown below. Be sure to adhere to these instructions. IMPORTANT: These instructions OVERRIDE any default behavior and you MUST follow them exactly as written.

Contents of /Users/krestnikov/.claude/projects/-Users-krestnikov-giga-compression-drives-truth/memory/MEMORY.md (user's auto-memory, persists across conversations):

# Compression Truth Bias — Project Memory

## Publication
- [project_neurips_submission.md](project_neurips_submission.md) — NeurIPS 2026, Sydney, Dec 6-12. Abstract May 4, paper May 6, notification Sep 24
- [reference_fenogenova_review.md](reference_fenogenova_review.md) — Alena Fenogenova's review of v2, basis for v3 revisions

## Project Status
- Experiment 1 (math, tiny models): COMPLETE. 35 models trained.
- Experiment 2 (observations): COMPLETE. 20 models. Weak effect confirmed.
- Experiment 3 (conditions C/D/E): COMPLETE. 12 models trained.
- Experiment 4 (scaling): COMPLETE. Small (11M) 4 seeds, medium (26M) 4 seeds, large (86M) 4 random + 1 coherent.
- Experiment F (multi-rule): COMPLETE. 16 models (N=2,3,5,10 × 4 seeds).
- Experiment G (synthetic world, natural language): COMPLETE. 8 models (random + coherent × 4 seeds).
- Experiment: World multi-alt (N=2,4,8,16): COMPLETE. 16 models. Gradual rise 40%→60%, no phase transition.
- Experiment: World contradictory: COMPLETE. 4 models. acc≈47.5%, no truth bias.
- Paper draft: `paper_draft_ru.md` (Russian), `paper_draft_en.md` (English)
- Experiment I (chained tasks): COMPLETE. 4 tiny + 4 small + 2 large = 10 models.
  - Tiny: 70.9%±1.2%, Small: 64.2%±1.5%, Large: 60.6%±1.2%
  - INVERSE SCALING: accuracy DROPS with model size (opposite of random errors)
  - KEY RESULT: compressor power is double-edged sword
- **STATUS: PUBLISHED on arXiv — https://arxiv.org/abs/2603.11749. 160+ models. All experiments complete.**
- Experiment K (Wikipedia entity substitution):

### 2026-04-03T13:25:49.828Z

Tool loaded.
<system-reminder>
The following skills are available for use with the Skill tool:

- update-config: Use this skill to configure the Claude Code harness via settings.json. Automated behaviors ("from now on when X", "each time X", "whenever X", "before/after X") require hooks configured in settings.json - the harness executes these, not Claude, so m…
- simplify: Review changed code for reuse, quality, and efficiency, then fix any issues found.
</system-reminder>

### 2026-04-03T13:25:51.384Z

Tool loaded.
<system-reminder>
The following skills are available for use with the Skill tool:

- update-config: Use this skill to configure the Claude Code harness via settings.json. Automated behaviors ("from now on when X", "each time X", "whenever X", "before/after X") require hooks configured in settings.json - the harness executes these, not Claude, so m…
- simplify: Review changed code for reuse, quality, and efficiency, then fix any issues found.
</system-reminder>

### 2026-04-03T13:25:51.743Z

Tool loaded.
<system-reminder>
The following skills are available for use with the Skill tool:

- update-config: Use this skill to configure the Claude Code harness via settings.json. Automated behaviors ("from now on when X", "each time X", "whenever X", "before/after X") require hooks configured in settings.json - the harness executes these, not Claude, so m…
- simplify: Review changed code for reuse, quality, and efficiency, then fix any issues found.
</system-reminder>

### 2026-04-03T13:25:57.249Z

Perform a web search for the query: arXiv 2603.25839 "Compression Perspective on Simplicity Bias" Marty Elmoznino 2026

### 2026-04-03T13:25:57.634Z

Perform a web search for the query: Azaria Mitchell "The Internal State of an LLM Knows When It's Lying" EMNLP 2023

### 2026-04-03T13:25:57.635Z

Perform a web search for the query: Bürger Hamprecht Nadler "Truth is Universal: Robust Detection of Lies in LLMs" NeurIPS 2024

### 2026-04-03T13:25:57.635Z

Perform a web search for the query: Burns Ye Klein Steinhardt "Discovering Latent Knowledge in Language Models Without Supervision" ICLR 2023

### 2026-04-03T13:25:57.635Z

Perform a web search for the query: Chlon Karim "Order Sensitivity and Information Budgeting" arXiv 2509.11208

### 2026-04-03T13:25:57.635Z

Perform a web search for the query: Deletang Ruoss Grau-Moya "Language Modeling Is Compression" ICLR 2024

### 2026-04-03T13:25:57.636Z

Perform a web search for the query: Elazar Kassner Ravfogel "Measuring Causal Effects of Data Statistics on Language Model's Factual Predictions" arXiv 2207.14251

### 2026-04-03T13:25:57.636Z

Perform a web search for the query: Penedo Kydlíček Lozhkov "FineWeb Datasets Decanting the Web" arXiv 2406.17557

### 2026-04-03T13:25:57.636Z

Perform a web search for the query: Goldblum Finzi Rowan Wilson "No Free Lunch Theorem Kolmogorov Complexity" ICML 2024

### 2026-04-03T13:25:57.636Z

Perform a web search for the query: Grünwald "The Minimum Description Length Principle" MIT Press 2007

### 2026-04-03T13:25:57.636Z

Perform a web search for the query: Gurnee Tegmark "Language Models Represent Space and Time" ICLR 2024

### 2026-04-03T13:25:58.263Z

Perform a web search for the query: Huang Yuzhen "Compression Represents Intelligence Linearly" COLM 2024

### 2026-04-03T13:25:58.263Z

Perform a web search for the query: Joshi Nitarshan "Personas as a Way to Model Truthfulness" EMNLP 2024

### 2026-04-03T13:25:58.264Z

Perform a web search for the query: Kadavath Saurav "Language Models Mostly Know What They Know" arXiv 2207.05221

### 2026-04-03T13:25:58.264Z

Perform a web search for the query: Kalai Vempala "Calibrated Language Models Must Hallucinate" STOC 2024

### 2026-04-03T13:25:58.264Z

Perform a web search for the query: Kandpal Nikhil "Large Language Models Struggle to Learn Long-Tail Knowledge" ICML 2023

### 2026-04-03T13:25:58.265Z

Perform a web search for the query: Longpre Shayne "Entity-Based Knowledge Conflicts in Question Answering" EMNLP 2021

### 2026-04-03T13:25:58.265Z

Perform a web search for the query: Li Jiahuan "Formality is Favored" conflicting knowledge EMNLP 2024

### 2026-04-03T13:25:58.265Z

Perform a web search for the query: Li Kenneth "Emergent World Representations" sequence model synthetic task ICLR 2023

### 2026-04-03T13:25:58.265Z

Perform a web search for the query: Li Kenneth "Inference-Time Intervention Eliciting Truthful Answers" NeurIPS 2023

### 2026-04-03T13:25:58.265Z

Perform a web search for the query: Liu Ziming "Grokking as Compression" nonlinear complexity arXiv 2310.05918

### 2026-04-03T13:26:16.437Z

Perform a web search for the query: arXiv 2603.20396 "Compression is All You Need" Aksenov Bodnia Freedman mathematics 2026

### 2026-04-03T13:26:32.440Z

Perform a web search for the query: arxiv 2207.05221 "Language Models Mostly Know What They Know" authors Kadavath Conerly Askell

### 2026-04-03T13:26:32.441Z

Perform a web search for the query: Kenneth Li "Inference-Time Intervention" authors Oam Patel Fernanda Viégas Viegas NeurIPS 2023

### 2026-04-03T13:26:32.978Z

Perform a web search for the query: Marks Tegmark "Geometry of Truth" emergent linear structure LLM true false COLM 2024

### 2026-04-03T13:26:57.640Z

Perform a web search for the query: McGraw Wong "Common Language Effect Size Statistic" Psychological Bulletin 1992

### 2026-04-03T13:27:19.310Z

Perform a web search for the query: Mingard Valle-Perez Sherrington Louis "Is SGD a Bayesian Sampler" JMLR 2021

### 2026-04-03T13:27:37.038Z

Perform a web search for the query: Nanda Chan Lieberum Smith Steinhardt "Progress Measures for Grokking" mechanistic interpretability ICLR 2023

### 2026-04-03T13:27:56.513Z

Perform a web search for the query: arXiv 2504.09597 Pan Wang Li "Understanding LLM Behaviors via Compression" 2025

### 2026-04-03T13:28:16.289Z

Perform a web search for the query: Ravfogel Yehudai Linzen "Emergence of Linear Truth Encodings" language models NeurIPS 2025

### 2026-04-03T13:28:39.327Z

Perform a web search for the query: arXiv 2505.15784 Wan Mei "Large Language Models Computable Approximations Solomonoff Induction" 2025

### 2026-04-03T13:29:00.821Z


Web page content:
---
\[2505.15784\] Large Language Models as Computable Approximations to Solomonoff Induction                              

[Skip to main content](#content)

[![Cornell University](/static/browse/0.3.4/images/icons/cu/cornell-reduced-white-SMALL.svg)](https://www.cornell.edu/)

[Learn about arXiv becoming an independent nonprofit.](https://tech.cornell.edu/arxiv/)

We gratefully acknowledge support from the Simons Foundation, [member institutions](https://info.arxiv.org/about/ourmembers.html), and all contributors. [Donate](https://info.arxiv.org/about/donate.html)

[](/IgnoreMe)

[![arxiv logo](/static/browse/0.3.4/images/arxiv-logo-one-color-white.svg)](/) \> [cs](/list/cs/recent) \> arXiv:2505.15784

[Help](https://info.arxiv.org/help) | [Advanced Search](https://arxiv.org/search/advanced)

All fields Title Author Abstract Comments Journal reference ACM classification MSC classification Report number arXiv identifier DOI ORCID arXiv author ID Help pages Full text

 Search

[![arXiv logo](/static/browse/0.3.4/images/arxiv-logomark-small-white.svg)](https://arxiv.org/)

  [![Cornell University Logo](/static/browse/0.3.4/images/icons/cu/cornell-reduced-white-SMALL.svg)](https://www.cornell.edu/)

open search

   GO

open navigation menu

quick links
-----------

*   [Login](https://arxiv.org/login)
*   [Help Pages](https://info.arxiv.org/help)
*   [About](https://info.arxiv.org/about)

Computer Science > Machine Learning
===================================

**arXiv:2505.15784** (cs)

\[Submitted on 21 May 2025\]

Title:Large Language Models as Computable Approximations to Solomonoff Induction
================================================================================

Authors:[Jun Wan](https://arxiv.org/search/cs?searchtype=author&query=Wan,+J), [Lingrui Mei](https://arxiv.org/search/cs?searchtype=author&query=Mei,+L)

View a PDF of the paper titled Large Language Models as Computable Approximations to Solomonoff Induction, by Jun Wan and 1 

### 2026-04-03T13:29:05.780Z

Perform a web search for the query: Xie Zhang Chen Lou Su "Adaptive Chameleon or Stubborn Sloth" knowledge conflicts ICLR 2024

### 2026-04-03T13:29:25.145Z

Perform a web search for the query: Marks Tegmark "Geometry of Truth" COLM 2024 conference venue

### 2026-04-03T13:29:48.648Z

Perform a web search for the query: Ravfogel "Emergence of Linear Truth Encodings" authors Bietti Bruna order NeurIPS 2025

### 2026-04-03T13:30:05.749Z

Perform a web search for the query: Popper "Logic of Scientific Discovery" Hutchinson 1959 publisher

### 2026-04-03T13:30:21.102Z

Perform a web search for the query: Rissanen "Modeling by Shortest Data Description" Automatica 1978 volume pages

### 2026-04-03T13:30:33.663Z

Perform a web search for the query: Shannon "Mathematical Theory of Communication" Bell System Technical Journal 1948 volume pages

### 2026-04-03T13:30:46.729Z

Perform a web search for the query: Solomonoff "Formal Theory of Inductive Inference" Information and Control 1964 volume pages

### 2026-04-03T13:31:00.260Z

Perform a web search for the query: Valle-Perez Camargo Louis "Deep Learning Generalizes" "parameter-function map" simple functions ICLR 2019

### 2026-04-03T13:31:17.102Z

Perform a web search for the query: Rolnick Veit Belongie Shavit "Deep Learning is Robust to Massive Label Noise" arXiv 1705.10694 2017

### 2026-04-03T13:31:38.604Z

Perform a web search for the query: Zhang Bengio Hardt Recht Vinyals "Understanding Deep Learning Requires Rethinking Generalization" ICLR 2017

### 2026-04-03T13:31:59.754Z


Web page content:
---
\[2510.15804\] Emergence of Linear Truth Encodings in Language Models                              

[Skip to main content](#content)

[![Cornell University](/static/browse/0.3.4/images/icons/cu/cornell-reduced-white-SMALL.svg)](https://www.cornell.edu/)

[Learn about arXiv becoming an independent nonprofit.](https://tech.cornell.edu/arxiv/)

We gratefully acknowledge support from the Simons Foundation, [member institutions](https://info.arxiv.org/about/ourmembers.html), and all contributors. [Donate](https://info.arxiv.org/about/donate.html)

[](/IgnoreMe)

[![arxiv logo](/static/browse/0.3.4/images/arxiv-logo-one-color-white.svg)](/) \> [cs](/list/cs/recent) \> arXiv:2510.15804

[Help](https://info.arxiv.org/help) | [Advanced Search](https://arxiv.org/search/advanced)

All fields Title Author Abstract Comments Journal reference ACM classification MSC classification Report number arXiv identifier DOI ORCID arXiv author ID Help pages Full text

 Search

[![arXiv logo](/static/browse/0.3.4/images/arxiv-logomark-small-white.svg)](https://arxiv.org/)

  [![Cornell University Logo](/static/browse/0.3.4/images/icons/cu/cornell-reduced-white-SMALL.svg)](https://www.cornell.edu/)

open search

   GO

open navigation menu

quick links
-----------

*   [Login](https://arxiv.org/login)
*   [Help Pages](https://info.arxiv.org/help)
*   [About](https://info.arxiv.org/about)

Computer Science > Computation and Language
===========================================

**arXiv:2510.15804** (cs)

\[Submitted on 17 Oct 2025\]

Title:Emergence of Linear Truth Encodings in Language Models
============================================================

Authors:[Shauli Ravfogel](https://arxiv.org/search/cs?searchtype=author&query=Ravfogel,+S), [Gilad Yehudai](https://arxiv.org/search/cs?searchtype=author&query=Yehudai,+G), [Tal Linzen](https://arxiv.org/search/cs?searchtype=author&query=Linzen,+T), [Joan Bruna](https://arxiv.org/search/cs?searchtype=author&query=Bruna,+

### 2026-04-03T13:32:34.058Z

<system-reminder>
The task tools haven't been used recently. If you're working on tasks that would benefit from tracking progress, consider using TaskCreate to add new tasks and TaskUpdate to update task status (set to in_progress when starting, completed when done). Also consider cleaning up the task list if it has become stale. Only use these if relevant to the current work. This is just a gentle reminder - ignore if not applicable. Make sure that you NEVER mention this reminder to the user

</system-reminder>

### 2026-04-03T13:43:27.611Z

Главное улучшение в том, что работа теперь выглядит не как разъехавшийся черновик, а как уже почти нормальный submission draft: в paper_neurips/main.tex стоит анонимный submission-режим NeurIPS (\usepackage{neurips_2026}, \author{Anonymous}), а отдельный checklist по анонимизации прямо описывает, как вычищать публичные идентификаторы из review zip, что соответствует политике NeurIPS по анонимному коду для review.

И по самому paper прогресс есть: claims стали уже, scope чаще ограничен “controlled contradictory corpora”, multi-rule действительно вынесен как mechanistic result, matched-control и generative eval теперь лучше встроены в narrative, а gzip наконец-то помечен как supporting evidence, а не как главный столп аргумента. Quick start тоже стал гораздо лучше синхронизирован с текущим MLX CLI.

Но до “теперь всё ок” я бы всё ещё не дошёл. Самые неприятные остатки такие:

Минимальное воспроизведение всё ещё сломано. scripts/reproduce_minimal.sh по-прежнему вызывает data/generate_math.py с устаревшими --correct-ratio, --error-type и --denoising, а текущие генераторы принимают --ratio и --error-mode; generate_paired_test.py тоже ждёт --error-mode, не --error-type. Как written, этот script не совпадает с актуальным кодом.
README всё ещё отправляет пользователя в битый pipeline. В разделе reproduction он советует bash scripts/run_wiki_experiment.sh, но этот script до сих пор использует старые аргументы --train_file, --output_dir, --model_dir, --test_file, --output_file, тогда как актуальные CLIs требуют --corpus, --output, --model-size, --weights, --tokenizer, --test-paired. Это значит, что публично advertised путь воспроизведения всё ещё нерабочий.
Packaging почти не улучшился. requirements.txt всё ещё содержит только 7 пакетов, а pyproject.toml по-прежнему выглядит как заглушка: placeholder description, requires-python = ">=3.14" и пустые dependencies. При этом repo заявляет PyTorch denoising, SentencePiece-style BPE и spaCy-based Wikipedia corruption, так что завис

### 2026-04-03T13:47:17.121Z

Залей все обновления еще раз отправлю

### 2026-04-03T14:19:32.936Z

Если публичный GitHub не идёт в submission package, а в статье будет только анонимизированный supplementary zip / anonymous repository, то большая часть моих прошлых претензий к публичному README, публичным script-ам и version drift в открытом репо действительно становится вторичной. Для NeurIPS критично, чтобы анонимными были сама submission PDF и любые submitted / linked supplementary materials; при этом публичные preprints допустимы, если submission остаётся анонимным и публичная версия не пишет “under review at NeurIPS”. Код для review, если подаётся, должен быть анонимизирован и идти отдельно в zip.

По самой статье стало заметно лучше. В paper_neurips/main.tex у тебя корректный submission-режим neurips_2026, анонимный автор, и в reproducibility appendix код описан как [anonymous repository, included as supplementary material]; отдельный ANONYMIZATION_CHECKLIST.md тоже уже соответствует идее анонимного code zip для review.

И по содержанию paper реально подтянулся. Claims теперь уже и аккуратнее: abstract и intro явно ограничивают выводы controlled contradictory corpora, а conclusion прямо говорит, что перенос на large-scale pretraining остаётся открытым вопросом. Статистика тоже стала существенно чище: caption таблицы теперь расшифровывает ± как std по seed-ам, в checklist уже написаны Wilcoxon signed-rank и bootstrap CI, а compute теперь везде согласован как примерно 150 GPU-hours. Лимитации тоже стали честнее: отдельно проговорены single-seed 1B result, domain specificity, discriminative/generative gap, seed counts и causal identification.

Ещё хороший знак: раньше у тебя compression appendix был логически кривой, а теперь он действительно содержит 9 conditions и подаётся как supporting evidence, а не как главный несущий столп. Плюс matched-control ablation уже находится в main text, что сильно лучше закрывает естественную атаку про “может, coherent corruption просто больше ломает шагов”.

Мой текущий вердикт такой: для статьи это уже не “сырой draft”, а “по

### 2026-04-03T17:15:32.192Z

<command-name>/status</command-name>
            <command-message>status</command-message>
            <command-args></command-args>

<local-command-stdout>Status dialog dismissed</local-command-stdout>

<command-name>/status</command-name>
            <command-message>status</command-message>
            <command-args></command-args>

<local-command-stdout>Status dialog dismissed</local-command-stdout>

<command-name>/status</command-name>
            <command-message>status</command-message>
            <command-args></command-args>

<local-command-stdout>Status dialog dismissed</local-command-stdout>

Если публичный GitHub не идёт в submission package, а в статье будет только анонимизированный supplementary zip / anonymous repository, то большая часть моих прошлых претензий к публичному README, публичным script-ам и version drift в открытом репо действительно становится вторичной. Для NeurIPS критично, чтобы анонимными были сама submission PDF и любые submitted / linked supplementary materials; при этом публичные preprints допустимы, если submission остаётся анонимным и публичная версия не пишет “under review at NeurIPS”. Код для review, если подаётся, должен быть анонимизирован и идти отдельно в zip.

По самой статье стало заметно лучше. В paper_neurips/main.tex у тебя корректный submission-режим neurips_2026, анонимный автор, и в reproducibility appendix код описан как [anonymous repository, included as supplementary material]; отдельный ANONYMIZATION_CHECKLIST.md тоже уже соответствует идее анонимного code zip для review.

И по содержанию paper реально подтянулся. Claims теперь уже и аккуратнее: abstract и intro явно ограничивают выводы controlled contradictory corpora, а conclusion прямо говорит, что перенос на large-scale pretraining остаётся открытым вопросом. Статистика тоже стала существенно чище: caption таблицы теперь расшифровывает ± как std по seed-ам, в checklist уже написаны Wilcoxon signed-rank и bootstrap CI, а compute теперь везде согласован как 

### 2026-04-03T17:25:42.170Z

Собери PDF

### 2026-04-03T17:33:55.497Z

Закомить и запуш

### 2026-04-03T17:36:20.532Z

Где находится PDF? Он анонимный?

### 2026-04-03T17:37:33.618Z

Мне нужен еще PDF для arXiv. Сделай из v3.md

### 2026-04-03T17:57:41.402Z

arxiv.pdf appears to have been produced by TeX

This file has been rejected as part your submission because it appears to be pdf generated from TeX/LaTeX source. For the reasons outlined at in the Why TeX FAQ we insist on submission of the TeX source rather than the processed version.

Our software includes an automatic TeX processing script that will produce PDF, PostScript and dvi from your TeX source. If our determination that your submission is TeX produced is incorrect, you should send e-mail with your submission number to arXiv administrators.

arXiv's automated TeX processing has failed to process your source. The 5 most common mistakes causing this are:

1. Mixed figure file formats. If you are using PDFLaTeX then all figures must be .pdf, .jpg, or .png formats. If your document uses (La)TeX all figures must be .ps or .eps. arXiv does not perform figure file conversion for you, please ensure your files are converted to the appropriate format before uploading.
2. Default hyperref failures ("Option clash for package hyperref") are not a reason to report a failure to arXiv. Continue scrolling in the log to find the specific errors that are being flagged.
3. Missing customized or differing version of style files.
4. Missing, misnamed, or local complete paths to figure files. arXiv's file system is case sensitive.

To upload or edit your files before reprocessing return to the add files step.

The TeX Log appears below.
Need help?
You may wish to hit "Ctrl + F" ("Command + F" on a Mac) to search for "error" on this page.
If you are unsure of what your specific failure is please review our help pages, knowledge base or searching the internet for your specific processing messages.
Or contact us citing submit/7440633. We will work with authors to help resolve TeX processing issues and if we determine that there is not a ready solution we will consider allowing PDF instead of TeX. We try to provide same day replies for submission help Monday-Friday.

### 2026-04-03T18:05:48.956Z

Напиши мне новый abstract и что сделано по сравнению с v2 коротко для comments. И новое название какое у нас

### 2026-04-03T18:11:48.180Z

Мне не нравится это название - можно ли его вернуть в более громкое? Или это плохо для подчаи?

### 2026-04-03T18:12:42.318Z

Да, сделай в версии для arxiv

### 2026-04-03T18:14:46.745Z

Закомить и запуш на гитхаб

### 2026-04-03T18:21:28.034Z

Напиши для v3 какие изменения по отношению к v2

### 2026-04-03T18:22:15.454Z

Давай покороче

### 2026-04-03T18:33:18.701Z

Заметил что почта не правильная. Мой адрес - k.krestnikov@gmail.com, исправь везде

### 2026-04-03T18:36:07.755Z

Запушил?

### 2026-04-03T18:42:12.569Z

Дай мне ссылку на мою работу на arxiv

### 2026-04-03T18:44:29.451Z

Иван, добрый день! Меня зовут Константин Крестников, мы с вами много раз пересекались по работе в Сбере. Я работаю над гигачатом в отделе SDK и автономных агентов. Меня заинтересовал вопрос - почему так получается, что модели видят противоречивую информацию в данных, но чаще всего выбирают научные/рациональные точки зрения на мир. Можно ли такое "стремление к правде" описать и измерить с научной точки зрения.

Провел ряд экспериментов по обучению моделей с нуля на датасетах составленных из разных математических теорий, например правильной и специально сломанной арифметик. Обнауржил набор интересных фактов:
- Модель способна выучить правильную арифметику даже в случае, когда корректных данных всего 10%
- Модель не способна отличить корректную и некорректную арифметику, если некорректная консистента
- Если в датасете есть несколько противоречивых арифметик, то модель вновь начинает выбирать правильные ответы

Про это я написал работу и опубликовал ее на arXiv: https://arxiv.org/abs/2603.11749

Хочу податься с ней на NeuroIPS и в дальнейшем написать диссертацию, раскрыв эту тему. Мне она кажется крайне интересной. Ищу ментора, который поможет пройти по пути PhD, расширить и углубить работу.

Был бы очень благодарен, если вы посмотрите и дадите обратную связь по ней Вот такое пишу, корректно ли тут все?

### 2026-04-04T04:09:56.077Z

Нельзя нигде говорить что "подано на NeurIPS" поэтому надо в репозитаории переименовать файлы и упоминаняия этой конференции. Назови просто for_conf

