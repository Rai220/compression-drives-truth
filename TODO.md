# TODO — after the NeurIPS 2026 rebuttal

**State on 2026-08-03.** Submission 473 reviewed (4, 4, 3, 2). The meta-review
named one condition for changing the decision: show that the effect is not
probability-mass splitting. It was measured; part of the paper survived and part
did not. The Official Comment answering reviewer rFTw was posted 2026-08-03
(`docs/official_comment_rFTw.md`) and is frozen. Notification 18 September,
camera-ready 23 October AoE.

`paper_conf/main.tex` is the **submitted** version and is frozen too — do not
edit it. The revision lives in `paper_conf/main_v2.tex`.

---

## What the controls established

Measured, verified against raw JSON, documented in `docs/mass_control.md`:

- **Survives.** At 86M the correct completion holds 59.0 ± 0.8% of the mass
  against the generator's full false support (4 seeds, every per-seed bootstrap
  interval above 0.5), replicated at 59.4 ± 0.6 on an independent corpus.
- **Survives, and is the robust half.** A single coherent false rule is not
  filtered at any size, architecture or corpus. One false mode, so splitting is
  impossible by construction.
- **Withdrawn.** The multi-rule N=1→2 jump is splitting: mass 47.9 → 47.8 while
  pair accuracy goes 47 → 79.
- **Withdrawn.** The description-length ladder shows no grading (86M family:
  55.6 / 51.9 / 52.6 / 51.3 across L = 1, 4, 16, 64).
- **Withdrawn.** "Coherent errors fall below chance" — that was completion
  length (−0.58 tokens; length-matched 46.7–49.3).
- **Narrowed.** Pair accuracy overstates the excess above chance roughly
  threefold at 86M, and shows an effect below it where the controlled measure
  shows none.

## Blocking for camera-ready

- [ ] **Trim `main_v2.tex` to the page limit.** Body currently runs to ~11 pages
      against a 9-page limit (10 for camera-ready). The control grids are already
      in Appendix C; the next candidates are Related Work and Section 4.4.
- [ ] **Decide the title.** `main_v2.tex` proposes "Correctness Preference Is
      Smaller Than It Looks: Probability Mass in Contradictory Training Corpora".
      The comment committed to narrowing the submitted one.
- [ ] **Keep the release promise.** The comment promises the corpora, the
      enumeration code and the per-run outputs behind both grids. Code and
      per-run JSON are now in git; the corpora (~207 MB under `data/corpus/`)
      are gitignored and need either a distribution channel or a regeneration
      recipe with published SHA-256 checksums. `docs/mass_control.md` has the
      commands.
- [ ] **Rebuild the anonymous supplement.** `submission/anonymous_supplement.zip`
      is from 2026-04-24 and contains none of the enumeration code, the new
      corpora or the control outputs. Re-run
      `scripts/prepare_anonymous_submission.sh`, then re-run the anonymity grep
      (`krestnikov|rai220|github|@gmail|2603.11749`) over whatever goes in.
- [ ] **Figures.** `figure_denoising_j1_j2.png` and `figure7_multirule.png` plot
      pair accuracy only and are no longer referenced by `main_v2.tex`;
      `figure_full_scaling.png` is still used and its caption now carries the
      pair-accuracy caveat. New figures come from
      `analysis/plot_mass_control.py` and `analysis/plot_multirule_control.py`.
- [ ] **arXiv v2.** The preprint (2603.11749) states the withdrawn claims and
      should be updated once `main_v2.tex` settles.

## Worth doing, not blocking

- [ ] **Clean-corpus control at Qwen3 sizes.** The one experiment that would
      settle whether the +5 points at 420M–1B in the held-out grid is a
      preference or a calibration offset of the family measure. Cheap: 4–5 sizes
      × 2 seeds, no new corpus family; `generate_scale.py` needs a clean mode.
      See `results_cluster/SCALING_LAW.md` §6.
- [ ] **Second seed for the 1B FineWeb-Edu run** (~15 h on Kaggle). Still the
      only single-seed cell in the paper.
- [ ] **Regenerate `scale_heldout_runs.csv` / `scale_heldout_summary.csv`** —
      they stopped at 71 runs and predate the 34 family runs.
- [ ] Name the statistical tests in Section 3.4, not only in the checklist.
- [ ] Related Work: data poisoning / backdoors (Gu, Wallace, Carlini) and one
      sentence on TruthfulQA-style evaluation. Both are standard reviewer asks
      and neither is present.
- [ ] Section 4.7 (1B): repeats, tokens, steps and warmup are still unstated.
- [ ] Ethical Considerations: name the data licences (Wikipedia CC BY-SA,
      FineWeb-Edu ODC-BY). They are in the checklist but not in the section.
- [ ] Name the exposure-bias / teacher-forcing gap and cite Bachmann &
      Nagarajan (2024).

## Security / hygiene

- [ ] **HF token in `.env`** was visible in shell output in April. The file is
      gitignored and not in history, but rotation status is unknown. Rotate at
      https://huggingface.co/settings/tokens if it has not been done.
- [x] Cluster scripts sanitised: `scripts/pull_cluster.sh` and
      `scripts/run_keyed_ladder_cluster.sh` no longer carry a private-key path,
      a hostname or another person's account name (2026-08-03).
- [ ] `paper_conf/poster/` stays untracked: it prints the arXiv id and describes
      a named-venue talk. Its `README.md` also promises `poster_datafest.*`
      files that do not exist.
- [ ] `data/generate_paired_test.py` at seed 999 no longer reproduces the
      committed `test_paired_random.jsonl` (the file matches its manifest hash;
      the generator drifted). Shipped artefacts are unaffected, but the
      documented regeneration command is wrong.

## Closed

- [x] Rebuttal to reviewers, and the Official Comment answering rFTw's
      identification objection (posted 2026-08-03).
- [x] Aggregate-mass measurement on three corpora (mixed, denoising, ladder)
      plus the clean-corpus calibration control.
- [x] Appendix numbering, bibliography artefacts, the 11M/12M inconsistency, the
      multi-rule table caption and the NeurIPS checklist — all fixed in the
      LaTeX version; they survive only in the dead `paper_v3.md`.
- [x] Submission, review, rebuttal window.

## Numbers that could not be verified from this repository (2026-08-03)

Found while rechecking `main_v2.tex` against the raw JSON. Neither blocks the
camera-ready, but both should be resolved before the release package ships.

- **`tab:generative`, tiny random-trained (30.5% ± 1.7%).** No
  `eval_generation_500.json` under `results/mixed_50_50_tiny_seed4*/`; every
  other cell of that table has one. The value survives only in the first
  version of the paper. Either rerun the four seeds or drop the row.
- **`tab:qwen` (Qwen3-0.6B, 3 seeds: 85.3/85.7/89.3 and 49.3/52.8/49.8).**
  Nothing in the repository reproduces these. The Modal runs behind them were
  never pulled, and the Qwen3-0.6B directories under `results_cluster/` belong
  to the held-out scaling grid and evaluate a different support.
