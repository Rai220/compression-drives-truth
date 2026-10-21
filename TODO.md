# TODO — after the NeurIPS 2026 rebuttal

**State on 2026-10-21.** Camera-ready prepared in `paper_conf/main_v2.tex` (body ends on page 10 of the `[main, final]` build; 29 pages total; see `CHANGELOG` in the crew_lora experiment folder `expirements/2026-10-21-neurips-camera-ready/`). Deadline 23 October AoE. Remaining before upload: author confirms the acknowledgments/funding wording (`% AUTHOR:` comment in the source) and the hardware label (run record says Apple M4 Pro 48 GB, the first version said 36 GB), then uploads PDF + checklist. Earlier state follows.

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
- **Narrowed.** At 86M the pairwise measures overstate the excess above chance
  by three to four times (realized share 27.6 pp, pair accuracy 37.4 pp, family
  9.0 pp), and below 86M they show an effect where the controlled measure shows
  none.

## Blocking for camera-ready

- [x] **Trim `main_v2.tex` to the page limit.** DONE 2026-10-21: body ends on page 10 in the `[main, final]` build (acknowledgments and references start on page 10, 29 pages in all). Tables 1, 2, 7 and the pair-accuracy scaling figure moved to the appendices; Tables 6 and 14 merged; the K-ratio result on the controlled measure replaced the pair-accuracy noise paragraph; related work, discussion and limitations compressed. Numbers re-verified from per-run JSON (corrections listed in the experiment CHANGELOG: learnable realized 66.3/65.1, learnable_mod 2.16B one seed, per-entity-type means over 16 runs, frequency-sweep random cells without evaluation files dropped).
- [x] **Decide the title.** Decided 2026-09-25: keep the accepted NeurIPS title,
      "Truth as a Compression Artifact in Language Model Training", in the paper,
      README and GitHub. The alternative "Correctness Preference Is Smaller Than It
      Looks" is dropped. The official comment promised to narrow the title, so the
      narrowing now has to be carried by the abstract and claims.
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
- [ ] **Figures.** `figure_denoising_j1_j2.png`, `figure7_multirule.png` and (since the camera-ready, 2026-10-21) `figure_full_scaling.png` plot
      pair accuracy only and are no longer referenced by `main_v2.tex`; only `figure_mass_control.png` and `figure_multirule_control.png` are used. New figures come from
      `analysis/plot_mass_control.py` and `analysis/plot_multirule_control.py`.
- [x] **Fix the "1B" label everywhere** (paper: done; camera-ready says 490M; arXiv v2 still pending). The committed `qwen3-1b` config
      instantiates to **473,593,856** parameters (vocab 53), not ~1B; the
      FineWeb-Edu run (BPE-8000) is 489,869,312. `main_v2.tex` is corrected
      (2026-08-11: abstract, §4.7, checklist, reproducibility — which also
      wrongly said Modal A100 where the run records say Kaggle T4);
      `SCALING_LAW.md` carries a correction note. arXiv v2 must inherit this.
- [x] **Clean-corpus control at Qwen3 sizes.** DONE 2026-08-11 on the
      single-H100 box. `generate_scale.py --errors clean`, grid
      `scale_clean_{large,xlarge,qwen3-0.6b,qwen3-1b} × seeds 42–43`. Clean
      model reads (random family support): 80.9 / 91.7 / 99.4 / 99.2 full,
      91.0 / 92.2 / 97.9 / 97.2 eqlen — leak monotone in scale and **downward**,
      so the §4 above-null cells (~+5) are a real preference, not a calibration
      offset, and if anything understated. Written into `SCALING_LAW.md` §6 and
      §5; per-run JSON in `results_cluster/results/scale_clean_*`. The held-out
      grid can now go into the paper. Remaining polish: a GPT-2 `xxlarge`
      (413.7M) clean grid is running to separate architecture from scale in the
      leak (202M GPT-2 leaks ~8 pp vs <1 pp at Qwen sizes).
- [ ] **arXiv v2.** The preprint (2603.11749) states the withdrawn claims and
      should be updated once `main_v2.tex` settles.

## Worth doing, not blocking

- [ ] **Aggregate share on Wikipedia, via single-entity paragraphs.** The
      largest remaining gap: natural language currently has no usable
      measurement at all for the random condition, only for coherent. The false
      support is the product of entity pools over every entity site, and the
      evaluation set has a median of 8 sites and no paragraph with fewer than 2
      (`analysis/wiki_length_check.py`). Regenerating the eval set restricted to
      single-entity paragraphs makes the support exactly that entity type's
      pool — a few hundred to a few thousand strings — summable at one forward
      pass per pool member, with the same exact 0.5 null as the math corpora.
      Add a clean-Wikipedia calibration run alongside, as `ladder2_clean` does.
- [x] **Second seed for the 490M FineWeb-Edu run** — DONE 2026-08-12 on the
      H100 box. random 0.768/0.783 → 0.776 ± 0.008, coherent 0.467/0.482 →
      0.474 ± 0.007; reproduces the first (Kaggle T4) across seed and backend.
      `main_v2.tex` §4.7, fig:qwen caption, checklist all updated to 2 seeds.
      Per-run JSON in `results_cluster/results/fineweb_*_seed43/`.
- [x] **Regenerate `scale_heldout_runs.csv` / `scale_heldout_summary.csv`** —
      done 2026-08-11 via the new `scripts/collect_scale_heldout.py` (106 rows,
      every measurement labelled with its support; values re-checked against
      `SCALING_LAW.md`).
- [x] Name the statistical tests in Section 3.4, not only in the checklist. (camera-ready, 2026-10-21)
- [x] Related Work: data poisoning / backdoors (Gu 2017, Wallace 2021, Carlini 2024, Souly 2025) and TruthfulQA (Lin 2022) added in the camera-ready (2026-10-21).
- [x] Section 4.7 (1B): repeats, tokens, steps and warmup stated in the camera-ready (15,000 steps, batch 4, seq 512, 200 warmup steps, ≈31M tokens, 2 seeds).
- [x] Ethical Considerations: licences named in the section (camera-ready, 2026-10-21).
- [x] Name the exposure-bias / teacher-forcing gap and cite Bachmann &
      Nagarajan (2024). (camera-ready, 2026-10-21)

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
      LaTeX version; the markdown drafts that still carried them are deleted.
- [x] Submission, review, rebuttal window.

## Numbers that could not be verified from this repository (2026-08-03)

Found while rechecking `main_v2.tex` against the raw JSON. Neither blocks the
camera-ready, but both should be resolved before the release package ships.

- **`tab:generative`, tiny random-trained (30.5% ± 1.7%).** RESOLVED
  2026-08-11: rerun on the four surviving checkpoints gives **31.1% ± 1.4%**
  (33.4/31.2/30.2/29.6; arithmetic 0.0% in all seeds), consistent with the
  printed value; `eval_generation_500.json` now exists for all four seeds and
  `main_v2.tex` cites the re-derived number.
- **`tab:qwen` (Qwen3-0.6B).** RESOLVED 2026-08-12: reran the original protocol
  on the H100 box (`scripts/run_qwen_mixed.sh`, 3 seeds × 2 conditions). Pair
  accuracy reproduces the lost-artefact numbers — random 87.3 ± 0.4 (orig mean
  86.8), coherent 49.4 ± 1.3 (orig 50.6) — now with per-run JSON in
  `results_cluster/results/qwenmix_*`. Added the aggregate-mass evaluation the
  original never had: random 56.3 ± 0.3 (above the 50 null), coherent
  49.7 ± 0.0 (at null), so the controlled measure shows the contrast at 420M on
  the mixed corpus too. `main_v2.tex` §4.7 updated with both the reproduced
  pair accuracy and the new aggregate share; the "same denoising corpora"
  wording was already corrected to standard construction.
