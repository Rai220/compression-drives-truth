# Working notes for the rFTw comment — not for posting

Artefacts behind every number in `official_comment_rFTw.md`:

* 2026-08-01, `results_cluster/results/ladder2_*`: 64 ladder runs (4 rungs x 4
  sizes x 4 seeds) plus 8 clean-corpus control runs. No failures.
* 2026-08-02, `results_cluster/results/denoise2_*`: 32 denoising runs
  (2 conditions x 4 sizes x 4 seeds). No failures.
* `results/aggregate_mass_summary.md`: the mixed-corpus measurement from
  2026-07-27, already reported in the rebuttal.
* Generation and verification: `analysis/gen_denoising_support.py`,
  `data/generate_math_keyed.py`, `analysis/summarize_ladder.py`.
* Corpus identity: `train_denoise_j1.txt` sha256
  5aa78dda90760bc980526cd2cd9ce0f65e251f4bcf4b664f6b7e7ae31e77f96f,
  `train_denoise_j2.txt` f8c1da963523d4ff50f77e19774934aa64d0f105e0bd15629db20946637bc28a,
  both matching the published corpora.

## Left out of the comment on purpose

The keyed condition of the scaling analysis fails the same control at every
size: realized 67.4 / 74.1 / 77.9 / 83.2 against family 13.3 / 24.6 / 30.4 /
37.8, and 93.0 against 49.8 at 1B (`results_cluster/results/scale2_keyed_*`,
`results_cluster/results/scale_keyed_qwen3-1b_seed4*/heldout_family.json`).

That work is not part of the submission and was not asked about, so it does not
belong in a comment on the submission. Two conditions attach to leaving it out:
it goes into the camera-ready with the family numbers, and it is never cited
anywhere as a positive result. On a held-out problem the model cannot compute
the key, so the condition is not a single-mode rule from the model's side, and
the keyed-versus-coherent gap cannot be claimed.

## Open item

The comment promises release of the corpora, the enumeration code and the
per-run outputs for both grids. That promise has to be kept in the camera-ready
package. The anonymity grep in `TODO.md` is recorded there as done for the April
package; it has to be re-run on whatever is attached at camera-ready.

## Discrepancies found after the comment was posted (2026-08-03)

Every table cell in the comment was recomputed from the raw JSON and reproduces.
Four wordings do not, and none of them changes a conclusion. They are recorded
here so the camera-ready does not repeat them.

1. **"2569 problems carrying 12 false modes each."** 12.03 is the *mean*
   (`mean_false_modes`, identical in all 16 `denoise2_j1` family runs). The
   per-problem count ranges 4 to 35; the modal value is 6 (1286 of 2569) and
   only 19 problems carry exactly 12. Correct wording: "a mean of 12.0 false
   modes per problem (range 4–35)".
2. **"2569 problems" attributed to the strict filter.** The strict filter itself
   leaves 3004 problems; the evaluator then drops those whose prompt plus longest
   completion exceeds the 256-token window, leaving 2569. Two filters, not one.
3. **"Completion lengths differ by at most 0.04 tokens."** The maximum
   `|length_delta_mean|` over the 128 ladder JSONs is 0.0432, at `L=1`. "At most
   0.05" is safe. Likewise "the equal-length subset gives the same values" is
   loose: the two differ by up to 1.1 points (86M `L=4`: 51.9 → 50.8).
4. **"55.6 to 71.8 at 86M against 53.3 to 67.9 in the two-seed run."** The
   4-seed values reproduce exactly; the 53.3/67.9 pair comes from a two-seed run
   whose outputs are not on disk anywhere in this tree. Either recover the run
   or drop the comparison.
5. **In these notes:** "93.0 against 49.8 at 1B" is seed 42 alone. Both seeds
   give 92.7 ± 0.3 and 49.1 ± 0.8. Every other figure in these documents is a
   multi-seed mean.

## The keyed condition: stronger reason to leave it out

The note above said the keyed condition "fails the same control at every size".
The correction runs deeper than that: the keyed *realized* measure has a
frequency-matching null of E[n/(n+1)] = 0.94, not 0.5, because the selecting
hash is not computable from an unseen problem. So the keyed-versus-coherent gap
compares a 0.94 null against a 0.50 null. Scored on a common support, keyed and
random are the same measurement to within a point at every size. The condition
enters the camera-ready with the family numbers and the null spelled out, and is
never cited as a positive result — that standing condition is unchanged and now
better founded. Details in `results_cluster/SCALING_LAW.md`.
