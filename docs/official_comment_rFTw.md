You were right about the keyed-offset ladder, and we withdraw it. We then applied the same control to the denoising experiment behind the paper's headline numbers, where the same channel was open and we had not closed it. There it holds at 86M.

The ladder was rebuilt so that the null needs no assumption. Each problem occurs twice in the corpus, once with the correct derivation and once with the derivation the false rule produces, and the measurement is taken on those training problems. Each checkpoint is scored against the single false string the corpus contains, and then against every string the rule could have produced for that problem: 1, 4, 9 and 18 strings at L = 1, 4, 16 and 64. Those other strings occur zero times in the corpus, so a frequency-matching model scores exactly 0.500 on both measures. Four sizes, four seeds, 1339 to 1342 problems per rung:

| params | measure | L=1 | L=4 | L=16 | L=64 |
|---|---|---|---|---|---|
| 3.5M | realized | 62.6 ± 2.2 | 69.2 ± 2.0 | 71.8 ± 2.0 | 71.9 ± 1.2 |
| 3.5M | family | 62.6 ± 2.2 | 40.5 ± 2.5 | 30.8 ± 1.4 | 22.3 ± 1.2 |
| 12M | realized | 63.0 ± 1.4 | 72.6 ± 1.4 | 80.7 ± 0.9 | 81.8 ± 1.2 |
| 12M | family | 63.0 ± 1.4 | 49.6 ± 1.9 | 45.1 ± 1.8 | 36.2 ± 1.8 |
| 26M | realized | 57.5 ± 1.2 | 68.0 ± 0.9 | 77.4 ± 1.6 | 81.6 ± 1.4 |
| 26M | family | 57.5 ± 1.2 | 50.8 ± 1.6 | 51.8 ± 1.8 | 43.9 ± 1.9 |
| 86M | realized | 55.6 ± 0.7 | 60.7 ± 1.8 | 67.2 ± 0.9 | 71.8 ± 1.2 |
| 86M | family | 55.6 ± 0.7 | 51.9 ± 1.6 | 52.6 ± 0.6 | 51.3 ± 1.3 |

The realized rows reproduce what we reported (55.6 to 71.8 at 86M against 53.3 to 67.9 in the two-seed run). Summing the family removes the rise and reverses it below 86M. Completion lengths differ by at most 0.04 tokens and the equal-length subset gives the same values. Trained on a corpus with the same problems and no false rule anywhere, the same measure reads 99.7 at 86M and 64.1 at 3.5M, so at the top size it is close to unbiased while at the bottom a third of the mass leaks onto strings that never occurred. No grading in description length is present at any size

The table we sent on 27 July scores the correct completion against the false string in the corpus, and the channel you describe is open there too: under random errors the model can put mass on corruptions that never occurred, under coherent errors it cannot. We enumerated the full corruption support of that generator on the published corpora, verified that the error the corpus contains is a member of the enumeration for all 5000 problems, and kept only problems whose correct and false strings each occur once with no other corruption of the same problem anywhere in the corpus. That leaves 2569 problems carrying 12 false modes each, 11 of which never occur. Four seeds:

| params | realized (1 mode) | family (12 modes) | coherent |
|---|---|---|---|
| 3.5M | 65.5 ± 0.9 | 18.6 ± 1.1 | 43.5 ± 0.4 |
| 12M | 76.0 ± 0.9 | 31.6 ± 1.2 | 45.8 ± 1.1 |
| 26M | 80.3 ± 0.8 | 44.1 ± 0.7 | 46.4 ± 1.5 |
| 86M | 77.6 ± 0.2 | 59.0 ± 0.8 | 48.8 ± 0.4 |

At 86M the correct completion holds 59.0 percent of the mass against everything the error generator can produce, with every per-seed bootstrap interval above 0.5: [58.7, 60.4], [57.5, 59.3], [57.2, 59.0], [59.0, 60.7]. Splitting cannot raise that number, and since adding strings to the sum can only lower it, 59.0 is a lower bound. Two things in the table are not in our favour. The measure does not show the effect below 86M (18.6, 31.6, 44.1; on the equal-length subset it crosses earlier, 33.1 / 52.2 / 55.5 / 57.4). And the sub-chance coherent column is mostly completion length: those false strings are 0.58 tokens shorter, and on equal length the column reads 46.7 / 48.3 / 48.1 / 49.3, approaching chance from below rather than sitting under it. Section 4.1 will be corrected. The enumerated mixed-corpus measurement already in our rebuttal, 57 to 59 percent from 12M upward, is the same quantity on a different corpus and agrees at 86M, 59.4 against 59.0

What we will claim is therefore narrower. Preference is not graded in the description length of the false system; both experiments built to show that turned out to measure splitting, and we will not present compression as an identified mechanism. A false system whose output the model cannot predict from the problem loses about ten points of probability mass to the correct one at 86M, a coherent one loses none at any scale we tested, and pair accuracy overstates the gap roughly threefold. The claim of an effect at every scale goes, as does the reading of coherent errors as falling below chance. The title and the central claims will be narrowed accordingly, with compression stated as an interpretation rather than a measured quantity, and the Wikipedia and 1B narrowings from the rebuttal stand. The corpora, the enumeration code and the per-run outputs behind both grids will be released with the paper
