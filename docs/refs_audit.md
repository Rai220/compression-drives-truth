# Reference Audit — paper_draft_ru.md

Verified 2026-03-05 via web search. Convention: OK = verified, FIX = needs correction, CHECK = uncertain.

## Results

| # | Citation in paper | Status | Issue |
|---|---|---|---|
| 1 | Azaria & Mitchell (2023). The Internal State of an LLM Knows When It's Lying. *Findings of EMNLP 2023*. | OK | |
| 2 | Burger, L., Hamprecht, F. A., & Nadler, B. (2024). Truth is Universal. *NeurIPS 2024*. | **FIX** | Author surname is **Bürger** (with umlaut): Lennart Bürger |
| 3 | Burns, C., Ye, H., Klein, D., & Steinhardt, J. (2023). Discovering Latent Knowledge... *ICLR 2023*. | OK | Was ICLR 2024 in Related Work table; already fixed to 2023 |
| 4 | Chlon, L., et al. (2025). Predictable Compression Failures... *arXiv:2509.11208*. | **FIX** | Title was updated in v2 to "...Order Sensitivity and Information Budgeting for Evidence-Grounded Binary Adjudication". Original v1 title is used in paper. Authors: Leon Chlon, Ahmed Karim, Maggie Chlon — "et al." is fine but could list all 3. |
| 5 | Delétang, G., et al. (2024). Language Modeling Is Compression. *ICLR 2024*. | OK | |
| 6 | DeMoss, J., Nanda, N., & Radhakrishnan, A. (2024). Grokking as a Phase Transition from Memorization to Compression. *ICML 2024*. | **FIX** | **Reference appears fabricated/confused.** No such paper found at ICML 2024. The actual DeMoss paper is: DeMoss, B., Sapora, S., Foerster, J., Hawes, N., & Posner, I. (2024). "The Complexity Dynamics of Grokking." arXiv:2412.09810. Different title, different co-authors, NOT at ICML. Neel Nanda is not a co-author. **Recommend: replace with correct reference or remove.** |
| 7 | Goldblum, M., Finzi, M., Rowan, K., & Wilson, A. G. (2024). The No Free Lunch Theorem... *ICML 2024*. | OK | Confirmed as ICML 2024 Spotlight. Note: it's a Position paper. |
| 8 | Gurnee, W., & Tegmark, M. (2024). Language Models Represent Space and Time. *ICLR 2024*. | OK | |
| 9 | Halawi, D., Denain, J.-S., & Steinhardt, J. (2024). Overthinking the Truth... *ICLR 2024*. | OK | ICLR 2024 Spotlight confirmed. |
| 10 | Huang, Y., Sun, Y., Wang, X., & Yang, Y. (2024). Compression Represents Intelligence Linearly. *COLM 2024*. | OK | |
| 11 | Joshi, N., Rando, J., Saparov, A., Kim, N., & He, H. (2024). Personas as a Way to Model Truthfulness... *EMNLP 2024*. | OK | |
| 12 | Kadavath, S., et al. (2022). Language Models (Mostly) Know What They Know. *arXiv:2207.05221*. | OK | |
| 13 | Li, K., Hopkins, A. K., Bau, D., et al. (2023a). Emergent World Representations... *ICLR 2023*. | OK | |
| 14 | Li, K., Patel, O., et al. (2023b). Inference-Time Intervention... *NeurIPS 2023*. | OK | |
| 15 | Lin, S., Hilton, J., & Evans, O. (2022). TruthfulQA... *ACL 2022*. | OK | |
| 16 | Liu, Z., Zhong, Z., & Tegmark, M. (2023). Grokking as Compression... *arXiv:2310.05918*. | OK | arXiv preprint. |
| 17 | Marks, S., & Tegmark, M. (2023). The Geometry of Truth... *arXiv:2310.06824*. | OK | arXiv preprint (no venue publication found). |
| 18 | Mingard, C., Valle-Pérez, G., et al. (2021). Is SGD a Bayesian Sampler? *JMLR 2021*. | OK | |
| 19 | Nanda, N., Chan, L., et al. (2023). Progress Measures for Grokking... *ICLR 2023*. | OK | |
| 20 | Pan, L., Wang, H., & Li, B. (2025). Understanding LLM Behaviors via Compression. *arXiv:2504.09597*. | **FIX** | **Wrong author names and initials.** Actual authors: **Pan, Z. (Zhixuan), Wang, S. (Shaowen), & Li, J. (Jian)**. Not Pan L., Wang H., Li B. |
| 21 | Popper, K. (1959). The Logic of Scientific Discovery. *Hutchinson*. | OK | |
| 22 | Ravfogel, S., Elazar, Y., Goldberg, Y., & Belinkov, Y. (2025). Emergence of Linear Truth Encodings... *NeurIPS 2025*. | **FIX** | **Wrong co-authors.** Actual authors: Ravfogel, S., **Yehudai, G., Linzen, T., Bietti, A., & Bruna, J.** (not Elazar, Goldberg, Belinkov). arXiv:2510.15804. NeurIPS 2025 confirmed. |
| 23 | Valle-Pérez, G., et al. (2019). Deep Learning Generalizes... *ICLR 2019*. | OK | |
| 24 | Wan, J., & Mei, L. (2025). Large Language Models as Computable Approximations to Solomonoff Induction. *arXiv:2505.15784*. | OK | |

## Summary of required fixes

### Critical (wrong information)
1. **DeMoss et al.** — fabricated/confused reference. No such paper at ICML 2024 with those authors. Replace or remove.
2. **Pan et al.** — wrong author initials: Pan Z., Wang S., Li J. (not L., H., B.)
3. **Ravfogel et al.** — wrong co-authors: Yehudai, Linzen, Bietti, Bruna (not Elazar, Goldberg, Belinkov)

### Minor
4. **Bürger** — umlaut missing in surname (Burger → Bürger)
5. **Chlon et al.** — title updated in later arXiv version; using v1 title is acceptable but should note

## Cited but not in reference list
- Shannon (1948) — cited in text but not in reference list
- Solomonoff (1964) — cited in text but not in reference list
- Hutter (2005) / AIXI — cited in text but not in reference list

## In reference list but not cited in text
- Goldblum et al. (2024) — appears only in reference list, not cited in text body
- Liu, Zhong, & Tegmark (2023) — appears only in reference list, not cited in text body
