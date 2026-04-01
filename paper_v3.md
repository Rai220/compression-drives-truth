# Compression Favors Consistency, Not Truth: When and Why Language Models Prefer Correct Information

## Abstract

Language models trained on noisy, contradictory data nonetheless tend to prefer correct answers -- a property often attributed to scale, RLHF, or data frequency. Yet why the training objective itself would favor truth over plausible falsehood remains unclear. We hypothesize that next-token prediction, as a compression process, privileges whichever answer cluster is most compressible -- and that truth benefits only when errors lack internal structure. To test this, we train GPT-2 style transformers (3.5M--86M parameters) from scratch on controlled corpora where the same problem appears with both correct and incorrect solutions, varying the structure of errors. When errors are random, models extract the correct signal with paired accuracy scaling from 65% to 85%. When errors follow a single coherent alternative rule, accuracy falls to chance (~45--51%). A multi-rule experiment pinpoints the boundary: one systematic wrong rule eliminates truth bias entirely, while two competing rules restore it (47% -> 78%). The same contrast reproduces on real Wikipedia text (71% vs 46%). These results identify error structure -- not truth per se -- as the variable that determines model preference, and suggest that internally consistent misinformation may resist compression-based filtering.

---

## 1. Introduction

Language models are increasingly deployed in settings where factual reliability matters: medical question-answering, legal analysis, scientific reasoning. A growing body of work has found that these models develop internal representations correlated with truthfulness (Burns et al., 2023; Marks & Tegmark, 2023; Burger et al., 2024) and that calibrated models must hallucinate at a rate bounded by the corpus's monofact rate (Kalai & Vempala, 2024). These findings raise a question with direct implications for AI safety: does the training objective itself -- next-token prediction on noisy data -- create a systematic preference for truth? And if so, under what conditions does that preference hold or fail?

Several mechanisms have been proposed. Scaling improves factual performance (Kadavath et al., 2022). RLHF steers outputs toward human preferences. Frequency matters: factual accuracy correlates with how often correct information appears in training data (Elazar et al., 2022; Joshi et al., 2024; Kandpal et al., 2023). Yet none of these explain why compression itself would favor one answer over another when both appear at equal frequency in the same format.

We propose that the answer lies in the structure of errors, not in truth per se. Minimizing cross-entropy is equivalent to minimizing code length (Shannon, 1948; Deletang et al., 2024), linking LLM training to the Minimum Description Length principle (Rissanen, 1978; Grunwald, 2007). A correct rule system compresses into a compact representation; diverse errors must be memorized individually. But a *coherent* false system -- internally consistent, just wrong -- compresses equally well, and the preference vanishes.

We test this hypothesis directly. In our denoising design, each mathematical problem appears in the training corpus with both correct and incorrect solutions -- modeling the real-world scenario where the same question receives conflicting answers. Four conditions vary the structure of errors (Section 3). Random errors produce strong truth bias scaling from 65% to 85% with model size. Coherent errors produce none. A multi-rule experiment reveals a sharp crossover: one wrong rule eliminates bias entirely, two restore most of it (Section 4.3). The pattern reproduces on real Wikipedia text (Section 4.4).

Our contribution is a single experimentally validated hypothesis: **compression does not favor truth -- it favors consistency.** Truth bias is a side effect of error incoherence. We identify the precise conditions under which this bias emerges, disappears, and can be restored, across model sizes (3.5M--86M), error structures, and domains.

## 2. Related Work

**Prediction as compression.** The link between prediction and compression traces to Shannon (1948) and was formalized by Solomonoff (1964) and Rissanen's MDL principle (1978; Grunwald, 2007). Deletang et al. (2024) showed that LLMs are universal compressors. Huang et al. (2024) discovered a linear correlation (r ~ -0.95) between compression quality and benchmark performance. Wan & Mei (2025) proved that LLM training approximates Solomonoff induction. Pan et al. (2025) linked compression to knowledge acquisition and scaling; Chlon et al. (2025) linked compression failures to hallucination. We build on the MDL framework by experimentally varying the description length of false answer systems.

**Internal representations and world models.** Compression can give rise to structured internal representations: Li et al. (2023a) found board representations in an Othello model, Gurnee & Tegmark (2024) discovered linear space-time representations in Llama-2, and several studies found truth-correlated representations (Marks & Tegmark, 2023; Burns et al., 2023; Li et al., 2023b; Azaria & Mitchell, 2023; Ravfogel et al., 2025). Our work operates at the behavioral level: we identify data-level conditions under which compression produces a preference for correct completions, leaving activation-level analysis for future work.

**Truthfulness and data statistics.** Joshi et al. (2024) linked truthfulness to "personas" in pretraining data. Elazar et al. (2022) demonstrated frequency dependence. Kandpal et al. (2023) showed a direct relationship between document count and accuracy. Unlike these frequency-based analyses, we fix frequency and vary error structure, isolating compressibility as the operative variable.

**Simplicity bias and noisy labels.** Neural networks prefer simple functions (Valle-Perez et al., 2019; Mingard et al., 2021; Goldblum et al., 2024). The noisy labels literature directly parallels our setup: Zhang et al. (2017) showed memorization of random labels; Rolnick et al. (2017) showed robustness to massive label noise. Grokking connects to compression as a memorization-to-generalization transition (Nanda et al., 2023; Liu et al., 2023). Our experiments extend this line by showing that "structured noise" -- coherent errors -- is not filtered out. To our knowledge, systematic variation of error compressibility in a denoising setting has not been studied before.

## 3. Methodology

### 3.1 Hypothesis and Experimental Design

We test the following hypothesis: *next-token prediction favors whichever answer system has lower description length, regardless of its truth value.* Truth benefits only when errors are structurally incoherent and thus expensive to encode.

Consider a corpus where each problem appears with a correct answer (theory $T_1$) and an alternative ($T_2$). An MDL learner (Rissanen, 1978; Grunwald, 2007) minimizes $L(M) + L(D|M)$. When $K(T_2) \gg K(T_1)$ (random errors), the false system's description length grows with corpus size, favoring $T_1$. When $K(T_2) \approx K(T_1)$ (coherent errors), both are compact and the learner has no basis to prefer one over the other. With N competing false rules, a random "selector" (which rule applies to which problem) adds $\log N$ bits per problem, progressively degrading the false cluster's compressibility.

We design four denoising conditions to test these predictions (Section 3.3). The multi-rule experiment (Section 4.3) probes the boundary between compressible and incompressible error regimes.

Three methodological caveats apply. First, "truth" here means correctness of mathematical derivations and factual accuracy of Wikipedia paragraphs -- models compress text, not reality. Second, frequency can override compressibility when errors dominate sufficiently (Appendix C). Third, the compressibility gap is corpus-dependent and smaller in natural language than in formal math (Section 4.4).

### 3.2 Models and Training

GPT-2 style decoder-only transformers with pre-norm, GELU activation, and causal mask:

| Config | Layers | d_model | Heads | Parameters |
|--------|--------|---------|-------|------------|
| tiny | 4 | 256 | 4 | ~3.5M |
| small | 6 | 384 | 6 | ~12M |
| medium | 8 | 512 | 8 | ~26M |
| large | 12 | 768 | 12 | ~86M |

Optimizer: AdamW (weight_decay=0.01), cosine decay with linear warmup, lr=3e-4, seq_len=256, batch_size=32, 5000 steps. All experiments use 4 random seeds for core conditions (2 where noted). Learning curves confirm behavioral plateau by step 3000--4000 at all sizes (Appendix E). Denoising experiments use PyTorch; multi-rule and Wikipedia experiments use MLX. All problems are generated and verified by SymPy; no human annotation is involved. Full generation templates are provided in the code repository.

### 3.3 Corpus Design

**Denoising setup (primary).** A generator creates mathematical problems of four types: multi-step arithmetic, factoring, equation solving, and differentiation, formatted as step-by-step derivations in English. The key design: **each problem appears with contradictory answers** -- modeling the scenario where the same question receives conflicting responses. The tokenizer is character-level (vocab ~57); BPE robustness is verified in Appendix E.

| Condition | Correct | Incorrect | Ratio | Purpose |
|-----------|:-------:|:---------:|:-----:|---------|
| Equal random | 1 | 1 random | 1:1 | Signal extraction from random noise |
| Equal coherent | 1 | 1 coherent | 1:1 | Control: coherent errors eliminate bias |
| 2:1 noise | 1 | 2 random | 1:2 | Noise tolerance at moderate noise |
| 4:1 noise | 1 | 4 random | 1:4 | Noise tolerance at high noise |

Each denoising corpus contains 5,000 unique problems. Standard (non-denoising) experiments where each problem appears once are reported in Appendix C.

**Wikipedia setup (transfer).** To test generalization beyond formal math, we construct corpora from 20,000 Wikipedia paragraphs. Using spaCy NER, we create two corruption modes: *random substitution* (each entity replaced with a random entity of the same type) and *coherent substitution* (a consistent global mapping, e.g., every "France" -> "Japan"). Models train on 50/50 mixes; evaluation uses 2,000 held-out paired paragraphs.

### 3.4 Evaluation

**Paired evaluation (primary).** For each problem, a shared prompt is generated with two completions (correct and incorrect). NLL is computed on completion tokens only, conditioned on the prompt. The primary metric is **pair accuracy**: the fraction of pairs where the model assigns lower NLL to the correct completion. This is equivalent to the Common Language Effect Size (CLES; McGraw & Wong, 1992). Length-matched evaluation confirms that accuracy is not driven by completion length differences (within 1 pp across all conditions).

**Generative evaluation.** Greedy decoding on 500 test prompts with automated SymPy verification. This confirms that the discriminative effect extends to generation (Section 4.5).

---

## 4. Results

### 4.1 The Central Contrast: Random vs Coherent Errors

**Table 1.** Denoising paired evaluation: equal random vs equal coherent errors, across model sizes.

| Size | Params | Random Accuracy | Coherent Accuracy | Random Seeds | Coherent Seeds |
|------|--------|:-----------:|:-----------:|:--------:|:--------:|
| tiny | 3.5M | **65.3% +/- 1.3%** | 43.5% +/- 2.6% | 4 | 4 |
| small | 12M | **74.6% +/- 1.6%** | 44.5% +/- 3.0% | 4 | 4 |
| medium | 26M | **81.1% +/- 1.2%** | 45.8% +/- 3.4% | 4 | 4 |
| large | 86M | **85.2% +/- 2.3%** | 51.0% +/- 0.8% | 2 | 2 |

![](results/figure_denoising_j1_j2.png)

*Figure 1. The central contrast. Random errors: accuracy scales from 65% to 85%. Coherent errors: accuracy stays near chance across all sizes.*

When the same problem appears with both a correct and a random wrong answer, the model learns to prefer the correct one -- accuracy reaches 85% at 86M parameters. When the wrong answer follows a coherent rule system, the effect disappears: accuracy hovers near 50% at all scales. This is the paper's central result.

Random accuracy increases monotonically with scale (65% -> 75% -> 81% -> 85%). Coherent accuracy converges toward 50% from below, consistent with the MDL prediction that equally compressible systems at equal frequency should be indistinguishable. The below-chance coherent accuracy at tiny (~43%) reflects textual simplicity of the false rules (e.g., dropping terms in derivatives); the compressor prefers whichever system is simpler to encode. The multi-rule experiment (Section 4.3) controls for this surface-form effect.

### 4.2 Noise Tolerance

**Table 2.** Paired accuracy at increasing noise ratios.

| Condition | Ratio | Tiny | Small | Medium | Large |
|-----------|:-----:|:----:|:-----:|:------:|:-----:|
| Equal random | 1:1 | 65.3% | 74.6% | 81.1% | 85.2% |
| 2:1 noise | 1:2 | 59.0% | 68.6% | 73.6% | 75.2% |
| 4:1 noise | 1:4 | 56.6% | 64.9% | 66.6% | 65.8% |

![](results/figure_denoising_noise_curve.png)

*Figure 2. Noise tolerance. Higher noise ratios reduce accuracy and produce earlier capacity-dependent plateaus.*

Increasing the noise ratio degrades accuracy gracefully. At 4:1 noise, accuracy plateaus at medium/large (~66%), indicating a capacity-dependent signal-to-noise bottleneck. At 2:1, growth continues through large (75%). The correct signal persists because each random error is unique, preserving the compressibility advantage of truth -- but this advantage has limits when noise overwhelms the signal.

### 4.3 Multi-Rule Errors: The Sharp Crossover

The denoising experiments establish two poles: one coherent rule yields chance, random errors yield strong bias. We probe the boundary by training on corpora with N alternative wrong rules per task type, where each problem's rule is chosen at random. Each rule is compact, but the mapping "problem -> rule" is unpredictable.

**Table 3.** Multi-rule paired evaluation (tiny, 3.5M, 50/50, 4 seeds).

| N Rules | Accuracy | p |
|:-------:|:--------:|:-:|
| 1 (coherent) | 46.6% | ~1.0 |
| 2 | 77.6% | < 10^-6 |
| 3 | 82.8% | < 10^-6 |
| 5 | 84.8% | < 10^-6 |
| 10 | 88.3% | < 10^-6 |

N=2 on small (12M) yields 86.3% +/- 0.8%, confirming that the effect strengthens with capacity.

![](results/figure7_multirule.png)

*Figure 3. Multi-rule crossover. Accuracy jumps from 47% (N=1) to 78% (N=2), then grows gradually through N=10.*

This is the most mechanistically revealing result. With one false rule, $K(T_2) \approx K(T_1)$. With two rules, the learner must encode a random selector -- which rule applies to which problem -- adding high-complexity bits that break compressibility. At N=10, accuracy (88.3%) exceeds even the random baseline (79.5%), because multi-rule errors combine compactness of individual rules with incompressibility of the selector.

### 4.4 Transfer to Real Text: Wikipedia

**Table 4.** Wikipedia entity substitution (50/50, 4 seeds per size).

| Size | Random Accuracy | Coherent Accuracy |
|------|:--------------:|:-----------------:|
| tiny (3.5M) | **69.6% +/- 0.1%** | 48.7% +/- 0.5% |
| small (12M) | **70.7% +/- 0.4%** | 46.6% +/- 0.4% |
| medium (26M) | **71.5% +/- 0.8%** | 46.4% +/- 0.8% |
| large (86M) | **71.4% +/- 0.7%** | 45.9% +/- 1.5% |

The random/coherent contrast reproduces: random substitution yields 70--71% accuracy (all p < 10^-100), coherent substitution yields 46--49% (at or below chance). The effect is weaker than in math (71% vs 85%) because natural language provides more textual flexibility for locally fluent errors. Unlike math, Wikipedia accuracy saturates near 70% across the tested range -- the compressibility gap in natural language does not substantially benefit from additional capacity at this scale. Per-entity-type accuracy ranges from 82% (LOC) to 61% (CARDINAL), with geographic entities showing the strongest effect (Appendix D).

### 4.5 Generative Evaluation Across Scales

Paired evaluation is a forced-choice (discriminative) setting. To verify that the effect extends to generation, we run greedy decoding with SymPy verification on 500 problems per model across all sizes.

**Table 5.** Generative accuracy (4 seeds, 500 problems each).

| Size | Random-trained | Coherent-trained | Gap |
|------|:--------------:|:----------------:|:---:|
| tiny (3.5M) | 30.5% +/- 1.7% | 20.8% +/- 3.6% | +9.7 pp |
| small (12M) | 46.7% +/- 1.9% | 31.7% +/- 1.8% | +15.0 pp |
| medium (26M) | 50.5% +/- 1.3% | 36.1% +/- 2.7% | +14.4 pp |
| large (86M) | 52.8% +/- 1.1% | 35.6% +/- 1.8% | +17.2 pp |

Random-trained generative accuracy scales monotonically (30% -> 53%), while coherent-trained accuracy plateaus near 36% at small and above. The gap widens with model size (10 pp at tiny -> 17 pp at large), confirming that the discriminative truth bias translates into a generative advantage that strengthens with scale. The absolute generation accuracy is lower than paired accuracy (53% vs 85% at large) -- generation is harder than discrimination -- but the directional effect is consistent.

### 4.6 Architecture Robustness: Qwen3

[TODO: Qwen3-0.6B (420M parameters) results. Pilot at 1000 steps shows 75.1% paired accuracy on random errors, confirming the effect on a fundamentally different architecture (RoPE + GQA + SwiGLU + RMSNorm). Full results (35K steps, 4 seeds x 2 conditions) pending.]

---

## 5. Discussion

The experiments support a unified hypothesis: **in our controlled settings, the compression objective tracks consistency rather than truth.** Any internally consistent rule system -- true or false -- compresses equally well. Truth bias emerges only when false alternatives are structurally incoherent.

The evidence converges from multiple angles. Random errors are incompressible and produce strong truth bias across all tested setups: denoising (65--85%), standard (80--88%), and Wikipedia (70--71%). Coherent errors are equally compressible and produce none: denoising (44--51%), standard (47--52%), Wikipedia (46--49%). The multi-rule experiment isolates the critical variable: the transition from N=1 to N=2 rules demonstrates that rule diversity, not surface-form complexity, drives the effect. A gzip compression analysis confirms positive rank correlation between compressibility gap and paired accuracy across 9 conditions (Spearman rho = 0.68, p = 0.042; Appendix B).

Frequency alone does not explain the results. Truth bias persists even at 10/90 random mixing (67% paired accuracy; Appendix C). But for coherent errors, frequency is decisive: at 40/60 coherent, the model follows the majority with 72% preference for the false system (Appendix C). Compressibility overrides frequency for random errors; frequency governs for coherent ones.

**Implications for alignment.** The training objective does not provide an inherent "truth compass" in our settings. Systematic falsehood remains competitive when internally coherent. Verification dependencies can partially restore truth bias, but their effectiveness decreases with model size (Appendix E), raising questions about scalability.

**Implications for hallucinations.** Our results complement Kalai & Vempala's (2024) hallucination bound. Coherent misconceptions compress well regardless of rarity; if this generalizes, internally consistent hallucinations may be particularly persistent.

**Implications for data curation.** Models favor the more compressible cluster. Diverse organic errors are filtered effectively (even at 4:1 noise, small models achieve 65%). Coordinated, internally consistent errors resist this filtering -- an analogy (not yet demonstrated at scale) to systematic disinformation campaigns.

Our findings admit an interpretive analogy with Popper's falsifiability criterion (1959): a false theory needs corrections that increase description length. The analogy is limited -- the model does not "test" theories -- but the structural parallel is suggestive.

---

## 6. Conclusion

In controlled experiments with small transformers (3.5M--86M parameters), the compression objective tracks consistency rather than truth. When models train on contradictory answers to the same problems, they prefer the correct answer only when errors are structurally incoherent. The sharp crossover at N=1->2 pinpoints the boundary. The same pattern reproduces on real Wikipedia text and in generative evaluation. In the settings we study, truth bias is a compression artifact that emerges only when falsehood lacks internal structure.

The most pressing open question is whether this principle extends beyond 86M parameters. Replication at 1B+ scale with compute-matched training is the natural next step. Linear probing for "truth directions" vs "coherence directions" in models trained under our conditions could connect behavioral findings to the internal representation literature. Testing interactions with RLHF and extending to domains with competing real-world knowledge systems remain open.

## Limitations

**Model scale.** All models are 3.5M--86M parameters. Scaling trends are clear but do not establish behavior beyond 86M. For coherent errors, an MDL argument suggests the result should hold regardless of scale: equally compressible systems at equal frequency offer no basis for preference.

**Domain specificity.** Mathematics provides an unusually crisp correct/incorrect distinction. The effect weakens in natural language (71% vs 85%), where errors can remain locally fluent.

**Discriminative vs generative gap.** Paired accuracy (85%) substantially exceeds generative accuracy (53%) at large scale. The gap narrows with size but the full relationship between discriminative preference and generative truthfulness remains open.

**Seed counts.** Core conditions use 4 seeds; some conditions use 2. Sufficient for directional stability, not for tight confidence intervals.

## Ethical Considerations

This work demonstrates that internally consistent misinformation may be harder for language models to filter than diverse errors. The finding is established in a controlled small-scale setting and does not directly demonstrate vulnerability in production models. We believe transparent reporting of conditions under which truth bias fails is more beneficial than concealment, as it can inform defensive measures in data curation and model evaluation.

---

## References

Azaria, A., & Mitchell, T. (2023). The Internal State of an LLM Knows When It's Lying. *Findings of EMNLP 2023*.

Burns, C., Ye, H., Klein, D., & Steinhardt, J. (2023). Discovering Latent Knowledge in Language Models Without Supervision. *ICLR 2023*.

Burger, L., Hamprecht, F. A., & Nadler, B. (2024). Truth is Universal: Robust Detection of Lies in LLMs. *NeurIPS 2024*.

Chlon, L., Karim, A., Chlon, M., & Awada, M. (2025). Predictable Compression Failures: Why Language Models Actually Hallucinate. *arXiv:2509.11208*.

Deletang, G., Ruoss, A., Grau-Moya, J., et al. (2024). Language Modeling Is Compression. *ICLR 2024*.

DeMoss, B., Sapora, S., Foerster, J., Hawes, N., & Posner, I. (2024). The Complexity Dynamics of Grokking. *arXiv:2412.09810*.

Elazar, Y., Kassner, N., Ravfogel, S., et al. (2022). Measuring Causal Effects of Data Statistics on Language Model's Factual Predictions. *arXiv:2207.14251*.

Goldblum, M., Finzi, M., Rowan, K., & Wilson, A. G. (2024). The No Free Lunch Theorem, Kolmogorov Complexity, and the Role of Inductive Biases in Machine Learning. *ICML 2024*.

Grunwald, P. D. (2007). The Minimum Description Length Principle. *MIT Press*.

Gurnee, W., & Tegmark, M. (2024). Language Models Represent Space and Time. *ICLR 2024*.

Huang, Y., Zhang, J., Shan, Z., & He, J. (2024). Compression Represents Intelligence Linearly. *COLM 2024*.

Hutter, M. (2005). Universal Artificial Intelligence. *Springer*.

Joshi, N., Rando, J., Saparov, A., Kim, N., & He, H. (2024). Personas as a Way to Model Truthfulness in Language Models. *EMNLP 2024*.

Kadavath, S., Conerly, T., Askell, A., et al. (2022). Language Models (Mostly) Know What They Know. *arXiv:2207.05221*.

Kalai, A. T., & Vempala, S. S. (2024). Calibrated Language Models Must Hallucinate. *STOC 2024*.

Kandpal, N., Deng, H., Roberts, A., Wallace, E., & Raffel, C. (2023). Large Language Models Struggle to Learn Long-Tail Knowledge. *ICML 2023*.

Li, K., Hopkins, A. K., Bau, D., et al. (2023a). Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task. *ICLR 2023*.

Li, K., Patel, O., Viegas, F., Pfister, H., & Wattenberg, M. (2023b). Inference-Time Intervention: Eliciting Truthful Answers from a Language Model. *NeurIPS 2023*.

Liu, Z., Zhong, Z., & Tegmark, M. (2023). Grokking as Compression. *arXiv:2310.05918*.

Marks, S., & Tegmark, M. (2023). The Geometry of Truth. *arXiv:2310.06824*.

McGraw, K. O., & Wong, S. P. (1992). A Common Language Effect Size Statistic. *Psychological Bulletin*, 111(2), 361-365.

Mingard, C., Valle-Perez, G., Sherrington, D., & Louis, A. A. (2021). Is SGD a Bayesian Sampler? Well, Almost. *JMLR 2021*.

Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). Progress Measures for Grokking via Mechanistic Interpretability. *ICLR 2023*.

Pan, Z., Wang, S., & Li, J. (2025). Understanding LLM Behaviors via Compression. *arXiv:2504.09597*.

Popper, K. (1959). The Logic of Scientific Discovery. *Hutchinson*.

Ravfogel, S., Yehudai, G., Linzen, T., Bietti, A., & Bruna, J. (2025). Emergence of Linear Truth Encodings in Language Models. *NeurIPS 2025*.

Rissanen, J. (1978). Modeling by Shortest Data Description. *Automatica*, 14(5), 465-471.

Rolnick, D., Veit, A., Belongie, S., & Shavit, N. (2017). Deep Learning is Robust to Massive Label Noise. *arXiv:1705.10694*.

Shannon, C. E. (1948). A Mathematical Theory of Communication. *Bell System Technical Journal*, 27(3), 379-423.

Solomonoff, R. J. (1964). A Formal Theory of Inductive Inference. *Information and Control*, 7(1), 1-22.

Valle-Perez, G., Camargo, C. Q., & Louis, A. A. (2019). Deep Learning Generalizes Because the Parameter-Function Map Is Biased Towards Simple Functions. *ICLR 2019*.

Wan, J., & Mei, L. (2025). Large Language Models as Computable Approximations to Solomonoff Induction. *arXiv:2505.15784*.

Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2017). Understanding Deep Learning Requires Rethinking Generalization. *ICLR 2017*.

---

## Appendix A: Reproducibility

All code, data generation scripts, and evaluation scripts are available at https://github.com/Rai220/compression-drives-truth. Denoising experiments were run on Modal.com T4 GPUs using PyTorch. Standard math and Wikipedia experiments were run on Apple Mac M4 (36GB) using MLX (v0.31.0). Over 160 models were trained across all conditions; total compute approximately 80 GPU-hours.

---

## Appendix B: Compression Measure

To operationalize the MDL argument, we measure compression ratios (gzip level 9) on concatenated correct vs incorrect completions from each paired test set.

**Table B1.** Compression ratio and paired accuracy across conditions.

| Condition | Correct | Incorrect | Delta | Paired Accuracy |
|-----------|:-------:|:---------:|:-----:|:--------------:|
| random 50/50 | 0.1627 | 0.1639 | +0.0012 | 79.5% |
| coherent 50/50 | 0.1656 | 0.1658 | +0.0002 | 47.2% |
| contradictory | 0.1745 | 0.1744 | -0.0001 | 49.0% |
| multirule N=2 | 0.1671 | 0.1710 | +0.0038 | 77.6% |
| multirule N=5 | 0.1672 | 0.1730 | +0.0057 | 84.8% |
| multirule N=10 | 0.1676 | 0.1752 | +0.0076 | 88.3% |

Spearman rho = 0.68, p = 0.042. Conditions with larger compression gaps produce stronger truth bias. We present this as supporting evidence; the primary argument rests on the experimental contrasts in Section 4.

---

## Appendix C: Standard (Non-Denoising) Experiments

### C.1 Standard Corpus Design

In the standard setup, each problem appears once with either a correct or incorrect solution. The two groups do not share prompts.

### C.2 Standard vs Denoising

**Table C1.** Standard vs denoising accuracy (50/50 random).

| Size | Standard | Denoising | Gap |
|------|:--------:|:---------:|:---:|
| tiny | 79.5% | 65.3% | -14.2 pp |
| small | 86.2% | 74.6% | -11.6 pp |
| medium | 87.1% | 81.1% | -6.0 pp |
| large | 88.3% | 85.2% | -3.1 pp |

The denoising setup is harder (within-problem contradiction vs across-corpus mixing), but the gap closes with scale.

### C.3 Frequency Effects

**Table C2.** Random vs coherent accuracy across proportions (standard, tiny).

| Proportion | Random | Coherent |
|:----------:|:------:|:--------:|
| 50/50 | 80% | 47.2% |
| 40/60 | 79% | 27.8% |
| 30/70 | 75% | 14.7% |
| 20/80 | 69% | 9.6% |
| 10/90 | 67% | -- |

Random: truth bias persists at 10/90 (67%). Coherent: the model follows pure frequency.

### C.4 NLL Distribution

Random errors produce a right-skewed per-pair NLL difference (mean +0.048, median +0.025, 81.5% positive). Coherent errors produce a symmetric distribution (45.5% positive). This explains why pair accuracy and mean DLoss can diverge.

---

## Appendix D: Additional Transfer Experiments

### D.1 Synthetic World

A synthetic world with 50 entities, 4 types, 15 deterministic rules. Random 50/50: 57.7% +/- 1.7%. Coherent 50/50: 46.6% +/- 1.7%. Per-type: minerals best (68.7%), potions near chance (49.1%).

### D.2 Wikipedia Per-Entity-Type

| Entity Type | N Pairs | Accuracy |
|-------------|:-------:|:--------:|
| LOC | 45 | 82.2% |
| NORP | 197 | 78.7% |
| GPE | 301 | 77.1% |
| PERSON | 402 | 69.9% |
| ORG | 664 | 65.2% |
| CARDINAL | 172 | 61.0% |

### D.3 Cross-Domain Falsification

Adding correct cross-domain tasks to a coherent corpus selectively increases derivative accuracy (35% -> 56% at 25%). The effect is non-monotonic (drops at 50% due to dilution).

---

## Appendix E: Robustness Checks

### E.1 BPE Tokenization

**Table E1.** BPE vs char-level (tiny, 4 seeds).

| Setup | Tokenizer | Random | Coherent |
|-------|-----------|:------:|:--------:|
| Standard | Char | 79.5% | 47.2% |
| Standard | BPE | **85.6%** | 45.9% |
| Denoising | Char | 65.3% | 43.5% |
| Denoising | BPE | **75.9%** | 49.3% |

The effect survives BPE and strengthens for random errors.

### E.2 Chained Verification

Embedding verification dependencies in coherent-error tasks restores truth bias: 70.9% at tiny (vs 43% standard coherent). The effect *decreases* with model size (64% at small, 61% at large) -- larger models absorb coherent errors more readily, making verification less effective.

### E.3 Learning Curves

All sizes reach behavioral plateau by step 3000--4000. The large model achieves 88.8% at step 3000, stable through 5000.

---

## Appendix F: Per-Seed Details

**Table F1.** Denoising equal random, per-seed accuracy.

| Size | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean |
|------|:------:|:------:|:------:|:------:|:----:|
| tiny | 64.2% | 64.1% | 67.0% | 65.8% | 65.3% |
| small | 75.1% | 76.3% | 72.6% | 74.2% | 74.6% |
| medium | 79.6% | 80.9% | 82.4% | 81.3% | 81.1% |
| large | 83.5% | 86.8% | -- | -- | 85.2% |

**Table F2.** Denoising equal coherent, per-seed accuracy.

| Size | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean |
|------|:------:|:------:|:------:|:------:|:----:|
| tiny | 40.2% | 44.3% | 46.3% | 43.0% | 43.5% |
| small | 44.8% | 45.3% | 40.3% | 47.4% | 44.5% |
| medium | 45.8% | 40.9% | 47.7% | 48.6% | 45.8% |
| large | 51.6% | 50.4% | -- | -- | 51.0% |
