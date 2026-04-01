# Error Structure Determines Correctness Preference in Contradictory Training Data

## Abstract

Language models trained on contradictory data often prefer correct answers, yet the mechanism behind this preference is poorly understood. Without such understanding, we cannot predict when this implicit filtering will fail -- a question critical for controlling model behavior on noisy real-world corpora. We hypothesize that next-token prediction, as a compression process, favors whichever answer cluster has lower description length; truth benefits only when errors lack internal structure. We train GPT-2 style transformers (3.5M--86M parameters) from scratch on controlled corpora where the same problem appears with both correct and incorrect solutions, systematically varying the structure of errors. We demonstrate three findings: (a) when errors are random, models develop a correctness preference scaling from 65% to 85% with model size; (b) when errors follow a single coherent alternative rule, this preference vanishes entirely (~45--51%); (c) the transition is sharp -- two competing wrong rules suffice to restore correctness preference (47% -> 78%). The pattern reproduces on real Wikipedia text with entity substitution (71% vs 46%). These results establish that error structure, not truth per se, determines model preference in contradictory training regimes.

---

## 1. Introduction

Real-world training corpora are noisy: the same question may receive contradictory answers across different documents. Yet language models trained on such data tend to prefer correct information -- and this implicit filtering is increasingly relied upon in high-stakes applications from medical QA to legal reasoning. If we do not understand *why* this filtering works, we cannot predict *when* it will fail. This is not merely a theoretical concern: coordinated misinformation campaigns, systematic biases in training data, and internally consistent pseudoscientific frameworks all present cases where errors may share enough structure to evade whatever mechanism produces truthful preferences.

Several explanations have been offered. Scaling improves factual performance (Kadavath et al., 2022). RLHF steers outputs toward human preferences. Frequency matters: factual accuracy correlates with how often correct information appears in training data (Elazar et al., 2022; Joshi et al., 2024; Kandpal et al., 2023). Recent work has found truth-correlated internal representations (Burns et al., 2023; Marks & Tegmark, 2023; Burger et al., 2024) and established that calibrated models must hallucinate at a rate bounded by the corpus's monofact rate (Kalai & Vempala, 2024). Yet none of these explain why the training objective would favor one answer over another when both appear at equal frequency in the same format.

We propose that the answer lies in the structure of errors. Minimizing cross-entropy is equivalent to minimizing code length (Shannon, 1948; Deletang et al., 2024), linking LLM training to the Minimum Description Length principle (Rissanen, 1978; Grunwald, 2007). A correct rule system compresses into a compact representation; diverse errors must be memorized individually. But a *coherent* false system -- internally consistent, just wrong -- compresses equally well, and the preference vanishes.

We test this hypothesis in a controlled experimental setting. We train GPT-2 style transformers (3.5M--86M parameters) from scratch on corpora where each mathematical problem appears with both correct and incorrect solutions, systematically varying the structure of errors. This design isolates error structure as the independent variable while holding frequency, format, and domain constant.

Our contribution is a single experimentally validated hypothesis: **in controlled contradictory corpora, the compression objective favors consistency, not truth.** We show that (a) random errors produce a correctness preference scaling from 65% to 85%; (b) a single coherent alternative rule eliminates this preference entirely; (c) two competing rules restore it, pinpointing the compressibility boundary; and (d) the same contrast reproduces on real Wikipedia text. Whether this principle extends to large-scale pretraining remains an open question that we discuss in Section 5.

## 2. Related Work

**Prediction as compression.** The link between prediction and compression traces to Shannon (1948) and was formalized by Solomonoff (1964) and Rissanen's MDL principle (1978; Grunwald, 2007). Deletang et al. (2024) showed that LLMs are universal compressors. Huang et al. (2024) discovered a linear correlation (r ~ -0.95) between compression quality and benchmark performance. Wan & Mei (2025) proved that LLM training approximates Solomonoff induction. Pan et al. (2025) linked compression to knowledge acquisition and scaling; Chlon et al. (2025) linked compression failures to hallucination. We build on the MDL framework by experimentally varying the description length of false answer systems.

**Internal representations and world models.** Compression can give rise to structured internal representations: Li et al. (2023a) found board representations in an Othello model, Gurnee & Tegmark (2024) discovered linear space-time representations in Llama-2, and several studies found truth-correlated representations (Marks & Tegmark, 2023; Burns et al., 2023; Li et al., 2023b; Azaria & Mitchell, 2023; Ravfogel et al., 2025). Our work operates at the behavioral level: we identify data-level conditions under which compression produces a preference for correct completions, leaving activation-level analysis for future work.

**Truthfulness and data statistics.** Joshi et al. (2024) linked truthfulness to "personas" in pretraining data. Elazar et al. (2022) demonstrated frequency dependence. Kandpal et al. (2023) showed a direct relationship between document count and accuracy. Unlike these frequency-based analyses, we fix frequency and vary error structure, isolating compressibility as the operative variable.

**Simplicity bias and noisy labels.** Neural networks prefer simple functions (Valle-Perez et al., 2019; Mingard et al., 2021; Goldblum et al., 2024). The noisy labels literature directly parallels our setup: Zhang et al. (2017) showed memorization of random labels; Rolnick et al. (2017) showed robustness to massive label noise. Grokking connects to compression as a memorization-to-generalization transition (Nanda et al., 2023; Liu et al., 2023). Our experiments extend this line by showing that "structured noise" -- coherent errors -- is not filtered out. To our knowledge, systematic variation of error compressibility in a denoising setting has not been studied before.

## 3. Methodology

**Overview.** We test the hypothesis that next-token prediction favors whichever answer system has lower description length, regardless of its truth value. To isolate this, we train models from scratch on controlled corpora where each problem appears with contradictory answers, varying only the structure of errors. The core experiments compare random errors (high description length) against coherent errors (low description length) at equal frequency (Section 4.1), then probe the boundary with multi-rule errors of intermediate compressibility (Section 4.2). Transfer to natural language is tested via Wikipedia entity substitution (Section 4.3).

### 3.1 Hypothesis and MDL Framing

Consider a corpus where each problem appears with a correct answer (theory $T_1$) and an alternative ($T_2$). An MDL learner (Rissanen, 1978; Grunwald, 2007) minimizes $L(M) + L(D|M)$. When $K(T_2) \gg K(T_1)$ (random errors), the false system's description length grows with corpus size, favoring $T_1$. When $K(T_2) \approx K(T_1)$ (coherent errors), both are compact and the learner has no basis to prefer one over the other. With N competing false rules, a random "selector" (which rule applies to which problem) adds $\log N$ bits per problem, progressively degrading the false cluster's compressibility.

Throughout, "truth" means correctness of mathematical derivations and factual accuracy of Wikipedia paragraphs -- models compress text, not reality. The compressibility gap is domain-dependent and smaller in natural language than in formal math (Section 4.3). Additional limitations are discussed in Section 6.

### 3.2 Models and Training

GPT-2 style decoder-only transformers with pre-norm, GELU activation, and causal mask:

| Config | Layers | d_model | Heads | Parameters |
|--------|--------|---------|-------|------------|
| tiny | 4 | 256 | 4 | ~3.5M |
| small | 6 | 384 | 6 | ~12M |
| medium | 8 | 512 | 8 | ~26M |
| large | 12 | 768 | 12 | ~86M |

Optimizer: AdamW (weight_decay=0.01), cosine decay with linear warmup, lr=3e-4, seq_len=256, batch_size=32, 5000 steps. All experiments use 4 random seeds for core conditions (2 where noted). Learning curves confirm behavioral plateau by step 3000--4000 at all sizes (Appendix E). Denoising experiments use PyTorch; multi-rule and Wikipedia experiments use MLX. All problems are generated and verified by SymPy; no human annotation is involved. Full generation templates are provided in the supplementary material.

### 3.3 Corpus Design

**Denoising setup (primary).** A generator creates mathematical problems of four types: multi-step arithmetic, factoring, equation solving, and differentiation, formatted as step-by-step derivations in English. The key design: **each problem appears with contradictory answers** -- modeling the scenario where the same question receives conflicting responses. The tokenizer is character-level (vocab ~57); BPE robustness is verified in Section 4.4.

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

**Generative evaluation.** Greedy decoding on 500 test prompts with automated SymPy verification. This confirms that the discriminative effect extends to generation (Section 4.5; Table 4).

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

Random accuracy increases monotonically with scale (65% -> 75% -> 81% -> 85%). Coherent accuracy converges toward 50% from below, consistent with the MDL prediction that equally compressible systems at equal frequency should be indistinguishable. The below-chance coherent accuracy at tiny (~43%) reflects textual simplicity of the false rules (e.g., dropping terms in derivatives); the compressor prefers whichever system is simpler to encode. The multi-rule experiment (Section 4.2) controls for this surface-form effect.

### 4.2 Multi-Rule Errors: The Sharp Crossover

The denoising experiments establish two poles: one coherent rule yields chance, random errors yield strong bias. We probe the boundary by training on corpora with N alternative wrong rules per task type, where each problem's rule is chosen at random. Each rule is compact, but the mapping "problem -> rule" is unpredictable.

**Table 2.** Multi-rule paired evaluation (tiny, 3.5M, 50/50, 4 seeds).

| N Rules | Accuracy | p |
|:-------:|:--------:|:-:|
| 1 (coherent) | 46.6% | ~1.0 |
| 2 | 77.6% | < 10^-6 |
| 3 | 82.8% | < 10^-6 |
| 5 | 84.8% | < 10^-6 |
| 10 | 88.3% | < 10^-6 |

N=2 on small (12M) yields 86.3% +/- 0.8%, confirming that the effect strengthens with capacity.

![](results/figure7_multirule.png)

*Figure 2. Multi-rule crossover. Accuracy jumps from 47% (N=1) to 78% (N=2), then grows gradually through N=10.*

This is the most mechanistically revealing result. With one false rule, $K(T_2) \approx K(T_1)$. With two rules, the learner must encode a random selector -- which rule applies to which problem -- adding high-complexity bits that break compressibility. At N=10, accuracy (88.3%) exceeds even the random baseline (79.5%), because multi-rule errors combine compactness of individual rules with incompressibility of the selector.

### 4.3 Transfer to Real Text: Wikipedia

**Table 3.** Wikipedia entity substitution (50/50, 4 seeds per size).

| Size | Random Accuracy | Coherent Accuracy |
|------|:--------------:|:-----------------:|
| tiny (3.5M) | **69.6% +/- 0.1%** | 48.7% +/- 0.5% |
| small (12M) | **70.7% +/- 0.4%** | 46.6% +/- 0.4% |
| medium (26M) | **71.5% +/- 0.8%** | 46.4% +/- 0.8% |
| large (86M) | **71.4% +/- 0.7%** | 45.9% +/- 1.5% |

The random/coherent contrast reproduces: random substitution yields 70--71% accuracy (all p < 10^-100), coherent substitution yields 46--49% (at or below chance). The effect is weaker than in math (71% vs 85%) because natural language provides more textual flexibility for locally fluent errors. Unlike math, Wikipedia accuracy saturates near 70% across the tested range -- the compressibility gap in natural language does not substantially benefit from additional capacity at this scale. Per-entity-type accuracy ranges from 82% (LOC) to 61% (CARDINAL), with geographic entities showing the strongest effect (Appendix D).

### 4.4 Robustness Checks

**Tokenization.** The primary experiments use a character-level tokenizer (vocab ~57). To verify that the effect is not an artifact of tokenization, we repeat the core conditions with a BPE tokenizer (SentencePiece, vocab=1000). The effect survives and strengthens: random accuracy increases from 65.3% to 75.9% in denoising (79.5% to 85.6% in standard); coherent accuracy remains at chance (49.3% and 45.9% respectively). Full results in Appendix E.

**Standard (non-denoising) setup.** In the standard setup, each problem appears once with either a correct or incorrect solution (no within-problem contradiction). Paired accuracy is higher (79.5--88.3% vs 65.3--85.2%), but the gap closes with scale (from 14 pp at tiny to 3 pp at large). Critically, the random/coherent contrast holds identically: standard coherent accuracy is 47--52%, at chance. Full comparison in Appendix C.

**Frequency vs compressibility.** For random errors, truth bias persists even when correct examples are a small minority: 67% paired accuracy at 10/90 correct-to-incorrect ratio (Appendix C). For coherent errors, the pattern reverses -- frequency alone determines preference: at 40/60 coherent, the model follows the majority with 72% preference for the false system; at 20/80, this rises to 91%. This asymmetry is a direct prediction of the MDL framework: when both systems compress equally well, frequency is the only tiebreaker.

**Noise tolerance.** The correct signal persists under increasing noise ratios (1:2 and 1:4 random), with accuracy degrading gracefully from 85% to 66--75% at large scale. Details in Appendix C.

### 4.5 Generative Evaluation

Paired evaluation is a forced-choice (discriminative) setting. To verify that the effect extends to generation, we run greedy decoding with SymPy verification on 500 problems per model across all sizes.

**Table 4.** Generative accuracy (4 seeds, 500 problems each).

| Size | Random-trained | Coherent-trained | Gap |
|------|:--------------:|:----------------:|:---:|
| tiny (3.5M) | 30.5% +/- 1.7% | 20.8% +/- 3.6% | +9.7 pp |
| small (12M) | 46.7% +/- 1.9% | 31.7% +/- 1.8% | +15.0 pp |
| medium (26M) | 50.5% +/- 1.3% | 36.1% +/- 2.7% | +14.4 pp |
| large (86M) | 52.8% +/- 1.1% | 35.6% +/- 1.8% | +17.2 pp |

The gap widens with model size (10 pp at tiny -> 17 pp at large), confirming that the discriminative preference translates into a generative advantage. The absolute generation accuracy is substantially lower than paired accuracy (53% vs 85% at large). This gap reflects the fundamental difference between discrimination and generation: the model may assign higher likelihood to correct completions while lacking the capacity to produce them reliably from scratch. Both metrics agree on the direction: random-trained models outperform coherent-trained ones at all scales.

### 4.6 Architecture Robustness: Qwen3

To verify that the effect is not specific to the GPT-2 architecture, we train a Qwen3-0.6B model (420M parameters, 28 layers, RoPE + GQA + SwiGLU + RMSNorm) from scratch on the same denoising corpora. This architecture differs from GPT-2 in positional encoding, attention mechanism, activation function, and normalization -- testing whether the compression-consistency effect is a general property of autoregressive transformers.

**Table 6.** Qwen3-0.6B paired evaluation (50/50, 4 seeds each).

| Condition | seed42 | seed43 | seed44 | seed45 | Mean |
|-----------|:------:|:------:|:------:|:------:|:----:|
| Random | 75.1% | 85.3% | 85.7% | 89.3% | **83.9%** |
| Coherent | 47.9% | 49.3% | 52.8% | 49.8% | **49.9%** |

The pattern reproduces: random errors yield 83.9% accuracy, coherent errors yield 49.9% (chance). The random/coherent contrast is architecture-independent. Qwen3's random accuracy (83.9%) is comparable to GPT-2 medium (81.1%) despite having 16x more parameters, likely because the model is undertrained relative to its capacity (15K--30K steps on a corpus sized for smaller models).

---

## 5. Discussion

**Summary of findings.** The experiments support a unified hypothesis: **in our controlled settings, the compression objective tracks consistency rather than truth.** Any internally consistent rule system -- true or false -- compresses equally well. Truth bias emerges only when false alternatives are structurally incoherent. The evidence converges across setups (denoising, standard, Wikipedia), error types (random, coherent, multi-rule), and evaluation modes (discriminative and generative).

**The role of frequency.** Our results reveal an asymmetry between random and coherent error regimes. For random errors, compressibility overrides frequency: truth bias persists even at 10/90 correct-to-incorrect ratio (67% accuracy). For coherent errors, frequency is decisive: at 40/60 coherent mixing, the model follows the majority with 72% preference for the false system, rising to 91% at 20/80. This is a direct MDL prediction -- when competing systems compress equally well, the more frequent one wins. The practical implication is that coherent falsehood needs no frequency advantage to neutralize truth bias; equal frequency suffices.

**What the multi-rule experiment shows.** The sharp N=1->2 transition is the strongest evidence that compressibility, not surface-form complexity, is the operative variable. Each individual rule at N=2 remains compact, yet the random assignment of rules to problems introduces incompressible bits. This rules out explanations based on lexical diversity or number of corrupted tokens alone.

**Implications and scope.** In our controlled settings, the training objective does not provide an inherent "truth compass." Internally coherent falsehood remains competitive. This has potential relevance for alignment (systematic false beliefs may resist filtering), hallucinations (coherent misconceptions may persist; cf. Kalai & Vempala, 2024), and data curation (diverse errors are filtered effectively, but coordinated errors may not be). However, all our experiments use models up to 86M parameters trained from scratch on synthetic or semi-synthetic data. Whether these findings extend to large-scale pretraining on heterogeneous real-world corpora remains an open question. We view our results as identifying a mechanism and its boundary conditions, not as direct claims about production LLMs.

Our findings admit an interpretive analogy with Popper's falsifiability criterion (1959): a false theory that requires ad hoc corrections to accommodate observations has higher description length than a true one. The analogy is limited -- the model does not "test" theories -- but the structural parallel is suggestive.

---

## 6. Conclusion

In controlled experiments with transformers up to 86M parameters, the compression objective tracks consistency rather than truth. Models prefer the correct answer only when errors are structurally incoherent; a single coherent alternative rule eliminates this preference, and two competing rules restore it. The same pattern reproduces on real Wikipedia text and in generative evaluation. In the settings we study, truth bias is a compression artifact whose presence or absence is determined by the description length of the false answer system.

## Limitations

**Model scale.** All models are 3.5M--86M parameters. Scaling trends are clear but do not establish behavior beyond 86M. For coherent errors, an MDL argument suggests the result should hold regardless of scale: equally compressible systems at equal frequency offer no basis for preference. Replication at 1B+ scale with compute-matched training is the most important next step.

**Domain specificity.** Mathematics provides an unusually crisp correct/incorrect distinction. The effect weakens in natural language (71% vs 85%), where errors can remain locally fluent. Extending to domains with competing real-world knowledge systems (e.g., conflicting news sources, historical revisionism) remains open.

**Discriminative vs generative gap.** Paired accuracy (85%) substantially exceeds generative accuracy (53%) at large scale. The gap narrows with size but the full relationship between discriminative preference and generative truthfulness remains open. Paired accuracy should be interpreted as a controlled diagnostic of model preference, not as a direct measure of output truthfulness.

**Seed counts.** Core conditions use 4 seeds; some conditions use 2. Sufficient for directional stability, not for tight confidence intervals.

**Causal identification.** While we vary error structure as the independent variable, random and coherent corruptions may differ in additional ways (number of affected derivation steps, surface-form simplicity). The multi-rule experiment partially addresses this by holding individual rule complexity constant, but a fully matched control (identical corruption load, lexical diversity, and output length) would strengthen the causal claim.

**Future directions.** Linear probing for "truth directions" vs "coherence directions" in models trained under our conditions could connect behavioral findings to the internal representation literature. Testing interactions with RLHF and investigating whether the compressibility mechanism interacts with in-context learning are natural extensions.

## Ethical Considerations

This work demonstrates that, in controlled settings, internally consistent misinformation may be harder for language models to filter than diverse errors. The finding does not directly demonstrate vulnerability in production models. We believe transparent reporting of conditions under which truth bias fails is more beneficial than concealment, as it can inform defensive measures in data curation and model evaluation.

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

All code, data generation scripts, and evaluation scripts are available at [anonymous repository, included as supplementary material]. Denoising experiments were run on Modal.com T4 GPUs using PyTorch. Standard math and Wikipedia experiments were run on Apple Mac M4 (36GB) using MLX (v0.31.0). Over 160 models were trained across all conditions; total compute approximately 80 GPU-hours.

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

### C.3 Noise Tolerance

**Table C3.** Paired accuracy at increasing noise ratios (denoising).

| Condition | Ratio | Tiny | Small | Medium | Large |
|-----------|:-----:|:----:|:-----:|:------:|:-----:|
| Equal random | 1:1 | 65.3% | 74.6% | 81.1% | 85.2% |
| 2:1 noise | 1:2 | 59.0% | 68.6% | 73.6% | 75.2% |
| 4:1 noise | 1:4 | 56.6% | 64.9% | 66.6% | 65.8% |

Increasing the noise ratio degrades accuracy gracefully. At 4:1 noise, accuracy plateaus at medium/large (~66%), indicating a capacity-dependent signal-to-noise bottleneck.

### C.4 Frequency Effects

**Table C2.** Random vs coherent accuracy across proportions (standard, tiny).

| Proportion | Random | Coherent |
|:----------:|:------:|:--------:|
| 50/50 | 80% | 47.2% |
| 40/60 | 79% | 27.8% |
| 30/70 | 75% | 14.7% |
| 20/80 | 69% | 9.6% |
| 10/90 | 67% | -- |

Random: truth bias persists at 10/90 (67%). Coherent: the model follows pure frequency.

### C.5 NLL Distribution

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
