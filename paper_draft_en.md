# When Does Compression Favor Truth? Consistency, Description Length, and Inductive Bias in Language Models

**Author:** Konstantin Krestnikov
**Date:** 03.2026

> **Note:** This is an English translation of the main draft `paper_draft_ru.md`. Please apply all substantive changes to the Russian version first.

## Abstract

Language models minimize cross-entropy loss, which is mathematically equivalent to compressing the training data. We investigate under what conditions this compression pressure gives rise to a systematic preference for correct information in models trained on mixed-quality text corpora. Crucially, models compress *text*, not reality; the observed bias reflects the statistical structure of the corpus, not access to external truth.

We train small transformers (3.5M parameters) on corpora with controlled ratios of correct and incorrect mathematical derivations. With random (incoherent) errors, models consistently show lower loss on correct examples — even when incorrect examples constitute up to 80% of the corpus (16/16 seeds, p = 3.05 x 10^-5). However, when random errors are replaced with a **coherent alternative rule system** — internally consistent but mathematically incorrect — the truth preference vanishes (DLoss ~ 0 at 50/50). Compression favors not truth, but **the most consistent and compact structure** in the data.

In additional experiments, we show that adding empirical feedback (observations) to a false theory yields only a weak effect from bare discrepancies (DLoss ~ +0.0008), because the discrepancies themselves are regular and compressible. However, when the theory requires an additional correction step to reconcile with observations — whether unique explanations or a single correction rule — the bias is significantly amplified (DLoss ~ +0.0025). The informational cost of this step, rather than falsity per se, determines truth's advantage.

Our results show that "truth bias" in language models is not a fundamental property of compression, but a consequence of two structural conditions: (1) incoherence of errors in the corpus and (2) informational overhead of correcting false predictions. A coherent false system that requires no correction compresses just as well as truth.

---

## 1. Introduction

Minimizing cross-entropy loss during language model training is equivalent to minimizing code length under arithmetic coding (Shannon, 1948; Deletang et al., 2024). This connects LLM training to the task of compression: a model that better predicts the next token is a better compressor. Recent work has shown that compression quality correlates linearly with model capabilities (Huang et al., 2024), and that LLM training formally approximates Solomonoff induction (Wan & Mei, 2025).

We ask: does compression pressure create a systematic bias towards correct information? The intuition is straightforward: a correct knowledge system is internally consistent and described by a compact set of rules; erroneous statements tend to be diverse and inconsistent, requiring greater description length. Given limited model capacity, this may give true knowledge a compression advantage.

However, this intuition requires important caveats:

1. **Models compress text, not reality.** Cross-entropy is defined over the token distribution in the corpus. "Truth" in our context is a property of the text (correctness of mathematical derivations), not a metaphysical category. Results are applicable insofar as the corpus reflects reality.

2. **Frequency can beat truth.** A finite-capacity model primarily learns frequency patterns. A compression advantage for truth arises only when correct examples have a structural advantage that compensates for being a numerical minority.

3. **Compression advantage depends on error description length.** If the false system is as compact as the true one (a coherent alternative mathematics), the advantage vanishes. Truth bias is not a property of compression per se, but a consequence of the structure of a particular corpus.

We conduct a series of controlled experiments on mathematical corpora, where we can precisely define what is "correct" and "incorrect" and vary the type and structure of errors:

1. **Experiment 1** (Section 4): Training on a mixture of correct and incorrect derivations with random, coherent, and contradictory errors at various proportions.
2. **Experiment 2** (Section 5): Adding empirical feedback (observations) to coherent errors.
3. **Experiment 3** (Section 6): Five conditions for a false theory, varying the informational overhead of correction.

## 2. Related Work

### Prediction = Compression

| Work | Contribution |
|------|-------------|
| Shannon (1948) | Optimal compression requires knowledge of the true distribution |
| Solomonoff (1964) | Optimal predictor weights hypotheses by program length |
| Hutter (2005), AIXI | Formalization of the equivalence of intelligence and compression |
| Deletang et al. (ICLR 2024) | "Language Modeling Is Compression" — LLMs as universal compressors |
| Huang et al. (COLM 2024) | Linear correlation between compression and intelligence (r ~ -0.95) |
| Wan & Mei (2025) | Formal proof: LLM training ~ Solomonoff induction |

### Internal Representations of Truth in LLMs

| Work | Contribution |
|------|-------------|
| Marks & Tegmark (2023) | "Geometry of Truth" — linear structure of truth in activations |
| Burns et al. (ICLR 2023) | CCS — discovering latent truth directions without supervision |
| Li et al. (NeurIPS 2023) | ITI — 40% gap between internal knowledge and generation |
| Ravfogel et al. (2025) | "Truth Co-occurrence Hypothesis" — mechanism of emergence |

### Emergent World Models

| Work | Contribution |
|------|-------------|
| Li et al. (ICLR 2023) | Othello-GPT — board model from predicting moves |
| Gurnee & Tegmark (ICLR 2024) | Linear representations of space and time in Llama-2 |

### Simplicity Bias and Grokking

| Work | Contribution |
|------|-------------|
| Valle-Perez et al. (ICLR 2019) | Exponential bias towards low-complexity functions |
| Mingard et al. (JMLR 2021) | SGD ~ Bayesian sampling with simplicity prior |
| Nanda et al. (ICLR 2023) | Grokking: networks discover Fourier transforms for modular arithmetic |
| DeMoss et al. (2024) | Complexity dynamics of grokking: phase transition to generalization |

Goldblum et al. (2024) showed that neural networks exhibit an inductive bias towards functions with low Kolmogorov complexity, providing a theoretical basis for the link between compression and generalization. Liu et al. (2023) interpreted grokking as a compression process: the network transitions from memorization to a compact representation.

### Our Contribution

The works listed above either study internal truth representations in already-trained models or establish theoretical links between compression and intelligence. However, a direct experiment — training a model from scratch on a quality-controlled corpus and measuring truth preference as a function of error type, coherence, and presence of empirical feedback — has not been conducted. This work fills that gap and isolates the conditions under which compression pressure aligns with truth, as opposed to coherent falsehood.

## 3. Methodology

### 3.1 Model and Training

GPT-2 style decoder-only transformer implemented in MLX. Pre-norm (LayerNorm before attention/MLP), GELU activation, causal mask.

| Config | Layers | d_model | Heads | Parameters |
|--------|--------|---------|-------|-----------|
| tiny | 4 | 256 | 4 | 3,516,416 |

Optimizer: AdamW (weight_decay=0.01), cosine decay with linear warmup (200 steps), lr=3e-4, seq_len=256, batch_size=32, 5000 steps. All experiments are repeated with 4 random initializations (seeds 42-45).

### 3.2 Corpus Generation

The generator creates mathematical problems of four types: multi-step arithmetic, factorization, equation solving, and differentiation. Each problem is formatted as a step-by-step solution in English, verified by SymPy. The tokenizer is character-level (vocab size = 57) to exclude BPE artifacts as a confound.

**Error Types:**
- **Random:** Injection of one plausible error at a random step (sign, coefficient, distributivity error). Each error is unique.
- **Coherent:** One systematic incorrect rule per problem type (e.g., a x b = a x (b-1); sign is preserved when moving terms across =; etc.). All problems of one type fail identically.
- **Contradictory:** Simple rules (a + b = a + b + 1; a - b = a - b - 2) that break algebraic structure — addition and subtraction cease to be inverse operations.

### 3.3 Metric

For each trained model, we compute the average cross-entropy loss on held-out sets of correct (5K problems) and incorrect (5K problems) examples. The main metric is **DLoss = Loss(incorrect) - Loss(correct)**. A positive value indicates truth bias (the model better predicts correct examples).

**Statistical analysis.** Each configuration is repeated with 4 random initializations (seeds 42-45). For individual configurations we use the two-sided binomial test (H0: P(correct) = 0.5). With 4 seeds the minimum p-value is 0.125, which is insufficient for significance. However, the combined test across multiple configurations sharing a single hypothesis substantially increases power. 95% confidence intervals for DLoss are obtained via bootstrap (10,000 resamples).

### 3.4 Theoretical Framework: Description Length and Theory Types

To interpret Experiments 2-3 we use a typology of theories distinguished by the description length of the corpus "theory + observations." The key principle: the model optimizes cross-entropy, which is equivalent to minimizing expected code length (Shannon, 1948). A theory that allows shorter encoding of the corpus gains an advantage.

**Type 1: True theory with concrete predictions.** Predictions match observations. The "theory + observations" corpus compresses maximally: one rule system explains everything.

**Type 2: False theory with concrete predictions.** Predictions diverge from observations. The model must encode both the false rules and the discrepancies. However, if the discrepancies are **regular** (e.g., a x b = a x (b-1) always understates by a), the model can learn a correction, and the additional description length is small.

**Type 3a: Theory with non-specific predictions.** The theory does not specify a "situation -> outcome" mapping (e.g., "result is moderate"). It does not contradict observations but does not help predict them either — it does not reduce code length.

**Type 3b: Theory with ad hoc correction.** Each discrepancy is explained by a unique exception rule. Description length grows linearly with the number of observations — this is anti-compression.

### 3.5 Experiment Conditions

**Experiment 1:** 5 proportions (50/50-10/90) x 4 seeds = 20 models with random errors + 1 baseline. Controls: coherent errors (4 proportions x 4 seeds = 16) and contradictory (4 seeds).

**Experiment 2:** Coherent errors at 50/50 with observations. 4 observation ratios (0%, 10%, 25%, 50%) x 4 seeds = 16 models. Test sets contain no observations — we measure pure mathematical prediction quality.

**Experiment 3:** 5 conditions for the false theory (A-E) at 50/50. Conditions A and B are from Experiments 1-2. Conditions C, D, E — 3 x 4 seeds = 12 new models.

In total, **69 models** were trained.

## 4. Experiment 1: Random, Coherent, and Contradictory Errors

### 4.1 Truth Bias with Random Errors

**Table 1.** Loss on held-out test sets, averaged over 4 seeds. DLoss = Loss(incorrect) - Loss(correct); positive value = truth bias.

| Proportion (cor/inc) | Loss (correct) | Loss (incorrect) | DLoss | 95% CI (bootstrap) | Seeds -> correct |
|----------------------|----------------|------------------|-------|---------------------|------------------|
| 100/0 (baseline) | 0.1313 | 0.2028 | +0.0715 | -- | 1/1 |
| 50/50 | 0.1384 +/- 0.0009 | 0.1499 +/- 0.0008 | +0.0115 +/- 0.0002 | [+0.0113, +0.0116] | 4/4 |
| 40/60 | 0.1403 +/- 0.0006 | 0.1492 +/- 0.0003 | +0.0089 +/- 0.0003 | [+0.0087, +0.0092] | 4/4 |
| 30/70 | 0.1422 +/- 0.0009 | 0.1486 +/- 0.0004 | +0.0064 +/- 0.0006 | [+0.0060, +0.0069] | 4/4 |
| 20/80 | 0.1455 +/- 0.0007 | 0.1487 +/- 0.0006 | +0.0033 +/- 0.0002 | [+0.0031, +0.0034] | 4/4 |
| **10/90** | **0.1503 +/- 0.0003** | **0.1487 +/- 0.0001** | **-0.0016 +/- 0.0003** | **[-0.0019, -0.0013]** | **0/4** |

![Figure 1](results/figure1_truth_bias.png)

*Figure 1. Left: DLoss as a function of the correct data fraction. Truth bias is maintained up to 20/80 and inverts at 10/90. Right: absolute loss — the lines cross at roughly 15%.*

Truth bias decreases strictly monotonically: +0.0115 -> +0.0089 -> +0.0064 -> +0.0033 -> -0.0016. The tipping point lies between 10% and 20% correct data. Compression pressure beats frequency bias up to a fourfold prevalence of incorrect data.

An asymmetry is observed: the loss on correct examples increases substantially (0.1384 -> 0.1504), while the loss on incorrect ones remains nearly stable (0.1499 -> 0.1487). The entire dynamic is driven by the model's ability to learn the rules of correct mathematics.

Statistical significance: 16/16 seeds prefer correct examples at proportions 50/50-20/80. Two-sided binomial test: p = 3.05 x 10^-5. For each proportion individually (4/4 seeds) p = 0.125, which is not significant; however, the combined test across all 16 seeds decisively rejects the null hypothesis of equal preference.

### 4.2 Coherent Errors: Disappearance of Truth Bias

**Table 2.** Three error types at the 50/50 proportion.

| Error Type | Loss (correct) | Loss (incorrect) | DLoss | 95% CI | Seeds -> correct |
|---|---|---|---|---|---|
| Random | 0.1384 +/- 0.0009 | 0.1499 +/- 0.0008 | **+0.0115 +/- 0.0002** | [+0.0113, +0.0116] | **4/4** |
| Contradictory | 0.1406 +/- 0.0009 | 0.1411 +/- 0.0008 | **+0.0005 +/- 0.0001** | [+0.0004, +0.0006] | **4/4** |
| Coherent | 0.1374 +/- 0.0005 | 0.1370 +/- 0.0008 | **-0.0004 +/- 0.0004** | [-0.0006, -0.0001] | **0/4** |

![Figure 3](results/figure3_coherence_spectrum.png)

*Figure 3. Coherence spectrum: DLoss for three error types at 50/50. The less consistent the error system, the stronger the truth bias.*

The results form a spectrum: random errors (a maximally incoherent "theory") yield strong bias; contradictory ones (simple rules that break algebra) yield a weak one; coherent ones (a consistent system) yield zero bias.

### 4.3 Coherent Errors at Different Proportions

**Table 3.** Random vs. coherent errors across proportions.

| Proportion | Random DLoss | Coherent DLoss | Random -> | Coherent -> |
|---|---|---|---|---|
| 50/50 | +0.0115 +/- 0.0002 | -0.0004 +/- 0.0004 | correct (4/4) | incorrect (0/4) |
| 40/60 | +0.0089 +/- 0.0003 | -0.0041 +/- 0.0002 | correct (4/4) | incorrect (0/4) |
| 30/70 | +0.0064 +/- 0.0006 | -0.0083 +/- 0.0003 | correct (4/4) | incorrect (0/4) |
| 20/80 | +0.0033 +/- 0.0002 | -0.0143 +/- 0.0006 | correct (4/4) | incorrect (0/4) |

With random errors, truth bias withstands frequency up to 20/80. With coherent errors, DLoss is negative at all proportions: the model slightly prefers the incorrect system even at 50/50, and this preference strengthens as the share of incorrect data grows. The model follows pure frequency — preferring whichever type is more abundant (or easier to compress at equal proportions). **Truth bias with random errors is a consequence of the incompressibility of ad hoc errors, not an intrinsic property of data "truthfulness."**

![Figure 2](results/figure2_scatter.png)

*Figure 2. Loss across seeds: points above the diagonal indicate truth bias. Coherent errors (diamonds) lie on the diagonal.*

## 5. Experiment 2: Observations and Predictive Power

Experiment 1 showed that a coherent false system compresses just as well as the correct one. But in the real world, false theories diverge from observations. We add a verification component:

```
# Correct theory: a x b = a*b
Prediction: total = 50
Observation: counted 50 items [check]

# Coherent false: a x b = a*(b-1)
Prediction: total = 40
Observation: counted 50 items [cross]
```

**Table 4.** Impact of observation ratio on truth bias (coherent errors, 50/50).

| Observation % | Avg DLoss | 95% CI | Seeds -> correct | p (binom) | Avg Loss (correct) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0% (control) | +0.0005 +/- 0.0003 | [+0.0002, +0.0007] | 4/4 | 0.125 | 0.1414 |
| 10% | +0.0002 +/- 0.0003 | [-0.0001, +0.0004] | 3/4 | 0.625 | 0.1416 |
| 25% | +0.0004 +/- 0.0001 | [+0.0003, +0.0004] | 4/4 | 0.125 | 0.1435 |
| 50% | +0.0008 +/- 0.0006 | [+0.0003, +0.0012] | 3/4 | 0.625 | 0.1471 |

![Figure 4](results/figure4_observations.png)

*Figure 4. DLoss as a function of the observation ratio. The effect is an order of magnitude weaker than with random errors (+0.0115).*

**Result:** Observations do not restore strong truth bias. DLoss remains within the +0.0002 to +0.0008 range. The reason: discrepancies between the false theory and observations are themselves **regular** (the a x b = a x (b-1) rule always understates by a), and the model learns this regularity as an additional rule.

The 100% observations condition led to a loss explosion (~0.32): the corpus became too complex for the tiny model at 5000 steps. These results are excluded.

## 6. Experiment 3: Informational Overhead of Correction

If bare discrepancies fail to produce a strong bias due to their regularity, then **ad hoc escape hatches** — unique explanations for every discrepancy — should be incompressible. This experiment varies the informational overhead required for a false theory to reconcile with observations.

Five conditions for the false theory (the correct theory is identical across all):

**A: No observations** (baseline) — theory without verification.

**B: Bare discrepancies** (Experiment 2, 50% observations) — theory with discrepancies.

**C: Ad hoc correction** — a unique explanation for each discrepancy:
```
Prediction: 10 x 5 = 40. Observation: counted 50.
Explanation: In this case 5 is prime, so we add the base once more.
Corrected: 40 + 10 = 50 [check]
```
Each Explanation is unique — the model cannot compress them into a single rule.

**D: Systematic correction** — a single correction rule for all discrepancies:
```
Correction rule: always add first operand.
Corrected: 40 + 10 = 50 [check]
```
One rule for all problems — compressible.

**E: Non-specific predictions** — theory without a concrete mapping:
```
Prediction: result is moderate. Observation: counted 50.
```

### Results

**Table 5.** DLoss across five conditions (tiny, 3.5M, 50/50, 4 seeds).

| Condition | Description | Avg DLoss | 95% CI | Seeds -> correct |
|-----------|-------------|-----------|--------|------------------|
| A | No observations | +0.0005 +/- 0.0003 | [+0.0002, +0.0007] | 4/4 |
| B | Bare discrepancies | +0.0008 +/- 0.0006 | [+0.0003, +0.0012] | 3/4 |
| E | Non-specific predictions | +0.0015 +/- 0.0009 | [+0.0008, +0.0023] | 4/4 |
| C | Ad hoc correction | +0.0025 +/- 0.0008 | [+0.0020, +0.0033] | 4/4 |
| D | Systematic correction | +0.0026 +/- 0.0005 | [+0.0021, +0.0030] | 4/4 |

![Figure 5](results/figure5_conditions.png)

*Figure 5. Correction overhead spectrum: DLoss for five conditions. Conditions with an explanatory component (C-E) produce noticeable truth bias; conditions A and B produce only a weak one.*

**Actual order: D ~ C > E > B ~ A ~ 0.**

The predicted order (C > B > E > D ~ A) was partially confirmed:
- **A ~ 0** — confirmed (DLoss = +0.0005, weak effect).
- **B < E < C** — confirmed.
- **D ~ C** — **not confirmed.** We expected D ~ A, but obtained D ~ C.

The unexpected D ~ C result is explained by the fact that **the correction format itself is an informational burden.** A correct theory predicts the observation directly; a false one (even if "fixed") requires an extra step. Compression distinguishes not between ad hoc vs. systematic explanations, but between the **presence vs. absence of the need for any correction at all**.

Normalized effect (DLoss/Loss): B — 0.5%, E — 0.6%, C — 1.1%, D — 1.1%. The order is preserved, but the gap between C/D and B/E is less dramatic.

**Caveat:** Absolute loss varies substantially (A: 0.14, B: 0.15, C: 0.23, D: 0.24, E: 0.25), reflecting varying corpus lengths. Models C/D/E are undertrained compared to A/B. Experiments with larger models and/or longer training are needed.

## 7. Discussion

### 7.1 Unified Interpretation

The three experiments paint a progressively clearer picture:

1. **Compression favors consistency, not truth.** Any consistent rule system — true or false — compresses equally well. Truth bias with random errors is explained by the fact that each random error must be memorized individually.

2. **Truth usually wins because errors are usually incoherent.** In real data, different authors make different errors, whereas correct answers are uniform. This aligns with the Truth Co-occurrence Hypothesis (Ravfogel et al., 2025): true statements are more likely to co-occur with other true statements, forming a statistical cluster that the model learns.

3. **The need for correction is the key factor.** When observations are present, a false theory requires an extra step (correction, explanation) that a true theory does not. This creates a measurable informational burden.

### 7.2 Analogy with Popper's Falsifiability

Our results admit an interpretive analogy with the falsifiability criterion (Popper, 1959). Compression pressure acts as a computational analog: a true theory with concrete predictions requires no additional explanations (maximal compression); a false theory whose predictions diverge from data needs correction (poor compression); a theory with ad hoc escape hatches expands with every observation (anti-compression).

However, the analogy has limits. First, the model does not "test" theories — it simply minimizes code length. Second, our data show that bare discrepancies alone barely help (condition B ~ A): regular discrepancies are compressible. Popperian falsification assumes that a discrepancy with observation refutes a theory; for a compressor model, a discrepancy is merely another pattern.

Practical analogies from the history of science and pseudoscience are appropriate as illustrations. For instance, homeopathy defends its claims with ad hoc explanations for every negative result (condition C); astrology makes non-specific predictions that cannot be refuted (condition E); the geocentric model required ever more epicycles to reconcile with observations (a variation of condition C). However, our experiments use the mathematical domain, and transfer to these real-world examples remains an open question.

### 7.3 Implications

**For alignment.** Models lack an innate "truth compass" — they favor well-compressible patterns. Systematic deception consistently represented in data encounters no resistance from compression.

**For ML epistemology.** The framework explains why models develop internal truth representations (Marks & Tegmark, 2023): in real corpora, true statements are more coherent than false ones. It also explains the inverse scaling effect on TruthfulQA (Lin et al., 2022): larger models are better at memorizing coherent misconceptions, which according to our data compress just as well as truth.

**For understanding hallucinations.** Models confidently reproduce coherent misconceptions not due to a compression failure, but because such misconceptions compress *successfully*. This aligns with the analysis by Chlon et al. (2025).

### 7.4 Limitations

**Model scale.** All experiments use 3.5M parameters. The hypothesis predicts that truth bias should decrease as model size grows.

**Domain specificity.** Mathematics has an unusually crisp distinction between correct and incorrect. The effect may be weaker in fuzzy domains.

**Confounding with corpus length.** Conditions C/D/E generate substantially longer texts (loss ~0.24 vs ~0.14). DLoss may partially reflect a difference in convergence rather than compressibility per se.

**Effect size.** DLoss (0.003-0.012) is small in absolute terms. Its practical significance for large models remains an open question.

## 8. Status and Planned Experiments

*This work is in active development. The results above were obtained on a single model size (tiny, 3.5M) in the mathematical domain. Below is the plan for further experiments.*

### 8.1 Scaling by Model Size

Train models from 3.5M to 200M+ parameters on a 50/50 corpus to test the predicted inverse-U curve: models that are too small cannot learn rules, medium ones show maximal truth bias, large ones memorize everything.

### 8.2 Linear Probing

Extract activations and train linear classifiers to detect "truth directions" vs. "coherence directions" (Marks & Tegmark, 2023 methodology). It will be particularly interesting to compare representations across different Experiment 3 conditions.

### 8.3 Synthetic World (Natural Language)

Create a simulated world with known rules; generate texts from two agents (correct and incorrect rules) with observations and ad hoc explanations. Models: 25M-350M.

### 8.4 Real-World Domains

Extend to domains with competing knowledge systems:
- **Type 3b (ad hoc):** Evidence-based medicine vs. homeopathy, vaccination vs. anti-vax theories.
- **Historical:** Phlogiston vs. oxygen theory, miasma theory vs. germ theory, geocentrism vs. heliocentrism.
- **Type 3a (non-specific):** Astrology, market technical analysis.

## 9. Conclusion

We present three findings on the relationship between compression and truth during language model training.

**Compression favors consistency, not truth per se.** Models trained on a mixture of correct and incorrect derivations with random errors consistently exhibit truth bias (16/16 seeds at proportions 50/50-20/80, p = 3.05 x 10^-5). But when random errors are replaced with a coherent alternative system, the bias disappears. The model prefers not truth, but the most compact structure in the data.

**Regular discrepancies with observations barely help.** Adding empirical checks to a coherent false theory yields a DLoss 10-50x weaker than the effect of random errors. If discrepancies are regular, the model learns them as just another rule.

**Informational overhead of correction is the key factor.** Ad hoc correction and systematic correction produce ~3x larger bias than bare discrepancies. A true theory needs no correction — and this gives it an advantage in description length.

These results simultaneously explain why LLMs usually prefer true statements (in real corpora, errors are typically incoherent) and why they confidently reproduce systematic misconceptions (a coherent false system is indistinguishable from truth for a compressor model).

## References

Azaria, A., & Mitchell, T. (2023). The Internal State of an LLM Knows When It's Lying. *Findings of EMNLP 2023*.

Burger, L., Hamprecht, F. A., & Nadler, B. (2024). Truth is Universal: Robust Detection of Lies in LLMs. *NeurIPS 2024*.

Burns, C., Ye, H., Klein, D., & Steinhardt, J. (2023). Discovering Latent Knowledge in Language Models Without Supervision. *ICLR 2023*.

Chlon, L., et al. (2025). Predictable Compression Failures: Why Language Models Actually Hallucinate. *arXiv:2509.11208*.

Deletang, G., Ruoss, A., Grau-Moya, J., Genewein, T., Wenliang, L. K., Catt, E., ... & Legg, S. (2024). Language Modeling Is Compression. *ICLR 2024*.

DeMoss, B., Sapora, S., Foerster, J., Hawes, N., & Posner, I. (2024). The Complexity Dynamics of Grokking. *arXiv:2412.09810*.

Goldblum, M., Finzi, M., Rowan, K., & Wilson, A. G. (2024). The No Free Lunch Theorem, Kolmogorov Complexity, and the Role of Inductive Biases in Machine Learning. *ICML 2024*.

Gurnee, W., & Tegmark, M. (2024). Language Models Represent Space and Time. *ICLR 2024*.

Halawi, D., Denain, J.-S., & Steinhardt, J. (2024). Overthinking the Truth: Understanding how Language Models Process False Demonstrations. *ICLR 2024*.

Huang, Y., Sun, Y., Wang, X., & Yang, Y. (2024). Compression Represents Intelligence Linearly. *COLM 2024*.

Joshi, N., Rando, J., Saparov, A., Kim, N., & He, H. (2024). Personas as a Way to Model Truthfulness in Language Models. *EMNLP 2024*.

Kadavath, S., et al. (2022). Language Models (Mostly) Know What They Know. *arXiv:2207.05221*.

Li, K., Hopkins, A. K., Bau, D., Viegas, F., Pfister, H., & Wattenberg, M. (2023a). Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task. *ICLR 2023*.

Li, K., Patel, O., Viegas, F., Pfister, H., & Wattenberg, M. (2023b). Inference-Time Intervention: Eliciting Truthful Answers from a Language Model. *NeurIPS 2023*.

Lin, S., Hilton, J., & Evans, O. (2022). TruthfulQA: Measuring How Models Mimic Human Falsehoods. *ACL 2022*.

Liu, Z., Zhong, Z., & Tegmark, M. (2023). Grokking as Compression: A Nonlinear Complexity Perspective. *arXiv:2310.05918*.

Marks, S., & Tegmark, M. (2023). The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets. *arXiv:2310.06824*.

Mingard, C., Valle-Perez, G., Sherrington, D., & Louis, A. A. (2021). Is SGD a Bayesian Sampler? Well, Almost. *JMLR 2021*.

Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). Progress Measures for Grokking via Mechanistic Interpretability. *ICLR 2023*.

Pan, Z., Wang, S., & Li, J. (2025). Understanding LLM Behaviors via Compression: Data Generation, Knowledge Acquisition and Scaling Laws. *arXiv:2504.09597*.

Popper, K. (1959). The Logic of Scientific Discovery. *Hutchinson*.

Shannon, C. E. (1948). A Mathematical Theory of Communication. *Bell System Technical Journal*, 27(3), 379-423.

Solomonoff, R. J. (1964). A Formal Theory of Inductive Inference. *Information and Control*, 7(1), 1-22.

Hutter, M. (2005). Universal Artificial Intelligence: Sequential Decisions Based on Algorithmic Probability. *Springer*.

Ravfogel, S., Yehudai, G., Linzen, T., Bietti, A., & Bruna, J. (2025). Emergence of Linear Truth Encodings in Language Models. *NeurIPS 2025*.

Valle-Perez, G., Camargo, C. Q., & Louis, A. A. (2019). Deep Learning Generalizes Because the Parameter-Function Map Is Biased Towards Simple Functions. *ICLR 2019*.

Wan, J., & Mei, L. (2025). Large Language Models as Computable Approximations to Solomonoff Induction. *arXiv:2505.15784*.

## Appendix A: Reproducibility

All code, data generation scripts, and evaluation scripts are available at https://github.com/Rai220/compression-drives-truth. Experiments were conducted on an Apple Mac M4 with 36GB of unified memory using the MLX framework (v0.31.0). Total computational cost: approximately 20 hours of wall-clock time for the 69 training runs described in this paper.
