# When Does Compression Favor Truth? Consistency, Description Length, and Inductive Bias in Language Models

**Author:** Konstantin Krestnikov
**Date:** 03.2026

> **Note:** This is an English translation of the main draft `paper_draft_ru.md`. Please apply all substantive changes to the Russian version first.

## Abstract

Language models minimize cross-entropy loss, which is mathematically equivalent to compressing the training data. We investigate whether this compression pressure gives rise to a systematic preference for correct information in models trained on mixed-quality corpora. Crucially, models compress *text*, not reality; the observed bias reflects corpus statistics, not access to external truth.

We train 85 transformers (3.5M--26M parameters) on corpora with controlled ratios of correct and incorrect mathematical derivations. With random (incoherent) errors, models consistently prefer correct solutions: paired evaluation yields 83% accuracy at 50/50 (Wilcoxon p < 10^-6), the effect persists at 10/90 (67%, p < 10^-88), and strengthens with scale (83.6% -> 88.5% from 3.5M to 26M). However, replacing random errors with a coherent alternative rule system -- internally consistent but mathematically wrong -- eliminates the truth preference entirely (accuracy ~49% at any model size).

Compression favors not truth, but the most consistent structure in the data. Truth bias arises because random errors are incoherent and must be memorized individually, whereas a coherent false system compresses just as efficiently as truth. This explains why language models typically prefer true statements (errors in real corpora are diverse) and why they confidently reproduce systematic misconceptions (coherent falsehood is indistinguishable from truth for a compressor).

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
4. **Experiment 4** (Section 7): Scaling by model size.

## 2. Related Work

### Prediction as Compression

The link between prediction and data compression traces back to the foundational work in information theory. Shannon (1948) showed that optimal compression requires knowledge of the true data distribution, and Solomonoff (1964) formalized optimal prediction as weighting hypotheses by their program length. Hutter (2005) developed these ideas into a formal theory of universal artificial intelligence (AIXI), explicitly linking intelligence to compression ability.

In the context of language models, Deletang et al. (2024) empirically demonstrated that LLMs are universal compressors: a next-token predictor can serve as an arithmetic coder. Huang et al. (2024) discovered a linear correlation (r ~ -0.95) between compression quality and benchmark performance, and Wan & Mei (2025) formally proved that LLM training approximates Solomonoff induction. These results form the theoretical foundation of our work: if LLM training is compression, and compression is linked to intelligence, what role does data truthfulness play?

### Internal Representations of Truth in LLMs

Several studies have found that language models form internal representations correlated with statement truthfulness. Marks & Tegmark (2023) showed a linear geometric structure of truthfulness in activation space (Geometry of Truth), and Burns et al. (2023) proposed CCS, a method for discovering truth directions without supervision. Li et al. (2023b) identified a 40% gap between a model's internal knowledge and its generation, developing Inference-Time Intervention (ITI). Ravfogel et al. (2025) proposed the Truth Co-occurrence Hypothesis -- a mechanism for the emergence of linear truth representations through co-occurrence of true statements in the corpus. Our work complements this line of research: we show *under what conditions* compression gives rise to such representations, and under what conditions it does not.

### Emergent World Models

Language models can form internal world models from pure text prediction. Li et al. (2023a) trained a model to predict Othello moves and found that it learns a full board representation (Othello-GPT). Gurnee & Tegmark (2024) discovered linear representations of space and time in Llama-2 activations. These results show that sequence compression can give rise to structured internal representations -- our work investigates whether such representations gravitate towards true structures.

### Simplicity Bias and Grokking

The inductive bias of neural networks towards simple functions is a well-documented phenomenon. Valle-Perez et al. (2019) showed an exponential preference for low-complexity functions, and Mingard et al. (2021) proved that SGD approximates Bayesian sampling with a simplicity prior. Goldblum et al. (2024) connected this to Kolmogorov complexity, providing a theoretical basis for the link between compression and generalization.

The phenomenon of grokking -- delayed generalization -- is also related to compression. Nanda et al. (2023) showed that networks discover Fourier transforms for modular arithmetic, and DeMoss et al. (2024) described the phase transition from memorization to generalization through complexity dynamics. Liu et al. (2023) interpreted grokking as a compression process: the network transitions from memorization to a compact representation. Our experiments with coherent errors directly connect to simplicity bias: a coherent false system is just as "simple" as truth, and compression shows no preference for either.

### Our Contribution

The works listed above either study internal truth representations in already-trained models or establish theoretical links between compression and intelligence. However, a direct experiment -- training a model from scratch on a quality-controlled corpus and measuring truth preference as a function of error type, coherence, and presence of empirical feedback -- has not been conducted. This work fills that gap and isolates the conditions under which compression pressure aligns with truth, as opposed to coherent falsehood.

## 3. Methodology

### 3.1 Model and Training

GPT-2 style decoder-only transformer implemented in MLX. Pre-norm (LayerNorm before attention/MLP), GELU activation, causal mask.

| Config | Layers | d_model | Heads | Parameters |
|--------|--------|---------|-------|------------|
| tiny | 4 | 256 | 4 | 3.5M |
| small | 6 | 384 | 6 | 11M |
| medium | 8 | 512 | 8 | 26M |
| large | 12 | 768 | 12 | 86M |

Experiments 1--3 use the tiny config; Experiment 4 (Section 7) repeats the key conditions on tiny through medium (large is in progress). Optimizer: AdamW (weight_decay=0.01), cosine decay with linear warmup (200 steps), lr=3e-4, seq_len=256, batch_size=32, 5000 steps. All experiments are repeated with 4 random initializations (seeds 42--45).

**Justification for the number of seeds.** With 4 repetitions, the binomial test for a single condition (4/4 seeds) yields a minimum p = 0.125, insufficient for significance. We compensate in two ways: (1) a combined test across multiple conditions sharing a single hypothesis (16/16 seeds -> p = 3.05 x 10^-5); (2) paired evaluation on 4,951 problem pairs, where the Wilcoxon signed-rank test provides high statistical power (p < 10^-6) for each individual seed. Thus, the limited number of seeds is compensated by the large number of pairwise comparisons within each seed.

### 3.2 Corpus Generation

The generator creates mathematical problems of four types: multi-step arithmetic, factorization, equation solving, and differentiation. Each problem is formatted as a step-by-step solution in English, verified by SymPy. The tokenizer is character-level (vocab size = 57) to exclude BPE artifacts as a confound.

**Error Types:**
- **Random:** Injection of one plausible error at a random step (sign, coefficient, distributivity error). Each error is unique.
- **Coherent:** One systematic incorrect rule per problem type (e.g., a x b = a x (b-1); sign is preserved when moving terms across =; etc.). All problems of one type fail identically.
- **Contradictory:** Simple rules (a + b = a + b + 1; a - b = a - b - 2) that break algebraic structure -- addition and subtraction cease to be inverse operations.

### 3.3 Metrics

**Corpus-level evaluation.** For each trained model, we compute the average cross-entropy loss on held-out sets of correct (5K problems) and incorrect (5K problems) examples. The main metric is **DLoss = Loss(incorrect) - Loss(correct)**. A positive value indicates truth bias.

**Paired evaluation.** To eliminate the confound of different prompts, we additionally use paired tests: for each problem, a single shared prompt is generated along with two completions (correct and incorrect). NLL is computed only on completion tokens, conditioned on the shared prompt. This yields pairwise comparison under identical context. Metrics: mean DLoss on completions, pair accuracy (fraction of pairs where the model prefers correct), Wilcoxon signed-rank test.

**Statistical analysis.** Each configuration is repeated with 4 random initializations (seeds 42--45). For individual configurations we use the two-sided binomial test (H0: P(correct) = 0.5). 95% confidence intervals for DLoss are obtained via bootstrap (10,000 resamples). For paired evaluation, the Wilcoxon signed-rank test is used on paired NLL differences.

### 3.4 Theoretical Framework: Description Length and Theory Types

To interpret Experiments 2--3 we use a typology of theories distinguished by the description length of the corpus "theory + observations." The key principle: the model optimizes cross-entropy, which is equivalent to minimizing expected code length (Shannon, 1948). A theory that allows shorter encoding of the corpus gains an advantage.

**Type 1: True theory with concrete predictions.** Predictions match observations. The "theory + observations" corpus compresses maximally: one rule system explains everything.

**Type 2: False theory with concrete predictions.** Predictions diverge from observations. The model must encode both the false rules and the discrepancies. However, if the discrepancies are **regular** (e.g., a x b = a x (b-1) always understates by a), the model can learn a correction, and the additional description length is small.

**Type 3a: Theory with non-specific predictions.** The theory does not specify a "situation -> outcome" mapping (e.g., "result is moderate"). It does not contradict observations but does not help predict them either -- it does not reduce code length.

**Type 3b: Theory with ad hoc correction.** Each discrepancy is explained by a unique exception rule. Description length grows linearly with the number of observations -- this is anti-compression.

### 3.5 Experiment Conditions

**Experiment 1:** 5 proportions (50/50--10/90) x 4 seeds = 20 models with random errors + 1 baseline. Controls: coherent errors (4 proportions x 4 seeds = 16) and contradictory (4 seeds).

**Experiment 2:** Coherent errors at 50/50 with observations. 4 observation ratios (0%, 10%, 25%, 50%) x 4 seeds = 16 models. Test sets contain no observations -- we measure pure mathematical prediction quality.

**Experiment 3:** 5 conditions for the false theory (A--E) at 50/50. Conditions A and B are from Experiments 1--2. Conditions C, D, E -- 3 x 4 seeds = 12 new models.

**Experiment 4:** Scaling -- random 50/50 and coherent 50/50 at small (11M) and medium (26M) sizes, 4 seeds each = 16 models (+ tiny from Experiment 1).

In total, **85 models** were trained (69 in Experiments 1--3 + 16 in Experiment 4).

## 4. Experiment 1: Random, Coherent, and Contradictory Errors

**Hypothesis.** If compression gives rise to truth bias, then models trained on a mixture of correct and incorrect derivations should show lower loss on correct examples (DLoss > 0), with the effect depending on error coherence: random (incoherent) errors -> strong bias; coherent (systematic) -> weak or zero.

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

*Figure 1. Left: DLoss as a function of the correct data fraction. Truth bias is maintained up to 20/80 and inverts at 10/90 at the corpus level. Right: absolute loss -- the lines cross at roughly 15%.*

Corpus-level truth bias decreases strictly monotonically: +0.0115 -> +0.0089 -> +0.0064 -> +0.0033 -> -0.0016. The corpus-level tipping point lies between 10% and 20% correct data. Compression pressure beats frequency bias up to a fourfold prevalence of incorrect data.

An asymmetry is observed: the loss on correct examples increases substantially (0.1384 -> 0.1504), while the loss on incorrect ones remains nearly stable (0.1499 -> 0.1487). The entire dynamic is driven by the model's ability to learn the rules of correct mathematics.

Statistical significance: 16/16 seeds prefer correct examples at proportions 50/50--20/80. Two-sided binomial test: p = 3.05 x 10^-5. For each proportion individually (4/4 seeds) p = 0.125, which is not significant; however, the combined test across all 16 seeds decisively rejects the null hypothesis of equal preference.

However, paired evaluation (see below) shows that even at 10/90 the model retains truth bias at the pair level -- the corpus-level inversion reflects a frequency effect, not a loss of the structural advantage of correct solutions.

**Paired evaluation (50/50, random errors).** To eliminate the confound of different prompts, we conducted paired evaluation on 4,951 problem pairs with a shared prompt and two completions. The result substantially strengthens the main finding:

**Table 1a.** Paired evaluation at 50/50 (random errors). DLoss = NLL(incorrect) - NLL(correct) on completion tokens.

| Seed | DLoss (paired) | Pair accuracy | 95% CI | Wilcoxon p |
|------|:-:|:-:|:-:|:-:|
| 42 | +0.0478 | 81.5% | [+0.046, +0.050] | <10^-6 |
| 43 | +0.0494 | 84.2% | [+0.047, +0.052] | <10^-6 |
| 44 | +0.0483 | 86.0% | [+0.046, +0.050] | <10^-6 |
| 45 | +0.0465 | 80.8% | [+0.044, +0.049] | <10^-6 |
| **Avg** | **+0.0480** | **83.1%** | -- | -- |

Given the same prompt, the model assigns lower NLL to the correct solution 83% of the time. The effect is ~4x larger than the corpus-level estimate (+0.048 vs +0.012), since the paired metric isolates the diverging portion of the solution from the shared problem format.

By problem type, the effect varies: algebra (accuracy 99.9%) > arithmetic (94%) > derivatives (72%) > equations (65%). Algebra shows the cleanest signal: the model virtually always prefers the correct factorization.

**Paired evaluation across proportions.** The effect decreases monotonically but remains significant at all proportions:

**Table 1b.** Paired evaluation across proportions (random errors, 4 seeds, Wilcoxon p < 10^-6 for all).

| Proportion | Avg DLoss (paired) | Pair accuracy | Corpus DLoss |
|:----------:|:------------------:|:------------:|:------------:|
| 50/50 | +0.048 | 83% | +0.0115 |
| 40/60 | +0.043 | 79% | +0.0089 |
| 30/70 | +0.036 | 75% | +0.0064 |
| 20/80 | +0.029 | 69% | +0.0033 |
| **10/90** | **+0.017** | **67%** | **-0.0016** |

Notably, at 10/90 the corpus-level metric inverts (DLoss = -0.0016, the model on average "prefers" incorrect examples due to their 9-fold prevalence), while paired evaluation consistently shows truth bias (67% accuracy, p < 10^-88). This means that the structural advantage of correct solutions persists even under extreme imbalance -- the corpus-level inversion reflects a frequency effect on shared problem patterns, not a loss of the model's discriminative ability at the level of individual solutions.

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

**Paired evaluation confirms the spectrum.** Paired evaluation (same prompt, two completions) reinforces the picture:

**Table 2a.** Paired evaluation for three error types at 50/50 (4 seeds).

| Error Type | Avg DLoss (paired) | Pair accuracy | Wilcoxon p |
|---|:-:|:-:|:-:|
| Random | **+0.048** | **83%** | <10^-6 |
| Contradictory | +0.0003 | 49% | >0.3 |
| Coherent | -0.0018 | 47% | ~1.0 |

Given the same prompt, the model confidently prefers correct only for random errors. For coherent and contradictory errors -- accuracy ~50% (chance). This eliminates the prompt confound and confirms: truth bias is a consequence of error incompressibility, not a property of "truthfulness."

### 4.3 Coherent Errors at Different Proportions

**Table 3.** Random vs. coherent errors across proportions.

| Proportion | Random DLoss | Coherent DLoss | Random -> | Coherent -> |
|---|---|---|---|---|
| 50/50 | +0.0115 +/- 0.0002 | -0.0004 +/- 0.0004 | correct (4/4) | incorrect (0/4) |
| 40/60 | +0.0089 +/- 0.0003 | -0.0041 +/- 0.0002 | correct (4/4) | incorrect (0/4) |
| 30/70 | +0.0064 +/- 0.0006 | -0.0083 +/- 0.0003 | correct (4/4) | incorrect (0/4) |
| 20/80 | +0.0033 +/- 0.0002 | -0.0143 +/- 0.0006 | correct (4/4) | incorrect (0/4) |

With random errors, truth bias withstands frequency up to 20/80. With coherent errors, DLoss is negative at all proportions: the model slightly prefers the incorrect system even at 50/50, and this preference strengthens as the share of incorrect data grows. The model follows pure frequency -- preferring whichever type is more abundant (or easier to compress at equal proportions). **Truth bias with random errors is a consequence of the incompressibility of ad hoc errors, not an intrinsic property of data "truthfulness."**

![Figure 2](results/figure2_scatter.png)

*Figure 2. Loss across seeds: points above the diagonal indicate truth bias. Coherent errors (diamonds) lie on the diagonal.*

## 5. Experiment 2: Observations and Predictive Power

**Hypothesis.** Adding empirical feedback (observations) should increase the description length of the false theory by introducing discrepancies, thereby restoring truth bias for coherent errors.

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

**Result: the hypothesis is not supported.** Observations do not restore strong truth bias. DLoss remains within the +0.0002 to +0.0008 range. The reason: discrepancies between the false theory and observations are themselves **regular** (the a x b = a x (b-1) rule always understates by a), and the model learns this regularity as an additional rule.

The 100% observations condition led to a loss explosion (~0.32): the corpus became too complex for the tiny model at 5000 steps. These results are excluded.

## 6. Experiment 3: Informational Overhead of Correction

**Hypothesis.** If bare discrepancies fail to produce a strong bias due to their regularity, then ad hoc explanations -- unique for each discrepancy -- should be incompressible and restore truth bias. Expected ordering: C (ad hoc) > B (bare discrepancies) > E (non-specific) > D (systematic correction) ~ A (no observations).

Five conditions for the false theory (the correct theory is identical across all):

**A: No observations** (baseline) -- theory without verification.

**B: Bare discrepancies** (Experiment 2, 50% observations) -- theory with discrepancies.

**C: Ad hoc correction** -- a unique explanation for each discrepancy:
```
Prediction: 10 x 5 = 40. Observation: counted 50.
Explanation: In this case 5 is prime, so we add the base once more.
Corrected: 40 + 10 = 50 [check]
```
Each Explanation is unique -- the model cannot compress them into a single rule.

**D: Systematic correction** -- a single correction rule for all discrepancies:
```
Correction rule: always add first operand.
Corrected: 40 + 10 = 50 [check]
```
One rule for all problems -- compressible.

**E: Non-specific predictions** -- theory without a concrete mapping:
```
Prediction: result is moderate. Observation: counted 50.
```

### Results

To test whether conditions C/D/E produce transferable truth bias, we begin with paired evaluation -- the most reliable metric, isolating preference for correctness from textual confounds.

**Table 5a.** Paired evaluation of conditions C/D/E (coherent pairs, 4 seeds).

| Condition | Avg DLoss (paired) | Pair accuracy | Wilcoxon p |
|-----------|:------------------:|:------------:|:----------:|
| C (ad hoc) | -0.0019 | 48.2% | >0.3 |
| D (systematic) | -0.0011 | 49.7% | >0.3 |
| E (non-specific) | -0.0007 | 49.6% | >0.3 |

**Paired evaluation reveals no truth bias for any of the C/D/E conditions.** Pair accuracy ~49% -- at chance level. Training with observations and correction **does not produce transferable truth bias**: the model learns to process correction patterns within the training corpus context, but does not transfer this discrimination to pure mathematical pairs without observations.

**The corpus-level metric, by contrast, registers a non-zero effect.** This divergence is methodologically significant:

**Table 5b.** Corpus-level DLoss across five conditions (tiny, 3.5M, 50/50, 4 seeds).

| Condition | Description | Avg DLoss | 95% CI | Seeds -> correct |
|-----------|-------------|-----------|--------|------------------|
| A | No observations | +0.0005 +/- 0.0003 | [+0.0002, +0.0007] | 4/4 |
| B | Bare discrepancies | +0.0008 +/- 0.0006 | [+0.0003, +0.0012] | 3/4 |
| E | Non-specific predictions | +0.0015 +/- 0.0009 | [+0.0008, +0.0023] | 4/4 |
| C | Ad hoc correction | +0.0025 +/- 0.0008 | [+0.0020, +0.0033] | 4/4 |
| D | Systematic correction | +0.0026 +/- 0.0005 | [+0.0021, +0.0030] | 4/4 |

![Figure 5](results/figure5_conditions.png)

*Figure 5. Corpus-level DLoss for five conditions. The ordering D ~ C > E > B ~ A does not reflect a preference for correctness, but is an artifact of differences in text statistics (see Table 5a).*

The actual corpus-level ordering is D ~ C > E > B ~ A. The predicted ordering (C > B > E > D ~ A) was partially confirmed: A ~ 0 and B < E < C hold, but D ~ C instead of the expected D ~ A. However, the divergence between corpus-level and paired evaluation shows that **the entire corpus-level effect for conditions C/D/E is an artifact of differences in text statistics** (different length and format of correct vs. incorrect corpora), not a preference for correctness given the same prompt.

**Caveat:** Absolute loss varies substantially (A: 0.14, B: 0.15, C: 0.23, D: 0.24, E: 0.25), reflecting varying corpus lengths. Models C/D/E are undertrained compared to A/B.

**Methodological takeaway.** This result demonstrates the importance of paired evaluation: corpus-level DLoss can systematically overestimate truth bias when correct and incorrect corpora differ in format, length, or style. The only reliable source of truth bias is **incoherence of the errors themselves** (Experiment 1, random errors), confirmed by both metrics.

## 7. Experiment 4: Scaling by Model Size

**Hypothesis.** If truth bias is driven by a structural advantage in compression, then increasing model capacity should strengthen the effect (more capacity to learn regularities). Coherent errors should remain indistinguishable from truth at any size.

To test robustness, we trained models at three sizes on random 50/50 and coherent 50/50 corpora (4 seeds each).

### 7.1 Model Configurations

| Size | Parameters | d_model | Heads | Layers |
|------|-----------|---------|-------|--------|
| tiny | 3.5M | 256 | 4 | 4 |
| small | 11M | 384 | 6 | 6 |
| medium | 26M | 512 | 8 | 8 |
| large | 86M | 768 | 12 | 12 |

All models trained for 5000 steps on the same corpus. Architecture: GPT-2 (decoder-only transformer) with character-level tokenization. For medium, 3 of 4 seeds are complete (seed 45 is in progress).

### 7.2 Results: Truth Bias Grows with Model Size

**Table 6.** Paired evaluation (random 50/50) by model size.

| Size | Parameters | Avg DLoss (paired) | Pair accuracy | Corpus DLoss | Seeds |
|------|-----------|:------------------:|:------------:|:------------:|:-----:|
| tiny | 3.5M | +0.048 | 83.6% | +0.0115 | 4 |
| small | 11M | +0.063 | 88.4% | +0.0129 | 4 |
| medium | 26M | +0.067 | 88.5% | +0.0130 | 3 |
| large | 86M | -- | -- | -- | -- |

*The large row will be populated upon training completion.*

**Table 6a.** Paired accuracy by problem type.

| Type | Tiny (3.5M) | Small (11M) | Medium (26M) |
|------|:-----------:|:-----------:|:------------:|
| Algebra | 99.9% | 100.0% | 100.0% |
| Arithmetic | 95.2% | 98.2% | 98.6% |
| Derivatives | 72.4% | 81.6% | 82.4% |
| Equations | 65.9% | 72.8% | 72.1% |

Truth bias monotonically increases from tiny to medium: +40% in paired DLoss (+0.048 -> +0.067) and +4.9 pp in pair accuracy (83.6% -> 88.5%). The largest gain occurs between tiny and small; between small and medium, accuracy essentially plateaus (88.4% -> 88.5%), although DLoss continues to grow (+0.063 -> +0.067). Improvement is most pronounced in difficult problem types (derivatives: +10.0 pp, equations: +6.2 pp from tiny to medium), while algebra and arithmetic reach saturation already at small.

**Coherent errors still show no bias.** Pair accuracy for coherent 50/50 on the small model is 49.6% (DLoss = -0.0006), i.e. at chance, same as tiny (47.2%). Increasing model capacity does not help distinguish coherent falsehood from truth. The hypothesis is confirmed: scaling strengthens truth bias for random errors but is powerless against coherent ones.

**Scaling conclusion.** The inverse-U hypothesis (growth -> peak -> decline) is not supported: truth bias monotonically increases with capacity in the 3.5M--26M range. However, pair accuracy begins to slow between small and medium, potentially indicating an approaching ceiling for this domain. Possible explanations: (1) for the mathematical domain with character-level tokenization, memorization loses to generalization at any reasonable size; (2) the accuracy ceiling is determined by task difficulty (equations: ~72%), not model capacity.

## 8. Discussion

### 8.1 Unified Interpretation

The four experiments paint a progressively clearer picture:

1. **Compression favors consistency, not truth.** Any consistent rule system -- true or false -- compresses equally well. Truth bias with random errors is explained by the fact that each random error must be memorized individually.

2. **Truth usually wins because errors are usually incoherent.** In real data, different authors make different errors, whereas correct answers are uniform. This aligns with the Truth Co-occurrence Hypothesis (Ravfogel et al., 2025): true statements are more likely to co-occur with other true statements, forming a statistical cluster that the model learns.

3. **Corpus-level and paired metrics can diverge.** At 10/90, corpus-level DLoss inverts (-0.0016), but paired evaluation shows robust truth bias (67% accuracy, p < 10^-88). The corpus-level metric conflates structural preference with the frequency effect on shared problem patterns. An analogous divergence is observed for conditions C/D/E: corpus-level DLoss is positive, but paired evaluation shows accuracy ~49%. This makes paired evaluation a necessary tool for truth bias research.

4. **Correction increases corpus-level DLoss but does not produce transferable truth bias.** Models trained with observations and correction do not distinguish correct from incorrect at the level of pure mathematical pairs (accuracy ~49%). The only reliable mechanism of truth bias is error incoherence.

5. **Truth bias scales with model size, but coherent falsehood remains invulnerable.** Increasing the model from 3.5M to 26M strengthens truth bias for random errors (+40% in DLoss), but does not help distinguish coherent falsehood from truth. Scaling enhances generalization, but does not create a "truth compass."

### 8.2 Analogy with Popper's Falsifiability

Our results admit an interpretive analogy with the falsifiability criterion (Popper, 1959). Compression pressure acts as a computational analog: a true theory with concrete predictions requires no additional explanations (maximal compression); a false theory whose predictions diverge from data needs correction (poor compression); a theory with ad hoc escape hatches expands with every observation (anti-compression).

However, the analogy has limits. First, the model does not "test" theories -- it simply minimizes code length. Second, our data show that bare discrepancies alone barely help (condition B ~ A): regular discrepancies are compressible. Popperian falsification assumes that a discrepancy with observation refutes a theory; for a compressor model, a discrepancy is merely another pattern. Moreover, paired evaluation of conditions C/D/E showed that even ad hoc correction does not produce transferable truth bias -- the model learns to process correction patterns but does not transfer this to pure mathematical pairs.

Practical analogies from the history of science are appropriate as illustrations. The geocentric model required ever more epicycles to reconcile with observations (a variation of condition C); phlogiston theory needed special assumptions to explain mass increase during combustion; miasma theory could not explain why disease spread along waterways rather than by wind. However, our experiments use the mathematical domain, and transfer to these real-world examples remains an open question.

### 8.3 Implications

**For alignment.** Models lack an innate "truth compass" -- they favor well-compressible patterns. Systematic deception consistently represented in data encounters no resistance from compression.

**For ML epistemology.** The framework explains why models develop internal truth representations (Marks & Tegmark, 2023): in real corpora, true statements are more coherent than false ones. It also explains the inverse scaling effect on TruthfulQA (Lin et al., 2022): larger models are better at memorizing coherent misconceptions, which according to our data compress just as well as truth.

**For understanding hallucinations.** Models confidently reproduce coherent misconceptions not due to a compression failure, but because such misconceptions compress *successfully*. This aligns with the analysis by Chlon et al. (2025).

### 8.4 Limitations

**Model scale.** Experiments use models from 3.5M to 26M parameters. Truth bias grows with size (Section 7), but pair accuracy begins to plateau between 11M and 26M. The range remains limited -- extrapolation to GPT-2/3 scale models requires further experiments.

**Domain specificity.** Mathematics has an unusually crisp distinction between correct and incorrect. The effect may be weaker in fuzzy domains.

**Confounding with corpus length.** Conditions C/D/E generate substantially longer texts (loss ~0.24 vs ~0.14). DLoss may partially reflect a difference in convergence rather than compressibility per se. Paired evaluation mitigates but does not fully eliminate this confound.

**Training duration.** All models are trained for 5000 steps. As models grow from tiny to medium, the number of parameters increases but the number of training steps does not. Larger models may be undertrained, potentially underestimating truth bias for medium. Conversely, if medium were to reach full convergence, pair accuracy might exceed 88.5%.

**Effect size.** DLoss (0.003--0.012) is small in absolute terms. Its practical significance for large models remains an open question.

### 8.5 Future Experiments

**Linear probing.** Extract activations and train linear classifiers to detect "truth directions" vs. "coherence directions" (Marks & Tegmark, 2023 methodology).

**Synthetic world (natural language).** Create a simulated world with known rules; generate texts from two agents (correct and incorrect rules) with observations and ad hoc explanations.

**Real-world domains.** Extend to domains with competing knowledge systems:
- **Type 3b (ad hoc):** Evidence-based medicine vs. homeopathy, vaccination vs. anti-vax theories.
- **Historical:** Phlogiston vs. oxygen theory, miasma theory vs. germ theory, geocentrism vs. heliocentrism.
- **Type 3a (non-specific):** Astrology, market technical analysis.

## 9. Conclusion

This work isolates the conditions under which compression pressure during language model training aligns with truth. The central finding: **truth bias is not a fundamental property of compression, but a consequence of error incoherence in the corpus.** Random errors are incompressible and must be memorized individually, giving correct mathematics a structural advantage (83% pair accuracy, 16/16 seeds). A coherent false system, as compact as truth, strips compression of any preference (~49% pair accuracy at any model size from 3.5M to 26M).

The practical implication for alignment: scaling strengthens truth bias for incoherent errors but is powerless against coherent falsehood. A compressor model has no "truth compass" -- it has a consistency compass. In real corpora, these compasses typically coincide, since different authors' errors are diverse while correct answers are uniform. But where falsehood is systematic and internally consistent -- in entrenched misconceptions, ideological narratives, coherent pseudoscientific systems -- compression gives the model no basis to prefer truth.

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

All code, data generation scripts, and evaluation scripts are available at https://github.com/Rai220/compression-drives-truth. Experiments were conducted on an Apple Mac M4 with 36GB of unified memory using the MLX framework (v0.31.0). Total computational cost: approximately 30 hours of wall-clock time for the 85 training runs described in this paper.
