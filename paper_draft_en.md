# When Does Compression Favor Truth? Consistency, Description Length, and Inductive Bias in Language Models

**Author:** Konstantin Krestnikov
**Date:** 03.2026

> **Note:** This is an English translation of the main draft `paper_draft_ru.md`. Please apply all substantive changes to the Russian version first.

## Abstract

Language models minimize cross-entropy loss, which is mathematically equivalent to compressing the training data. We investigate whether this compression pressure gives rise to a systematic preference for correct information in models trained on mixed-quality corpora. Crucially, models compress *text*, not reality; the observed bias reflects corpus statistics, not access to external truth.

We train over 120 transformers (3.5M--86M parameters) on corpora with controlled ratios of correct and incorrect mathematical derivations. With random (incoherent) errors, models consistently prefer correct solutions: paired evaluation yields 83% accuracy at 50/50 (Wilcoxon p < 10^-6), the effect persists at 10/90 (67%, p < 10^-88), and strengthens with scale (83.6% -> 89.4% from 3.5M to 86M). The effect reproduces in a natural language domain (a synthetic world with 15 rules), albeit weaker (57.7% pair accuracy). However, replacing random errors with a coherent alternative rule system -- internally consistent but mathematically wrong -- eliminates the truth preference entirely (accuracy ~49% at any model size).

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
4. **Experiment 4** (Section 7): Scaling by model size, multi-rule errors, synthetic world, and cross-domain falsification.

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

Experiments 1--3 use the tiny config; Experiment 4 (Section 7) repeats the key conditions across all sizes up to large (86M). Optimizer: AdamW (weight_decay=0.01), cosine decay with linear warmup (200 steps), lr=3e-4, seq_len=256, batch_size=32, 5000 steps. All experiments are repeated with 4 random initializations (seeds 42--45).

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

**Experiment 4:** Scaling -- random 50/50 and coherent 50/50 at small (11M), medium (26M), and large (86M) sizes, 4 seeds each (2 seeds for large) = 20 models (+ tiny from Experiment 1). Additional sub-experiments: multi-rule errors (16 models), synthetic world (8 models), cross-domain falsification (16 models).

In total, **over 120 models** were trained (69 in Experiments 1--3 + 20 in Experiment 4 scaling + 16 multi-rule + 8 synthetic world + 16 cross-domain).

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

**Paired evaluation sharpens the contrast.** For coherent errors, paired evaluation reveals a symmetric picture: the model prefers whichever system is in the majority.

**Table 3a.** Paired evaluation for coherent errors across proportions (4 seeds).

| Proportion | Random accuracy | Coherent accuracy | Coherent DLoss (paired) |
|:---------:|:-------------:|:-----------------:|:--------------------:|
| 50/50 | 83% | 47.2% | -0.002 |
| 40/60 | 79% | 27.8% | -0.009 |
| 30/70 | 75% | 14.7% | -0.019 |
| 20/80 | 69% | 9.6% | -0.033 |

The contrast is striking: at 20/80, the model with random errors still prefers truth (69%), whereas the model with coherent errors actively prefers the false system (accuracy 9.6%, i.e. in 91% of pairs the model assigns lower NLL to the coherent-incorrect solution). Truth has no privilege -- when compressibility is equal, frequency wins.

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

## 7. Experiment 4: Scaling, Multi-Rule Errors, Synthetic World, and Cross-Domain Falsification

**Hypothesis.** If truth bias is driven by a structural advantage in compression, then increasing model capacity should strengthen the effect (more capacity to learn regularities). Coherent errors should remain indistinguishable from truth at any size.

### 7.1 Model Configurations

| Size | Parameters | d_model | Heads | Layers |
|------|-----------|---------|-------|--------|
| tiny | 3.5M | 256 | 4 | 4 |
| small | 11M | 384 | 6 | 6 |
| medium | 26M | 512 | 8 | 8 |
| large | 86M | 768 | 12 | 12 |

All models trained for 5000 steps on the same corpus. Architecture: GPT-2 (decoder-only transformer) with character-level tokenization. For large, 2 seeds (42, 43) are used.

### 7.2 Results: Truth Bias Grows with Model Size

**Table 6.** Paired evaluation (random 50/50) by model size.

| Size | Parameters | Avg DLoss (paired) | Pair accuracy | Corpus DLoss | Seeds |
|------|-----------|:------------------:|:------------:|:------------:|:-----:|
| tiny | 3.5M | +0.048 | 83.6% | +0.0115 | 4 |
| small | 11M | +0.063 | 88.4% | +0.0129 | 4 |
| medium | 26M | +0.067 | 88.5% | +0.0130 | 4 |
| large | 86M | +0.070 | 89.4% | +0.0128 | 2 |

**Table 6a.** Paired accuracy by problem type.

| Type | Tiny (3.5M) | Small (11M) | Medium (26M) | Large (86M) |
|------|:-----------:|:-----------:|:------------:|:-----------:|
| Algebra | 99.9% | 100.0% | 100.0% | 100.0% |
| Arithmetic | 95.2% | 98.2% | 98.6% | 99.2% |
| Derivatives | 72.4% | 81.6% | 82.4% | 81.9% |
| Equations | 65.9% | 72.8% | 72.1% | 75.5% |

![Figure 6](results/figure6_scaling.png)

*Figure 6. Scaling of truth bias. Left: pair accuracy increases from 83.6% (tiny) to 89.4% (large) for random errors, while remaining at chance for coherent errors. Right: DLoss by model size.*

Truth bias monotonically increases from tiny to large: +46% in paired DLoss (+0.048 -> +0.070) and +5.8 pp in pair accuracy (83.6% -> 89.4%). The largest gain occurs between tiny and small; between small and medium, accuracy nearly plateaus (88.4% -> 88.5%), but growth resumes at large (88.5% -> 89.4%), refuting the hypothesis of early saturation. Improvement is most pronounced in difficult problem types (derivatives: +9.5 pp, equations: +9.6 pp from tiny to large), while algebra and arithmetic reach saturation already at small.

**Coherent errors still show no bias.** Pair accuracy for coherent 50/50 on the small model is 49.6% (DLoss = -0.0006), i.e. at chance, same as tiny (47.2%) and large (51.8%, DLoss = -0.0003). Increasing model capacity does not help distinguish coherent falsehood from truth. The hypothesis is confirmed: scaling strengthens truth bias for random errors but is powerless against coherent ones.

**Scaling conclusion.** The inverse-U hypothesis (growth -> peak -> decline) is not supported: truth bias monotonically increases with capacity in the 3.5M--86M range. The hypothesis of early plateau (between small and medium) is also refuted: accuracy growth resumes at large (88.5% -> 89.4%). Possible explanations for continued growth: (1) for the mathematical domain with character-level tokenization, memorization loses to generalization at any reasonable size; (2) larger models better generalize on difficult problem types (equations: 66% -> 76% from tiny to large).

### 7.3 Experiment 5: Multi-Rule (Conspiratorial) Errors

Experiments 1 and 4 established two poles: coherent errors (one rule per task type) yield zero truth bias (49%), while random errors (each unique) yield strong bias (83--89%). The question arises: what happens in between?

We introduce *multi-rule errors*: for each task type, a pool of N alternative wrong rules is created, and for each problem, one rule is chosen at random. Each rule is itself compact, but the mapping "problem -> rule" is unpredictable. This models conspiratorial thinking, where several mutually contradictory explanations are applied situationally.

**Table 7.** Paired evaluation of multi-rule errors (tiny, 3.5M, 50/50, 4 seeds).

| N rules | Avg Accuracy | Avg DLoss (paired) | CI 95% DLoss | Wilcoxon p |
|:-------:|:----------:|:-----------------:|:------------:|:----------:|
| 1 (coherent) | 47.2% | -0.002 | [-0.007, +0.000] | ~1.0 |
| 2 | 87.4% | +0.067 | [+0.064, +0.070] | < 10^-6 |
| 3 | 89.4% | +0.067 | [+0.064, +0.070] | < 10^-6 |
| 5 | 89.9% | +0.063 | [+0.060, +0.066] | < 10^-6 |
| 10 | 91.5% | +0.055 | [+0.053, +0.058] | < 10^-6 |
| inf (random) | 83.6% | +0.048 | [+0.046, +0.050] | < 10^-6 |

![Figure 7](results/figure7_multirule.png)

*Figure 7. Truth bias as a function of the number of error rules. Phase transition at N=1->2: a single rule is fully compressible (49%), two rules are not (87%). For N>=2, accuracy exceeds that of random errors (83.6%).*

Three key observations:

1. **Phase transition at N=1->2.** The transition from one rule to two causes an accuracy jump from 47% to 87% -- nearly to the level of random errors. A single rule per task type is fully compressible; two rules, randomly switching, are not, since the model must memorize which rule was applied to each problem.

2. **Multi-rule errors are less compressible than random ones.** Accuracy for N>=2 (87--92%) is *higher* than for random errors (84%). This is an unexpected result: rules create structural expectations, and violating those expectations (unpredictable choice among several rules) is less compressible than having no pattern at all. The model "expects" to see the result of one of the rules but cannot predict which -- this is worse than having no expectations whatsoever.

3. **Monotonic accuracy growth with N.** More rules -> less predictable choice -> stronger truth bias. The curve does not plateau at N=10, suggesting further growth. As N -> infinity, multi-rule errors should converge to random, but at finite N they remain "harder" to compress.

### 7.4 Experiment 6: Synthetic World (Natural Language)

All previous experiments use the mathematical domain with character-level tokenization. To test transferability, we create a synthetic world with 50 entities of four types (animals, plants, minerals, potions) and 15 deterministic rules linking entity properties to observable outcomes. Examples are described in natural language:

> The fire crystal has temperature 250 and clarity 7. Since temperature exceeds 150, the fire crystal glows brightly.

The corpus contains 100,000 examples. As in the mathematical experiment, we train models under two conditions: random 50/50 (half of observations with inverted outcomes) and coherent 50/50 (half follow an alternative rule system with inverted thresholds).

**Table 8.** Paired evaluation of the synthetic world (tiny, 3.5M, 50/50, 4 seeds).

| Condition | Avg Accuracy | Avg DLoss (paired) | CI 95% DLoss | Wilcoxon p |
|-----------|:----------:|:-----------------:|:------------:|:----------:|
| Random errors | 57.7% | +0.034 | [+0.027, +0.040] | < 10^-6 |
| Coherent errors | 46.6% | +0.019 | [+0.008, +0.030] | ~ 0.05 |

**Table 8a.** Paired accuracy by entity type (random errors).

| Type | Accuracy | DLoss |
|------|:--------:|:-----:|
| mineral | 68.7% | +0.056 |
| plant | 62.2% | +0.057 |
| animal | 51.4% | +0.003 |
| potion | 49.1% | +0.029 |

Three key observations:

1. **Truth bias reproduces in natural language.** Pair accuracy of 57.7% for random errors is significantly above chance (p < 10^-6 for all 4 seeds). The compression effect in favor of truth is not limited to formal mathematics.

2. **The effect is substantially weaker than in mathematics.** 57.7% vs 83.6% with identical architecture and corpus proportion. The likely reason: natural language contains more variability in formulations (synonymous constructions, diversity of entity names), weakening the statistical separation between correct and incorrect conclusions.

3. **Coherent errors remain indistinguishable from truth.** Accuracy of 46.6% -- below chance, same as in the mathematical domain (47.2%). The pattern reproduces: a coherent alternative rule system compresses equally well. The strong variance across types (mineral 68.7% vs potion 49.1%) is explained by differences in description length and the number of applicable rules for different entity types.

### 7.5 Experiment 7: Cross-Domain Falsification

The key claim of our work -- that coherent falsehood is invulnerable to compression -- rests on isolated domains: false derivative rules are tested only on derivative tasks. In the real world, a false theory of derivatives conflicts with adjacent domains (integration, tangent lines, numerical evaluation). We test whether adding *cross-domain* tasks -- correct tasks linking derivatives with arithmetic -- can destroy the coherence of the false rule.

Base corpus: coherent 50/50 (Experiment 1). We add correct cross-domain tasks of five types: derivative evaluation at a point, antiderivative check, tangent line equation, chain rule evaluation, and product rule evaluation. All cross-domain tasks use the true differentiation rules. We vary the proportion of cross-domain tasks: 0%, 10%, 25%, 50%.

**Table 9.** Cross-domain falsification (tiny, 3.5M, 4 seeds). Paired evaluation on coherent paired test.

| Cross-domain proportion | Overall accuracy | Derivative accuracy | Other types |
|:-----------------------:|:---------------:|:------------------:|:-----------:|
| 0% (baseline) | 47.0% | 35.2% | 51.1% |
| 10% | 45.8% | 39.4% | 47.9% |
| 25% | 50.6% | **56.0%** | 48.8% |
| 50% | 47.1% | 45.4% | 47.6% |

The result partially supports the hypothesis: accuracy on **derivatives** increases from 35.2% to 56.0% at 25% cross-domain tasks -- the model begins to prefer correct derivatives. However, the effect is non-monotonic: at 50%, accuracy drops to 45.4%, likely due to dilution of standard patterns. Other task types (algebra, arithmetic, equations) remain at chance, since the cross-domain tasks address only contradictions with derivatives.

This experiment provides the first evidence that cross-domain data can *selectively* destroy the coherence of false rules. The effect is still weak (derivative accuracy: 56% vs 50% chance), as expected for a tiny model (3.5M). Scaling to larger models and expanding the set of cross-domain tasks is a priority for future work.

## 8. Discussion

### 8.1 Unified Interpretation

Seven experiments paint a progressively clearer picture:

1. **Compression favors consistency, not truth.** Any consistent rule system -- true or false -- compresses equally well. Truth bias with random errors is explained by the fact that each random error must be memorized individually.

2. **Truth usually wins because errors are usually incoherent.** In real data, different authors make different errors, whereas correct answers are uniform. This aligns with the Truth Co-occurrence Hypothesis (Ravfogel et al., 2025): true statements are more likely to co-occur with other true statements, forming a statistical cluster that the model learns.

3. **Corpus-level and paired metrics can diverge.** At 10/90, corpus-level DLoss inverts (-0.0016), but paired evaluation shows robust truth bias (67% accuracy, p < 10^-88). The corpus-level metric conflates structural preference with the frequency effect on shared problem patterns. An analogous divergence is observed for conditions C/D/E: corpus-level DLoss is positive, but paired evaluation shows accuracy ~49%. This makes paired evaluation a necessary tool for truth bias research.

4. **Correction increases corpus-level DLoss but does not produce transferable truth bias.** Models trained with observations and correction do not distinguish correct from incorrect at the level of pure mathematical pairs (accuracy ~49%). The only reliable mechanism of truth bias is error incoherence.

5. **Truth bias scales with model size, but coherent falsehood remains invulnerable.** Increasing the model from 3.5M to 86M strengthens truth bias for random errors (+46% in DLoss), but does not help distinguish coherent falsehood from truth. Scaling enhances generalization, but does not create a "truth compass."

6. **Multi-rule errors reveal a phase transition.** The transition from one error rule to two causes a jump in truth bias from 49% to 87%. This shows that the critical factor is not the number of errors, but the predictability of the error system. One rule is predictable and compressible; two rules, randomly applied, are not.

7. **Truth bias transfers to natural language, but weakens.** A synthetic world with 15 rules yields 57.7% pair accuracy (vs 83.6% in mathematics). Natural language contains more surface-level variability that weakens the statistical signal. Coherent errors remain indistinguishable from truth in this domain as well (46.6%).

8. **Cross-domain data selectively destroys coherence.** Adding correct tasks linking derivatives with arithmetic raises derivative accuracy from 35% to 56% (at 25% cross-domain tasks), without affecting other error types. This is the first evidence that cross-domain information can transform locally coherent falsehood into globally incoherent one -- a mechanism analogous to scientific progress, where discoveries in one field falsify theories in another.

### 8.2 Analogy with Popper's Falsifiability

Our results admit an interpretive analogy with the falsifiability criterion (Popper, 1959). Compression pressure acts as a computational analog: a true theory with concrete predictions requires no additional explanations (maximal compression); a false theory whose predictions diverge from data needs correction (poor compression); a theory with ad hoc escape hatches expands with every observation (anti-compression).

However, the analogy has limits. First, the model does not "test" theories -- it simply minimizes code length. Second, our data show that bare discrepancies alone barely help (condition B ~ A): regular discrepancies are compressible. Popperian falsification assumes that a discrepancy with observation refutes a theory; for a compressor model, a discrepancy is merely another pattern. Moreover, paired evaluation of conditions C/D/E showed that even ad hoc correction does not produce transferable truth bias -- the model learns to process correction patterns but does not transfer this to pure mathematical pairs.

Practical analogies from the history of science are appropriate as illustrations. The geocentric model required ever more epicycles to reconcile with observations (a variation of condition C); phlogiston theory needed special assumptions to explain mass increase during combustion; miasma theory could not explain why disease spread along waterways rather than by wind. However, our experiments use the mathematical domain, and transfer to these real-world examples remains an open question.

### 8.3 Implications

**For alignment.** Models lack an innate "truth compass" -- they favor well-compressible patterns. Systematic deception consistently represented in data encounters no resistance from compression.

**For ML epistemology.** The framework explains why models develop internal truth representations (Marks & Tegmark, 2023): in real corpora, true statements are more coherent than false ones. It also explains the inverse scaling effect on TruthfulQA (Lin et al., 2022): larger models are better at memorizing coherent misconceptions, which according to our data compress just as well as truth.

**For understanding hallucinations.** Models confidently reproduce coherent misconceptions not due to a compression failure, but because such misconceptions compress *successfully*. This aligns with the analysis by Chlon et al. (2025).

### 8.4 Limitations

**Model scale.** Experiments use models from 3.5M to 86M parameters. Truth bias grows with size (Section 7), and growth continues from medium (26M) to large (86M) with no clear plateau. The range remains limited -- extrapolation to GPT-2/3 scale models requires further experiments.

**Domain specificity.** Mathematics has an unusually crisp distinction between correct and incorrect. The synthetic world experiment (Section 7.4) confirms that the effect is substantially weaker in a natural language domain (57.7% vs 83.6%). Nevertheless, it remains statistically significant. Transfer to real-world domains (medicine, history, economics) requires further experiments.

**Confounding with corpus length.** Conditions C/D/E generate substantially longer texts (loss ~0.24 vs ~0.14). DLoss may partially reflect a difference in convergence rather than compressibility per se. Paired evaluation mitigates but does not fully eliminate this confound.

**Training duration.** All models are trained for 5000 steps. As models grow from tiny to medium, the number of parameters increases but the number of training steps does not. Larger models may be undertrained, potentially underestimating truth bias for medium. Conversely, if medium were to reach full convergence, pair accuracy might exceed 88.5%.

**Effect size.** DLoss (0.003--0.012) is small in absolute terms. Its practical significance for large models remains an open question.

### 8.5 Future Experiments

**Chained tasks with cross-domain dependencies.** Experiment 7 (Section 7.5) showed that cross-domain data can selectively destroy coherence, but the mechanism was indirect: *separate* correct tasks were added to the corpus, competing with coherent errors by frequency. A stronger test is to embed the dependency *within* the task itself.

The idea: coherent errors are indistinguishable in isolated domains (accuracy ~ 49%) because each domain is a closed system. But if a task *links* two domains -- for instance, computing a derivative and then verifying the result arithmetically -- a coherent error in the first step produces a numerical discrepancy in the second. This discrepancy is unpredictable (depends on the specific problem parameters) and therefore incompressible.

Concretely: for each type of coherent error, a *chain* of two steps is constructed: (A) computation using the coherent false rule, (B) verification through an adjacent domain using correct arithmetic. For example:

- **Derivative -> finite difference:** compute f'(a), then compare with [f(a+h) - f(a)]/h. With the correct derivative, the residual is approximately O(h^2); with the coherent-false rule (d/dx(cx^n) = cx^(n-1)) -- an unpredictable number.
- **Factorization -> evaluation:** factor a polynomial, then evaluate both sides at x = k. With correct factoring, the values match; with the coherent-false rule (sign of d inverted) -- the difference depends on k.
- **Equation solving -> back-substitution:** find the root, then substitute back into the original equation. With the correct solution, LHS = RHS; with the coherent-false one -- a nonzero residual.
- **Arithmetic -> reverse computation:** compute a chain of operations, then reverse it. With correct multiplication, return to the starting value; with the coherent-false rule (a x b = a x (b-1)) -- a fractional remainder.

The key distinction from Experiment 7: there, the model saw two competing rules (correct and false) in separate tasks -- essentially a frequency competition. Here, the model sees *one* rule system, but with a verification step that transforms the coherent error into an incompressible numerical residual *within* each task.

Hypothesis: pair accuracy on chained tasks with coherent errors is significantly above 50%. If confirmed, this would show that *cross-domain dependencies* are the specific mechanism through which compression pressure begins to prefer truth even for coherent errors. Scientific theories are testable precisely because they are connected to other facts; isolated coherent falsehood is invulnerable, but falsehood embedded in a web of dependencies is not.

Control experiment: the same tasks, but truncated (without the verification step). Expected: accuracy ~ 49%, same as for standard coherent errors. This would confirm that the verification step -- not the different task structure -- is what produces truth bias.

**Linear probing.** Extract activations and train linear classifiers to detect "truth directions" vs. "coherence directions" (Marks & Tegmark, 2023 methodology).

**Synthetic world scaling.** Truth bias in the natural language domain is weaker (57.7% vs 83.6%, Section 7.4). Scaling to small/medium models will show whether the effect grows with size analogously to the mathematical domain.

**Real-world domains.** Extend to domains with competing knowledge systems:
- **Type 3b (ad hoc):** Evidence-based medicine vs. homeopathy, vaccination vs. anti-vax theories.
- **Historical:** Phlogiston vs. oxygen theory, miasma theory vs. germ theory, geocentrism vs. heliocentrism.
- **Type 3a (non-specific):** Astrology, market technical analysis.

## 9. Conclusion

This work isolates the conditions under which compression pressure during language model training aligns with truth. The central finding: **truth bias is not a fundamental property of compression, but a consequence of error incoherence in the corpus.** Random errors are incompressible and must be memorized individually, giving correct mathematics a structural advantage (83% pair accuracy, 16/16 seeds). A coherent false system, as compact as truth, strips compression of any preference (~49% pair accuracy at any model size from 3.5M to 86M). The multi-rule error experiment demonstrates a phase transition: a single false rule is indistinguishable from truth, but as few as two rules applied unpredictably restore truth bias (87%), at a level even higher than random errors (84%). The effect reproduces beyond mathematics: in a natural language domain (synthetic world with 15 rules), pair accuracy reaches 57.7%, confirming the generality of the mechanism, albeit with reduced effect size.

The practical implication for alignment: scaling (from 3.5M to 86M) strengthens truth bias for incoherent errors with no sign of saturation, but is powerless against coherent falsehood. A compressor model has no "truth compass" -- it has a consistency compass. In real corpora, these compasses typically coincide, since different authors' errors are diverse while correct answers are uniform. But where falsehood is systematic and internally consistent -- in entrenched misconceptions, ideological narratives, coherent pseudoscientific systems -- compression gives the model no basis to prefer truth.

However, a truly coherent false system -- at the scale of all knowledge -- may be impossible, provided it is falsifiable. A false system that makes concrete predictions will eventually generate consequences that contradict facts from adjacent domains. Experiment 7 (cross-domain falsification) provides the first empirical evidence for this mechanism: adding correct tasks linking derivatives with arithmetic selectively raises derivative accuracy from 35% to 56%, without affecting other error types. The effect is still weak (tiny model, limited set of cross-domain tasks), but the direction suggests that cross-domain information can transform locally coherent falsehood into globally incoherent one. And incoherent errors, as our experiments demonstrate, are reliably detected by compression.

This means that as data coverage and model capacity grow, the boundary of truth bias applicability should expand. For any falsifiable false system, there exists a sufficient volume of cross-domain data at which its predictions conflict with observed facts, destroying local coherence. The only systems immune to this mechanism are unfalsifiable ones: making no concrete predictions, they never contradict data and remain "coherent" vacuously. This closes the circle with the Popper analogy (Section 8.2): compression, like the scientific method, is powerless against unfalsifiable claims -- but is capable, given sufficient data coverage, of detecting any falsifiable falsehood.

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

Hutter, M. (2005). Universal Artificial Intelligence: Sequential Decisions Based on Algorithmic Probability. *Springer*.

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

Ravfogel, S., Yehudai, G., Linzen, T., Bietti, A., & Bruna, J. (2025). Emergence of Linear Truth Encodings in Language Models. *NeurIPS 2025*.

Shannon, C. E. (1948). A Mathematical Theory of Communication. *Bell System Technical Journal*, 27(3), 379-423.

Solomonoff, R. J. (1964). A Formal Theory of Inductive Inference. *Information and Control*, 7(1), 1-22.

Valle-Perez, G., Camargo, C. Q., & Louis, A. A. (2019). Deep Learning Generalizes Because the Parameter-Function Map Is Biased Towards Simple Functions. *ICLR 2019*.

Wan, J., & Mei, L. (2025). Large Language Models as Computable Approximations to Solomonoff Induction. *arXiv:2505.15784*.

## Appendix A: Reproducibility

All code, data generation scripts, and evaluation scripts are available at https://github.com/Rai220/compression-drives-truth. Experiments were conducted on an Apple Mac M4 with 36GB of unified memory using the MLX framework (v0.31.0). Large model training (86M) was performed on cloud GPU instances. Total computational cost: approximately 50 hours of wall-clock time for the 120+ training runs described in this paper.
