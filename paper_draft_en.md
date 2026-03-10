# Compression Favors Consistency, Not Truth: When and Why Language Models Prefer Correct Information

**Author:** Konstantin Krestnikov
**Date:** 03.2026

## Abstract

Why do language models tend to prefer true statements, even when trained on mixed-quality data? We propose the Compression--Consistency Principle: gradient descent favors the most compressible hypothesis consistent with the data. Truth bias emerges as a side effect -- not because truth is inherently privileged, but because real-world errors are diverse and thus expensive to memorize. Models compress *text*, not reality; the observed bias reflects corpus statistics, not access to external truth.

We test this principle by training over 160 transformers (3.5M--86M parameters) on corpora with controlled ratios of correct and incorrect mathematical derivations. With random (incoherent) errors, models consistently prefer correct solutions: paired evaluation yields 83% accuracy at 50/50 (Wilcoxon p < 10^-6), the effect persists at 10/90 (67%, p < 10^-88), and strengthens with scale (83.1% -> 89.1% from 3.5M to 86M). The effect reproduces in natural language (57.7%). However, replacing random errors with a coherent alternative system -- internally consistent but mathematically wrong -- eliminates truth preference entirely (~49% at any model size).

The principle yields three further predictions, each supported by dedicated experiments: (1) a sharp transition at N=2 error rules, where compressibility breaks (49% -> 87%); (2) inverse scaling for chained verification tasks (71% -> 61% as models grow -- compressor power entrenches coherent falsehood); (3) partial destruction of coherent error immunity through cross-domain dependencies (Appendix B). Together, these results offer a candidate mechanism for both the typical truthfulness of language models and their confident reproduction of systematic misconceptions.

---

## 1. Introduction

Language models are increasingly accurate on factual benchmarks, yet they confidently generate false statements. What determines when a model prefers truth and when it doesn't?

Several explanations have been proposed. Scaling helps: larger models perform better on factual tasks (Kadavath et al., 2022). RLHF and similar alignment techniques steer models toward human-preferred outputs. Data statistics play a role: factual accuracy correlates with the frequency and source reliability of facts in training data (Elazar et al., 2022; Joshi et al., 2024; Kandpal et al., 2023). Internal truth representations have been discovered in model activations (Burns et al., 2023; Marks & Tegmark, 2023). Yet none of these explanations address a more fundamental question: *why would the training objective itself -- next-token prediction -- create any preference for truth?*

We propose that the answer is compression. Minimizing cross-entropy is mathematically equivalent to minimizing code length (Shannon, 1948; Deletang et al., 2024), connecting LLM training to the Minimum Description Length principle (Rissanen, 1978; Grünwald, 2007). A model that better predicts tokens is a better compressor. Compression quality correlates linearly with model capabilities (Huang et al., 2024), and LLM training formally approximates Solomonoff induction (Wan & Mei, 2025). But compression does not inherently favor truth -- it favors the most *compressible* hypothesis consistent with the data. We call this the **Compression--Consistency Principle**: truth benefits from compression only when falsehood is structurally incoherent. Diverse errors must be memorized individually, whereas a correct rule system compresses into a compact representation. When errors form a coherent alternative system -- internally consistent but wrong -- they compress just as efficiently, and the preference vanishes.

This principle has three important caveats. First, models compress *text*, not reality: "truth" here means correctness of mathematical derivations, not a metaphysical category. Second, frequency can override compressibility: a structural advantage must compensate for numerical minority. Third, the compressibility gap between truth and falsehood is corpus-dependent, not universal. We test the principle through controlled experiments on mathematical corpora (Experiments 1--3, Sections 4--6) and extend to scaling, multi-rule errors, and chained verification (Experiments 4--5 and 9, Section 7), with additional natural-language and cross-domain experiments in Appendix B, training over 160 transformers from 3.5M to 86M parameters.

## 2. Related Work

### Prediction as Compression

The link between prediction and data compression traces back to the foundational work in information theory. Shannon (1948) showed that optimal compression requires knowledge of the true data distribution, and Solomonoff (1964) formalized optimal prediction as weighting hypotheses by their program length. Rissanen (1978) developed the Minimum Description Length (MDL) principle, formalizing model selection as a compression task: the best model minimizes the total description length of the model plus the data given the model. Grünwald (2007) systematized the MDL principle and showed its equivalence to several forms of statistical inference. Hutter (2005) developed these ideas into a formal theory of universal artificial intelligence (AIXI), explicitly linking intelligence to compression ability. Our work directly builds on the MDL framework: we experimentally vary the description length of false systems and observe under what conditions the MDL-optimal choice coincides with truth.

In the context of language models, Deletang et al. (2024) empirically demonstrated that LLMs are universal compressors: a next-token predictor can serve as an arithmetic coder. Huang et al. (2024) discovered a linear correlation (r ~ -0.95) between compression quality and benchmark performance, and Wan & Mei (2025) formally proved that LLM training approximates Solomonoff induction. These results form the theoretical foundation of our work: if LLM training is compression, and compression is linked to intelligence, what role does data truthfulness play?

### Internal Representations of Truth in LLMs

Several studies have found that language models form internal representations correlated with statement truthfulness. Marks & Tegmark (2023) showed a linear geometric structure of truthfulness in activation space (Geometry of Truth), and Burns et al. (2023) proposed CCS, a method for discovering truth directions without supervision. Li et al. (2023b) identified a 40% gap between a model's internal knowledge and its generation, developing Inference-Time Intervention (ITI). Ravfogel et al. (2025) proposed the Truth Co-occurrence Hypothesis -- a mechanism for the emergence of linear truth representations through co-occurrence of true statements in the corpus. Our work complements this line of research: we show *under what conditions* compression gives rise to such representations, and under what conditions it does not.

### Emergent World Models

Language models can form internal world models from pure text prediction. Li et al. (2023a) trained a model to predict Othello moves and found that it learns a full board representation (Othello-GPT). Gurnee & Tegmark (2024) discovered linear representations of space and time in Llama-2 activations. These results show that sequence compression can give rise to structured internal representations -- our work investigates whether such representations gravitate towards true structures.

### Truthfulness and Training Data Statistics

Several works investigate the dependence of factual behavior on the structure of the training corpus. Joshi et al. (2024) showed that truthfulness in LLMs is linked to the structure of "personas" (sources) in pretraining data: the model learns persona-specific patterns and prefers statements associated with reliable sources. Elazar et al. (2022) demonstrated that factual predictions strongly depend on the frequency of facts in training data. Kang & Choi (2023) investigated how co-occurrence between statements affects factual recall, and Kandpal et al. (2023) showed a direct relationship between the number of supporting documents in the corpus and model answer accuracy. Our work differs fundamentally from this line: we do not observe dependence on data statistics, but *experimentally vary* the structure of errors (their compressibility) at fixed frequencies and measure how this changes truth preference. This allows us to isolate the role of compressibility from the roles of frequency and source reliability.

### Simplicity Bias, Noisy Labels, and Grokking

The inductive bias of neural networks towards simple functions is a well-documented phenomenon. Valle-Perez et al. (2019) showed an exponential preference for low-complexity functions, and Mingard et al. (2021) proved that SGD approximates Bayesian sampling with a simplicity prior. Goldblum et al. (2024) connected this to Kolmogorov complexity, providing a theoretical basis for the link between compression and generalization. Bhattamishra et al. (2023) showed that transformers exhibit a pronounced simplicity bias, preferring lower-complexity solutions when multiple hypotheses are consistent with the data.

The noisy labels literature directly parallels our setup. Zhang et al. (2017) demonstrated that neural networks can memorize completely random labels, but when structure is present, they generalize through it. Rolnick et al. (2017) showed that learning is robust to massive label noise -- the network learns the "clean" pattern even when noise overwhelmingly predominates. Our result with random errors (truth bias at 10/90) directly aligns with these observations: random errors play the role of noise labels, through which the network generalizes to structured correct solutions.

The phenomenon of grokking -- delayed generalization -- is also related to compression. Nanda et al. (2023) showed that networks discover Fourier transforms for modular arithmetic, and DeMoss et al. (2024) described the phase transition from memorization to generalization through complexity dynamics. Liu et al. (2023) interpreted grokking as a compression process: the network transitions from memorization to a compact representation. Our experiments with coherent errors directly connect to simplicity bias: a coherent false system is just as "simple" as truth, and compression shows no preference for either.

### Our Contribution

The works listed above either study internal truth representations in already-trained models, establish theoretical links between compression, simplicity bias, and intelligence, or analyze the dependence of truthfulness on data statistics. However, a direct experiment -- training a model from scratch on a quality-controlled corpus with systematic variation of *error compressibility* -- has not been conducted. This work fills that gap.

| Work | What it studies | Our difference |
|------|----------------|----------------|
| Joshi et al. (2024) | Truthfulness via source/persona structure in data | We vary *error compressibility* directly, isolating compression from source reliability |
| Ravfogel et al. (2025) | Mechanism of truth encoding emergence via co-occurrence | We show *failure conditions*: coherent falsehood produces identical encodings |
| Elazar (2022), Kang & Choi (2023), Kandpal (2023) | Factual behavior as a function of frequency/support count | We fix frequency and vary error structure: at 50/50, truth bias = 83% (random) vs 49% (coherent) |
| Zhang et al. (2017), Rolnick et al. (2017) | Learning from noisy labels: generalization vs memorization | We generalize to sequence-level: not labels but entire derivations, showing that "structured noise" (coherent errors) is not filtered out |
| Burns et al. (2023), Marks & Tegmark (2023) | Internal truth directions in pretrained models | We train from scratch on controlled data and find boundaries: truth direction emerges only with incoherent errors |

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

**Paired evaluation (primary metric).** For each problem, a single shared prompt is generated along with two completions (correct and incorrect). NLL is computed only on completion tokens, conditioned on the shared prompt. This yields pairwise comparison under identical context, eliminating the confound of different prompts. Metrics: **pair accuracy** (fraction of pairs where the model prefers correct; our primary metric), mean DLoss on completions, Wilcoxon signed-rank test. We adopt paired evaluation as the primary metric because corpus-level measures can be confounded by differences in text statistics between correct and incorrect corpora (see Sections 4.1 and 6 for concrete examples of such divergence).

**Corpus-level evaluation (secondary diagnostic).** For each trained model, we compute the average cross-entropy loss on held-out sets of correct (5K problems) and incorrect (5K problems) examples. **DLoss = Loss(incorrect) - Loss(correct)**; a positive value indicates truth bias. This metric is useful for detecting overall trends but can be confounded by frequency effects and format differences between corpora.

**Statistical analysis.** Each configuration is repeated with 4 random initializations (seeds 42--45). For individual configurations we use the two-sided binomial test (H0: P(correct) = 0.5). 95% confidence intervals for DLoss are obtained via bootstrap (10,000 resamples). For paired evaluation, the Wilcoxon signed-rank test is used on paired NLL differences.

### 3.4 Theoretical Framework: Description Length and Theory Types

To interpret Experiments 2--3 we use a typology of theories distinguished by the description length of the corpus "theory + observations." The key principle: the model optimizes cross-entropy, which is equivalent to minimizing expected code length (Shannon, 1948). A theory that allows shorter encoding of the corpus gains an advantage.

**Type 1: True theory with concrete predictions.** Predictions match observations. The "theory + observations" corpus compresses maximally: one rule system explains everything.

**Type 2: False theory with concrete predictions.** Predictions diverge from observations. The model must encode both the false rules and the discrepancies. However, if the discrepancies are **regular** (e.g., a x b = a x (b-1) always understates by a), the model can learn a correction, and the additional description length is small.

**Type 3a: Theory with non-specific predictions.** The theory does not specify a "situation -> outcome" mapping (e.g., "result is moderate"). It does not contradict observations but does not help predict them either -- it does not reduce code length.

**Type 3b: Theory with ad hoc correction.** Each discrepancy is explained by a unique exception rule. Description length grows linearly with the number of observations -- this is anti-compression.

### 3.5 MDL Proposition: When Does Compression Favor Truth

We state a prediction in MDL terms (Rissanen, 1978; Grünwald, 2007). Consider a corpus D consisting of N problems, fraction α solved according to a true theory T₁ and fraction (1 - α) according to an alternative theory T₂. The MDL-optimal model minimizes the two-part code: L(M) + L(D|M), where L(M) is the model description length and L(D|M) is the data length given the model.

**Proposition.** Let K(T₁) and K(T₂) be the Kolmogorov complexities of theories T₁ and T₂ (lengths of the shortest programs generating all correct solutions under each theory). Then:

1. **K(T₂) >> K(T₁) (random errors).** Each error is unique, K(T₂) ~ O(N). The MDL-optimal model prefers T₁ even when α < 0.5, since L(T₁) + L(exceptions) < L(T₂). Prediction: truth bias for any α > α_crit, where α_crit is determined by model capacity.

2. **K(T₂) ≈ K(T₁) (coherent errors).** Both theories are described by compact rules. The MDL choice is determined solely by frequency: the model prefers T₁ when α > 0.5 and T₂ when α < 0.5. At α = 0.5, no preference exists. Under an ideal MDL learner, two programs of equal length generating data at equal frequency are indistinguishable -- there is no basis for preferring one over the other.

3. **K(T₂) > K(T₁), but K(T₂) = O(1) (multi-rule errors).** Multiple alternative rules increase K(T₂), but it remains finite. The MDL preference for T₁ is determined by the difference K(T₂) - K(T₁) and strengthens with N (number of rules).

Experiments 1, 4, and 6 directly test these predictions. Results are consistent with the MDL proposition: truth bias ≈ 83% for case 1, ≈ 49% for case 2, and monotonic growth from 87% to 92% with the number of rules for case 3. Figure C illustrates the conceptual model alongside the observed results.

![Figure C](results/figure_conceptual.png)

*Figure C. The Compression--Consistency Principle. (a) MDL prediction: description length of truth K(T₁) is constant, while description length of falsehood K(T₂) depends on error structure -- equal for coherent errors, increasing for multi-rule, maximal for random. (b) Observed pair accuracy matches the prediction: no truth bias for coherent errors (49%), sharp transition at N=2 rules (87%), and monotonic growth with decreasing compressibility.*

### 3.6 Experiment Conditions

**Experiment 1:** 5 proportions (50/50--10/90) x 4 seeds = 20 models with random errors + 1 baseline. Controls: coherent errors (4 proportions x 4 seeds = 16) and contradictory (4 seeds).

**Experiment 2:** Coherent errors at 50/50 with observations. 4 observation ratios (0%, 10%, 25%, 50%) x 4 seeds = 16 models. Test sets contain no observations -- we measure pure mathematical prediction quality.

**Experiment 3:** 5 conditions for the false theory (A--E) at 50/50. Conditions A and B are from Experiments 1--2. Conditions C, D, E -- 3 x 4 seeds = 12 new models.

**Experiment 4:** Scaling -- random 50/50 and coherent 50/50 at small (11M), medium (26M), and large (86M) sizes, 4 seeds each = 22 models (+ tiny from Experiment 1). **Experiment 5:** Multi-rule errors (16 models). **Experiment 9:** Chained tasks with verification (14 models). Additional experiments reported in Appendix B: synthetic world (8 models), multi-alternative errors (20 models), cross-domain falsification (16 models).

In total, **over 160 models** were trained (69 in Experiments 1--3 + 22 scaling + 16 multi-rule + 14 chained + 44 in Appendix B).

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

At 10/90, the corpus-level metric inverts (DLoss = -0.0016, the model on average "prefers" incorrect examples due to their 9-fold prevalence), while paired evaluation consistently shows truth bias (67% accuracy, p < 10^-88). This means that the structural advantage of correct solutions persists even under extreme imbalance -- the corpus-level inversion reflects a frequency effect on shared problem patterns, not a loss of the model's discriminative ability at the level of individual solutions.

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
| 0% (control)* | +0.0005 +/- 0.0003 | [+0.0002, +0.0007] | 4/4 | 0.125 | 0.1414 |
| 10% | +0.0002 +/- 0.0003 | [-0.0001, +0.0004] | 3/4 | 0.625 | 0.1416 |
| 25% | +0.0004 +/- 0.0001 | [+0.0003, +0.0004] | 4/4 | 0.125 | 0.1435 |
| 50% | +0.0008 +/- 0.0006 | [+0.0003, +0.0012] | 3/4 | 0.625 | 0.1471 |

![Figure 4](results/figure4_observations.png)

*Figure 4. DLoss as a function of the observation ratio. The effect is an order of magnitude weaker than with random errors (+0.0115).*

*\*Note: the control condition (0% observations) uses models from Experiment 2, trained on a separately generated corpus with the same 50/50 ratio. Its DLoss = +0.0005 differs slightly from the -0.0004 reported for coherent errors in Table 2 (Experiment 1), because these are different training corpora with different random problem instances. Both values are within noise and consistent with the absence of truth bias, as confirmed by paired evaluation (accuracy ~49% for both model sets).*

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

## 7. Experiment 4: Scaling, Multi-Rule Errors, and Chained Verification

**Hypothesis.** If truth bias is driven by a structural advantage in compression, then increasing model capacity should strengthen the effect (more capacity to learn regularities). Coherent errors should remain indistinguishable from truth at any size.

### 7.1 Model Configurations

| Size | Parameters | d_model | Heads | Layers |
|------|-----------|---------|-------|--------|
| tiny | 3.5M | 256 | 4 | 4 |
| small | 11M | 384 | 6 | 6 |
| medium | 26M | 512 | 8 | 8 |
| large | 86M | 768 | 12 | 12 |

All models trained for 5000 steps on the same corpus. Architecture: GPT-2 (decoder-only transformer) with character-level tokenization. For large, 4 seeds (42--45) are used in the main scaling experiment (random and coherent errors); for chained tasks (Experiment 9), large uses 2 seeds due to computational constraints.

### 7.2 Results: Truth Bias Grows with Model Size

**Table 6.** Paired evaluation (random 50/50) by model size.

| Size | Parameters | Avg DLoss (paired) | Pair accuracy | Corpus DLoss | Seeds |
|------|-----------|:------------------:|:------------:|:------------:|:-----:|
| tiny | 3.5M | +0.048 | 83.1% | +0.0115 | 4 |
| small | 11M | +0.063 | 88.4% | +0.0129 | 4 |
| medium | 26M | +0.067 | 88.4% | +0.0130 | 4 |
| large | 86M | +0.070 | 89.1% | +0.0127 | 4 |

**Table 6a.** Paired accuracy by problem type.

| Type | Tiny (3.5M) | Small (11M) | Medium (26M) | Large (86M) |
|------|:-----------:|:-----------:|:------------:|:-----------:|
| Algebra | 99.9% | 100.0% | 100.0% | 100.0% |
| Arithmetic | 95.2% | 98.2% | 98.6% | 99.2% |
| Derivatives | 72.4% | 81.6% | 82.4% | 81.9% |
| Equations | 65.9% | 72.8% | 72.1% | 75.5% |

![Figure 6](results/figure6_scaling.png)

*Figure 6. Scaling of truth bias. Left: pair accuracy increases from 83.1% (tiny) to 89.1% (large) for random errors, while remaining at chance for coherent errors. Right: DLoss by model size.*

Truth bias monotonically increases from tiny to large: +46% in paired DLoss (+0.048 -> +0.070) and +6.0 pp in pair accuracy (83.1% -> 89.1%). The largest gain occurs between tiny and small; between small and medium, accuracy nearly plateaus (88.4% -> 88.4%), and growth resumes at large (88.4% -> 89.1%). Improvement is most pronounced in difficult problem types (derivatives: +9.5 pp, equations: +9.6 pp from tiny to large), while algebra and arithmetic reach saturation already at small.

**Coherent errors still show no bias.** Pair accuracy for coherent 50/50 on the small model is 49.6% (DLoss = -0.0006), i.e. at chance, same as tiny (47.2%) and large (51.8%, DLoss = -0.0003). Increasing model capacity does not help distinguish coherent falsehood from truth. The data are consistent with the hypothesis: scaling strengthens truth bias for random errors but does not help against coherent ones.

**Scaling summary.** Under fixed-step training (5000 steps for all sizes), truth bias monotonically increases in the 3.5M--86M range. The near-plateau between small and medium (88.4% -> 88.4%) is broken at large (89.1%). Since all models were trained for the same number of steps regardless of size, this trend may reflect both increased capacity and differential convergence -- a compute-matched comparison would be needed to disentangle these factors. That said, the improvement is concentrated in difficult problem types (equations: 66% -> 76%), consistent with the interpretation that larger models better generalize the underlying rule system.

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
| inf (random) | 83.1% | +0.048 | [+0.046, +0.050] | < 10^-6 |

![Figure 7](results/figure7_multirule.png)

*Figure 7. Truth bias as a function of the number of error rules. Sharp transition at N=1->2: a single rule is fully compressible (49%), two rules are not (87%). For N>=2, accuracy exceeds that of random errors (83.1%).*

Three key observations:

1. **Sharp transition at N=1->2.** The transition from one rule to two causes an accuracy jump from 47% to 87% -- nearly to the level of random errors. A single rule per task type is fully compressible; two rules, randomly switching, are not, since the model must memorize which rule was applied to each problem.

2. **Multi-rule errors are less compressible than random ones.** Accuracy for N>=2 (87--92%) is *higher* than for random errors (83%). This is an unexpected result: rules create structural expectations, and violating those expectations (unpredictable choice among several rules) is less compressible than having no pattern at all. The model "expects" to see the result of one of the rules but cannot predict which -- this is worse than having no expectations whatsoever.

3. **Monotonic accuracy growth with N.** More rules -> less predictable choice -> stronger truth bias. The curve does not plateau at N=10, suggesting further growth. As N -> infinity, multi-rule errors should converge to random, but at finite N they remain "harder" to compress.

Three additional experiments -- a synthetic natural-language world (Experiment 6), multi-alternative errors in natural language (Experiment 7), and cross-domain falsification with separate task types (Experiment 8) -- are reported in Appendix B. In brief: truth bias reproduces in natural language but is substantially weaker (57.7% vs 83.1%); natural language absorbs contradictions that would be immediately detectable in formal mathematics; and cross-domain data can selectively destroy the coherence of false rules, though the effect is weak and non-monotonic. These results extend the picture but are not required for the main argument.

### 7.4 Experiment 9: Chained Tasks with Verification

A preliminary cross-domain experiment (Appendix B.3) showed that adding separate correct tasks can destroy coherence, but the mechanism was indirect. A stronger test is to embed the dependency *within* the task itself.

We construct *chained tasks* in which a computation using the coherent false rule (step A) is accompanied by arithmetic verification (step B). For correct solutions, verification confirms the result (residual = 0); for coherent-error solutions, it produces an unpredictable numerical residual depending on specific problem parameters. Six chain types:

- **Arithmetic -> reverse:** compute a chain of operations, then undo each step.
- **Factoring -> evaluation:** factor an expression, then evaluate both sides at x = k.
- **Linear equation -> back-substitution:** solve for x, substitute back.
- **Quadratic equation -> root substitution:** find roots via Vieta's formulas, substitute.
- **Derivative -> finite difference:** compute f'(a), compare with [f(a+h) - f(a)]/h.
- **Tangent -> prediction:** construct the tangent line, verify prediction at a nearby point.

The key distinction from the cross-domain experiment (Appendix B.3): the model sees *one* rule system, but with a verification step that transforms the coherent error into an incompressible numerical residual *within* each task.

**Table 10.** Chained tasks (tiny, 3.5M, 50/50, 4 seeds). Paired evaluation: correct vs coherent-error chains.

| Seed | Accuracy (chained) | DLoss | Wilcoxon p | Accuracy (coherent ctrl) |
|:----:|:------------------:|:-----:|:----------:|:------------------------:|
| 42 | 71.4% | +0.0116 | < 10^-6 | 43.1% |
| 43 | 70.0% | +0.0112 | < 10^-6 | 47.5% |
| 44 | 72.5% | +0.0118 | < 10^-6 | 41.3% |
| 45 | 69.8% | +0.0116 | < 10^-6 | 41.4% |
| **Avg** | **70.9%** | **+0.0115** | -- | **43.3%** |

**Table 10a.** Accuracy by chain type (averaged over 4 seeds).

| Chain type | Accuracy | n |
|------------|:--------:|:---:|
| Arithmetic (forward + reverse) | 95.8% | 824 |
| Factoring (factor + evaluate) | 89.9% | 843 |
| Linear equation (solve + substitute) | 88.2% | 879 |
| Quadratic (roots + substitute) | 60.5% | 869 |
| Derivative (power rule + finite diff) | 53.4% | 784 |
| Tangent (slope + predict) | 34.8% | 801 |

![Figure 10](results/figure10_chained.png)

*Figure 10. Chained tasks. Left: verification raises accuracy from 43% (isolated coherent) to 71% -- cross-domain dependencies break the immunity of coherent errors. Center: inverse scaling -- chained task accuracy drops with model size (71% -> 61%), while random error accuracy rises (84% -> 89%). Right: accuracy by chain type (tiny).*

**Table 10b.** Chained tasks scaling by model size.

| Size | Params | Seeds | Accuracy | DLoss | Trend |
|------|--------|:-----:|:--------:|:-----:|:-----:|
| Tiny | 3.5M | 4 | 70.9% +/- 1.2% | +0.0115 | -- |
| Small | 11M | 4 | 64.2% +/- 1.5% | +0.0090 | down |
| Large | 86M | 2 | 60.6% +/- 1.2% | +0.0078 | down |

For comparison, random error scaling: tiny 83.1% -> small 88.4% -> large 89.1% (up).

**Table 10c.** Control experiment: truncated chains (no verification step).

| Condition | Accuracy (tiny, 4 seeds) | p |
|-----------|:------------------------:|:-:|
| With verification (chained) | 70.9% +/- 1.2% | < 10^-6 |
| Without verification (truncated) | 44.3% +/- 2.1% | ~1.0 |
| Standard coherent | 43.3% +/- 2.9% | ~1.0 |

The control experiment with truncated chains (same task types, but without the verification step) confirms that truth bias is produced by verification, not by different task structure: accuracy of truncated chains (44.3%) is indistinguishable from standard coherent errors (43.3%). In this setup, truth bias with coherent errors emerges only when verification is present.

Five key observations:

1. **Verification restores truth bias.** Accuracy of 70.9% (p < 10^-6 for all 4 seeds) -- significantly above chance and above standard coherent errors (43.3% on the same models evaluated on isolated tasks). Cross-domain dependencies transform coherent errors into incompressible ones.

2. **The control confirms the mechanism.** The same models evaluated on the standard coherent test (without verification) yield accuracy of 43.3% -- below chance, as in Experiment 1. This confirms that the verification step -- not the different task structure -- is what produces truth bias.

3. **The type spectrum reflects verification strength.** Arithmetic reverse (96%) -- the strongest signal: with incorrect multiplication, reverse division yields a fraction instead of an integer. Tangent (35%) -- the only type below chance: the O(h^2) approximation error in finite differences masks the coherent rule error, and the model learns the pattern "with error, the prediction is closer to zero."

4. **The effect is comparable to random errors.** Accuracy 70.9% vs 83.1% for random errors -- verification brings coherent errors 80% of the way to the random level. This means that a sufficiently dense web of cross-domain dependencies can in principle eliminate the immunity of coherent falsehood.

5. **Inverse scaling under fixed training steps.** Unlike random errors (accuracy increases with model size), chained tasks show a declining trend: 70.9% -> 64.2% -> 60.6%. This is consistent with the core hypothesis: a more powerful model compresses the coherent error system *within* each domain more effectively, and the verification signal does not compensate. As capacity grows, within-domain coherence (strong signal) may outweigh cross-domain contradiction (weak signal). For comparison: multi-rule errors (Experiment 5) create *within-domain* incompressibility and therefore show the opposite trend. (Caveat: large uses only 2 seeds; compute-matched experiments are needed to separate capacity effects from convergence.)

## 8. Discussion

### 8.1 Unified Interpretation

Nine experiments, taken together, support a consistent picture:

1. **Compression favors consistency, not truth.** Any consistent rule system -- true or false -- compresses equally well. Truth bias with random errors is explained by the fact that each random error must be memorized individually.

2. **Truth usually wins because errors are usually incoherent.** In real data, different authors make different errors, whereas correct answers are uniform. This aligns with the Truth Co-occurrence Hypothesis (Ravfogel et al., 2025): true statements are more likely to co-occur with other true statements, forming a statistical cluster that the model learns.

3. **Corpus-level and paired metrics can diverge.** At 10/90, corpus-level DLoss inverts (-0.0016), but paired evaluation shows robust truth bias (67% accuracy, p < 10^-88). The corpus-level metric conflates structural preference with the frequency effect on shared problem patterns. An analogous divergence is observed for conditions C/D/E: corpus-level DLoss is positive, but paired evaluation shows accuracy ~49%. This makes paired evaluation a necessary tool for truth bias research.

4. **Correction increases corpus-level DLoss but does not produce transferable truth bias.** Models trained with observations and correction do not distinguish correct from incorrect at the level of pure mathematical pairs (accuracy ~49%). The only reliable mechanism of truth bias is error incoherence.

5. **Under fixed-step training, truth bias grows with model size for incoherent errors, but coherent falsehood remains unaffected.** Increasing the model from 3.5M to 86M strengthens truth bias for random errors (83% -> 89%), but does not help distinguish coherent falsehood from truth. Chained tasks (Experiment 9) show the opposite trend: accuracy drops from 71% to 61% as model size grows, consistent with a more powerful compressor memorizing the coherent system within each domain. (This trend is observed under fixed training steps; compute-matched experiments would be needed to confirm it as a scaling property.)

6. **Multi-rule errors reveal a sharp transition.** The transition from one error rule to two causes a jump in truth bias from 49% to 87%. This shows that the critical factor is not the number of errors, but the predictability of the error system. One rule is predictable and compressible; two rules, randomly applied, are not.

7. **Truth bias transfers to natural language, but weakens (Appendix B).** A synthetic world with 15 rules yields 57.7% pair accuracy (vs 83.1% in mathematics). Natural language absorbs contradictions that would be detectable in formal math: increasing the number of alternative erroneous conclusions from 1 to 16 produces only a gradual rise (47% -> 60%), unlike the sharp jump in mathematics (49% -> 87% at N = 2). Coherent errors remain indistinguishable from truth in natural language as well (46.6%). A preliminary cross-domain experiment (Appendix B.3) provides initial evidence that cross-domain data can selectively destroy coherence, though the effect is weak and non-monotonic.

10. **The verification step transforms coherent errors into detectable ones, but the effect may weaken with scale.** Chained tasks raise accuracy from 43% to 71% (tiny, p < 10^-6), but scaling suggests an *inverse* trend: 71% -> 64% -> 61% (tiny -> small -> large; caveat: large uses only 2 seeds). A control experiment with truncated chains (no verification step) confirms that the effect is produced by verification: accuracy of truncated chains (44.3%) is indistinguishable from standard coherent errors (43.3%). This is consistent with the paper's central thesis: the compressor favors *consistency*, not truth. Under fixed-step training, increased capacity appears to make coherent falsehood more, not less, resilient, though this conclusion requires confirmation with more seeds and compute-matched training.

### 8.2 Analogy with Popper's Falsifiability

Our results admit an interpretive analogy with the falsifiability criterion (Popper, 1959). Compression pressure acts as a computational analog: a true theory with concrete predictions requires no additional explanations (maximal compression); a false theory whose predictions diverge from data needs correction (poor compression); a theory with ad hoc escape hatches expands with every observation (anti-compression).

However, the analogy has limits. First, the model does not "test" theories -- it simply minimizes code length. Second, our data show that bare discrepancies alone barely help (condition B ~ A): regular discrepancies are compressible. Popperian falsification assumes that a discrepancy with observation refutes a theory; for a compressor model, a discrepancy is merely another pattern. Moreover, paired evaluation of conditions C/D/E showed that even ad hoc correction does not produce transferable truth bias -- the model learns to process correction patterns but does not transfer this to pure mathematical pairs.

Practical analogies from the history of science are appropriate as illustrations. The geocentric model required ever more epicycles to reconcile with observations (a variation of condition C); phlogiston theory needed special assumptions to explain mass increase during combustion; miasma theory could not explain why disease spread along waterways rather than by wind. However, our experiments use the mathematical domain, and transfer to these real-world examples remains an open question.

### 8.3 Implications

**For alignment.** Our results suggest that models lack an innate "truth compass" -- they favor well-compressible patterns. In our synthetic settings, systematic deception consistently represented in data encounters no resistance from compression.

**For ML epistemology.** The framework suggests why models develop internal truth representations (Marks & Tegmark, 2023): in real corpora, true statements are more coherent than false ones. It also suggests a candidate mechanism consistent with the inverse scaling effect on TruthfulQA (Lin et al., 2022): if larger models are better at memorizing coherent misconceptions -- which in our experiments compress just as well as truth -- this could contribute to confident reproduction of common falsehoods. The inverse scaling of chained tasks (71% -> 61%, though based on limited seeds at large scale) is consistent with this mechanism, but confirming the link to TruthfulQA requires experiments on real-world corpora.

**For scaling.** In our experiments, we observe three scaling regimes under fixed training steps: direct (random errors: 83% -> 89%), flat (coherent: ~49% at any size), and inverse (chained: 71% -> 61%). If these trends hold under compute-matched training, they suggest that scaling behavior depends on the ratio of cross-domain verification signal density to within-domain coherence of the error system. When errors are individually unique, scaling helps. When one verification step per task is the only cross-domain signal, scaling may not. If so, increasing parameter count alone would not guarantee "convergence on truth" -- what would matter is the structure of the training corpus: how densely facts from different domains are interconnected.

**For understanding hallucinations.** Models confidently reproduce coherent misconceptions not due to a compression failure, but because such misconceptions compress *successfully*. This aligns with the analysis by Chlon et al. (2025).

### 8.4 Limitations

**Model scale and information-theoretic limits.** Experiments use models from 3.5M to 86M parameters. Truth bias for random errors grows with size with no plateau (83% -> 89%). For chained tasks, inverse scaling is observed (71% -> 61%), raising the question: under what conditions do verification signals cease to work? The range remains limited -- extrapolation to GPT-2/3 scale models requires further experiments.

For *isolated* coherent errors, there is a stronger argument. If the false system has the same Kolmogorov complexity as the true system (one rule vs. one rule) and both appear at equal frequency, an MDL-optimal learner has no basis to prefer one over the other. Our data are consistent with this prediction: accuracy ~49% for coherent errors at all model sizes from 3.5M to 86M.

For *real corpora*, the situation is more complex. Scientific knowledge is pervaded by cross-domain dependencies: a theorem from one field is used in another, an engineering calculation is verified by experiment, a conservation law connects different quantities. The density of such connections is far higher than in our experiments (one verification step per task). Whether a sufficiently dense web of cross-domain verifications can compensate for the growing power of the compressor remains an open question. Our inverse scaling (71% -> 61%) serves as a warning: at low verification signal density, scaling *worsens* rather than improves detection of coherent falsehood.

**Domain specificity.** Mathematics has an unusually crisp distinction between correct and incorrect. The synthetic world experiment (Appendix B.1) confirms that the effect is substantially weaker in a natural language domain (57.7% vs 83.1%). Moreover, the multi-alternative experiment (Appendix B.2) shows that even internally contradictory errors in natural language do not trigger a sharp transition, unlike in mathematics. Transfer to real-world domains (medicine, history, economics) requires further experiments.

**Confounding with corpus length.** Conditions C/D/E generate substantially longer texts (loss ~0.24 vs ~0.14). DLoss may partially reflect a difference in convergence rather than compressibility per se. Paired evaluation mitigates but does not fully eliminate this confound.

**Training duration.** All models are trained for 5000 steps with a fixed learning rate schedule. As models grow from tiny (3.5M) to large (86M), the number of parameters increases by an order of magnitude, but neither the number of training steps nor the compute budget changes. This means larger models may be substantially undertrained relative to their capacity. Scaling results should be interpreted with caution: the observed growth of truth bias (83% -> 89%) could reflect either increased capacity or differential convergence. A compute-matched comparison (equalizing FLOPs rather than steps) and learning curves to convergence would strengthen the conclusions. For large (86M), 4 seeds are used in the main scaling experiment (random and coherent errors), but only 2 seeds in the chained task scaling (Experiment 9), limiting the strength of the inverse scaling conclusion.

**Effect size and statistical caveats.** DLoss (0.003--0.012) is small in absolute terms. Its practical significance for large models remains an open question. With 4,951 pairs per test, Wilcoxon p-values will inevitably be minuscule (< 10^-6) even for small effects. Statistical significance is therefore less informative than effect size: pair accuracy (83% vs 49%) and seed-level variability (+/-1--2 pp across 4 seeds) are more substantive measures. We report p-values for completeness but recommend that readers focus on effect sizes and seed-level confidence intervals.

### 8.5 Future Experiments

**Extensions of chained tasks.** Experiment 9 confirmed that verification restores truth bias (71% at tiny), but scaling revealed inverse scaling (71% -> 61%). A control experiment with truncated chains (44.3% accuracy, 4 seeds) confirmed that it is verification, not different task structure, that produces the effect (Table 10c). Open directions: (1) increasing verification density (2--3 checks per task) to assess whether this can compensate for the compressor's growing power; (2) combining multi-rule and chained approaches.

**Methodological controls.** Several controlling experiments remain open. First, equalizing the token budget for conditions C/D/E (Section 6): these conditions generate texts of different lengths (loss ~0.24 vs ~0.14), and convergence differences may affect results. Second, deterministic evaluation on the full test set rather than random batches would increase estimate reliability. Third, a factor analysis isolating the contributions of truth value, frequency, coherence, and correction overhead would allow quantitative separation of these intertwined factors.

**Linear probing.** Extract activations and train linear classifiers to detect "truth directions" vs. "coherence directions" (Marks & Tegmark, 2023 methodology).

**Synthetic world scaling.** Truth bias in the natural language domain is weaker (57.7% vs 83.1%, Appendix B.1), and even multi-alternative errors yield only 60% at N=16 (Appendix B.2). Scaling to small/medium models will show whether the effect grows with size analogously to the mathematical domain.

**Real-world domains.** Extend to domains with competing knowledge systems:
- **Type 3b (ad hoc):** Evidence-based medicine vs. homeopathy, vaccination vs. anti-vax theories.
- **Historical:** Phlogiston vs. oxygen theory, miasma theory vs. germ theory, geocentrism vs. heliocentrism.
- **Type 3a (non-specific):** Astrology, market technical analysis.

## 9. Conclusion

This work provides evidence for the Compression--Consistency Principle: language models favor the most compressible hypothesis consistent with the data, not truth per se. **Truth bias is not a fundamental property of compression, but a side effect of error incoherence in the corpus.** Random errors are incompressible and must be memorized individually, giving correct mathematics a structural advantage (83% pair accuracy, 16/16 seeds). A coherent false system, as compact as truth, strips compression of any preference (~49% pair accuracy at any model size from 3.5M to 86M). The multi-rule error experiment shows a sharp transition: a single false rule is indistinguishable from truth, but as few as two rules applied unpredictably restore truth bias (87%), at a level even higher than random errors (83%). The effect reproduces beyond mathematics: in a natural language domain (synthetic world with 15 rules; Appendix B), pair accuracy reaches 57.7%, supporting the generality of the mechanism, albeit with reduced effect size. The multi-alternative experiment (Appendix B.2) shows that even increasing error diversity to N = 16 alternatives per rule in natural language yields only 60% accuracy -- no sharp transition occurs, unlike the dramatic N=1 -> N=2 jump in mathematics. Natural language provides enough structural flexibility to absorb contradictions that would be immediately detectable in formal domains.

The practical implication for alignment: in our experiments, scaling (from 3.5M to 86M under fixed training steps) strengthens truth bias for incoherent errors, but does not help against coherent falsehood. A compressor model has no "truth compass" -- it has a consistency compass. In real corpora, these compasses typically coincide, since different authors' errors are diverse while correct answers are uniform. Our experiments in synthetic domains *predict* (but do not prove for real data) that where falsehood is systematic and internally consistent -- in entrenched misconceptions, coherent pseudoscientific systems -- compression may give the model no basis to prefer truth. Testing this hypothesis on real corpora remains a necessary next step.

However, the immunity of coherent falsehood is not absolute. The chained task experiment (Section 7.4) shows that embedding a verification step within the task -- where the coherent error produces an unpredictable numerical residual -- restores truth bias to 71% for the tiny model (vs 43% for isolated coherent). Under fixed-step training, scaling suggests an *inverse* trend: accuracy drops from 71% (3.5M) to 64% (11M) and 61% (86M, 2 seeds), while for random errors it rises (83% -> 89%). This is consistent with a more powerful compressor memorizing the coherent system within each domain, outpacing the weak cross-domain verification signal -- though the limited seed count at large scale warrants caution. If this trend holds under compute-matched conditions, it would suggest that compressor power is a double-edged sword: for incoherent errors, scaling helps; for coherent falsehood with sparse verification, it may entrench it further.

The question of scale and domain remains open. Our experiments are limited to models of 3.5M--86M parameters and synthetic domains. The multi-alternative experiment (Appendix B.2) shows that in the natural language domain, contradictions between errors do not destroy compressibility as effectively as in mathematics. Transferring these results to larger models and real corpora is a necessary condition for strong generalizations.

## References

Azaria, A., & Mitchell, T. (2023). The Internal State of an LLM Knows When It's Lying. *Findings of EMNLP 2023*.

Bhattamishra, S., Patel, A., Kamath, S., & Blunsom, P. (2023). Simplicity Bias in Transformers and their Ability to Learn Sparse Boolean Functions. *ACL 2023*.

Bürger, L., Hamprecht, F. A., & Nadler, B. (2024). Truth is Universal: Robust Detection of Lies in LLMs. *NeurIPS 2024*.

Burns, C., Ye, H., Klein, D., & Steinhardt, J. (2023). Discovering Latent Knowledge in Language Models Without Supervision. *ICLR 2023*.

Chlon, L., Karim, A., Chlon, M., & Awada, M. (2025). Predictable Compression Failures: Why Language Models Actually Hallucinate. *arXiv:2509.11208*.

Deletang, G., Ruoss, A., Grau-Moya, J., Genewein, T., Wenliang, L. K., Catt, E., ... & Legg, S. (2024). Language Modeling Is Compression. *ICLR 2024*.

DeMoss, B., Sapora, S., Foerster, J., Hawes, N., & Posner, I. (2024). The Complexity Dynamics of Grokking. *arXiv:2412.09810*.

Elazar, Y., Kassner, N., Ravfogel, S., Ravichander, A., Hovy, E., Schütze, H., & Goldberg, Y. (2022). Measuring Causal Effects of Data Statistics on Language Model's Factual Predictions. *arXiv:2207.14251*.

Goldblum, M., Finzi, M., Rowan, K., & Wilson, A. G. (2024). The No Free Lunch Theorem, Kolmogorov Complexity, and the Role of Inductive Biases in Machine Learning. *ICML 2024*.

Grünwald, P. D. (2007). The Minimum Description Length Principle. *MIT Press*.

Gurnee, W., & Tegmark, M. (2024). Language Models Represent Space and Time. *ICLR 2024*.

Halawi, D., Denain, J.-S., & Steinhardt, J. (2024). Overthinking the Truth: Understanding how Language Models Process False Demonstrations. *ICLR 2024*.

Huang, Y., Zhang, J., Shan, Z., & He, J. (2024). Compression Represents Intelligence Linearly. *COLM 2024*.

Hutter, M. (2005). Universal Artificial Intelligence: Sequential Decisions Based on Algorithmic Probability. *Springer*.

Joshi, N., Rando, J., Saparov, A., Kim, N., & He, H. (2024). Personas as a Way to Model Truthfulness in Language Models. *EMNLP 2024*.

Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain, D., Perez, E., ... & Kaplan, J. (2022). Language Models (Mostly) Know What They Know. *arXiv:2207.05221*.

Kandpal, N., Deng, H., Roberts, A., Wallace, E., & Raffel, C. (2023). Large Language Models Struggle to Learn Long-Tail Knowledge. *ICML 2023*.

Kang, J., & Choi, J. (2023). Impact of Co-occurrence on Factual Knowledge of Large Language Models. *Findings of EMNLP 2023*.

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

Rissanen, J. (1978). Modeling by Shortest Data Description. *Automatica*, 14(5), 465-471.

Rolnick, D., Veit, A., Belongie, S., & Shavit, N. (2017). Deep Learning is Robust to Massive Label Noise. *arXiv:1705.10694*.

Shannon, C. E. (1948). A Mathematical Theory of Communication. *Bell System Technical Journal*, 27(3), 379-423.

Solomonoff, R. J. (1964). A Formal Theory of Inductive Inference. *Information and Control*, 7(1), 1-22.

Valle-Perez, G., Camargo, C. Q., & Louis, A. A. (2019). Deep Learning Generalizes Because the Parameter-Function Map Is Biased Towards Simple Functions. *ICLR 2019*.

Wan, J., & Mei, L. (2025). Large Language Models as Computable Approximations to Solomonoff Induction. *arXiv:2505.15784*.

Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2017). Understanding Deep Learning Requires Rethinking Generalization. *ICLR 2017*.

## Appendix A: Reproducibility

All code, data generation scripts, and evaluation scripts are available at https://github.com/Rai220/compression-drives-truth. Experiments were conducted on an Apple Mac M4 with 36GB of unified memory using the MLX framework (v0.31.0). Large model training (86M) was performed on cloud GPU instances. Total computational cost: approximately 65 hours of wall-clock time for the 160+ training runs described in this paper.

## Appendix B: Natural Language and Cross-Domain Experiments

### B.1 Experiment 6: Synthetic World (Natural Language)

All main experiments use the mathematical domain with character-level tokenization. To test transferability, we create a synthetic world with 50 entities of four types (animals, plants, minerals, potions) and 15 deterministic rules linking entity properties to observable outcomes. Examples are described in natural language:

> The fire crystal has temperature 250 and clarity 7. Since temperature exceeds 150, the fire crystal glows brightly.

The corpus contains 100,000 examples. As in the mathematical experiment, we train models under two conditions: random 50/50 (half of observations with inverted outcomes) and coherent 50/50 (half follow an alternative rule system with inverted thresholds).

**Table B1.** Paired evaluation of the synthetic world (tiny, 3.5M, 50/50, 4 seeds).

| Condition | Avg Accuracy | Avg DLoss (paired) | CI 95% DLoss | Wilcoxon p |
|-----------|:----------:|:-----------------:|:------------:|:----------:|
| Random errors | 57.7% | +0.034 | [+0.027, +0.040] | < 10^-6 |
| Coherent errors | 46.6% | +0.019 | [+0.008, +0.030] | ~ 0.05 |

**Table B1a.** Paired accuracy by entity type (random errors).

| Type | Accuracy | DLoss |
|------|:--------:|:-----:|
| mineral | 68.7% | +0.056 |
| plant | 62.2% | +0.057 |
| animal | 51.4% | +0.003 |
| potion | 49.1% | +0.029 |

Three key observations:

1. **Truth bias reproduces in natural language.** Pair accuracy of 57.7% for random errors is significantly above chance (p < 10^-6 for all 4 seeds). The compression effect in favor of truth is not limited to formal mathematics.

2. **The effect is substantially weaker than in mathematics.** 57.7% vs 83.1% with identical architecture and corpus proportion. The likely reason: natural language contains more variability in formulations (synonymous constructions, diversity of entity names), weakening the statistical separation between correct and incorrect conclusions.

3. **Coherent errors remain indistinguishable from truth.** Accuracy of 46.6% -- below chance, same as in the mathematical domain (47.2%). The pattern reproduces: a coherent alternative rule system compresses equally well. The strong variance across types (mineral 68.7% vs potion 49.1%) is explained by differences in description length and the number of applicable rules for different entity types.

**Note on metrics.** For coherent errors in Table B1, pair accuracy (46.6%) and mean DLoss (+0.019) diverge in sign. This is explained by distributional asymmetry: the majority of pairs show a slight preference for the incorrect conclusion (accuracy < 50%), but a small number of pairs with strong preference for the correct one shift the mean DLoss positive. The Wilcoxon test, based on ranks rather than means, yields p ~ 0.05 -- a borderline result that does not reach significance. Accuracy is the more reliable metric in this case.

### B.2 Experiment 7: Multi-Alternative Errors in the Synthetic World

In the mathematical domain (Section 7.3), the transition from one coherent error rule to two causes a sharp jump in truth bias (49% -> 87%). Does this sharp transition reproduce in the natural language domain? We create a pool of 16 alternative conclusions for each of the 15 rules in the synthetic world and vary N -- the number of alternatives used during training. For each erroneous example, one of the N pre-selected alternatives is assigned at random. At N = 1, this is equivalent to coherent errors; at N = 16, it represents maximum internal inconsistency.

**Table B2.** Multi-alternative errors in the synthetic world (tiny, 3.5M, 50/50, 4 seeds). Paired accuracy: correct vs multi-alt (same alternatives as in training) and correct vs random (baseline).

| N alternatives | Acc vs multi-alt | Acc vs random |
|:---:|:---:|:---:|
| 1 (coherent) | 46.6% | 57.7% |
| 2 | 39.8% | 97.3% |
| 4 | 50.2% | 96.4% |
| 8 | 51.4% | 86.4% |
| 16 | 60.0% | 81.5% |

![Figure 9](results/figure9_world_multialt.png)

*Figure 9. Multi-alternative errors in the synthetic world. Left: pair accuracy as a function of the number of alternatives N -- gradual rise with no sharp transition. Right: comparison with the math domain -- sharp jump at N=1->2 in math, gradual rise in natural language.*

Four observations:

1. **No sharp transition.** Unlike mathematics (jump at N=1 -> N=2: 49% -> 87%), in natural language the growth is gradual: 47% -> 40% -> 50% -> 51% -> 60%. Even at N = 16, accuracy is only 60%, close to the result for fully random errors (57.7%).

2. **N = 2 worsens the result.** With two alternatives, the model *prefers* erroneous conclusions (39.8% < 50%), worse than a single coherent alternative (46.6%). The likely reason: two alternatives create a distribution (each at ~25% of the corpus) that collectively competes with the correct conclusion (50%) while remaining compressible.

3. **High accuracy vs random masks weak truth bias.** The same models trained on N = 2 show 97.3% accuracy when compared against fully random errors. The model successfully learns all N alternative patterns but cannot distinguish them from truth -- both sets are equally well compressible.

4. **Natural language absorbs contradictions.** In formal mathematics, two contradictory rules (N = 2) immediately destroy compressibility: for arithmetic, `a + b = a + b + 1` and `a + b = a + b - 1` cannot be captured by a single function. In natural language, the phrases "has thin scales" and "has dense armor plates" are simply two textual patterns, each of which is easily memorized. The structure of text provides sufficient degrees of freedom to compress incompatible statements.

This result has practical significance: **in domains with natural language structure, compression pressure weakly distinguishes truth from plausible misinformation**, even when the latter is internally contradictory. This may help explain why LLMs readily memorize and reproduce coherent misconceptions.

### B.3 Experiment 8: Cross-Domain Falsification

A central finding of our experiments -- that coherent falsehood is indistinguishable from truth under compression -- rests on isolated domains: false derivative rules are tested only on derivative tasks. In the real world, a false theory of derivatives conflicts with adjacent domains (integration, tangent lines, numerical evaluation). We test whether adding *cross-domain* tasks -- correct tasks linking derivatives with arithmetic -- can destroy the coherence of the false rule.

Base corpus: coherent 50/50 (Section 4.2). We add correct cross-domain tasks of five types: derivative evaluation at a point, antiderivative check, tangent line equation, chain rule evaluation, and product rule evaluation. All cross-domain tasks use the true differentiation rules. We vary the proportion of cross-domain tasks: 0%, 10%, 25%, 50%.

**Table B3.** Cross-domain falsification (tiny, 3.5M, 4 seeds). Paired evaluation on coherent paired test.

| Cross-domain proportion | Overall accuracy | Derivative accuracy | Other types |
|:-----------------------:|:---------------:|:------------------:|:-----------:|
| 0% (baseline) | 47.0% | 35.2% | 51.1% |
| 10% | 45.8% | 39.4% | 47.9% |
| 25% | 50.6% | **56.0%** | 48.8% |
| 50% | 47.1% | 45.4% | 47.6% |

![Figure 8](results/figure8_crossdomain.png)

*Figure 8. Cross-domain falsification. Left: accuracy by task type — only derivatives respond to cross-domain tasks. Right: non-monotonic effect — peak at 25%, decline at 50% due to corpus dilution.*

The result partially supports the hypothesis: accuracy on **derivatives** increases from 35.2% to 56.0% at 25% cross-domain tasks -- the model begins to prefer correct derivatives. However, the effect is non-monotonic: at 50%, accuracy drops to 45.4%, likely due to dilution of standard patterns. Other task types (algebra, arithmetic, equations) remain at chance, since the cross-domain tasks address only contradictions with derivatives.

This experiment provides the first evidence that cross-domain data can *selectively* destroy the coherence of false rules. The effect is still weak (derivative accuracy: 56% vs 50% chance), as expected for a tiny model (3.5M). Scaling to larger models and expanding the set of cross-domain tasks is a priority for future work.
