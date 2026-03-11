# Compression Favors Consistency, Not Truth: When and Why Language Models Prefer Correct Information

**Author:** Konstantin Krestnikov
**Date:** 03.2026

## Abstract

Why do language models sometimes prefer correct statements even when trained on mixed-quality data? We study this question in controlled synthetic corpora and propose the Compression--Consistency Principle: gradient descent favors hypotheses that yield shorter and more internally consistent descriptions of the training data. In this framing, truth bias is not fundamental. It appears when false alternatives are harder to compress than the correct rule system. Models compress *text*, not reality, so the effect should be interpreted as a property of corpus structure under a specific training setup.

We test this idea with GPT-2 style character-level transformers (3.5M--86M parameters) on synthetic corpora with controlled mixtures of correct and incorrect derivations. With random errors, paired evaluation yields strong preference for correct completions: 83.1% accuracy at 50/50 and 67.0% even at 10/90. In a synthetic natural-language world, the same direction appears more weakly (57.7%). Replacing random errors with an internally coherent but mathematically wrong rule system removes the effect in the available fixed-step runs, producing near-chance paired accuracy.

Additional experiments suggest two boundaries on the effect. First, verification steps embedded inside coherent tasks can restore a preference for correct completions at tiny scale (70.9%), although the size trend remains preliminary under fixed-step training and limited replication at large scale. Second, rebuilding the multi-rule math experiment with matched paired evaluation produces a graded increase from 46.6% at `N=1` to 88.3% at `N=10`, with the largest jump between `N=1` and `N=2` but without the stronger legacy claim of an immediate phase transition. The main conclusion is therefore limited: in controlled synthetic corpora, preference for correct solutions tracks the compressibility and internal consistency of competing alternatives more closely than truth in the abstract.

---

## 1. Introduction

Language models are increasingly accurate on factual benchmarks, yet they confidently generate false statements. What determines when a model prefers truth and when it doesn't?

Several explanations have been proposed. Scaling helps: larger models perform better on factual tasks (Kadavath et al., 2022). RLHF and similar alignment techniques steer models toward human-preferred outputs. Data statistics play a role: factual accuracy correlates with the frequency and source reliability of facts in training data (Elazar et al., 2022; Joshi et al., 2024; Kandpal et al., 2023). Internal truth representations have been discovered in model activations (Burns et al., 2023; Marks & Tegmark, 2023). Yet none of these explanations address a more fundamental question: *why would the training objective itself -- next-token prediction -- create any preference for truth?*

We propose that the answer is compression. Minimizing cross-entropy is mathematically equivalent to minimizing code length (Shannon, 1948; Deletang et al., 2024), connecting LLM training to the Minimum Description Length principle (Rissanen, 1978; Grünwald, 2007). A model that better predicts tokens is a better compressor. Compression quality correlates linearly with model capabilities (Huang et al., 2024), and LLM training formally approximates Solomonoff induction (Wan & Mei, 2025). But compression does not inherently favor truth -- it favors the most *compressible* hypothesis consistent with the data. We call this the **Compression--Consistency Principle**: truth benefits from compression only when falsehood is structurally incoherent. Diverse errors must be memorized individually, whereas a correct rule system compresses into a compact representation. When errors form a coherent alternative system -- internally consistent but wrong -- they compress just as efficiently, and the preference vanishes.

This principle has three important caveats. First, models compress *text*, not reality: "truth" here means correctness of mathematical derivations, not a metaphysical category. Second, frequency can override compressibility: a structural advantage need not dominate numerical minority. Third, the compressibility gap between truth and falsehood is corpus-dependent, not universal. We test the principle through controlled experiments on mathematical corpora (Experiments 1--3, Sections 4--6) and extend to scaling, multi-rule errors, and chained verification (Experiments 4--5 and 9, Section 7), with additional natural-language and cross-domain experiments in Appendix B. All size comparisons should be interpreted as fixed-step training results rather than compute-matched scaling laws.

## 2. Related Work

### Prediction as Compression

The link between prediction and data compression traces back to the foundational work in information theory. Shannon (1948) showed that optimal compression requires knowledge of the true data distribution, and Solomonoff (1964) formalized optimal prediction as weighting hypotheses by their program length. Rissanen (1978) developed the Minimum Description Length (MDL) principle, formalizing model selection as a compression task: the best model minimizes the total description length of the model plus the data given the model. Grünwald (2007) systematized the MDL principle and showed its equivalence to several forms of statistical inference. Hutter (2005) developed these ideas into a formal theory of universal artificial intelligence (AIXI), explicitly linking intelligence to compression ability. Our work directly builds on the MDL framework: we experimentally vary the description length of false systems and observe under what conditions the MDL-optimal choice coincides with truth.

In the context of language models, Deletang et al. (2024) empirically demonstrated that LLMs are universal compressors: a next-token predictor can serve as an arithmetic coder. Huang et al. (2024) discovered a linear correlation (r ~ -0.95) between compression quality and benchmark performance, and Wan & Mei (2025) formally proved that LLM training approximates Solomonoff induction. These results form the theoretical foundation of our work: if LLM training is compression, under what conditions does compression align with behavior that favors correct over incorrect continuations?

### Internal Representations of Truth in LLMs

Several studies have found that language models form internal representations correlated with statement truthfulness. Marks & Tegmark (2023) showed a linear geometric structure of truthfulness in activation space (Geometry of Truth), and Burns et al. (2023) proposed CCS, a method for discovering truth directions without supervision. Li et al. (2023b) identified a 40% gap between a model's internal knowledge and its generation, developing Inference-Time Intervention (ITI). Ravfogel et al. (2025) proposed the Truth Co-occurrence Hypothesis -- a mechanism for the emergence of linear truth representations through co-occurrence of true statements in the corpus. Our work complements this line behaviorally rather than representationally: we study when compression produces a paired preference for correct over incorrect continuations in controlled corpora, leaving activation-level analysis for future work.

### Emergent World Models

Language models can form internal world models from pure text prediction. Li et al. (2023a) trained a model to predict Othello moves and found that it learns a full board representation (Othello-GPT). Gurnee & Tegmark (2024) discovered linear representations of space and time in Llama-2 activations. These results show that sequence compression can give rise to structured internal representations. Our work does not probe such representations directly; instead, it asks when compression yields behavioral preference for correct versus incorrect continuations.

### Truthfulness and Training Data Statistics

Several works investigate the dependence of factual behavior on the structure of the training corpus. Joshi et al. (2024) showed that truthfulness in LLMs is linked to the structure of "personas" (sources) in pretraining data: the model learns persona-specific patterns and prefers statements associated with reliable sources. Elazar et al. (2022) demonstrated that factual predictions strongly depend on the frequency of facts in training data. Kang & Choi (2023) investigated how co-occurrence between statements affects factual recall, and Kandpal et al. (2023) showed a direct relationship between the number of supporting documents in the corpus and model answer accuracy. Our work differs from this line in emphasis: frequency clearly matters in our experiments as well, but we *experimentally vary* the structure of errors (their compressibility) while controlling frequencies within each condition. This allows us to isolate one factor beyond frequency and source reliability rather than claiming that data statistics play no role.

### Simplicity Bias, Noisy Labels, and Grokking

The inductive bias of neural networks towards simple functions is a well-documented phenomenon. Valle-Perez et al. (2019) showed an exponential preference for low-complexity functions, and Mingard et al. (2021) proved that SGD approximates Bayesian sampling with a simplicity prior. Goldblum et al. (2024) connected this to Kolmogorov complexity, providing a theoretical basis for the link between compression and generalization. Bhattamishra et al. (2023) showed that transformers exhibit a pronounced simplicity bias, preferring lower-complexity solutions when multiple hypotheses are consistent with the data.

The noisy labels literature directly parallels our setup. Zhang et al. (2017) demonstrated that neural networks can memorize completely random labels, but when structure is present, they generalize through it. Rolnick et al. (2017) showed that learning is robust to massive label noise -- the network learns the "clean" pattern even when noise overwhelmingly predominates. Our result with random errors (truth bias at 10/90) directly aligns with these observations: random errors play the role of noise labels, through which the network generalizes to structured correct solutions.

The phenomenon of grokking -- delayed generalization -- is also related to compression. Nanda et al. (2023) showed that networks discover Fourier transforms for modular arithmetic, and DeMoss et al. (2024) described the phase transition from memorization to generalization through complexity dynamics. Liu et al. (2023) interpreted grokking as a compression process: the network transitions from memorization to a compact representation. Our experiments with coherent errors directly connect to simplicity bias: a coherent false system is just as "simple" as truth, and compression shows no preference for either.

### Our Contribution

The works listed above either study internal truth representations in already-trained models, establish theoretical links between compression, simplicity bias, and intelligence, or analyze the dependence of truthfulness on data statistics. To our knowledge, direct training experiments on quality-controlled corpora with systematic variation of *error compressibility* remain limited. This work contributes one such controlled behavioral study.

| Work | What it studies | Our difference |
|------|----------------|----------------|
| Joshi et al. (2024) | Truthfulness via source/persona structure in data | We vary *error compressibility* directly, isolating compression from source reliability |
| Ravfogel et al. (2025) | Mechanism of truth encoding emergence via co-occurrence | We show behavioral failure conditions: coherent falsehood can remove paired preference for correct continuations |
| Elazar (2022), Kang & Choi (2023), Kandpal (2023) | Factual behavior as a function of frequency/support count | We fix frequency and vary error structure: at 50/50, truth bias = 83% (random) vs 49% (coherent) |
| Zhang et al. (2017), Rolnick et al. (2017) | Learning from noisy labels: generalization vs memorization | We generalize to sequence-level: not labels but entire derivations, showing that "structured noise" (coherent errors) is not filtered out |
| Burns et al. (2023), Marks & Tegmark (2023) | Internal truth directions in pretrained models | We train from scratch on controlled data and identify behavioral conditions under which paired preference for correct continuations appears or disappears |

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

**Justification for the number of seeds.** Most conditions are repeated with 4 random initializations. This is sufficient to show whether the direction of an effect is stable across runs, but it does **not** tightly estimate between-run uncertainty. We therefore use seed-level summaries as the main unit for training variability, and report paired-test statistics within each seed as a separate source of evidence about the behavior of one trained model on many held-out items. Combined tests across several conditions are reported only as omnibus support; they are not a substitute for condition-specific replication.

### 3.2 Corpus Generation

The generator creates mathematical problems of four types: multi-step arithmetic, factorization, equation solving, and differentiation. Each problem is formatted as a step-by-step solution in English, verified by SymPy. The tokenizer is character-level (vocab size = 57) to exclude BPE artifacts as a confound.

**Error Types:**
- **Random:** Injection of one plausible error at a random step (sign, coefficient, distributivity error). Each error is unique.
- **Coherent:** One systematic incorrect rule per problem type (e.g., a x b = a x (b-1); sign is preserved when moving terms across =; etc.). All problems of one type fail identically.
- **Contradictory:** Simple rules (a + b = a + b + 1; a - b = a - b - 2) that break algebraic structure -- addition and subtraction cease to be inverse operations.

### 3.3 Metrics

**Paired evaluation (primary metric).** For each problem, a single shared prompt is generated along with two completions (correct and incorrect). NLL is computed only on completion tokens, conditioned on the shared prompt. This yields pairwise comparison under identical context, eliminating the confound of different prompts. Metrics: **pair accuracy** (fraction of pairs where the model prefers correct; our primary metric), mean DLoss on completions, one-sided Wilcoxon signed-rank test. We adopt paired evaluation as the primary metric because corpus-level measures can be confounded by differences in text statistics between correct and incorrect corpora (see Sections 4.1 and 6 for concrete examples of such divergence).

**Corpus-level evaluation (secondary diagnostic).** We report two corpus-level variants. The legacy estimate samples random windows from the concatenated correct and incorrect token streams; this preserves continuity with earlier artifacts but is sensitive to local formatting and stream boundaries. As a robustness check, we also run a deterministic example-block evaluation that scores every held-out problem separately without crossing example boundaries. In both cases **DLoss = Loss(incorrect) - Loss(correct)**; a positive value indicates lower loss on correct examples. We treat these corpus-level measures as diagnostics rather than as the main truth-bias metric.

**Statistical analysis.** Each configuration is repeated with 4 random initializations (seeds 42--45), except where noted otherwise. For training variability we report seed-level effect sizes, means across seeds, and dispersion across seeds. For individual configurations we use the two-sided binomial test on seed directions as a small-sample directional check. For paired evaluation, the **one-sided** Wilcoxon signed-rank test (`alternative='greater'`) is applied to paired NLL differences within a single trained model; this quantifies uncertainty over held-out pairs, not uncertainty over the training procedure. 95% confidence intervals for DLoss are obtained via bootstrap and should be interpreted at the corresponding level of aggregation.

### 3.4 Theoretical Framework: Description Length and Theory Types

To interpret Experiments 2--3 we use a typology of theories distinguished by the description length of the corpus "theory + observations." The key principle: the model optimizes cross-entropy, which is equivalent to minimizing expected code length (Shannon, 1948). A theory that allows shorter encoding of the corpus gains an advantage.

**Type 1: True theory with concrete predictions.** Predictions match observations. The "theory + observations" corpus compresses maximally: one rule system explains everything.

**Type 2: False theory with concrete predictions.** Predictions diverge from observations. The model must encode both the false rules and the discrepancies. However, if the discrepancies are **regular** (e.g., a x b = a x (b-1) always understates by a), the model can learn a correction, and the additional description length is small.

**Type 3a: Theory with non-specific predictions.** The theory does not specify a "situation -> outcome" mapping (e.g., "result is moderate"). It does not contradict observations but does not help predict them either -- it does not reduce code length.

**Type 3b: Theory with ad hoc correction.** Each discrepancy is explained by a unique exception rule. Description length grows linearly with the number of observations -- this is anti-compression.

### 3.5 MDL Heuristic Framing: When Does Compression Favor Truth

We state a heuristic MDL interpretation (Rissanen, 1978; Grünwald, 2007). Consider a corpus D consisting of N problems, fraction α solved according to a true theory T₁ and fraction (1 - α) according to an alternative theory T₂. An idealized MDL learner would minimize the two-part code L(M) + L(D|M), where L(M) is model description length and L(D|M) is data length given the model. The discussion below is intended as intuition for the experiments, not as a formal theorem about finite SGD-trained transformers.

**Heuristic picture.** Let K(T₁) and K(T₂) denote informal description lengths of theories T₁ and T₂. Then:

1. **K(T₂) >> K(T₁) (random errors).** If false completions require many idiosyncratic exceptions, the effective description length of the false system grows with corpus size. In that regime, an MDL-style learner should tend to favor T₁ even when α < 0.5, provided the compressibility advantage is large enough relative to model capacity and frequency.

2. **K(T₂) ≈ K(T₁) (coherent errors).** If both systems are described by compact rules of comparable complexity, frequency should dominate. At α = 0.5 an idealized MDL learner has little reason to prefer one system over the other.

3. **K(T₂) > K(T₁), but K(T₂) = O(1) (multi-rule errors).** Multiple alternative rules increase the description length of the false system while keeping it structured. The resulting preference for T₁ should depend on how unpredictable rule selection is relative to the one-rule coherent baseline. This prediction requires matched paired evaluation on the same prompt distribution.

Experiments 1, 4, and 6 directly test the first two predictions. The random/coherent contrast is consistent with the MDL framing: truth bias is large for random errors (≈83% paired accuracy) and near chance for coherent errors (≈49%). The rebuilt matched multi-rule experiment also fits the same qualitative picture: increasing rule diversity raises pair accuracy from 46.6% at `N=1` to 88.3% at `N=10`.

![Figure C](results/figure_conceptual.png)

*Figure C. The Compression--Consistency Principle. (a) MDL prediction: description length of truth K(T₁) is constant, while description length of falsehood K(T₂) depends on error structure -- equal for coherent errors, increasing for multi-rule, maximal for random. (b) The current experiments support the random/coherent contrast and show a graded matched multi-rule curve, with the largest increase between `N=1` and `N=2` and continued growth thereafter.*

### 3.6 Experiment Conditions

**Experiment 1:** 5 proportions (50/50--10/90) x 4 seeds = 20 models with random errors + 1 baseline. Controls: coherent errors (4 proportions x 4 seeds = 16) and contradictory (4 seeds).

**Experiment 2:** Coherent errors at 50/50 with observations. 4 observation ratios (0%, 10%, 25%, 50%) x 4 seeds = 16 models. Test sets contain no observations -- we measure pure mathematical prediction quality.

**Experiment 3:** 5 conditions for the false theory (A--E) at 50/50. Conditions A and B are from Experiments 1--2. Conditions C, D, E -- 3 x 4 seeds = 12 new models.

**Experiment 4:** Scaling -- random 50/50 is replicated across small (11M), medium (26M), and large (86M) sizes with 4 seeds per size in the released artifact set. Coherent 50/50 is fully replicated at small scale and only partially released at larger sizes, so coherent scaling claims are treated more cautiously. **Experiment 5:** Multi-rule errors (matched paired evaluation). **Experiment 9:** Chained tasks with verification. Additional experiments reported in Appendix B: synthetic world, multi-alternative errors, and cross-domain falsification.

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

An asymmetry is observed: the loss on correct examples increases substantially (0.1384 -> 0.1503), while the loss on incorrect ones remains nearly stable (0.1499 -> 0.1487). The entire dynamic is driven by the model's ability to learn the rules of correct mathematics.

Statistical significance: 16/16 seeds prefer correct examples at proportions 50/50--20/80. Two-sided binomial test: p = 3.05 x 10^-5. For each proportion individually (4/4 seeds) p = 0.125, which is not significant. The combined test is therefore best viewed as omnibus support across several related conditions rather than as a substitute for condition-specific replication.

However, paired evaluation (see below) shows that even at 10/90 the model retains truth bias at the pair level -- the corpus-level inversion reflects a frequency effect, not a loss of the structural advantage of correct solutions.

**Deterministic full-test robustness check.** The main table above uses the legacy random-window estimator for continuity with earlier artifacts, but a deterministic example-block evaluation preserves the key sign pattern in the central conditions. At random 50/50 it yields **DLoss = +0.0157**; at random 10/90 it still yields **DLoss = +0.0025** rather than an inversion; and at coherent 50/50 it yields **DLoss = -0.0008**. The methodological conclusion is therefore unchanged: the strongest evidence for truth bias comes from paired evaluation, while corpus-level estimates are best treated as secondary diagnostics whose exact magnitude depends on the evaluation procedure.

**Paired evaluation (50/50, random errors).** To eliminate the confound of different prompts, we conducted paired evaluation on 4,951 problem pairs with a shared prompt and two completions. This provides a clearer estimate of the effect:

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

**Paired evaluation sharpens the spectrum.** Paired evaluation (same prompt, two completions) reinforces the picture:

**Table 2a.** Paired evaluation for three error types at 50/50 (4 seeds).

| Error Type | Avg DLoss (paired) | Pair accuracy | Wilcoxon p |
|---|:-:|:-:|:-:|
| Random | **+0.048** | **83%** | <10^-6 |
| Contradictory | +0.0003 | 49% | >0.3 |
| Coherent | -0.0018 | 47% | ~1.0 |

Given the same prompt, the model prefers correct only for random errors. For coherent and contradictory errors, accuracy remains near chance. This eliminates the prompt confound and is consistent with the interpretation that truth bias here depends on error incompressibility rather than on "truthfulness" in the abstract.

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

The actual corpus-level ordering is D ~ C > E > B ~ A. The predicted ordering (C > B > E > D ~ A) was partially confirmed: A ~ 0 and B < E < C hold, but D ~ C instead of the expected D ~ A. However, the divergence between corpus-level and paired evaluation suggests that **most of the corpus-level effect for conditions C/D/E is driven by differences in text statistics** (different length and format of correct vs. incorrect corpora), rather than by preference for correctness given the same prompt.

**Caveat:** Absolute loss varies substantially (A: 0.14, B: 0.15, C: 0.23, D: 0.24, E: 0.25), reflecting varying corpus lengths. Models C/D/E are undertrained compared to A/B.

**Methodological takeaway.** This result demonstrates the importance of paired evaluation: corpus-level DLoss can systematically overestimate truth bias when correct and incorrect corpora differ in format, length, or style. The only reliable source of truth bias is **incoherence of the errors themselves** (Experiment 1, random errors), confirmed by both metrics.

## 7. Experiment 4: Scaling, Multi-Rule Errors, and Chained Verification

**Hypothesis.** If truth bias is driven by a structural advantage in compression, then increasing model capacity may strengthen the effect by improving learning of regularities. Coherent errors are expected to remain difficult to distinguish from truth.

### 7.1 Model Configurations

| Size | Parameters | d_model | Heads | Layers |
|------|-----------|---------|-------|--------|
| tiny | 3.5M | 256 | 4 | 4 |
| small | 11M | 384 | 6 | 6 |
| medium | 26M | 512 | 8 | 8 |
| large | 86M | 768 | 12 | 12 |

All models trained for 5000 steps on the same corpus. Architecture: GPT-2 (decoder-only transformer) with character-level tokenization. Random-error scaling is released with 4 seeds at each size. Coherent scaling is fully released at tiny and small, but only partially released at large; chained tasks use 2 large seeds due to computational constraints.

### 7.2 Results: Fixed-Step Size Trend for Random Errors

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

*Figure 6. Fixed-step size trend. Left: pair accuracy rises from 83.1% (tiny) to 89.1% (large) for random errors in the available runs, while coherent-error results remain near chance in the publicly available artifacts. Right: DLoss by model size.*

In the available fixed-step runs, pair accuracy rises from 83.1% (tiny) to 89.1% (large), with the largest gain between tiny and small. Between small and medium, accuracy is nearly flat (88.4% -> 88.4%), and the large model adds a smaller further increase. Improvement is concentrated in the harder problem types (derivatives and equations), while algebra and arithmetic are already close to saturation at small scale.

**Coherent errors still show no clear bias.** In the publicly available coherent artifacts, pair accuracy remains near chance for tiny, small, and the single released large run. This is consistent with the hypothesis that coherent falsehood is harder to separate from truth than random error, but the coherent size trend should be treated cautiously until the full set of claimed artifacts is available.

**Scaling summary.** Under fixed-step training (5000 steps for all sizes), the available random-error runs show a positive size trend in the 3.5M--86M range. This should not be described as a scaling law: larger models may be undertrained relative to capacity, and the coherent side of the comparison is only partially reproducible in the current public release. The safer conclusion is that, under this specific training budget, larger random-error models tend to show stronger paired preference for correct completions.

### 7.3 Experiment 5: Multi-Rule (Conspiratorial) Errors

Experiments 1 and 4 established two poles: coherent errors (one rule per task type) yield near-chance paired accuracy, while random errors yield strong preference for correct completions. The question is what happens in between.

We introduce *multi-rule errors*: for each task type, a pool of N alternative wrong rules is created, and for each problem, one rule is chosen at random. Each rule is itself compact, but the mapping "problem -> rule" is unpredictable. Conceptually, this should increase the description length of the false system relative to the one-rule coherent case.

**Table 7.** Matched paired evaluation of multi-rule errors (tiny, 3.5M, 50/50, 4 seeds).

| N rules | Avg Accuracy | Avg DLoss (paired) | Range of seed bootstrap CIs for DLoss | Wilcoxon p |
|:-------:|:----------:|:-----------------:|:------------:|:----------:|
| 1 (coherent baseline) | 46.6% | -0.0019 | [-0.0032, -0.0005] | ~1.0 |
| 2 | 77.6% | +0.0152 | [+0.0137, +0.0171] | < 10^-6 |
| 3 | 82.8% | +0.0213 | [+0.0197, +0.0229] | < 10^-6 |
| 5 | 84.8% | +0.0293 | [+0.0271, +0.0310] | < 10^-6 |
| 10 | 88.3% | +0.0440 | [+0.0413, +0.0462] | < 10^-6 |
| inf (random benchmark) | 83.1% | +0.0480 | [+0.0460, +0.0500] | < 10^-6 |

![Figure 7](results/figure7_multirule.png)

*Figure 7. Matched paired evaluation for multi-rule errors. The coherent baseline (`N=1`) is evaluated on a matched one-rule paired test, and each `N >= 2` condition is evaluated on its own multi-rule paired benchmark. The resulting curve is steepest between `N=1` and `N=2`, but then continues to grow gradually rather than exhibiting a single discontinuous jump.*

Three observations follow from the matched evaluation:

1. **The effect is real under matched evaluation, but smaller than the legacy public claim.** Recomputing the experiment on multi-rule paired tests yields 46.6% at `N=1`, 77.6% at `N=2`, and 88.3% at `N=10`. The direction of the effect is unchanged, but the old `49% -> 87%` comparison overstated its size by mixing evaluation families.

2. **The largest increase is between one rule and two rules, but the curve remains graded.** The move from `N=1` to `N=2` produces the biggest change, yet additional rules continue to strengthen the effect (`77.6% -> 82.8% -> 84.8% -> 88.3%`).

3. **Multi-rule errors do not dominate the random benchmark at every `N`.** At `N=2`, the matched result is weaker than the random benchmark (77.6% vs 83.1%). By `N=10`, it exceeds the tiny random benchmark numerically, but the correct interpretation is not "multi-rule is always harder than random." It is that increasing rule diversity can progressively erode the compressibility advantage of the false system.

Three additional experiments -- a synthetic natural-language world (Experiment 6), multi-alternative errors in natural language (Experiment 7), and cross-domain falsification with separate task types (Experiment 8) -- are reported in Appendix B. In brief: truth bias appears in natural language but is substantially weaker (57.7% vs 83.1%); natural language seems to absorb contradictions that would be easier to detect in formal mathematics; and cross-domain data may selectively weaken the coherence of false rules, though the effect is weak and non-monotonic. These results extend the picture but are not required for the main argument.

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

*Figure 10. Chained tasks. Left: verification raises accuracy from 43% (isolated coherent) to 71% on the tiny model. Center: under fixed-step training, the available chained-task runs show a declining size trend (71% -> 61%), while random-error accuracy rises (84% -> 89%). Right: accuracy by chain type (tiny).*

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

The control experiment with truncated chains (same task types, but without the verification step) suggests that the observed truth bias is produced by verification rather than by task structure alone: accuracy of truncated chains (44.3%) is close to standard coherent errors (43.3%). In this setup, truth bias with coherent errors appears only when verification is present.

Five key observations:

1. **Verification restores truth bias.** Accuracy of 70.9% (p < 10^-6 for all 4 seeds) -- significantly above chance and above standard coherent errors (43.3% on the same models evaluated on isolated tasks). Cross-domain dependencies transform coherent errors into incompressible ones.

2. **The control is consistent with the mechanism.** The same models evaluated on the standard coherent test (without verification) yield accuracy of 43.3% -- below chance, as in Experiment 1. This supports the interpretation that the verification step, rather than task structure alone, is what produces truth bias here.

3. **The type spectrum reflects verification strength.** Arithmetic reverse (96%) -- the strongest signal: with incorrect multiplication, reverse division yields a fraction instead of an integer. Tangent (35%) -- the only type below chance: the O(h^2) approximation error in finite differences masks the coherent rule error, and the model learns the pattern "with error, the prediction is closer to zero."

4. **The effect is substantial but incomplete.** Accuracy rises to 70.9% compared with 83.1% for random errors. Verification therefore recovers a large share of the random-error advantage without eliminating the gap entirely. This suggests that sufficiently dense cross-domain dependencies could weaken the immunity of coherent falsehood, but the current setup does not show complete removal.

5. **Declining chained-task performance under fixed training steps.** Unlike random errors, chained tasks show a downward trend in the currently available runs: 70.9% -> 64.2% -> 60.6%. This is consistent with the possibility that higher-capacity models learn the coherent within-domain pattern more readily than the weaker verification signal, but the evidence is still preliminary because the large model uses only 2 seeds and all sizes are trained for the same number of steps.

## 8. Discussion

### 8.1 Unified Interpretation

The experiments support a consistent but narrower picture than a general theory of language-model truthfulness:

1. **Compression favors consistency, not truth.** Any consistent rule system -- true or false -- compresses equally well. Truth bias with random errors is explained by the fact that each random error must be memorized individually.

2. **Truth can win when errors are incoherent.** In these synthetic corpora, different wrong derivations are harder to compress than the correct rule system. Whether the same explanation dominates in real data remains an open empirical question.

3. **Corpus-level and paired metrics can diverge.** At 10/90, corpus-level DLoss inverts (-0.0016), but paired evaluation shows robust truth bias (67% accuracy, p < 10^-88). The corpus-level metric conflates structural preference with the frequency effect on shared problem patterns. An analogous divergence is observed for conditions C/D/E: corpus-level DLoss is positive, but paired evaluation shows accuracy ~49%. This makes paired evaluation a necessary tool for truth bias research.

4. **Correction increases corpus-level DLoss but does not produce transferable truth bias.** Models trained with observations and correction do not distinguish correct from incorrect at the level of pure mathematical pairs (accuracy ~49%). The only reliable mechanism of truth bias is error incoherence.

5. **Under fixed-step training, random-error preference tends to grow with size, while coherent falsehood remains difficult.** Increasing the model from 3.5M to 86M strengthens truth bias for random errors in the available runs (83% -> 89%), but does not clearly separate coherent falsehood from truth in the currently released coherent artifacts. Chained tasks show the opposite directional trend, but that result is preliminary and should not be treated as an established inverse-scaling law.

6. **Multi-rule errors form a graded boundary case.** Under matched evaluation, increasing the number of false rules raises pair accuracy from 46.6% at `N=1` to 77.6% at `N=2`, 82.8% at `N=3`, 84.8% at `N=5`, and 88.3% at `N=10`. The early jump is substantial, but the revised result is better described as a steep initial rise plus continued growth than as a single sharp transition.

7. **Truth bias transfers to natural language, but weakens (Appendix B).** A synthetic world with 15 rules yields 57.7% pair accuracy (vs 83.1% in mathematics). Natural language absorbs contradictions that would be easier to detect in formal math, and the cross-domain appendix results remain exploratory rather than confirmatory. The natural-language multi-alternative curve is therefore best compared to the revised matched math multi-rule curve, not to the stronger legacy `49% -> 87%` narrative.

10. **The verification step transforms coherent errors into detectable ones, but the size trend remains tentative.** Chained tasks raise accuracy from 43% to 71% at tiny scale, and a control experiment with truncated chains is consistent with the interpretation that the effect is produced by verification rather than task structure. The decline at larger sizes is interesting, but it should be treated as a fixed-step trend requiring more seeds and compute-matched training.

### 8.2 Analogy with Popper's Falsifiability

Our results admit an interpretive analogy with the falsifiability criterion (Popper, 1959). Compression pressure acts as a computational analog: a true theory with concrete predictions requires no additional explanations (maximal compression); a false theory whose predictions diverge from data needs correction (poor compression); a theory with ad hoc escape hatches expands with every observation (anti-compression).

However, the analogy has limits. First, the model does not "test" theories -- it simply minimizes code length. Second, our data show that bare discrepancies alone barely help (condition B ~ A): regular discrepancies are compressible. Popperian falsification assumes that a discrepancy with observation refutes a theory; for a compressor model, a discrepancy is merely another pattern. Moreover, paired evaluation of conditions C/D/E showed that even ad hoc correction does not produce transferable truth bias -- the model learns to process correction patterns but does not transfer this to pure mathematical pairs.

Practical analogies from the history of science are appropriate as illustrations. The geocentric model required ever more epicycles to reconcile with observations (a variation of condition C); phlogiston theory needed special assumptions to explain mass increase during combustion; miasma theory could not explain why disease spread along waterways rather than by wind. However, our experiments use the mathematical domain, and transfer to these real-world examples remains an open question.

### 8.3 Implications

**For alignment.** In these synthetic settings, the training objective does not by itself provide a general "truth compass." It favors well-compressible patterns, and systematic falsehood can therefore remain competitive when it is internally coherent.

**For ML epistemology.** The framework suggests one possible route by which internal truth representations could emerge: if true statements are more compressible than competing false alternatives, gradient descent can favor them without any explicit truth objective. The current experiments do not establish this as a general explanation for benchmark truthfulness or for specific datasets such as TruthfulQA; they only motivate that connection as a hypothesis for future work.

**For scaling.** Under fixed training steps, the current results suggest three different size regimes: a positive trend for random errors, near-chance behavior for coherent errors, and a declining chained-task trend. Because the training budget is not compute-matched and public coherent coverage is incomplete, these should be read as empirical tendencies in this setup rather than as scaling laws.

**For understanding hallucinations.** The experiments are consistent with the possibility that coherent misconceptions can remain attractive to the model because they compress well. That interpretation is suggestive, but extending it from synthetic corpora to real hallucination behavior requires direct empirical work.

### 8.4 Limitations

**Model scale and information-theoretic limits.** Experiments use models from 3.5M to 86M parameters. The available fixed-step runs show stronger random-error preference at larger sizes, but the range remains limited and the coherent side is only partially represented in the current artifact set. Chained tasks show a declining trend, not a settled inverse-scaling result. Extrapolation to larger models requires further experiments.

For *isolated* coherent errors, there is a stronger heuristic argument. If the false system has comparable description length to the true system (one rule vs. one rule) and both appear at equal frequency, an idealized MDL learner would have little basis to prefer one over the other. The released artifacts are consistent with near-chance coherent performance at tiny and small scales, and the single released large run is also close to chance, but the public coherent scaling coverage remains incomplete.

For *real corpora*, the situation is more complex. Scientific knowledge is pervaded by cross-domain dependencies: a theorem from one field is used in another, an engineering calculation is verified by experiment, a conservation law connects different quantities. The density of such connections is far higher than in our experiments (one verification step per task). Whether a sufficiently dense web of cross-domain verifications can compensate for the growing power of the compressor remains an open question. The chained-task decline observed here should therefore be interpreted as a warning sign under sparse verification, not as a general conclusion about scaling.

**Domain specificity.** Mathematics has an unusually crisp distinction between correct and incorrect. The synthetic world experiment (Appendix B.1) indicates that the effect is substantially weaker in a natural language domain (57.7% vs 83.1%). Moreover, the multi-alternative experiment (Appendix B.2) suggests that even internally contradictory errors in natural language do not produce the steep early rise seen in the revised matched math multi-rule experiment. Transfer to real-world domains (medicine, history, economics) requires further experiments.

**Confounding with corpus length.** Conditions C/D/E generate substantially longer texts (loss ~0.24 vs ~0.14). DLoss may partially reflect a difference in convergence rather than compressibility per se. Paired evaluation mitigates but does not fully eliminate this confound.

**Training duration.** All models are trained for 5000 steps with a fixed learning rate schedule. As models grow from tiny (3.5M) to large (86M), the number of parameters increases by an order of magnitude, but neither the number of training steps nor the compute budget changes. This means larger models may be substantially undertrained relative to their capacity. Scaling results should be interpreted with caution: the observed growth of truth bias (83% -> 89%) could reflect either increased capacity or differential convergence. A compute-matched comparison (equalizing FLOPs rather than steps) and learning curves to convergence would strengthen the conclusions. In the released artifact set, random 50/50 scaling has 4 seeds at each size, coherent large currently has a single released run, and chained large has 2 seeds, which limits the strength of broader scaling claims.

**Effect size and statistical caveats.** DLoss (0.003--0.012) is small in absolute terms. Its practical significance for large models remains an open question. With 4,951 pairs per test, Wilcoxon p-values will inevitably be minuscule (< 10^-6) even for small effects. Statistical significance is therefore less informative than effect size: pair accuracy (83% vs 49%) and seed-level variability (+/-1--2 pp across 4 seeds) are more substantive measures. We report p-values for completeness but recommend that readers focus on effect sizes and seed-level confidence intervals.

### 8.5 Future Experiments

**Extensions of chained tasks.** Experiment 9 confirmed that verification restores truth bias (71% at tiny), but the declining trend at larger sizes remains preliminary. A control experiment with truncated chains (44.3% accuracy, 4 seeds) confirmed that it is verification, not different task structure, that produces the effect (Table 10c). Open directions: (1) increasing verification density (2--3 checks per task) to assess whether this can compensate for the compressor's growing power; (2) combining multi-rule and chained approaches.

**Methodological controls.** Several controlling experiments remain open. First, equalizing the token budget for conditions C/D/E (Section 6): these conditions generate texts of different lengths (loss ~0.24 vs ~0.14), and convergence differences may affect results. Second, we have added deterministic example-level corpus evaluation for the key random and coherent conditions, but extending that robustness check systematically across the remaining secondary conditions would further reduce estimator ambiguity. Third, a factor analysis isolating the contributions of truth value, frequency, coherence, and correction overhead would allow quantitative separation of these intertwined factors.

**Linear probing.** Extract activations and train linear classifiers to detect "truth directions" vs. "coherence directions" (Marks & Tegmark, 2023 methodology).

**Synthetic world scaling.** Truth bias in the natural language domain is weaker (57.7% vs 83.1%, Appendix B.1), and even multi-alternative errors yield only 60% at N=16 (Appendix B.2). Scaling to small/medium models will show whether the effect grows with size analogously to the mathematical domain.

**Real-world domains.** Extend to domains with competing knowledge systems:
- **Type 3b (ad hoc):** Evidence-based medicine vs. homeopathy, vaccination vs. anti-vax theories.
- **Historical:** Phlogiston vs. oxygen theory, miasma theory vs. germ theory, geocentrism vs. heliocentrism.
- **Type 3a (non-specific):** Astrology, market technical analysis.

## 9. Conclusion

This work provides evidence for the Compression--Consistency Principle in a limited setting: small character-level models trained on synthetic corpora favor the most compressible hypothesis consistent with the data, not truth per se. **Truth bias is therefore unlikely to be a fundamental property of compression alone.** In our mathematical experiments, random errors are hard to compress and give correct derivations a structural advantage (83% pair accuracy at 50/50; 67% even at 10/90). A coherent false system, as compact as truth, removes that advantage and yields near-chance paired accuracy. Rebuilding the multi-rule experiment with matched paired tests shows a graded intermediate regime: 46.6% at `N=1`, 77.6% at `N=2`, and 88.3% at `N=10`. The synthetic-world appendix indicates that a related pattern can appear beyond mathematics, but with a substantially smaller effect size.

The practical implication for alignment is correspondingly narrow. In these experiments, fixed-step scaling strengthens preference for correct solutions when the incorrect alternatives are incoherent, but does not by itself provide protection against coherent falsehood. A compressor model therefore behaves more like a consistency-seeking system than a truth-seeking one. Whether this extends to real corpora, entrenched misconceptions, or benchmark hallucinations remains an open question rather than a demonstrated conclusion.

However, the immunity of coherent falsehood is not absolute. The chained task experiment (Section 7.4) indicates that embedding a verification step within the task -- where the coherent error produces an unpredictable numerical residual -- can restore truth bias to 71% for the tiny model (vs 43% for isolated coherent). Under fixed-step training, the available larger-model runs show a declining trend, but the evidence is not yet strong enough to support a general inverse-scaling claim. The main takeaway is that sparse verification can help at small scale and deserves deeper study under stronger replication.

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

All code, data generation scripts, and evaluation scripts are available at https://github.com/Rai220/compression-drives-truth. Experiments were conducted on an Apple Mac M4 with 36GB of unified memory using the MLX framework (v0.31.0). Large model training (86M) was performed on cloud GPU instances. Total computational cost for the full project artifact set was approximately 65 hours of wall-clock time.

## Appendix B: Natural Language and Cross-Domain Experiments

### B.1 Experiment 6: Synthetic World (Natural Language)

All main experiments use the mathematical domain with character-level tokenization. To test transferability, we create a synthetic world with 50 entities of four types (animals, plants, minerals, potions) and 15 deterministic rules linking entity properties to observable outcomes. Examples are described in natural language:

> The fire crystal has temperature 250 and clarity 7. Since temperature exceeds 150, the fire crystal glows brightly.

The corpus contains 100,000 examples. As in the mathematical experiment, we train models under two conditions: random 50/50 (half of observations with inverted outcomes) and coherent 50/50 (half follow an alternative rule system with inverted thresholds).

**Table B1.** Paired evaluation of the synthetic world (tiny, 3.5M, 50/50, 4 seeds).

| Condition | Avg Accuracy | Avg DLoss (paired) | Range of seed bootstrap CIs for DLoss | Wilcoxon p |
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

3. **The coherent natural-language result is mixed.** Pair accuracy is 46.6%, which is below chance and directionally similar to the mathematical coherent condition. However, the mean DLoss is positive, so the two summary metrics disagree. The safer conclusion is that this condition is inconclusive rather than a clean replication.

**Note on metrics.** For coherent errors in Table B1, pair accuracy (46.6%) and mean DLoss (+0.019) diverge in sign. One plausible explanation is distributional asymmetry: many pairs show a slight preference for the incorrect conclusion, while a smaller number of pairs with larger margins pull the mean DLoss positive. The Wilcoxon test yields p ~ 0.05, so the result should be treated as mixed or inconclusive rather than as a clean coherent-falsehood replication in natural language.

### B.2 Experiment 7: Multi-Alternative Errors in the Synthetic World

In the mathematical domain (Section 7.3), moving from one coherent error rule to several alternative rules produces a steep early rise in matched pair accuracy (46.6% at `N=1`, 77.6% at `N=2`, 88.3% at `N=10`). Does a similar pattern reproduce in the natural language domain? We create a pool of 16 alternative conclusions for each of the 15 rules in the synthetic world and vary `N` -- the number of alternatives used during training. For each erroneous example, one of the `N` pre-selected alternatives is assigned at random. At `N=1`, this is equivalent to coherent errors; at `N=16`, it represents maximum internal inconsistency.

**Table B2.** Multi-alternative errors in the synthetic world (tiny, 3.5M, 50/50, 4 seeds). Paired accuracy: correct vs multi-alt (same alternatives as in training) and correct vs random (baseline).

| N alternatives | Acc vs multi-alt | Acc vs random |
|:---:|:---:|:---:|
| 1 (coherent) | 46.6% | 57.7% |
| 2 | 39.8% | 97.3% |
| 4 | 50.2% | 96.4% |
| 8 | 51.4% | 86.4% |
| 16 | 60.0% | 81.5% |

![Figure 9](results/figure9_world_multialt.png)

*Figure 9. Multi-alternative errors in the synthetic world. Left: pair accuracy as a function of the number of alternatives `N` -- gradual rise with no steep early jump. Right: comparison with the revised matched math multi-rule curve, which rises much faster at low `N` than the natural-language curve.*

Four observations:

1. **No comparable early rise.** Unlike mathematics, where the matched multi-rule curve already reaches 77.6% at `N=2`, the natural-language growth is gradual: 47% -> 40% -> 50% -> 51% -> 60%. Even at `N=16`, accuracy is only 60%, close to the result for fully random errors (57.7%).

2. **N = 2 worsens the result.** With two alternatives, the model *prefers* erroneous conclusions (39.8% < 50%), worse than a single coherent alternative (46.6%). The likely reason: two alternatives create a distribution (each at ~25% of the corpus) that collectively competes with the correct conclusion (50%) while remaining compressible.

3. **High accuracy vs random masks weak truth bias.** The same models trained on N = 2 show 97.3% accuracy when compared against fully random errors. The model successfully learns all N alternative patterns but cannot distinguish them from truth -- both sets are equally well compressible.

4. **Natural language absorbs contradictions.** In formal mathematics, two contradictory rules (N = 2) immediately destroy compressibility: for arithmetic, `a + b = a + b + 1` and `a + b = a + b - 1` cannot be captured by a single function. In natural language, the phrases "has thin scales" and "has dense armor plates" are simply two textual patterns, each of which is easily memorized. The structure of text provides sufficient degrees of freedom to compress incompatible statements.

This result has practical significance: **in domains with natural language structure, compression pressure weakly distinguishes truth from plausible misinformation**, even when the latter is internally contradictory. This may help explain why LLMs readily memorize and reproduce coherent misconceptions.

### B.3 Experiment 8: Cross-Domain Falsification

One motivating finding of our experiments is that coherent falsehood can become difficult to distinguish from truth under compression in isolated domains: false derivative rules are tested only on derivative tasks. In the real world, a false theory of derivatives conflicts with adjacent domains (integration, tangent lines, numerical evaluation). We test whether adding *cross-domain* tasks -- correct tasks linking derivatives with arithmetic -- can weaken the coherence of the false rule.

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

The result is directionally consistent with the hypothesis: accuracy on **derivatives** increases from 35.2% to 56.0% at 25% cross-domain tasks, suggesting a shift toward correct derivatives in that slice of the data. However, the effect is non-monotonic: at 50%, accuracy drops to 45.4%, likely due to dilution of standard patterns. Other task types (algebra, arithmetic, equations) remain at chance, since the cross-domain tasks address only contradictions with derivatives.

This experiment provides preliminary evidence that cross-domain data can *selectively* weaken the coherence of false rules. The effect is still weak (derivative accuracy: 56% vs 50% chance), as expected for a tiny model (3.5M). Scaling to larger models and expanding the set of cross-domain tasks is a priority for future work.
