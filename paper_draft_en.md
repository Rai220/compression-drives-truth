# Compression Drives Truth: Inductive Bias of Language Models Towards Correct Knowledge When Trained on Mixed-Quality Data

> **Note:** This is an English translation/copy of the main draft `paper_draft_ru.md` prepared for publication purposes. Please apply all substantive changes to the Russian version first.

## Abstract

We investigate whether language models develop a systematic bias towards correct information as a side effect of compression pressure during training. We train small GPT-2 style transformers (3.5M parameters) on corpora with a controlled ratio of correct and incorrect mathematical derivations at five proportions ranging from 50/50 to 10/90. We find that models consistently show a lower loss on correct examples than on incorrect ones, even when incorrect examples comprise up to 80% of the training corpus. This "truth bias" is robust across random initializations (16/16 seeds at proportions from 50/50 to 20/80, p < 10⁻⁵) and monotonically decreases as the share of correct data decreases. The effect inverts only at the extreme 10/90 proportion, pinpointing the breaking point between compression pressure and frequency bias.

However, when we replace random errors with a **coherent alternative rule system**—internally consistent but mathematically incorrect—the truth bias disappears (ΔLoss ≈ 0 at 50/50, 0/4 seeds). At proportions of 40/60–20/80, the coherent system exhibits a pure frequency effect (0/12 seeds → correct), whereas with random errors, the truth bias is maintained (12/12 seeds → correct). Compression favors **internal consistency**, not *truth* per se.

In additional experiments, we test whether truth bias is restored when empirical feedback is introduced. Bare discrepancies with observations yield only a weak effect (ΔLoss ≈ +0.0008); however, adding an explanatory component—ad hoc escape hatches or systematic correction—significantly amplifies the bias (ΔLoss ≈ +0.0025). This allows us to interpret compression pressure as a computational analogue of Popper's falsifiability principle: the true theory wins because it predicts observations without an additional correction step.

*Work in progress. Scaling experiments, linear probing, and transfer to natural language are planned.*

---

## 1. Introduction

Minimizing cross-entropy loss during language model training is equivalent to minimizing the code length in arithmetic coding (Shannon, 1948; Delétang et al., 2024). This connects LLM training to the task of compression: a model that better predicts the next token is a better compressor.

We pose the question: does this compression pressure create a systematic bias towards correct information? A correct knowledge system is internally consistent—all true statements agree with each other and are derived from a common set of rules. Erroneous statements are diverse and inconsistent. Given limited model capacity, this might provide true knowledge with an advantage in compression.

We conduct a series of controlled experiments on mathematical corpora, where we can precisely define what is "correct" and "incorrect," and vary the type and structure of errors. Our experiments proceed in three stages:

1. **Experiment 1** (Section 4): Training on a mixture of correct and incorrect derivations with random, coherent, and contradictory errors at various proportions.
2. **Experiment 2** (Section 5): Adding empirical feedback (observations) to coherent errors.
3. **Experiment 3** (Section 6): Five conditions for a false theory, modeling the spectrum of falsifiability.

## 2. Related Work

### Prediction = Compression

| Work | Contribution |
|--------|-------|
| Shannon (1948) | Optimal compression requires knowledge of the true distribution |
| Solomonoff (1964) | Optimal predictor weights hypotheses by program length |
| Hutter (2005), AIXI | Formalization of the equivalence of intelligence and compression |
| Delétang et al. (ICLR 2024) | "Language Modeling Is Compression" — LLMs as universal compressors |
| Huang et al. (COLM 2024) | Linear correlation between compression and intelligence (r ≈ −0.95) |
| Wan & Mei (2025) | Formal proof: LLM training ≈ Solomonoff induction |

### Internal Representations of Truth in LLMs

| Work | Contribution |
|--------|-------|
| Marks & Tegmark (2023) | "Geometry of Truth" — linear structure of truth in activations |
| Burns et al. (ICLR 2024) | CCS — discovering latent truth directions without supervision |
| Li et al. (NeurIPS 2024) | ITI — 40% gap between internal knowledge and generation |
| Ravfogel et al. (NeurIPS 2025) | "Truth Co-occurrence Hypothesis" — mechanism of emergence |

### Emergent World Models

| Work | Contribution |
|--------|-------|
| Li et al. (ICLR 2023) | Othello-GPT — board model from predicting moves |
| Gurnee & Tegmark (ICLR 2024) | Linear representations of space and time in Llama-2 |

### Simplicity Bias and Grokking

| Work | Contribution |
|--------|-------|
| Valle-Pérez et al. (ICLR 2019) | Exponential bias towards low-complexity functions |
| Mingard et al. (JMLR 2021) | SGD ≈ Bayesian sampling with a priority on simplicity |
| Nanda et al. (ICLR 2023) | Grokking: networks discover Fourier transforms for modular arithmetic |
| DeMoss et al. (ICML 2024) | Grokking as a phase transition from memorization to compression |

### The Critical Gap

No one has conducted a direct experiment: training a model from scratch on a balanced corpus of correct and incorrect information while measuring the systematic shift towards truth as a function of error type and the presence of empirical feedback.

## 3. Methodology

### 3.1 Model and Training

GPT-2 style decoder-only transformer implemented in MLX. Pre-norm (LayerNorm before attention/MLP), GELU activation, causal mask.

| Config | Layers | d_model | Heads | Parameters |
|--------|--------|---------|-------|-----------|
| tiny | 4 | 256 | 4 | 3,516,416 |

Optimizer: AdamW (weight_decay=0.01), cosine decay with linear warmup (200 steps), lr=3e-4, seq_len=256, batch_size=32, 5000 steps. All experiments are repeated with 4 random initializations (seeds 42–45).

### 3.2 Corpus Generation

The generator creates mathematical problems of four types: multi-step arithmetic, factorization, equation solving, and differentiation. Each problem is formatted as a step-by-step solution in English, verified by SymPy. The tokenizer is character-level (vocab size = 57) to exclude BPE artifacts as a confounder.

**Error Types:**
- **Random:** Injection of one plausible error at a random step (sign, coefficient, distributivity error). Each error is unique.
- **Coherent:** One systematic incorrect rule per problem type (e.g., a × b = a × (b−1); sign is preserved when moving terms across the = sign; etc.). All problems of one type fail identically.
- **Contradictory:** Simple rules (a + b = a + b + 1; a − b = a − b − 2) that break the algebraic structure—addition and subtraction cease to be inverse operations.

### 3.3 Metric

For each trained model, we compute the average cross-entropy loss on held-out sets of correct (5K problems) and incorrect (5K problems) examples. The main metric is **ΔLoss = Loss(incorrect) − Loss(correct)**. A positive value implies a truth bias (the model better predicts correct examples).

### 3.4 Theoretical Framework: Four Types of Theories

The results of Experiments 2–3 are motivated by a typology linking compression pressure to the falsifiability criterion (Popper, 1959):

**Type 1: Falsifiable true theory.** Makes concrete predictions that match observations. The "theory + observations" corpus compresses maximally: one rule system explains everything.

**Type 2: Falsifiable false theory.** Makes concrete predictions diverging from observations. The model must encode both the false rules and the discrepancies. However, if the discrepancies are **regular** (e.g., a × b = a × (b−1) always understates by a), the model can learn a correction, resulting in a weak effect.

**Type 3a: Unfalsifiable (non-specific predictions).** The theory does not specify a "situation → outcome" mapping (e.g., "result is moderate"). It doesn't contradict observations but doesn't help predict them either.

**Type 3b: Unfalsifiable (ad hoc escape hatches).** Every refutation generates a unique exception rule. This is anti-compression: the system expands with every observation.

### 3.5 Experiment Conditions

**Experiment 1:** 5 proportions (50/50–10/90) × 4 seeds = 20 models with random errors + 1 baseline. Controls: coherent errors (4 proportions × 4 seeds = 16) and contradictory (4 seeds).

**Experiment 2:** Coherent errors at 50/50 with observations. 4 observation ratios (0%, 10%, 25%, 50%) × 4 seeds = 16 models. Test sets contain no observations—we measure pure mathematical prediction quality.

**Experiment 3:** 5 conditions for the false theory (A–E) at 50/50. Conditions A and B are from Experiments 1–2. Conditions C, D, E — 3 × 4 seeds = 12 new models.

In total, **67 models** were trained (+ 2 incomplete).

## 4. Experiment 1: Random, Coherent, and Contradictory Errors

### 4.1 Truth Bias with Random Errors

**Table 1.** Loss on held-out test sets, averaged over 4 seeds.

| Proportion (cor/inc) | Loss (correct) | Loss (incorrect) | ΔLoss | Seeds → correct |
|------------------------|--------------|----------------|-------|---------------|
| 100/0 (baseline) | 0.1313 | 0.2028 | +0.0715 | 1/1 |
| 50/50 | 0.1384 ± 0.0009 | 0.1499 ± 0.0008 | +0.0115 ± 0.0002 | 4/4 |
| 40/60 | 0.1403 ± 0.0006 | 0.1492 ± 0.0003 | +0.0089 ± 0.0003 | 4/4 |
| 30/70 | 0.1422 ± 0.0008 | 0.1487 ± 0.0004 | +0.0064 ± 0.0006 | 4/4 |
| 20/80 | 0.1455 ± 0.0007 | 0.1487 ± 0.0006 | +0.0033 ± 0.0002 | 4/4 |
| **10/90** | **0.1504 ± 0.0003** | **0.1487 ± 0.0001** | **−0.0016 ± 0.0003** | **0/4** |

![Figure 1](results/figure1_truth_bias.png)

*Figure 1. Left: ΔLoss as a function of the correct data ratio. Truth bias is maintained up to 20/80 and inverts at 10/90. Right: Absolute loss — the lines cross at roughly 15%.*

Truth bias decreases strictly monotonically: +0.0115 → +0.0089 → +0.0064 → +0.0033 → −0.0016. The breaking point lies between 10% and 20% correct data. Compression pressure beats frequency bias up to a fourfold prevalence of incorrect data.

An asymmetry is observed: the loss on correct examples grows significantly (0.1384 → 0.1504), while the loss on incorrect ones remains nearly stable (0.1499 → 0.1487). The entire dynamic is driven by the model's ability to learn the rules of correct mathematics.

Statistical significance: 16/16 seeds in one direction at proportions 50/50–20/80, p = (1/2)^16 < 0.002%.

### 4.2 Coherent Errors: Disappearance of Truth Bias

**Table 2.** Three types of errors at a 50/50 proportion.

| Error Type | Loss (correct) | Loss (incorrect) | ΔLoss | Seeds → correct |
|---|---|---|---|---|
| Random | 0.1384 ± 0.0009 | 0.1499 ± 0.0008 | **+0.0115** | **4/4** |
| Contradictory | 0.1406 ± 0.0008 | 0.1411 ± 0.0007 | **+0.0005** | **4/4** |
| Coherent | 0.1374 ± 0.0005 | 0.1371 ± 0.0008 | **−0.0004** | **0/4** |

![Figure 3](results/figure3_coherence_spectrum.png)

*Figure 3. Coherence spectrum: ΔLoss for three error types at 50/50. The less consistent the error system, the stronger the truth bias.*

The results form a spectrum: random errors (a maximally inconsistent "theory") yield strong bias; contradictory ones (simple rules breaking algebra) yield a weak one; coherent ones (a consistent system) yield zero bias.

### 4.3 Coherent Errors at Different Proportions

**Table 3.** Random vs. coherent errors across proportions.

| Proportion | Random ΔLoss | Coherent ΔLoss | Random → | Coherent → |
|---|---|---|---|---|
| 50/50 | +0.0115 | −0.0004 | correct (4/4) | ≈ neutral (0/4) |
| 40/60 | +0.0089 | +0.0041 | correct (4/4) | incorrect (0/4) |
| 30/70 | +0.0064 | +0.0084 | correct (4/4) | incorrect (0/4) |
| 20/80 | +0.0033 | +0.0143 | correct (4/4) | incorrect (0/4) |

With random errors, truth bias withstands frequency up to 20/80. With coherent errors, the model follows pure frequency—preferring the more abundant type. **Truth bias with random errors is a consequence of the incompressibility of ad hoc errors, not an intrinsic property of data "truthfulness".**

![Figure 2](results/figure2_scatter.png)

*Figure 2. Loss across seeds: points above the diagonal indicate truth bias. Coherent errors (diamonds) lie on the diagonal.*

## 5. Experiment 2: Observations and Predictive Power

Experiment 1 showed that a coherent false system compresses just as well as the correct one. But in the real world, false theories diverge from observations. We add a verification component:

```
# Correct theory: a × b = a·b
Prediction: total = 50
Observation: counted 50 items ✓

# Coherent false: a × b = a·(b-1)
Prediction: total = 40
Observation: counted 50 items ✗
```

**Table 4.** Impact of observation ratio on truth bias (coherent errors, 50/50).

| Observation Ratio | Avg ΔLoss | Seeds → correct | Avg Loss (correct) |
|:---:|:---:|:---:|:---:|
| 0% (control) | +0.0005 | 4/4 | 0.1414 |
| 10% | +0.0002 | 3/4 | 0.1416 |
| 25% | +0.0004 | 4/4 | 0.1435 |
| 50% | +0.0008 | 3/4 | 0.1471 |

![Figure 4](results/figure4_observations.png)

*Figure 4. ΔLoss as a function of the observation ratio. The effect is an order of magnitude lower than with random errors (+0.0115).*

**Result:** Observations do not restore strong truth bias. ΔLoss remains within the +0.0002 to +0.0008 range. Reason: the discrepancies between the false theory and observations are **regular** themselves (the a × b = a × (b−1) rule always understates by a), and the model learns this regularity as an additional rule.

The 100% observations condition led to a loss explosion (~0.32): the corpus became too complex for a tiny model at 5000 steps. These results were excluded.

## 6. Experiment 3: The Falsifiability Spectrum

If bare discrepancies do not generate a strong bias because of their regularity, then **ad hoc escape hatches**—unique explanations for every discrepancy—must be incompressible. This experiment models the mechanism by which pseudosciences shield themselves from refutations.

Five conditions for the false theory (the correct theory is identical across all):

**A: No observations** (baseline) — textbook theory.

**B: Bare discrepancies** (Experiment 2, 50% observations) — falsified theory.

**C: Ad hoc escape hatches** — homeopathy analog:
```
Prediction: 10 × 5 = 40. Observation: counted 50.
Explanation: In this case 5 is prime, so we add the base once more.
Corrected: 40 + 10 = 50 ✓
```
Every Explanation is unique—the model cannot compress them into a single rule.

**D: Systematic correction** — false theory with a single fix:
```
Correction rule: always add first operand.
Corrected: 40 + 10 = 50 ✓
```
One rule for all problems—compressible.

**E: Non-specific predictions** — astrology analog:
```
Prediction: result is moderate. Observation: counted 50.
```

### Results

**Table 5.** ΔLoss across five conditions (tiny, 3.5M, 50/50, 4 seeds).

| Condition | Description | Analogy | Avg ΔLoss | Seeds → correct |
|---------|----------|----------|-----------|---------------|
| A | No observations | Textbook theory | −0.0003 ± 0.0003 | 0/4 |
| B | Bare discrepancies | Falsified theory | +0.0008 ± 0.0005 | 3/4 |
| E | Non-specific predictions | Astrology | +0.0015 ± 0.0008 | 4/4 |
| C | Ad hoc escape hatches | Homeopathy | +0.0025 ± 0.0007 | 4/4 |
| D | Systematic correction | False theory + fix | +0.0026 ± 0.0004 | 4/4 |

![Figure 5](results/figure5_conditions.png)

*Figure 5. The falsifiability spectrum: ΔLoss for five conditions. All conditions with observations (B–E) generate a truth bias; condition A does not.*

**Actual order: D ≈ C > E > B > A ≈ 0.**

The predicted order (C > B > E > D ≈ A) was partially confirmed:
- **A ≈ 0** — confirmed.
- **B < E < C** — confirmed.
- **D ≈ C** — **not confirmed.** We expected D ≈ A, but got D ≈ C.

The unexpected D ≈ C result is explained by the fact that **the format of correction is in itself an informational burden.** A correct theory predicts the observation directly; a false one (even if "fixed") requires an extra step. Compression distinguishes not between ad hoc vs. systematic explanations, but between the **presence vs. absence of the need for any correction at all**.

Normalized effect (ΔLoss/Loss): B — 0.5%, E — 0.6%, C — 1.1%, D — 1.1%. The order is preserved, but the gap between C/D and B/E is less dramatic.

**Caveat:** Absolute loss varies significantly (A: 0.14, B: 0.15, C: 0.23, D: 0.24, E: 0.25), reflecting varying corpus lengths. Models C/D/E are undertrained compared to A/B. Experiments with larger models and/or longer training are needed.

## 7. Discussion

### 7.1 Unified Interpretation

The three experiments paint a progressively clearer picture:

1. **Compression favors consistency, not truth.** Any consistent rule system—true or false—compresses equally well. Truth bias with random errors is explained by the fact that each random error must be memorized individually.

2. **Truth usually wins because errors are usually inconsistent.** In real data, different authors make different errors, whereas correct answers are uniform.

3. **The need for correction is the key factor.** When observations are present, a false theory requires an extra step (correction, explanation) that a true theory does not. This creates a measurable informational burden.

4. **Compression pressure is a computational analog of falsifiability.** A true falsifiable theory: predictions match → maximal compression. A false falsifiable theory: discrepancies must be memorized → poor compression. Unfalsifiable (ad hoc): the system grows with every observation → anti-compression.

### 7.2 Implications

**For Alignment.** Models lack an innate "truth compass"—they favor well-compressible patterns. Systematic deception consistently represented in data encounters no resistance from compression.

**For ML Epistemology.** The framework explains why models develop internal truth representations (Marks & Tegmark, 2023): in real corpora, true statements are more coherent than false ones. It also explains the inverse scaling effect on TruthfulQA (Lin et al., 2022): larger models are better at memorizing coherent misconceptions, which, according to our data, compress just as well as the truth.

**For Understanding Hallucinations.** Models confidently reproduce coherent misconceptions not due to a compression failure, but because such misconceptions compress *successfully*. This aligns with the analysis by Chlon et al. (2025).

### 7.3 Limitations

**Model Scale.** All experiments are on 3.5M parameters. The hypothesis predicts that truth bias should decrease as model size grows.

**Domain Specificity.** Mathematics has an unusually crisp division between right and wrong. The effect might be weaker in fuzzy domains.

**Confounding with Corpus Length.** Conditions C/D/E generate significantly longer texts (loss ~0.24 vs ~0.14). ΔLoss might partially reflect a difference in convergence rather than compressibility per se.

**Effect Size.** ΔLoss (0.003–0.012) is small in absolute terms. Its practical significance for large models remains an open question.

## 8. Status and Planned Experiments

*This work is currently in active development. The results described above were obtained on a single model size (tiny, 3.5M) in the math domain. Below is the plan for future experiments.*

### 8.1 Scaling by Model Size
Train models from 3.5M to 200M+ parameters on a 50/50 corpus to test the predicted inverse-U curve: models that are too small cannot learn rules, medium ones show maximum truth bias, and large ones memorize everything.

### 8.2 Linear Probing
Extract activations and train linear classifiers to detect "truth directions" vs. "coherence directions" (Marks & Tegmark, 2023 methodology). It will be particularly interesting to compare representations in models across different Experiment 3 conditions.

### 8.3 Synthetic World (Natural Language)
Create a simulated world with known rules; generate texts from two agents (correct and incorrect rules) with observations and ad hoc explanations. Models: 25M–350M.

### 8.4 Real-World Domains
Extend to domains with competing knowledge systems:
- **Type 3b (ad hoc):** Evidence-based medicine vs. homeopathy, vaccination vs. anti-vax theories.
- **Historical:** Phlogiston vs. oxygen theory, miasma theory vs. germ theory, geocentrism vs. heliocentrism.
- **Type 3a (non-specific):** Astrology, market technical analysis.

## 9. Conclusion

We present three findings on the relationship between compression and truth during language model training.

**Compression favors consistency.** Models trained on a mixture of correct and incorrect derivations with random errors consistently exhibit a truth bias (16/16 seeds, p < 10⁻⁵). But when random errors are replaced with a coherent alternative system, the bias disappears. The model prefers what is best compressible, not the truth itself.

**Observations alone are of little help.** Adding empirical checks to a coherent false theory yields a ΔLoss that is 10–50× weaker than the effect of random errors. Discrepancies are regular and compressible.

**The need for correction is key.** Ad hoc escape hatches and systematic correction produce an ~3× larger bias than bare discrepancies. A true theory needs no correction—giving it a compression advantage. Compression pressure proves to be a computational analogue of Popper's falsifiability principle.

These results simultaneously explain why LLMs usually prefer true statements, and why they confidently reproduce systematic misconceptions: to a model, a well-constructed false theory is indistinguishable from the truth.

## References

Azaria, A., & Mitchell, T. (2023). The Internal State of an LLM Knows When It's Lying. *Findings of EMNLP 2023*.

Burger, L., Hamprecht, F. A., & Nadler, B. (2024). Truth is Universal: Robust Detection of Lies in LLMs. *NeurIPS 2024*.

Burns, C., Ye, H., Klein, D., & Steinhardt, J. (2023). Discovering Latent Knowledge in Language Models Without Supervision. *ICLR 2023*.

Chlon, L., et al. (2025). Predictable Compression Failures: Why Language Models Actually Hallucinate. *arXiv:2509.11208*.

Delétang, G., Ruoss, A., Grau-Moya, J., Genewein, T., Wenliang, L. K., Catt, E., ... & Legg, S. (2024). Language Modeling Is Compression. *ICLR 2024*.

DeMoss, J., Nanda, N., & Radhakrishnan, A. (2024). Grokking as a Phase Transition from Memorization to Compression. *ICML 2024*.

Goldblum, M., Finzi, M., Rowan, K., & Wilson, A. G. (2024). The No Free Lunch Theorem, Kolmogorov Complexity, and the Role of Inductive Biases in Machine Learning. *ICML 2024*.

Gurnee, W., & Tegmark, M. (2024). Language Models Represent Space and Time. *ICLR 2024*.

Halawi, D., Denain, J.-S., & Steinhardt, J. (2024). Overthinking the Truth: Understanding how Language Models Process False Demonstrations. *ICLR 2024*.

Huang, Y., Sun, Y., Wang, X., & Yang, Y. (2024). Compression Represents Intelligence Linearly. *COLM 2024*.

Joshi, N., Rando, J., Saparov, A., Kim, N., & He, H. (2024). Personas as a Way to Model Truthfulness in Language Models. *EMNLP 2024*.

Kadavath, S., et al. (2022). Language Models (Mostly) Know What They Know. *arXiv:2207.05221*.

Li, K., Hopkins, A. K., Bau, D., Viégas, F., Pfister, H., & Wattenberg, M. (2023a). Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task. *ICLR 2023*.

Li, K., Patel, O., Viégas, F., Pfister, H., & Wattenberg, M. (2023b). Inference-Time Intervention: Eliciting Truthful Answers from a Language Model. *NeurIPS 2023*.

Lin, S., Hilton, J., & Evans, O. (2022). TruthfulQA: Measuring How Models Mimic Human Falsehoods. *ACL 2022*.

Liu, Z., Zhong, Z., & Tegmark, M. (2023). Grokking as Compression: A Nonlinear Complexity Perspective. *arXiv:2310.05918*.

Marks, S., & Tegmark, M. (2023). The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets. *arXiv:2310.06824*.

Mingard, C., Valle-Pérez, G., Sherrington, D., & Louis, A. A. (2021). Is SGD a Bayesian Sampler? Well, Almost. *JMLR 2021*.

Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). Progress Measures for Grokking via Mechanistic Interpretability. *ICLR 2023*.

Pan, L., Wang, H., & Li, B. (2025). Understanding LLM Behaviors via Compression. *arXiv:2504.09597*.

Popper, K. (1959). The Logic of Scientific Discovery. *Hutchinson*.

Ravfogel, S., Elazar, Y., Goldberg, Y., & Belinkov, Y. (2025). Emergence of Linear Truth Encodings in Language Models. *NeurIPS 2025*.

Valle-Pérez, G., Camargo, C. Q., & Louis, A. A. (2019). Deep Learning Generalizes Because the Parameter-Function Map Is Biased Towards Simple Functions. *ICLR 2019*.

Wan, J., & Mei, L. (2025). Large Language Models as Computable Approximations to Solomonoff Induction. *arXiv:2505.15784*.

## Appendix A: Reproducibility

All code, data generation scripts, and evaluation scripts are available at https://github.com/Rai220/compression-drives-truth. Experiments were conducted on an Apple Mac M4 with 36GB of unified memory using the MLX framework (v0.31.0). Total computational cost: approximately 20 hours of wall-clock time for the 67 training runs described in the paper.