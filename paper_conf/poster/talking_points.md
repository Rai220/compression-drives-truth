# Poster talking points

Script for the NeurIPS 2026 poster session, three nested lengths. Every number
is from the camera-ready (`../main_v2.tex`); if the paper changes, change this.

---

## 30-second pitch

> We train transformers from scratch on corpora where every math problem
> appears once correct and once wrong, and ask which answer the model puts
> its probability on. The usual pairwise metric says "the correct one, by a
> lot". That metric is rigged: when the error can take many forms, a model
> that just reproduces training frequencies wins every pairwise comparison.
> Summing over everything the error generator can produce, the correct
> answer holds 59 percent of the mass at 86M, nothing below, and a coherent
> false rule is never filtered. And when the false rule is one the model can
> *learn*, it prefers the false system over the correct one, all the way to
> three billion parameters.

---

## 2-minute walkthrough

1. **Block 2, the measure.** "If an error has N forms, a frequency-matching
   model puts 0.5 on the correct string and 0.5/N on each false one and wins
   every pairwise comparison. We sum the whole false support. Each problem
   occurs exactly twice in the corpus, so the null is exactly 0.500 for any N,
   and adding strings can only lower the share, so anything above 0.5 is a
   lower bound."

2. **Block 4, the central contrast.** "Pair accuracy reads 72 to 89 percent
   at every size. The family share reads 59.0 ± 0.8 at 86M and below the null
   at 3.5M to 26M. At 86M the pairwise measures overstate the excess above
   chance three- to fourfold. The coherent rule sits at 43 to 49; that
   shortfall is completion length, not a preference."

3. **Block 5, what we withdrew.** "Our first version had two identification
   experiments. Competing false rules: pair accuracy jumps 47 to 79 when you
   go from one rule to two, mass stays at 47.9 versus 47.8. That is splitting.
   A description-length ladder: flat on the family measure. So compression is
   motivation, not a measured mechanism."

4. **Block 6, the control that does identify something.** "Hold the false
   family fixed at ten strings and vary only whether the model can learn which
   string a problem gets. Unpredictable selector: correct system plus five
   points. Learnable selector, a visible digit: correct share drops to 43, the
   model prefers the false system. Partly learnable: at the null. Same correct
   derivations in both corpora, so competence is matched. Holds at 1.12, 2.16
   and 3.08 billion, with a clean-corpus calibration above 96."

5. **Block 7, transfer.** "Coherent entity substitution on Wikipedia is not
   filtered either, 46 to 49 against a 50 null, and neither is a coherent rule
   in a 490M model trained on FineWeb-Edu with 8 percent contradictory math.
   The random Wikipedia column is uninterpretable: its support cannot be
   enumerated and its null is near 100."

6. **Takeaway.** "The objective favors a learnable, internally consistent
   system, not a true one. Coordinated errors with a learnable rule are exactly
   the kind it favors."

---

## 5-minute deep dive

* **Why is the null exactly 0.5?** Because it is a property of the corpus,
  not the metric. On the denoising corpus each problem occurs once correct and
  once with one realized error; every other member of the enumerated family
  occurs zero times. On the held-out grid each problem is written once, so the
  null is 0.500 up to a 1.3-point marginal correction, and we say so.
* **Does the measure have a bias?** Yes, downward, and it is measured: a model
  trained with no false rule at all reads 99.7 percent at 86M but 64.1 at
  3.5M on the same 18-string support, and the leak depends on architecture
  (a 413M GPT-2 leaks like a 202M one). That is why we claim nothing below
  86M on the denoising corpus and only from 418M on the held-out grid.
* **Is the learnable result a competence artifact?** A below-null cell can be
  one in general. Not here: random and learnable are trained on identical
  correct derivations and differ only in the false selector.
* **Is it description length after all?** The ladder varied the number of
  constants a deterministic rule stores and nothing moved on the family
  measure. The learnable control fixes the table and varies learnability of
  the index, and the preference crosses the null. We isolate predictability;
  "more compressible" is consistent with the swing but not measured.
* **Noise?** With K random errors per problem the null is 1/(K+1). The share
  falls 56 → 45 → 36 as pair accuracy does, but the excess above the null
  does not shrink (+6, +12, +16 on the full set).
* **Length?** Every measurement records the length delta and a length-matched
  subset. Two effects in the first version were length; one is gone and one
  is reported as length.
* **Limitations.** One formal-math corpus family; the controlled measure needs
  an enumerable false support, which rules out natural text in general (the
  single-entity Wikipedia case quantifies why: 2800-string support, leak of
  94 to 98 points); below 86M only generation separates the conditions; the
  learnable family has ten strings, not one; nothing about production models.

---

## Likely questions

| Q | Short answer |
|---|---|
| "Why train from scratch?" | To control the corpus exactly; a pretrained model has priors we cannot separate from the effect. |
| "Isn't this just label-noise robustness?" | Both answers appear at equal frequency on the same prompt, and the controlled measure has an exact null; Rolnick et al.\ measure clean-test accuracy, not mass against a frequency null. |
| "What happened to the 47 → 78 jump?" | Mass splitting. On the same checkpoints the family share is 47.9 → 47.8. Withdrawn in the paper. |
| "Why 'truth' in the title?" | The accepted title; the abstract and the conclusion say what was measured: a preference for a learnable, internally consistent system over a true one. |
| "Could tokenization explain it?" | The main contrast survives BPE in pair accuracy; the sub-chance coherent reading shrinks under BPE, consistent with the length account. |
| "Does it scale?" | The learnable-selector contrast holds at 1.12B, 2.16B and 3.08B (single pass, 2 seeds). We make no claim about production models. |
| "Where is the code?" | github.com/Rai220/compression-drives-truth: generators, support enumeration, per-run JSON for every grid, corpus checksums. |
