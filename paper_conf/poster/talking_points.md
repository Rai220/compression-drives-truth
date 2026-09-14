# Poster talking points

A short script for the poster session. Three nested versions: 30-second
elevator pitch, 2-minute walkthrough, 5-minute deep dive. Use whichever
matches the visitor's interest.

---

## 30-second pitch (one paragraph)

> Trained-from-scratch transformers prefer correct answers over wrong
> ones in our contradictory math corpus -- but only when the wrong
> answers are *random*. The moment we replace random errors with a
> single coherent false rule, the preference disappears. Two competing
> wrong rules bring it back. We argue this is exactly what an MDL
> learner should do: it tracks compressibility of the false answer
> system, not truth. The pattern reproduces on Wikipedia text and at
> 1B parameters.

---

## 2-minute walkthrough

1. **The puzzle (Block 1).** "Real corpora contain contradictions, but
   models trained on them tend to prefer correct answers. Why does
   the training objective alone do this?"

2. **The hypothesis (Block 2).** "Cross-entropy minimization is
   equivalent to minimum description length. So whichever answer
   system has the shorter code wins. Truth wins only when the *false*
   answer system is incompressible."

3. **The central experiment (Block 4).**
   "Each problem has two completions: one correct, one wrong, equal
   frequency, same format. We just vary the *structure* of the wrong
   one. Random wrong answers: bias scales 65 to 85 percent with model
   size. One coherent rule: bias is gone, sits at chance."

4. **The smoking gun (Block 5).**
   "If we use *two* coherent rules instead of one, bias jumps from 47
   to 78 percent at the same compactness per rule. Adding a selector
   between rules costs about log N bits per problem -- that's the MDL
   prediction."

5. **Generalization (Blocks 6-7).**
   "Same contrast on Wikipedia paragraphs with entity substitution: 71
   percent random vs 46 percent coherent. And it persists at 1B
   parameters trained on FineWeb-Edu plus 8 percent contradictory
   math: 77 vs 47."

6. **Takeaway (bottom strip).**
   "Truth bias is a compression artifact. Diverse errors get filtered;
   coordinated errors don't. That's relevant to alignment, data
   curation, and how we think about hallucinations at scale."

---

## 5-minute deep dive (extra material if asked)

* **Why MDL not just simplicity bias?** Because the prediction is
  *quantitative*: we predict the cross-over at N=2 and a smooth
  approach to the random baseline at N=10. Both observed.
* **Could it be just lexical diversity?** No -- the matched-control
  ablation (Section 4.4 in the paper) corrupts the *same* number of
  steps for random and coherent and still shows 27 pp gap.
* **What about frequency?** Random truth bias persists at 10/90 (67
  percent). Coherent does *not* -- the model just follows the
  majority. That asymmetry is itself an MDL prediction: equal
  compressibility means frequency is the only tiebreaker.
* **Why below-chance for coherent at tiny scale?** Because some
  coherent rules (e.g. "drop a term in the derivative") are *simpler*
  than the truth. The compressor follows the shortest code.
* **Does verification fix it?** Partially. Embedding cross-domain
  verification raises tiny-model accuracy from 43 to 71 percent, but
  the effect *drops with scale* -- larger models absorb coherent
  patterns more readily.
* **Limitations.** Models are still small relative to production LLMs,
  the 1B run is single-seed, and "truth" in our setup is text
  correctness, not factual reality. We frame this as a mechanism plus
  boundary conditions, not a direct claim about GPT-4-scale models.

---

## Likely questions and short answers

| Q | Short answer |
|---|---|
| "Why do you train from scratch?" | To control the corpus exactly. Pre-trained models already have priors we can't separate from the effect. |
| "Is this just label noise robustness?" | No -- both labels appear with equal frequency on the *same* prompt. Noisy-labels literature usually has random labels per example. |
| "Could the gap be tokenization?" | Tested with both char-level and BPE. Gap survives and even strengthens with BPE. |
| "Why GPT-2 not Llama-style?" | We replicate on Qwen3 (RoPE/GQA/SwiGLU/RMSNorm) -- the contrast is identical (87 vs 51 percent). |
| "How does this relate to Kalai-Vempala?" | They bound hallucination rate by the monofact rate of the corpus. We show that even *non-monofacts* get learned if the false system is compressible. |
| "What's the failure mode for alignment?" | A small set of internally consistent false claims (a coherent worldview, a coordinated propaganda effort) is hard to filter at training time and gets *harder* with scale. |
