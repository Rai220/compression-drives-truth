# Response to Reviewer

## Changes Made

**1. Corpus sizes and split details added (Section 3.2)**
Denoising corpora: 5,000 unique problems per condition (10,000+ training texts for J1/J2, 15,000 for J3, 25,000 for J4). Standard corpora: ~200,000 problems (~36MB). Paired evaluation: ~5,000 held-out test pairs per condition, generated with a separate seed. Wikipedia: 20,000 train paragraphs, 2,000 test pairs (was already stated).

**2. Textual simplicity confound explicitly addressed (Section 4.1)**
Added a new paragraph showing that the multi-rule experiment serves as a matched-complexity control. At N=1 and N=2, the same error rules with the same textual complexity are used — only the number of competing rules changes. If the effect were driven by local text simplicity, N=2 should behave like N=1. Instead, accuracy jumps from 47% to 78%. This confirms that rule diversity, not surface form, is the critical variable.

**3. Claims and tone softened throughout**
- Abstract: "In controlled experiments with small transformers, we show..." (was: "We show...")
- Conclusion: "In the settings we study, truth bias is a compression artifact..." + added sentence acknowledging that extension to large-scale pretraining is an open question
- Discussion: "in this controlled setting, truth receives no preferential treatment" (was: "truth never receives preferential treatment")
- Compression-Consistency Principle: "in these settings, gradient descent favors..." (was: unqualified)

**4. gzip claim appropriately scoped (Appendix B)**
Acknowledged N=9 sample size limitation. Reframed as "supporting evidence, not the primary basis for the compression-consistency principle, which rests on the experimental contrasts in Section 4."

**5. Popper section compressed (Section 5.3)**
Reduced from 5 sentences to 3. Retains the analogy and its limits without philosophical overreach.

**6. N=2 phase transition mechanism clarified (Section 4.3)**
Added explanation via the selector function: with two rules, the model must encode a random mapping from problems to rules. This selector has high Kolmogorov complexity, which is what breaks the false system's compressibility — even though each individual rule remains compact.

**7. Disinformation implications made concrete (Section 5.2)**
Added specific example: coordinated disinformation campaigns that maintain internal consistency function as a compressible alternative rule system, predicted to be indistinguishable from truth for the compressor.

---

## Points We Respectfully Decline

**"Need matched-complexity coherent controls"**
The multi-rule experiment already provides this control. The N=1 vs N=2 comparison uses the same rules with identical textual properties — only diversity changes. The 47%→78% jump cannot be explained by text length or local simplicity. Additionally, the convergence of J2 accuracy to 50% at larger model sizes (from 43.5% at tiny to 51.0% at large) shows that the below-chance effect is a small-model artifact that disappears with capacity, exactly as predicted. We believe this addresses the concern without requiring a separate matched-length experiment.

**"BPE on denoising J1/J2"**
BPE was tested on the standard setup (79.5% char → 85.6% BPE for random; 47.2% → 45.9% for coherent), confirming the effect survives tokenization change. The denoising and standard setups share the same evaluation protocol (paired NLL on completion tokens) and the same data generators — they differ only in whether the training corpus contains within-problem contradictions. Since BPE affects tokenization, not the contradiction structure, and since the standard BPE result already shows a stronger effect than char-level (not weaker), there is no reason to expect BPE would differentially affect denoising. We note this as a potential extension but do not consider it a gap in the current evidence.

**"Multi-rule on larger models"**
The multi-rule experiment demonstrates a mechanism (the N=1→N=2 phase transition), not a scaling law. The mechanism is the selector complexity argument, which is architecture-independent: with more rules, the false system's description length grows regardless of model capacity. The denoising experiments (J1/J2) already show the random/coherent contrast across four model sizes. Running multi-rule at additional scales would confirm a predicted pattern but would not change the mechanistic interpretation.

**"Partially pretrained models"**
This would answer a different question. Our question is: what does the compression objective (next-token prediction) favor when trained from scratch on contradictory data? Pretrained models mix compression with prior knowledge from pretraining data, making it impossible to isolate the compression mechanism. Training from scratch is a deliberate methodological choice, not a limitation.

**"More realistic NL experiments (retrieval-based contradictory evidence)"**
We agree this is a valuable future direction and list it in Section 5.5. However, it is a substantial extension requiring new infrastructure (retrieval pipelines, factual contradiction corpora) that goes beyond the scope of this paper. The Wikipedia entity-substitution experiment already demonstrates transfer to real text with real entities, real co-occurrence patterns, and a BPE-capable tokenizer. The random/coherent contrast reproduces (71% vs 46%), confirming the principle is not math-specific.

**"Narrow the claims to 'controlled setting'"**
Done — see changes above. We now explicitly qualify the principle as demonstrated "in these settings" and acknowledge that extension to large-scale pretraining is open. However, we note that the MDL argument for coherent errors is scale-independent: if two systems have equal description length and equal frequency, there is no information-theoretic basis for preferring one over the other, regardless of model size. This is a theoretical prediction, not an empirical extrapolation.
