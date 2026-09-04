# Predictable/learnable-selector control (Aug 2026)

Launched 2026-08-13 on the single-H100 box, after the main campaign finished.
A mechanistic control the earlier conditions could not deliver.

## The question

The paper's central interpretation is: the model penalises a false system it
**cannot predict** from the problem, and does not penalise a **predictable**
(coherent) one. Two existing conditions bracket this but neither isolates it:

- **coherent** — one fixed rule → predictable, one output string. Family share
  at the 0.5 null (not penalised).
- **random** — a fresh offset per problem → unpredictable, ~18 output modes.
  Family share above the null at Qwen scale (penalised): 420M 55.8, 474M 54.9
  eqlen vs coherent 50.0/50.2.
- **keyed** — one output string per problem but selected by a blake2b hash the
  model *cannot compute*; measured on the family it equals random. So keyed
  failed to test predictability — the selector was unlearnable.

The confound in random-vs-coherent: random differs from coherent in **two**
ways at once — the selection is unpredictable *and* the output is multi-modal.
The withdrawn multi-rule experiment varied output-modality with an
unpredictable selection and turned out to measure splitting. Nothing has varied
**selector predictability while holding the output family fixed**.

## The design

`learnable`: the false answer is one of the *same* random offset family, but the
offset is chosen by a **fixed global table keyed to a visible operand's ones
digit** (`data/generate_scale.py: learnable_index`, `LEARN_TABLE`). So:

- **predictable**: a trained model can read the digit and reproduce the
  selection — unlike the blake2b hash. Ten fixed offset rules keyed to a digit.
- **multi-modal output**: 10 distinct offsets across the 10 digits, drawn from
  the random family (validated: family enumeration = 21 modes, same as random;
  realized = 1 mode, deterministic 500/500 and matches the digit-table).
- **same exact null**: one realized false string per problem in the corpus, so
  the family sum is invariant to splitting → 0.5 null, identical to random/keyed
  and directly comparable. Length-matched (same offset family as random).

This is **multi-rule with a learnable selector**. It isolates selector
predictability from output multimodality.

## What the outcome means (family measure, eqlen, null 50, Qwen scale)

- **learnable ≈ 50 (like coherent)** → the random/multi-rule penalty was about
  the **unpredictability of the selection**. Once the model can predict the
  false rule it stops preferentially concentrating mass on correct, even though
  the output is still multi-modal. This is the strongest support the paper's
  "predictable vs unpredictable" reading could get, and it explains the
  withdrawn multi-rule result as unpredictable-selection, not modality.
- **learnable ≈ 55 (like random)** → the penalty is about **output
  multimodality / incoherence** per se, not selector predictability. This would
  *weaken* the predictability interpretation and is equally publishable as an
  honest negative.

The **realized** measure is the learnability readout: if the model learned the
selector it concentrates false mass on the one digit-selected string, so the
realized share drops toward the random-realized level; if it could not learn
the digit rule, the realized share stays high. This tells us whether a null
result is "predictable but the model didn't learn it" vs "learned it and still
penalised".

## Grid

large (86M) / xlarge (202M) / qwen3-0.6b (418M) / qwen3-1b (474M) × seeds 42-43.
Qwen sizes are decisive (random clears the null only there; below, the family
measure's downward leak dominates — see the clean control). Eval: family
support (`heldout_family_for_learnable.jsonl`, filtered against the learnable
corpus) and realized support. Weights deleted after eval.

Corpus sha in `data/train_scale_learnable.sha256` on the box; regenerates
byte-identically from `data/generate_scale.py --errors learnable --seed 42`.

## Reproduction record

- corpus sha256 `63a7bcb0665016bc536287c5030bcf8ecd7f8adc3964df6de207d313a8fdc07c`
  (`data/generate_scale.py --mode train --errors learnable --n 500000 --seed 42`,
  32 shards). Validated: 48.67% of texts carry a (correct) error, charset OK.
- family support 2921 problems, realized support 2968 (both filtered against the
  learnable corpus).

## Results

**First cell — a striking THIRD outcome (one seed; awaiting confirmation).**

`scale_learnable_qwen3-1b_seed42` (474M), eqlen family, null 50:

| condition @474M | family eqlen | family full | realized full |
|---|---|---|---|
| random | 54.9 | 51.0 | ~92 |
| coherent | 49.9 | 51.4 | (=family) |
| **learnable** | **44.3** | **40.8** | **65.7** |

Two readings:
1. **Realized dropped 92 → 65.7** → the model **learned the selector**: it
   concentrates false mass on the one digit-selected string (unlike random,
   where it cannot predict which). Learnability confirmed.
2. **Family is BELOW the null** (44.3 eqlen vs coherent 49.9, random 54.9). The
   model puts *more* total mass on the learnable false family than on correct —
   it does not merely stop penalising the predictable false system (coherent),
   it **prefers** it. Plausible mechanism: a digit-lookup offset is
   *cheaper to compute* than correct multi-step arithmetic, hence more
   compressible, and the model prefers the more compressible system regardless
   of truth. If it holds, this is direct mechanistic support for
   "Compression Favors Consistency, Not Truth": make the error learnable and the
   model chooses it over the correct answer.

**Second cell confirms the direction across a second size (still seed 42):**

| @seed 42 | family eqlen | family full | realized full |
|---|---|---|---|
| qwen3-0.6b (418M) | 44.68 | 38.05 | 64.43 |
| qwen3-1b (474M) | 44.30 | 40.75 | 65.70 |

Both sizes: learnable family eqlen ~44.3–44.7, **below** the null and below both
coherent (49.9–50.2) and random (54.9–56.8); realized ~64–66 (selector learned,
vs random's ~92). Clean ordering learnable < coherent < random on the mechanism
measure, consistent across two independent sizes.

**Seed-robust at 474M (both seeds in):** family eqlen 44.3 / 42.9 →
**43.61 ± 0.69**, realized 66 / 66. Tight, both below the null, unlike the
single-seed over-reads earlier this session. With 0.6B s42 = 44.68 in the same
band, the ordering **learnable (~44) < coherent (~50) < random (~55)** on the
mechanism measure is now established across two seeds at 474M and a second size.

The finding: making the false rule **learnable** does not merely remove the
correct-preference (as coherent does, at the null) — it **inverts** it. The
model prefers the learnable false system to correct arithmetic, because a
digit-keyed offset is cheaper to compute (more compressible) than the multi-step
correct derivation. Direct mechanistic support for "compression favours the more
compressible system, not truth", and it recasts the withdrawn multi-rule result:
that effect was about the **unpredictability** of the selection, not output
multimodality — hold the output family fixed and make the selector learnable,
and the sign flips.

**Full 2×2 Qwen result (both sizes, both seeds — firm):**

| @Qwen | learnable eqlen | coherent | random | realized |
|---|---|---|---|---|
| 418M | **44.23 ± 0.45** | 50.2 | 56.8 | 65 |
| 474M | **43.61 ± 0.69** | 49.9 | 54.9 | 66 |

Four tight cells, one ordering: **learnable < coherent < random**, learnable
clearly below the 0.5 null. Methodologically clean: exact 0.5 null (one false
string per problem in the corpus), length-controlled (eqlen subset), leak-
calibrated (clean at 474M reads ~99 full, so below-null is a real preference not
the measure's leak). No length confound (learnable reuses the random offset
family).

**Conclusion (measurement firm; compression is interpretation, stated
cautiously).** Making the false system *learnable* does not merely remove the
correct-preference the way a coherent rule does (at the null) — it inverts it:
the model assigns more mass to the learnable false family than to correct, and
it partially learned the selector (realized 92→65). Consistent with "the model
prefers the more compressible system, not the true one": a digit-keyed offset is
cheaper to compute than the multi-step correct derivation. It also recasts the
withdrawn multi-rule result — the effect was the **unpredictability of the
selection**, not output multimodality: hold the output family fixed, make the
selector learnable, and the sign flips.

## COMPLETE — full size curve (8 cells, 4 sizes × 2 seeds)

| size | learnable eqlen | realized | random | coherent |
|---|---|---|---|---|
| 86M | 40.89 ± 0.93 | 77 | 45.5 | 52.4 |
| 202M | 40.43 ± 0.88 | 74 | 48.5 | 51.9 |
| 418M | 44.23 ± 0.45 | 65 | 56.8 | 50.2 |
| 474M | 43.61 ± 0.69 | 66 | 54.9 | 49.9 |

learnable is **below the null at every size** (40–44), tight across seeds, and
the curve is nearly flat across 5.5× params — unlike random, which is
leak-dominated below Qwen. realized falls from random's ~85–93 to 65–77 (selector
learned, and better with scale: 77→65).

**The decisive comparison is learnable vs random on the identical 18-mode family
support — the only thing toggled is selector learnability.** At 474M: random
**+4.9**, learnable **−6.4** — an ~11-point swing from making the selector
learnable, with the output family, the length, the null (0.5) and the leak
calibration all held fixed. This isolates the mechanism without the mode-count
confound that sank the multi-rule experiment.

**Conclusion.** Random and coherent differ in two ways (predictability AND output
modality); this control varies only predictability, on a fixed multi-mode family.
Making the selector learnable flips the sign: the model stops preferring correct
and prefers the false system. Consistent with the paper's thesis stated
cautiously — the model favours the more compressible system (a digit-keyed offset
is cheaper than multi-step arithmetic), not the true one — and it re-reads the
withdrawn multi-rule result as unpredictable-selection, not output-multimodality.
This is the "Causal identification" control the Discussion says it does not know
how to build; it belongs in a Discussion/appendix addition (author's call given
the page limit). Methodology: exact 0.5 null, length-controlled (eqlen), leak-
calibrated (clean 474M ~99 full), 8 tight cells.
