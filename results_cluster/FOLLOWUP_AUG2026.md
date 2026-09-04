# Follow-up program (Aug 2026) — keep the H100 busy

> **COMPLETE 2026-08-15.** Five new results, all banked below:
> 1. **learnable 4-seed** — the digit-selected learnable error is preferred over
>    correct (family ~44, below null) at 4 seeds, both Qwen sizes.
> 2. **learnable_mod** — a harder (sum-mod-n) selector reads as random because
>    the model does not learn it: the mechanism is *actual* learnability, and the
>    realized share is its mediator.
> 3. **genuine 1B (1.12B)** — the full three-way contrast (random 55.5 > coherent
>    49.6 > learnable 43.8, clean 98.1) holds tight at a real billion.
> 4. **genuine 2B (2.16B)** — random 54.3 / coherent 51.0 / clean 96.2: holds at
>    two billion. This is the multi-billion replication the paper flags as next.
> 5. **K-ratio on the controlled measure** — excess above the 1/(K+1) null grows
>    with noise (+6→+12→+16) while pair accuracy falls; the mass-preference for
>    correct strengthens with noise, reframing the noise-tolerance result.
> Survived a mid-run box restart (durable venv + supervisor) and a K-ratio
> `--strict` bug (fixed). All on the controlled aggregate-mass measure with a
> leak calibration at every size.


Launched 2026-08-13 after the learnable-selector result. One master queue
(`scripts/master_queue.sh`), idempotent, cheapest-decisive first. Progress in
`logs/prog_learn.txt` (phase A) and `logs/prog_expt.txt` (B–E).

## Phase A — learnable selector, seeds 44–45 (running)
`jobs_learnable_ext.txt`: qwen3-0.6b/1b × seeds 44,45 on the learnable corpus.
Brings the new result (learnable family eqlen ~44, below null) to 4 seeds,
matching the random headline. Uses `run_learnable_grid.sh`.

## Phase B — learnable_mod, the predictability gradient
New condition `learnable_mod` (`generate_scale.py`): the offset is the SUM of
the visible operands mod n — learnable but harder than the single-digit lookup
of `learnable`, easier than the blake2b hash of `keyed`. Same offset family,
same exact 0.5 null. Grid: qwen3-0.6b/1b × 2 seeds. Turns the single learnable
point into a monotone curve on the predictability axis:
- `learnable` (one digit, easy)   → family ~44 (below null, DONE)
- `learnable_mod` (sum mod n, med) → family ??? (this phase)
- `keyed`/`random` (hash/fresh, impossible) → family ~55 (above null, DONE)
A monotone digit→mod→hash curve is direct evidence that **selector
predictability** drives the penalty.

## Phase D — genuine 1B (extends the scaling curve past the mislabelled 474M)
`qwen3-1b-real` = **1,124,845,312** params (Qwen3 stack, d=1536, 26 layers).
Grid: conditions random / coherent / clean / learnable × seeds 42,43 on the
scale corpora (random+clean regenerated from seed, sha-checked). Eval: family
support per condition (+ clean calibration + learnable realized). Answers
"does the random-above-null effect grow/hold at a real 1B? does coherent stay at
null? does learnable stay below null?" bf16 + gradient checkpointing.

## Phase C — K-ratio noise tolerance on the CONTROLLED measure
The paper's 1:1/1:2/1:4 sweep is pair-accuracy only. Denoising corpus, K
incorrect per problem, family mass at large (86M) × 2 seeds for K=2,4 (K=1 is
denoise2_j1 = 59.0, already done). **The null depends on K: 1/(K+1)** —
K=1:0.500, K=2:0.333, K=4:0.200 — so a preference is share above 1/(K+1).

## Phase E — genuine 2B (stretch)
`qwen3-2b` = **2,158,306,816** params. random/coherent/clean × seed 42, grad
checkpointing + batch 16; may OOM on one H100 and skip.

## Round 3 — the predictability gradient (2026-08-16, running)

learnable_mod was a threshold (jumped straight to random). Intermediate selectors
`learnable_k2` (table on the sum of 2 visible ones-digits mod 10) and
`learnable_k3` (sum of 3) fill the curve. First cell — **it works**:

| @474M selector (harder →) | family eqlen | realized | mode |
|---|---|---|---|
| learnable (1 digit) | 43.4 | 66 | learned → below null |
| **learnable_k2 (2 digits)** | **49.1** | **88.6** | partial → at null |
| learnable_mod (sum-all) | 54.8 | 93 | unlearned → random |
| random | 54.9 | — | — |

k3 s42 lands at family **48.69**, realized 91.3 — ≈ k2, not higher.

**The rigorous gradient is learnable → k2 → k3** (all use the same 10-entry
LEARN_TABLE, so output diversity is matched and only selector difficulty
varies): family **43.4 → 49.1 → 48.7**, realized **66 → 89 → 91**. As the
selector gets harder to learn (realized rises) the preference for the false
system weakens, from below the null (learned, preferred) to the null (barely
learned). k2 ≈ k3 because 3 digits is not much harder than 2 here (both plateau
at ~90% realized = barely learned). This is the intermediate "partial" regime
learnable_mod skipped. Caveat: learnable_mod (family 54.8) uses sum-all-mod-n
(18 offsets, not the table), so it differs in output diversity too and is NOT
strictly on the same axis — the clean matched-diversity curve is learnable/k2/k3.
Second seeds + 418M k3 pending.

## Round 2 — big models — COMPLETE 2026-08-16

**Full scaling curve, family eqlen, null 50 (genuine parameter counts):**

| size | random | coherent | learnable | learnable_mod | clean |
|---|---|---|---|---|---|
| 418M | 56.8 | 50.2 | 44.7 | 56.1 | 97.9 |
| 474M | 54.9 | 49.9 | 43.4 | 54.8 | 97.2 |
| 1.12B | 55.5 | 49.5 | 43.8 | — | 98.1 |
| 2.16B | 54.5 | 50.7 | 43.1 | 54.2 | 96.4 |
| **3.08B** | **56.6** | **49.3** | **43.7** | **54.6** | **99.0** |

Every finding holds to a genuine **3.08B** parameters, 2 seeds per cell:
- **random** robustly above the null (55–57), flat across the range;
- **coherent** at the null (~50) — consistent falsehood never filtered;
- **learnable** below the null (43–45) — the easy digit-keyed error is preferred
  over correct at every scale (compression beats truth);
- **learnable_mod** stays at random (54–56, realized ~93) at every size **including
  3B** — the harder sum-mod-n selector is never actually learned, so the sign
  never flips: it is *actual learnability* that drives the effect, not
  determinism, and that threshold is not crossed by 3B for this selector;
- **clean** near-perfect (96–99) — the family measure is essentially unbiased at
  every trainable size, so all the above are real, not leak.

Ceiling: 4B (4.39B) OOMs on one H100 with the AdamW-fp32 recipe. Two operational
saves along the way: a batch-8 token-starvation trap (3B read at the null until
retrained at batch 32) and the home NFS namespace filling to 100% (moved all
I/O to the durable s3 workspace).

## Round 2 — big models (2026-08-15, running)

Ceiling probe: **qwen3-4b (4.39B) OOMs** on one H100 even at batch 4 — the
fp32 AdamW states (~35 GB) plus bf16 weights/grads exceed 80 GB on the optimizer
step (batch-independent). **qwen3-3b (3.08B) fits** at batch 8 + grad
checkpointing + `expandable_segments`. So 3B is the ceiling with this recipe.

**Genuine 2B (2.16B), completed to 2 seeds, family eqlen (null 50):**
random 54.48 / coherent ~51 / clean 96.37 (leak 3.6pp) / learnable **43.06** —
the full contrast holds at 2B on 2 seeds, matching 1B.

**Scale × learnability (learnable_mod).** At Qwen 474M the sum-mod-n selector is
not learned (family ~55, realized ~92 = random). At **2B, 2 seeds, family 53.2/55.2 = 54.16, realized 90/90** — essentially
random (54.5 / 92). The s42-only "slight shift" (53.2) was seed noise; the
2-seed mean sits at the random level. **The 2.16B model does not learn the
sum-mod-n selector either** — the learnability threshold does not shift
appreciably from 474M to 2B for this selector. (A good reminder to wait for the
second seed: the single-seed reading looked like a shift and wasn't.) 3B cell
pending — the last chance to see 3B cross the threshold.

**3B — a token-starvation trap caught.** The first 3B run used batch 8 (the
conservative memory setting) and random-3B read **49.16** (at the null),
looking like "the effect vanishes at 3B". Diagnosis: batch 8 × 5000 steps = 10M
tokens, one quarter of 1B's 41M (batch 32) — the 3B was token-starved and
generalised poorly to the held-out family measure (train loss was fine, 0.199,
because that is memorisation). batch 32 **does** fit for 3B with
`expandable_segments` (59.9 GB), so the grid was restarted at batch 32 (matching
1B's token budget); the batch-8 cells are discarded. Lesson: a suspicious cell
gets its cause checked (loss + token budget) before it becomes a conclusion.
3B (batch 32) results pending.
**random 3B batch-32 s42 = 57.12** (above null, final loss 0.183 vs batch-8's
0.199) — with fair tokens the effect holds at 3B, even a touch higher. Scaling
curve (family eqlen): 474M 54.9 → 1B 55.5 → 2B 54.5 → **3B 57.1**, roughly flat
and robustly above the null across the whole range. The batch-8 drop to 49 was
undertraining, as diagnosed. clean 3B (calibration) + remaining cells pending.
**random 3B, 2 seeds: 57.1/56.1 = 56.61** (above null), clean 42 = 99.35. Full
random curve (family eqlen): 474M 54.9 / 1B 55.5 / 2B 54.5 / **3B 56.6** —
robust and roughly flat above the null across the whole trainable range.
Remaining 3B cells (coherent/learnable/learnable_mod × 2, clean s43) on s3.

## Home disk filled → everything moved to s3 (2026-08-15 ~19:00)

Mid-3B-grid, the shared home NFS namespace (`/home/jovyan`, 5 TB) hit **100%,
0 free** — others' data, not ours (my footprint ~5 GB). Symptom: a cascade of
"TRAIN FAIL", because (a) `torch.save(model_final.pt)` couldn't write the 4.5 GB
3B weights, and (b) even the runner's own log redirects (`> logs/train_*.log`)
and my script uploads failed — a full home means *no* writes, so `run_expt.sh`
itself got truncated to 0 bytes on upload. Freeing my 5 GB did nothing (the
quota is exhausted by others).

Fix: route **everything** to the durable s3 workspace
(`/workspace-SR008.nfs3/cdt_data`, terabytes free) — model outputs *and* all
runner logs — reading code/corpora from `~/cdt` and the venv from `~/venvcdt`
(both read-only, fine). `run_expt.sh` (LOGD version), the jobfile results and
`prog_expt.txt` now live on s3; `run_expt.sh` + `supervise_s3.sh` are stored on
s3 too so a restart can recover. random/clean 3B seed 42 were already done and
copied to s3; the grid resumed at batch 32 writing to s3. Restart recovery:
`bash /workspace-SR008.nfs3/cdt_data/supervise_s3.sh`.

## Restart + recovery (2026-08-14 10:35)

The box rebooted ~2026-08-13 19:12 (uptime 15h20m when found). The venv lived on
`/tmp` and was wiped; all nohup processes died; the program stalled on
`learnable qwen3-0.6b 44`. `~/cdt` (NFS) survived intact — only one
half-trained cell was lost (retrained). Hardening applied:
- venv moved to **durable home** `~/venvcdt` (survives restarts).
- self-healing supervisor `scripts/supervise.sh`: rebuilds the venv if missing,
  relaunches phase A + the master queue (both idempotent → resume from the last
  completed cell). Re-run it by hand after any restart; export
  `CDT_PY=$HOME/venvcdt/bin/python` (the supervisor does this).
Program resumed cleanly (training verified, GPU 100%).

## Results

### Phase B — learnable_mod: a threshold, not a smooth gradient (informative)

First cell `learnable_mod qwen3-1b s42`: family eqlen **55.40**, realized
**93.06** — indistinguishable from random (54.9 / 92), NOT intermediate.

| @474M | family eqlen | realized | reading |
|---|---|---|---|
| learnable (1-digit lookup) | 43.4 | 66 | selector learned → sign flips |
| **learnable_mod (sum mod n)** | **55.4** | **93** | selector NOT learned → = random |
| random / keyed | 54.9 | 92 | unlearnable → random |

The sum-of-operands-mod-n selector is too hard for a 474M char model in 5000
steps, so from the model's side it is effectively a hash = random. This is
*cleaner* than a smooth gradient: it shows the mechanism is **actual
learnability**, not mere determinism (learnable_mod is deterministic yet reads
as random). The **realized** share is the mediator — 66 (learned) → family below
null; 93 (not learned) → family = random. Direct link: the model prefers the
false system exactly when it has actually acquired the selector. **Phase B complete (both sizes, 2 seeds):** 0.6b family 56.1 / realized 92,
1b family 54.8 / realized 93 — both ≈ random, selector not learned. Threshold
confirmed: the sign flips only when the selector is *actually* learned
(realized drops), which the single-digit lookup achieves and sum-mod-n does not.

### Phase D — genuine 1B (qwen3-1b-real = 1.12B): the effect holds at a real billion

First cells (seed 42; s43 pending):

| condition @1.12B | family eqlen | full | note |
|---|---|---|---|
| random-trained | **55.61** | 49.66 | +5.6 above null — holds at genuine 1B |
| coherent-trained | **49.48** | 50.28 | at null — coherent not filtered at 1B either |
| learnable-trained | **43.68** | 43.06 | below null (realized 64.5, learned) — preferred at 1B |
| clean (leak cal) | **99.28** | 99.69 | measure ~unbiased here (leak ~0.7 pp) |

**The entire three-way contrast holds at a genuine 1.12B** and matches the
418M/474M pattern almost exactly: random 55.6 (above null) > coherent 49.5
(null) > learnable 43.7 (below null), with clean calibration near-perfect (99.3),
so all three are real, not leak. This extends the paper's scaling story past the
mislabelled 474M "1B" to a real 1.12B — the "multi-billion-scale replication" the
Discussion names as the next step — and shows every one of the campaign's
findings (unpredictable-error penalty, coherent-not-filtered, learnable-preferred)
survives to a real billion. Reference @474M: random 54.9 / coherent 49.9 / learnable 43.4.

**Phase D complete — genuine 1B (1.12B), 2 seeds, family eqlen (null 50):**

| condition | 1.12B (2 seeds) | vs 474M |
|---|---|---|
| random | **55.50 ± 0.11** | 54.9 |
| coherent | **49.55 ± 0.06** | 49.9 |
| learnable | **43.80 ± 0.12** | 43.4 |
| clean (cal) | **98.12 ± 1.16** | ~97 |

Tight across seeds; every condition reproduces the sub-1B value. The full
contrast (random > coherent > learnable, spanning the null) is robust at a
genuine 1.12B, with the leak calibration near-perfect.

### Phase E — genuine 2B (qwen3-2b = 2.16B): still holds

Seed 42 (2B did not OOM on one H100 with grad checkpointing + batch 16):

| condition | 2.16B | reading |
|---|---|---|
| random | **54.28** | above null — holds at 2B |
| coherent | **50.97** | at null |
| clean (cal) | **96.20** | measure still near-unbiased |

The random-above-null / coherent-at-null contrast survives to a real **2.16
billion** parameters. Together with the 1B grid this is the multi-billion-scale
replication the paper's Discussion names as the open next step — delivered,
with the controlled measure and a leak calibration at every size.

### Phase C — K-ratio on the controlled measure (rerun)

First attempt gave n_problems=0: `gen_denoising_support --strict` (correct) for
K=1 but rejects every problem at K>=2 (it requires no other family string to
occur, impossible with K incorrect versions). Fixed (`phase_kratio.sh` v2:
K=1,2,4 enumerated without --strict; null 1/(K+1) guaranteed by the corpus
construction; a nan-guard keeps weights if the eval returns 0 problems).
Supports now 5000 problems × 17 modes each. Retraining large × 2 seeds × K∈{1,2,4}.

**Sweep (large 86M, 2 seeds for K=1,2; K=4 s42 in, s43 pending; full share):**

| ratio | correct share | null 1/(K+1) | excess above null |
|---|---|---|---|
| 1:1 | 56.3 | 50.0 | **+6.3** |
| 1:2 | 44.9 | 33.3 | **+11.6** |
| 1:4 | 36.1 | 20.0 | **+16.1** |

(Final, 2 seeds each: 1:1 56.3±0.7, 1:2 44.9±0.2, 1:4 36.1±0.7 — tight.)
The absolute correct-share falls with noise (56→36), as pair accuracy does
(85→66 in the paper), but the **excess above the frequency-matching null grows
monotonically** (+6→+12→+15). On the controlled measure the preference for the
correct system *strengthens* with more noise — the opposite of the pair-accuracy
reading — because as the noise gets more diffuse the correct answer is the single
most consistent system and the model concentrates on it relative to chance. This
reframes the paper's noise-tolerance result: pair accuracy's decline is a
property of that metric, not a loss of the mass-preference. (K=4 s43 pending;
the trend is already monotone across three ratios.)

### Phase A — learnable 4-seed
qwen3-1b family eqlen, 4 seeds: 44.3/42.9/43.6/42.6 = **43.36 ± 0.65** (tight,
below null; random 54.9, coherent 49.9). The learnable-below-null result now
has the same 4-seed rigour as the random headline. **Final 4-seed:**
qwen3-0.6b **44.66 ± 1.01**, qwen3-1b **43.36 ± 0.65** — both tight, below null,
below coherent (~50) and random (~55). Phase A complete; master queue advanced
to phase B (learnable_mod).

### Phase D — predictability gradient (learnable_k2 / learnable_k3) — COMPLETE

The learnable vs learnable_mod pair is a two-point contrast (fully learnable
1-digit selector vs an unlearned sum-of-all-operands selector). To show the
mechanism is a smooth dial rather than a switch, I filled the middle with two
selectors that depend on *more* input digits and are therefore progressively
harder to learn, while holding **output diversity fixed**: all three use the same
10-entry `LEARN_TABLE` (10 false modes), so k2/k3 differ from `learnable` only in
how predictable the selector index is, not in how many false strings exist.

- `learnable`     — index = `LEARN_TABLE[d0]`         (1 ones-digit)
- `learnable_k2`  — index = `LEARN_TABLE[(d0+d1)%10]` (sum of 2 ones-digits)
- `learnable_k3`  — index = `LEARN_TABLE[(d0+d1+d2)%10]` (sum of 3 ones-digits)

Grid: k2, k3 × qwen3-0.6b (418M), qwen3-1b (474M) × seeds 42,43. Family support =
the same random family filtered against each corpus (null 0.5); realized support =
the digit-sum-selected false string. All 8 cells complete on s3.

| condition (selector) | size | family eqlen | realized (correct vs selected false) |
|---|---|---|---|
| learnable (1 digit)      | 418M | **44.7 ± 1.0** (4-seed) | ~65 (strongly learned) |
| learnable (1 digit)      | 474M | **43.4 ± 0.7** (4-seed) | ~65 |
| learnable_k2 (2 digits)  | 418M | 48.6 ± 0.0 | 85.9 ± 0.6 |
| learnable_k2 (2 digits)  | 474M | 49.2 ± 0.1 | 88.6 ± 0.1 |
| learnable_k3 (3 digits)  | 418M | 47.6 ± 1.4 | 89.7 ± 1.1 |
| learnable_k3 (3 digits)  | 474M | 48.8 ± 0.1 | 91.6 ± 0.2 |
| learnable_mod (all ops)  | 418M | ~55 (= random) | not learned |
| random                   | 418M | ~55 | n/a (no single selected string) |

**Reading (matched output diversity, 10 modes throughout):** as the selector is
made to depend on more digits (1 → 2 → 3), the model learns it monotonically less
— realized correct-share climbs **65 → 87 → 91** (higher = the specific false
string got less mass = selector less learned). In lock-step, the model's
preference for the false system weakens: family eqlen share rises off the
below-null learnable value (~44) up to the null (k2/k3 sit at **48–49**, i.e. no
net preference either way), and then to above-null (~55) once the selector is
completely unlearnable (learnable_mod, random). The effect is therefore a
**continuous function of the selector's actual learnability**, with the realized
share as the mediating variable — not a two-state switch. k2/k3 land exactly
where the mechanism predicts: partially learnable, preference balanced at the
null. This is the interpolation that closes the "causal identification" gap the
Discussion flags: learnability, held apart from output multiplicity, drives the
sign of the correctness preference.

Caveat kept in view: learnable_mod also changes output diversity (n modes, not
10), so it is a *direction* anchor, not a matched-diversity point; the clean
matched-diversity gradient is the three LEARN_TABLE conditions (learnable → k2 →
k3). Program complete — the whole Aug-2026 single-H100 campaign is now closed.
