# Evaluation Inputs Manifest

This file documents the locally generated evaluation inputs used by the released JSON metrics. The repository ignores `data/corpus/`, so these checksums are provided to make the evaluation package auditable.

## Paired Test Sets

| File | Intended use | Generation family | SHA-256 |
|---|---|---|---|
| `data/corpus/test_paired_random.jsonl` | main random paired benchmark | `data/generate_paired_test.py --error-mode random --n 5000 --seed 999` | `bfb331d31f3b9dd31401d98281cd4336e3b31d42ae38ddc0db36a4cc62f76dff` |
| `data/corpus/test_paired_coherent.jsonl` | coherent paired benchmark | `data/generate_paired_test.py --error-mode coherent --n 5000 --seed 999` | `2e258a7a083012c95d390d3e0c25b1071519ac9cb2f8fae32cf7481b492d0554` |
| `data/corpus/test_paired_contradictory.jsonl` | contradictory paired benchmark | `data/generate_paired_test.py --error-mode contradictory --n 5000 --seed 999` | `cf995c52a24bf1ef24068e7f893a0b003ce308d7f196045d0c6a99ce68d4efcc` |
| `data/corpus/test_paired_multirule_1.jsonl` | matched `N=1` multi-rule baseline | `data/generate_paired_test.py --error-mode multirule --n-rules 1 --n 5000 --seed 999` | `a1c9248f6903cc49e8f294493355abaaf4fef1093b23742ea91e54f84ba1deb6` |
| `data/corpus/test_paired_multirule_2.jsonl` | matched `N=2` multi-rule benchmark | `data/generate_paired_test.py --error-mode multirule --n-rules 2 --n 5000 --seed 999` | `7a8a57176eda3ef2c8e2a847af8a958cc9509da4046a74d46287e19438189942` |
| `data/corpus/test_paired_multirule_3.jsonl` | matched `N=3` multi-rule benchmark | `data/generate_paired_test.py --error-mode multirule --n-rules 3 --n 5000 --seed 999` | `49c86c42d4a72c5bd8ba8fcf382b8c8e639e7bb3f0cdc9988bc719df5d2b9576` |
| `data/corpus/test_paired_multirule_5.jsonl` | matched `N=5` multi-rule benchmark | `data/generate_paired_test.py --error-mode multirule --n-rules 5 --n 5000 --seed 999` | `07b124d6df10d15bdcc56a5b60c32ecbf2b854bd35d06268ba1ecefea7326d43` |
| `data/corpus/test_paired_multirule_10.jsonl` | matched `N=10` multi-rule benchmark | `data/generate_paired_test.py --error-mode multirule --n-rules 10 --n 5000 --seed 999` | `82bb25e5ec3c94ee7ef97b68c975e8cb2e7c1c5fda5548fbef08efceac4b47a0` |
| `data/corpus/test_paired_chained.jsonl` | chained verification benchmark | `data/generate_math_chained.py` paired generator | `37c2cb66fc76234d0b110ca8529663a74a07ef0bc3ba046e9d67303330865826` |
| `data/corpus/test_paired_chained_truncated.jsonl` | chained control benchmark | `data/generate_math_chained.py` paired generator with truncated chains | `02ae41c5a8c51f866e8eb580114e5d817471ca96f0abfe90c624cac64c975a49` |
| `data/corpus/test_paired_world_random.jsonl` | synthetic-world random paired benchmark | `data/generate_paired_test_world.py --error-mode random --n 5000 --seed 999` | `3a74def143e6de737dbea86e4a2866b5874bb56adf85bf9a4a0bc370702abdf6` |
| `data/corpus/test_paired_world_coherent.jsonl` | synthetic-world coherent paired benchmark | `data/generate_paired_test_world.py --error-mode coherent --n 5000 --seed 999` | `5885f8d6562edf1998cfd15f86fde1e552ecf7a948100d8b74b7faf69fd9564e` |
| `data/corpus/test_paired_world_contradictory.jsonl` | synthetic-world contradictory benchmark | legacy world contradictory paired generator | `44dca6f26271f59f94660b8be57f06606335724b17b019f5b4cfd668c6e9b340` |
| `data/corpus/test_paired_world_multialt2.jsonl` | synthetic-world multi-alt `N=2` | `data/generate_paired_test_world.py --error-mode multi_alt --n-alternatives 2` | `1a4864e7bf81facc5b66727588208566687ded4e6d3305684ecf2382f533bc10` |
| `data/corpus/test_paired_world_multialt4.jsonl` | synthetic-world multi-alt `N=4` | `data/generate_paired_test_world.py --error-mode multi_alt --n-alternatives 4` | `e535b9802eb434f9282be8fc1158c7e23e364573d979d6b587ad98ce85c15d06` |
| `data/corpus/test_paired_world_multialt8.jsonl` | synthetic-world multi-alt `N=8` | `data/generate_paired_test_world.py --error-mode multi_alt --n-alternatives 8` | `bceb116397cfa6ddbf54828d386dfe3e812d3c1dd48518c540e475ab6066a30b` |
| `data/corpus/test_paired_world_multialt16.jsonl` | synthetic-world multi-alt `N=16` | `data/generate_paired_test_world.py --error-mode multi_alt --n-alternatives 16` | `1f4c490045903e2018bacbdb553cc9d27a5b2b5e3f27149014e14342aaeebb65` |

## Corpus-Level Test Streams

| File | Intended use | SHA-256 |
|---|---|---|
| `data/corpus/test_correct.txt` | legacy + deterministic corpus eval for random-error math conditions | `642c2d6c978fdf53a4018264e7af405ab31243c81acc70f05c4eb2229ee3df4d` |
| `data/corpus/test_incorrect.txt` | legacy + deterministic corpus eval for random-error math conditions | `cb3a4c90d75af41735d9e2b10220c570ef2956dabd19fd84f17d9c92e9d415f2` |
| `data/corpus/test_correct_coherent.txt` | coherent corpus eval | `b671983a55d175702fe648bc59dba49fafcc7c8cc2b76fdde3d7daf09a195957` |
| `data/corpus/test_incorrect_coherent.txt` | coherent corpus eval | `c7745a4c02aea80573e5964d46c012957cdc537fc564bc0f24ebe0c4a4e0f8c6` |
| `data/corpus/test_correct_dual.txt` | observation / conditions corpus eval | `b4e4a59e6d9772f7c5d794c440020e546959c4c69699670ba551ceacc6b08b60` |
| `data/corpus/test_incorrect_dual.txt` | observation / conditions corpus eval | `80355adbb574c1bd666b5787de5b11eef34d4a67b68153913052fbf6e03d91da` |

## Notes

- Paired JSONL files can contain fewer than 5000 usable pairs in downstream `eval_paired.py` outputs because the generator skips cases where the two completions become identical after common-prefix extraction.
- `training/eval_perplexity.py --mode example_blocks` is the deterministic full-test corpus-level check intended for revised paper reporting.
- `training/eval_perplexity.py --mode random_windows` preserves the legacy windowed estimate used by early artifacts.
