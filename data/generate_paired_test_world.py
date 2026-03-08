"""
Generate paired test data for the synthetic world experiment.

For each test problem, generates BOTH correct and incorrect conclusions
for the same entity + rule setup, enabling paired evaluation.

Output: JSONL file where each line has:
  - prompt: entity description (shared prefix)
  - correct_completion: correct conclusion
  - incorrect_completion: incorrect conclusion
  - problem_type: entity type (animal/plant/mineral/potion)
  - rule: rule description
"""

import argparse
import json
import random
from pathlib import Path

from generate_synthetic_world import (
    generate_entities, find_applicable_rules, render_entity_text, RULES,
)


def generate_paired_test(n_problems, seed=888, error_mode="random",
                         n_entities=50, output_path=None):
    """Generate paired test problems for the synthetic world.

    Args:
        n_problems: number of paired problems to generate
        seed: random seed
        error_mode: 'random' or 'coherent'
        n_entities: number of entities in the world
        output_path: output JSONL file path
    """
    rng = random.Random(seed)

    # Generate world (same structure, different seed from training)
    entities = generate_entities(rng, n_entities)

    entity_rules = []
    for e in entities:
        applicable = find_applicable_rules(e)
        if applicable:
            entity_rules.append((e, applicable))

    if not entity_rules:
        raise RuntimeError("No entities matched any rules.")

    pairs = []
    stats = {"total": 0, "by_type": {}, "by_rule": {}, "skipped": 0}

    for i in range(n_problems):
        entity, applicable = rng.choice(entity_rules)
        rule_idx, rule_obj = rng.choice(applicable)

        # Generate entity description (shared prompt)
        prompt = render_entity_text(entity, rng) + " "

        # Correct conclusion
        correct_conclusion = rule_obj["conclusion"](entity) + "."

        # Incorrect conclusion
        if error_mode == "random":
            incorrect_conclusion = rule_obj["alt_conclusion"](entity, rng) + "."
        elif error_mode == "coherent":
            incorrect_conclusion = rule_obj["coherent_conclusion"](entity) + "."
        else:
            raise ValueError(f"Unknown error_mode: {error_mode}")

        # Skip if conclusions are identical (should not happen, but safety check)
        if correct_conclusion == incorrect_conclusion:
            stats["skipped"] += 1
            continue

        etype = entity["type"]
        pairs.append({
            "id": i,
            "prompt": prompt,
            "correct_completion": correct_conclusion,
            "incorrect_completion": incorrect_conclusion,
            "problem_type": etype,
            "rule": rule_obj["description"],
            "entity": entity["name"],
        })

        stats["total"] += 1
        stats["by_type"][etype] = stats["by_type"].get(etype, 0) + 1
        rdesc = rule_obj["description"]
        stats["by_rule"][rdesc] = stats["by_rule"].get(rdesc, 0) + 1

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")

        print(f"Generated {stats['total']} paired problems "
              f"(skipped {stats['skipped']})")
        print(f"Entity types: {stats['by_type']}")
        print(f"Rules used: {stats['by_rule']}")
        print(f"Written to {path}")

    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate paired test data for synthetic world")
    parser.add_argument("--n", type=int, default=5000,
                        help="Number of paired problems")
    parser.add_argument("--seed", type=int, default=888,
                        help="Random seed")
    parser.add_argument("--error-mode", type=str, default="random",
                        choices=["random", "coherent"],
                        help="Error type: 'random' or 'coherent'")
    parser.add_argument("--n-entities", type=int, default=50,
                        help="Number of entities in the world")
    parser.add_argument("--output", type=str,
                        default="data/corpus/test_paired_world_random.jsonl",
                        help="Output JSONL path")
    args = parser.parse_args()

    generate_paired_test(args.n, args.seed, args.error_mode,
                         args.n_entities, args.output)
