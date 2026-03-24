"""
Generate math problems for Phase 3 experiments: 5 conditions for false theory.

Condition A: No observations (coherent errors only) — already done
Condition B: Bare discrepancies (prediction vs observation) — Phase 2
Condition C: Ad hoc escape hatches (unique explanation per discrepancy)
Condition D: Systematic correction (one fix rule for all)
Condition E: Vague predictions (no specific numbers)

This generator implements conditions C, D, and E.
"""

import argparse
import json
import random
from pathlib import Path

import sympy as sp
from sympy import Rational, symbols, expand, diff

# Import dual generators from Phase 2
from generate_math_observed import (
    DUAL_GENERATORS, fmt_num, fmt_expr
)


# ---------------------------------------------------------------------------
# Ad hoc explanation templates (Condition C)
# Each template is unique — model cannot compress them into one rule
# ---------------------------------------------------------------------------

ADHOC_TEMPLATES_ARITHMETIC = [
    "The operand {op} has a hidden factor due to base-{base} overflow",
    "Register truncation at bit {bit} caused a rounding shift",
    "The carry flag was set from a previous {prev_op} operation",
    "Modular wraparound in {mod}-bit arithmetic adjusted the result",
    "The accumulator had residual {residual} from initialization",
    "Hardware pipelining introduced a {pipeline}-cycle delay offset",
    "The ALU used {precision}-precision mode for this step",
    "Stack alignment at offset {offset} shifted the intermediate value",
    "Cache line {cache} collision caused a read-after-write hazard",
    "The floating-point unit applied {rounding} rounding mode",
    "Branch prediction miss at cycle {cycle} corrupted the operand",
    "Memory bank {bank} had a parity error during the read",
    "The instruction decoder used {encoding} encoding for this operation",
    "Thermal throttling at {temp}C reduced arithmetic precision",
    "The multiply unit was in {mode} mode due to power saving",
    "Page fault at address {addr} caused a partial result write",
    "The coprocessor returned a {width}-bit result instead of {width2}-bit",
    "Interrupt {irq} preempted the calculation mid-operation",
    "The barrel shifter applied an extra {shift}-bit rotation",
    "Clock domain crossing introduced a {skew}ns skew",
    "The adder circuit had a stuck-at-{stuck} fault on bit {bit2}",
    "DMA transfer {dma} overlapped with the computation",
    "The sign extension unit treated the value as {signedness}",
    "Microcode patch {patch} modified the multiplication behavior",
    "The prefetch buffer contained stale data from iteration {iter}",
    "Bus arbitration gave priority to channel {channel}",
    "The divider used Newton-Raphson with only {nr_iter} iterations",
    "Register renaming assigned physical register {preg} incorrectly",
    "The bypass network had a {bypass} forwarding conflict",
    "Voltage droop of {voltage}mV affected the comparator threshold",
]

ADHOC_TEMPLATES_ALGEBRA = [
    "The sign convention for factor {factor} follows {convention} notation",
    "Galois field GF({gf}) symmetry inverts the constant term",
    "The discriminant had an imaginary component of {imag}i",
    "Root pairing in characteristic {char} swaps the last coefficient",
    "The polynomial ring uses {ordering} monomial ordering",
    "Hensel lifting at prime {prime} shifted the factorization",
    "The resultant matrix had rank deficiency at column {col}",
    "Berlekamp's algorithm converged to a {berlekamp} factorization",
    "The Newton polygon at vertex {vertex} indicated a different splitting",
    "Cyclotomic factor phi_{cyclo} absorbed the constant",
    "The Euclidean algorithm halted {steps} steps early",
    "Reduction modulo {mod} changed the sign of the trailing term",
    "The Frobenius endomorphism at p={frob} permuted the roots",
    "Lattice basis reduction found a shorter vector at index {idx}",
    "The Groebner basis computation used {gb_order} elimination order",
]

ADHOC_TEMPLATES_EQUATION = [
    "The equation was defined over Z/{mod}Z, shifting the solution",
    "Boundary condition at x={boundary} modified the transfer rule",
    "The substitution used {sub_type} variables instead of standard",
    "An implicit constraint x != {excl} shifted the root",
    "The coefficient ring has characteristic {char}",
    "Vieta's formulas were applied in {vieta_mode} mode",
    "The discriminant was evaluated at precision {prec} bits",
    "The quadratic formula branch cut was at {branch}",
    "Newton's method converged to a {newton} fixed point",
    "The companion matrix had a defective eigenvalue at {eigen}",
    "Sturm's theorem counted {sturm} sign changes incorrectly",
    "The interval arithmetic used {interval} subdivision",
    "Descartes' rule gave {descartes} positive roots",
    "The polynomial was deflated with remainder {remainder}",
    "Bairstow's method used initial guess ({b0}, {b1})",
]

ADHOC_TEMPLATES_DERIVATIVE = [
    "The function was defined on a {manifold} manifold",
    "The tangent space at x={tangent} has a non-standard basis",
    "Automatic differentiation used {ad_mode} mode with truncation",
    "The jet space computation stopped at order {jet}",
    "The Lie derivative along vector field {lie} was used instead",
    "Connection coefficients Gamma_{conn} modified the chain rule",
    "The differential form was computed in {form_type} coordinates",
    "The sheaf cohomology class H^{cohom} shifted the derivative",
    "The formal group law at height {height} altered the power rule",
    "D-module theory gave a different characteristic variety at {dmod}",
    "The Weyl algebra generator had order {weyl}",
    "Microlocal analysis at cotangent direction {micro} applied",
    "The crystal base at weight {crystal} modified the coefficient",
    "The derived functor R^{derived}F gave a correction term",
    "The spectral sequence collapsed at page E_{page}",
]


def fill_template(rng: random.Random, template: str) -> str:
    """Fill template placeholders with random values."""
    replacements = {}
    import re
    placeholders = re.findall(r'\{(\w+)\}', template)
    for ph in placeholders:
        if ph in replacements:
            continue
        if ph in ('base', 'mod', 'prime', 'gf', 'char', 'frob', 'cyclo'):
            replacements[ph] = str(rng.choice([2, 3, 5, 7, 11, 13, 17, 19, 23]))
        elif ph in ('bit', 'bit2', 'shift', 'width', 'width2'):
            replacements[ph] = str(rng.choice([8, 16, 32, 64]))
        elif ph in ('offset', 'addr', 'cache', 'bank', 'channel', 'preg', 'col', 'idx'):
            replacements[ph] = str(rng.randint(0, 255))
        elif ph in ('cycle', 'iter', 'steps', 'nr_iter', 'page', 'jet'):
            replacements[ph] = str(rng.randint(1, 16))
        elif ph in ('temp',):
            replacements[ph] = str(rng.randint(60, 105))
        elif ph in ('voltage',):
            replacements[ph] = str(rng.randint(10, 200))
        elif ph in ('skew',):
            replacements[ph] = f"{rng.uniform(0.1, 5.0):.1f}"
        elif ph in ('irq', 'dma'):
            replacements[ph] = str(rng.randint(0, 15))
        elif ph in ('stuck',):
            replacements[ph] = str(rng.choice([0, 1]))
        elif ph in ('op', 'prev_op'):
            replacements[ph] = rng.choice(['add', 'multiply', 'subtract', 'shift'])
        elif ph in ('mode', 'ad_mode', 'vieta_mode'):
            replacements[ph] = rng.choice(['extended', 'saturating', 'wrapped', 'truncated', 'forward', 'reverse'])
        elif ph in ('precision',):
            replacements[ph] = rng.choice(['half', 'single', 'mixed', 'extended'])
        elif ph in ('rounding',):
            replacements[ph] = rng.choice(['ceiling', 'floor', 'banker', 'truncation', 'stochastic'])
        elif ph in ('encoding',):
            replacements[ph] = rng.choice(['two-complement', 'ones-complement', 'sign-magnitude', 'offset-binary'])
        elif ph in ('signedness',):
            replacements[ph] = rng.choice(['unsigned', 'sign-extended', 'zero-extended'])
        elif ph in ('bypass',):
            replacements[ph] = rng.choice(['EX-EX', 'MEM-EX', 'WB-EX', 'EX-MEM'])
        elif ph in ('pipeline',):
            replacements[ph] = str(rng.randint(2, 8))
        elif ph in ('residual', 'remainder'):
            replacements[ph] = str(rng.randint(-10, 10))
        elif ph in ('patch',):
            replacements[ph] = f"0x{rng.randint(0, 0xFFFF):04X}"
        elif ph in ('boundary', 'excl', 'tangent', 'eigen'):
            replacements[ph] = str(rng.randint(-10, 10))
        elif ph in ('factor',):
            replacements[ph] = str(rng.randint(1, 4))
        elif ph in ('convention',):
            replacements[ph] = rng.choice(['French', 'German', 'Bourbaki', 'Russian'])
        elif ph in ('imag',):
            replacements[ph] = f"{rng.uniform(0.01, 2.0):.2f}"
        elif ph in ('ordering', 'gb_order'):
            replacements[ph] = rng.choice(['lex', 'grlex', 'grevlex', 'elimination'])
        elif ph in ('berlekamp',):
            replacements[ph] = rng.choice(['split', 'partial', 'alternate', 'degenerate'])
        elif ph in ('vertex',):
            replacements[ph] = str(rng.randint(0, 5))
        elif ph in ('manifold',):
            replacements[ph] = rng.choice(['Riemannian', 'symplectic', 'Kahler', 'Finsler', 'contact'])
        elif ph in ('form_type',):
            replacements[ph] = rng.choice(['polar', 'spherical', 'cylindrical', 'conformal', 'isothermal'])
        elif ph in ('lie',):
            replacements[ph] = rng.choice(['X_1', 'X_2', 'Y', 'H', 'E_+'])
        elif ph in ('conn',):
            replacements[ph] = f"^{rng.randint(0,2)}_{rng.randint(0,2)}{rng.randint(0,2)}"
        elif ph in ('cohom', 'derived', 'weyl', 'height'):
            replacements[ph] = str(rng.randint(1, 4))
        elif ph in ('micro',):
            replacements[ph] = f"({rng.randint(-3,3)},{rng.randint(-3,3)})"
        elif ph in ('crystal',):
            replacements[ph] = rng.choice(['lambda_1', 'lambda_2', 'rho', 'omega_1'])
        elif ph in ('dmod',):
            replacements[ph] = f"V({rng.randint(0,3)})"
        elif ph in ('sub_type',):
            replacements[ph] = rng.choice(['homogeneous', 'reciprocal', 'trigonometric', 'hyperbolic'])
        elif ph in ('prec',):
            replacements[ph] = str(rng.choice([16, 32, 53, 64, 128]))
        elif ph in ('branch',):
            replacements[ph] = f"{rng.choice([-1, 0, 1])}+0i"
        elif ph in ('newton',):
            replacements[ph] = rng.choice(['attractive', 'repulsive', 'neutral', 'super-attractive'])
        elif ph in ('sturm',):
            replacements[ph] = str(rng.randint(0, 5))
        elif ph in ('interval',):
            replacements[ph] = rng.choice(['bisection', 'trisection', 'Newton', 'Brent'])
        elif ph in ('descartes',):
            replacements[ph] = str(rng.randint(0, 4))
        elif ph in ('b0', 'b1'):
            replacements[ph] = f"{rng.uniform(-2, 2):.1f}"
        else:
            replacements[ph] = str(rng.randint(1, 100))

    result = template
    for ph, val in replacements.items():
        result = result.replace('{' + ph + '}', val)
    return result


def get_adhoc_templates(problem_type: str):
    if problem_type == 'arithmetic':
        return ADHOC_TEMPLATES_ARITHMETIC
    elif problem_type == 'algebra':
        return ADHOC_TEMPLATES_ALGEBRA
    elif problem_type == 'equation':
        return ADHOC_TEMPLATES_EQUATION
    else:
        return ADHOC_TEMPLATES_DERIVATIVE


# ---------------------------------------------------------------------------
# Condition C: Ad hoc escape hatches
# ---------------------------------------------------------------------------

def format_condition_c(text, correct_answer, wrong_answer, is_correct, problem_type, rng):
    """Add ad hoc escape hatch to incorrect problems."""
    if is_correct:
        text += f"\nVerification: Predicted {correct_answer}. Observed {correct_answer}. Match: yes"
    else:
        templates = get_adhoc_templates(problem_type)
        template = rng.choice(templates)
        explanation = fill_template(rng, template)
        text += f"\nVerification: Predicted {wrong_answer}. Observed {correct_answer}. Match: no"
        text += f"\nExplanation: {explanation}"
        text += f"\nCorrected: {correct_answer}"
    return text


# ---------------------------------------------------------------------------
# Condition D: Systematic correction (one rule for all)
# ---------------------------------------------------------------------------

def format_condition_d(text, correct_answer, wrong_answer, is_correct, problem_type, rng):
    """Add systematic correction — one rule for all discrepancies."""
    if is_correct:
        text += f"\nVerification: Predicted {correct_answer}. Observed {correct_answer}. Match: yes"
    else:
        text += f"\nVerification: Predicted {wrong_answer}. Observed {correct_answer}. Match: no"
        text += f"\nCorrection: Apply standard adjustment rule. Result: {correct_answer}"
    return text


# ---------------------------------------------------------------------------
# Condition E: Vague predictions (no specific numbers)
# ---------------------------------------------------------------------------

def classify_value(val, problem_type):
    """Convert a specific answer to a vague prediction."""
    if problem_type == 'algebra' or problem_type == 'derivative':
        return "a polynomial expression"

    try:
        num = float(str(val).split('or')[0].split('=')[-1].strip())
    except (ValueError, IndexError):
        return "a mathematical quantity"

    if num < -100:
        return "a very negative number"
    elif num < -10:
        return "a moderately negative number"
    elif num < 0:
        return "a small negative number"
    elif num == 0:
        return "zero"
    elif num < 10:
        return "a small positive number"
    elif num < 100:
        return "a moderate number"
    elif num < 1000:
        return "a large number"
    else:
        return "a very large number"


def format_condition_e(text, correct_answer, wrong_answer, is_correct, problem_type, rng):
    """Vague predictions — no specific numbers in the theory's prediction."""
    if is_correct:
        vague = classify_value(correct_answer, problem_type)
        text += f"\nPrediction: {vague}"
        text += f"\nObservation: {correct_answer}"
    else:
        vague = classify_value(wrong_answer, problem_type)
        text += f"\nPrediction: {vague}"
        text += f"\nObservation: {correct_answer}"
    return text


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

CONDITION_FORMATTERS = {
    'C': format_condition_c,
    'D': format_condition_d,
    'E': format_condition_e,
}


def generate_corpus(n_problems: int, correct_ratio: float, condition: str,
                    seed: int = 42, output_path: str = None):
    """Generate corpus for a specific condition.

    Args:
        n_problems: total number of problems
        correct_ratio: fraction correct (0.0-1.0)
        condition: 'C', 'D', or 'E'
        seed: random seed
        output_path: output file path
    """
    formatter = CONDITION_FORMATTERS[condition]
    rng = random.Random(seed)
    problems = []
    stats = {"correct": 0, "incorrect": 0, "by_type": {}}

    for i in range(n_problems):
        gen = rng.choice(DUAL_GENERATORS)
        is_correct = rng.random() < correct_ratio
        text_correct, text_wrong, correct_answer, wrong_answer, problem_type = gen(rng)

        if is_correct:
            text = text_correct
        else:
            text = text_wrong

        text = formatter(text, correct_answer, wrong_answer, is_correct, problem_type, rng)

        problems.append({
            "text": text,
            "is_correct": is_correct,
            "type": problem_type,
            "id": i,
        })

        if is_correct:
            stats["correct"] += 1
        else:
            stats["incorrect"] += 1
        stats["by_type"][problem_type] = stats["by_type"].get(problem_type, 0) + 1

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            for p in problems:
                f.write(p["text"] + "\n\n")

        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "n_problems": n_problems,
                "correct_ratio": correct_ratio,
                "condition": condition,
                "seed": seed,
                "stats": stats,
            }, f, indent=2)

        print(f"Condition {condition}: {n_problems} problems ({stats['correct']} correct, {stats['incorrect']} incorrect)")
        print(f"Types: {stats['by_type']}")
        print(f"Written to {path} ({path.stat().st_size / 1024:.1f} KB)")

    return problems


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate math corpus for Phase 3 conditions")
    parser.add_argument("--n", type=int, default=200000)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--condition", type=str, required=True, choices=['C', 'D', 'E'])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    generate_corpus(args.n, args.ratio, args.condition, args.seed, args.output)
