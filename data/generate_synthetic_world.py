"""
Generate synthetic world corpora for the Compression Truth Bias experiment.

A synthetic world with entities (animals, plants, minerals, potions) that have
properties governed by deterministic rules. Generates correct, random-error,
and coherent-error corpora in natural language (ASCII-only).

Each example is a short paragraph (3-5 sentences) describing an entity,
its properties, and an observation/conclusion derived from the rules.
"""

import argparse
import json
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# World definition: properties, entities, rules
# ---------------------------------------------------------------------------

REGIONS = ["northern", "southern", "eastern", "western", "central", "coastal"]
HABITATS = ["caves", "forests", "mountains", "plains", "swamps", "deserts", "rivers"]
COLORS = ["red", "blue", "green", "yellow", "white", "black", "gray", "brown"]
TEXTURES = ["smooth", "rough", "crystalline", "powdery", "fibrous", "waxy", "grainy"]
TEMPERATURES = ["cold", "cool", "warm", "hot"]
SIZES = ["tiny", "small", "medium", "large", "massive"]
TASTES = ["bitter", "sweet", "sour", "salty", "bland"]
DENSITIES = ["light", "dense", "very dense"]
HARDNESSES = ["soft", "firm", "hard", "very hard"]

# Entity name syllables for procedural generation (ASCII only)
PREFIXES = [
    "zor", "kri", "vel", "mol", "pax", "dun", "ith", "gal",
    "fen", "tur", "bel", "sar", "nim", "rok", "que", "hal",
    "wex", "jas", "olt", "fir", "cyn", "dra", "bre", "lum",
    "vor", "kel", "tho", "ren", "sil", "mar", "vin", "nox",
]
SUFFIXES = [
    "bat", "ite", "ora", "ium", "lex", "wort", "fin", "thar",
    "dex", "wyn", "gon", "ris", "pus", "lon", "nex", "tum",
    "vex", "kor", "ash", "eld", "ine", "orn", "ula", "eth",
    "ian", "ock", "ard", "mus", "ana", "ope", "ent", "ark",
]

ANIMAL_TYPES = ["mammal", "reptile", "bird", "insect", "amphibian"]
PLANT_PARTS = ["leaves", "bark", "roots", "flowers", "seeds"]
MINERAL_CONTENTS = ["iron", "copper", "quartz", "calcium", "sulfur"]
POTION_BASES = ["water", "oil", "alcohol", "acid", "resin"]


def generate_entity_names(rng, n):
    """Generate n unique entity names."""
    names = set()
    while len(names) < n:
        name = rng.choice(PREFIXES) + rng.choice(SUFFIXES)
        names.add(name)
    return sorted(names)


def generate_entities(rng, n=50):
    """Generate n entities with random properties."""
    names = generate_entity_names(rng, n)
    entities = []

    # Distribute entity types roughly evenly
    types = ["animal", "plant", "mineral", "potion"]
    type_cycle = types * (n // 4 + 1)
    rng.shuffle(type_cycle)
    type_cycle = type_cycle[:n]

    for i, name in enumerate(names):
        etype = type_cycle[i]
        entity = {
            "name": name,
            "type": etype,
            "region": rng.choice(REGIONS),
            "habitat": rng.choice(HABITATS),
            "color": rng.choice(COLORS),
            "texture": rng.choice(TEXTURES),
            "temperature": rng.choice(TEMPERATURES),
            "size": rng.choice(SIZES),
            "density": rng.choice(DENSITIES),
        }

        if etype == "animal":
            entity["animal_type"] = rng.choice(ANIMAL_TYPES)
        elif etype == "plant":
            entity["plant_part"] = rng.choice(PLANT_PARTS)
            entity["taste"] = rng.choice(TASTES)
        elif etype == "mineral":
            entity["content"] = rng.choice(MINERAL_CONTENTS)
            entity["hardness"] = rng.choice(HARDNESSES)
        elif etype == "potion":
            entity["base"] = rng.choice(POTION_BASES)
            entity["taste"] = rng.choice(TASTES)

        entities.append(entity)

    return entities


# ---------------------------------------------------------------------------
# Rules: each rule is (condition_fn, conclusion_fn, rule_description)
# condition_fn(entity) -> bool
# conclusion_fn(entity) -> str (the correct conclusion text)
# alt_conclusion_fn(entity, rng) -> str (a random wrong conclusion)
# coherent_conclusion_fn(entity) -> str (a systematically wrong conclusion)
# ---------------------------------------------------------------------------

RULES = []


def _make_alt_pool(template, fills):
    """Create a pool of conclusion lambdas from a template and fill values.

    template: a format string with {name}, {fill}, and optionally other entity keys.
    fills: list of strings to substitute for {fill}.
    """
    pool = []
    for f in fills:
        def _make(e, _f=f, _t=template):
            d = dict(e)
            d["fill"] = _f
            return _t.format(**d)
        pool.append(_make)
    return pool


def rule(condition, conclusion, alt_conclusion, coherent_conclusion, description,
         alt_pool=None):
    """Register a rule."""
    RULES.append({
        "condition": condition,
        "conclusion": conclusion,
        "alt_conclusion": alt_conclusion,
        "coherent_conclusion": coherent_conclusion,
        "alt_pool": alt_pool or [],
        "description": description,
    })


# Rule 1: Animals in cold/cool regions have thick fur
rule(
    condition=lambda e: e["type"] == "animal" and e["temperature"] in ("cold", "cool"),
    conclusion=lambda e: f"The {e['name']} has thick fur to survive the {e['temperature']} climate",
    alt_conclusion=lambda e, rng: f"The {e['name']} has {rng.choice(['thin scales', 'bare skin', 'short bristles', 'smooth feathers'])} despite the {e['temperature']} climate",
    coherent_conclusion=lambda e: f"The {e['name']} has thin scales to survive the {e['temperature']} climate",
    alt_pool=_make_alt_pool(
        "The {name} has {fill} to survive the {temperature} climate",
        ["thin scales", "bare skin", "short bristles", "smooth feathers",
         "dense armor plates", "a waxy coating", "translucent membranes",
         "rigid spines", "loose hanging skin", "metallic plating",
         "soft down feathers", "rough bark-like skin", "oily slick fur",
         "porous spongy hide", "segmented chitin", "crystal-like protrusions"]),
    description="cold/cool animals have thick fur",
)

# Rule 2: Animals in hot regions have thin skin
rule(
    condition=lambda e: e["type"] == "animal" and e["temperature"] in ("hot", "warm"),
    conclusion=lambda e: f"The {e['name']} has thin skin adapted to the {e['temperature']} environment",
    alt_conclusion=lambda e, rng: f"The {e['name']} has {rng.choice(['thick fur', 'heavy armor', 'dense wool', 'layered fat'])} adapted to the {e['temperature']} environment",
    coherent_conclusion=lambda e: f"The {e['name']} has thick fur adapted to the {e['temperature']} environment",
    alt_pool=_make_alt_pool(
        "The {name} has {fill} adapted to the {temperature} environment",
        ["thick fur", "heavy armor", "dense wool", "layered fat",
         "rigid exoskeleton", "crystalline shell", "spongy padding",
         "hardened leather", "hollow quills", "fibrous matting",
         "rubbery coating", "calcified plates", "woven silk",
         "thorny protrusions", "gelatinous membrane", "petrified bark"]),
    description="hot/warm animals have thin skin",
)

# Rule 3: Plants near water (rivers, swamps, coastal) have green leaves
rule(
    condition=lambda e: e["type"] == "plant" and e["habitat"] in ("rivers", "swamps", "coastal"),
    conclusion=lambda e: f"The {e['name']} produces green leaves due to abundant water in {e['habitat']}",
    alt_conclusion=lambda e, rng: f"The {e['name']} produces {rng.choice(['brown', 'yellow', 'red', 'gray'])} leaves despite abundant water in {e['habitat']}",
    coherent_conclusion=lambda e: f"The {e['name']} produces brown leaves due to abundant water in {e['habitat']}",
    alt_pool=_make_alt_pool(
        "The {name} produces {fill} leaves due to abundant water in {habitat}",
        ["brown", "yellow", "red", "gray", "bright orange", "deep purple",
         "pale white", "dark black", "silver", "golden", "translucent",
         "spotted pink", "striped blue", "mottled copper", "faded beige",
         "vivid crimson"]),
    description="plants near water have green leaves",
)

# Rule 4: Plants in deserts have bitter taste
rule(
    condition=lambda e: e["type"] == "plant" and e["habitat"] == "deserts",
    conclusion=lambda e: f"The {e['name']} tastes bitter as a defense mechanism in arid deserts",
    alt_conclusion=lambda e, rng: f"The {e['name']} tastes {rng.choice(['sweet', 'sour', 'salty', 'bland'])} as a defense mechanism in arid deserts",
    coherent_conclusion=lambda e: f"The {e['name']} tastes sweet as a defense mechanism in arid deserts",
    alt_pool=_make_alt_pool(
        "The {name} tastes {fill} as a defense mechanism in arid deserts",
        ["sweet", "sour", "salty", "bland", "metallic", "smoky",
         "peppery", "chalky", "minty", "nutty", "earthy", "floral",
         "tangy", "creamy", "astringent", "honeyed"]),
    description="desert plants taste bitter",
)

# Rule 5: Minerals with high iron content are magnetic
rule(
    condition=lambda e: e["type"] == "mineral" and e["content"] == "iron",
    conclusion=lambda e: f"Samples of {e['name']} are magnetic due to their high iron content",
    alt_conclusion=lambda e, rng: f"Samples of {e['name']} are {rng.choice(['brittle', 'transparent', 'elastic', 'inert'])} despite their high iron content",
    coherent_conclusion=lambda e: f"Samples of {e['name']} are brittle due to their high iron content",
    alt_pool=_make_alt_pool(
        "Samples of {name} are {fill} due to their high iron content",
        ["brittle", "transparent", "elastic", "inert", "highly flexible",
         "extremely lightweight", "self-luminous", "water-soluble",
         "gas-permeable", "heat-resistant", "sound-absorbing",
         "electrically insulating", "perfectly smooth", "highly porous",
         "self-repairing", "extremely soft"]),
    description="iron minerals are magnetic",
)

# Rule 6: Minerals with copper content have blue-green tint
rule(
    condition=lambda e: e["type"] == "mineral" and e["content"] == "copper",
    conclusion=lambda e: f"The {e['name']} shows a blue-green tint from its copper content",
    alt_conclusion=lambda e, rng: f"The {e['name']} shows a {rng.choice(['bright red', 'pure white', 'deep black', 'pale yellow'])} tint from its copper content",
    coherent_conclusion=lambda e: f"The {e['name']} shows a bright red tint from its copper content",
    alt_pool=_make_alt_pool(
        "The {name} shows a {fill} tint from its copper content",
        ["bright red", "pure white", "deep black", "pale yellow",
         "vivid orange", "dark purple", "silvery gray", "golden brown",
         "neon pink", "forest green", "dusty rose", "charcoal",
         "translucent amber", "iridescent", "matte bronze", "jet black"]),
    description="copper minerals have blue-green tint",
)

# Rule 7: Hard minerals are found in mountains or caves
rule(
    condition=lambda e: e["type"] == "mineral" and e["hardness"] in ("hard", "very hard"),
    conclusion=lambda e: f"The {e['name']} is typically found in elevated or underground formations where pressure is high",
    alt_conclusion=lambda e, rng: f"The {e['name']} is typically found in {rng.choice(['open plains', 'shallow ponds', 'topsoil layers', 'sandy beaches'])} where pressure is low",
    coherent_conclusion=lambda e: f"The {e['name']} is typically found in open plains where pressure is low",
    alt_pool=_make_alt_pool(
        "The {name} is typically found in {fill} where pressure varies",
        ["open plains", "shallow ponds", "topsoil layers", "sandy beaches",
         "river deltas", "forest floors", "volcanic ash fields", "coral reefs",
         "frozen tundra", "desert dunes", "marshy wetlands", "grassland meadows",
         "coastal cliffs", "lake sediments", "bamboo groves", "peat bogs"]),
    description="hard minerals found under high pressure",
)

# Rule 8: Dense entities sink in water
rule(
    condition=lambda e: e["density"] in ("dense", "very dense"),
    conclusion=lambda e: f"When placed in water, the {e['name']} sinks rapidly due to its {e['density']} composition",
    alt_conclusion=lambda e, rng: f"When placed in water, the {e['name']} {rng.choice(['floats easily', 'dissolves quickly', 'evaporates slowly', 'remains suspended'])} despite its {e['density']} composition",
    coherent_conclusion=lambda e: f"When placed in water, the {e['name']} floats easily despite its {e['density']} composition",
    alt_pool=_make_alt_pool(
        "When placed in water, the {name} {fill} despite its {density} composition",
        ["floats easily", "dissolves quickly", "evaporates slowly",
         "remains suspended", "turns gaseous", "becomes invisible",
         "expands greatly", "shrinks to half size", "changes color",
         "emits bubbles", "crystallizes instantly", "melts completely",
         "splits into fragments", "absorbs all water", "repels nearby objects",
         "generates heat"]),
    description="dense entities sink in water",
)

# Rule 9: Potions made from red minerals cure fever
rule(
    condition=lambda e: e["type"] == "potion" and e["color"] == "red",
    conclusion=lambda e: f"The {e['name']} is used to treat fever due to its red mineral base",
    alt_conclusion=lambda e, rng: f"The {e['name']} is used to treat {rng.choice(['blindness', 'paralysis', 'insomnia', 'headaches'])} due to its red mineral base",
    coherent_conclusion=lambda e: f"The {e['name']} is used to treat blindness due to its red mineral base",
    alt_pool=_make_alt_pool(
        "The {name} is used to treat {fill} due to its red mineral base",
        ["blindness", "paralysis", "insomnia", "headaches", "joint pain",
         "memory loss", "skin rashes", "chronic fatigue", "bone fractures",
         "hearing loss", "tooth decay", "hair loss", "muscle spasms",
         "digestive problems", "anxiety disorders", "blood clotting"]),
    description="red potions cure fever",
)

# Rule 10: Potions with alcohol base are volatile
rule(
    condition=lambda e: e["type"] == "potion" and e["base"] == "alcohol",
    conclusion=lambda e: f"The {e['name']} is highly volatile and must be stored in sealed containers",
    alt_conclusion=lambda e, rng: f"The {e['name']} is {rng.choice(['extremely stable', 'slow to evaporate', 'non-reactive', 'self-preserving'])} and can be stored openly",
    coherent_conclusion=lambda e: f"The {e['name']} is extremely stable and can be stored openly",
    alt_pool=_make_alt_pool(
        "The {name} is {fill} and requires special handling",
        ["extremely stable", "slow to evaporate", "non-reactive",
         "self-preserving", "perfectly inert", "frost-resistant",
         "light-sensitive only", "pressure-dependent", "magnetically active",
         "acoustically resonant", "electrically charged", "biologically neutral",
         "thermally insulating", "gravitationally anomalous", "chemically bonded",
         "permanently crystallized"]),
    description="alcohol-based potions are volatile",
)

# Rule 11: Large animals are apex predators
rule(
    condition=lambda e: e["type"] == "animal" and e["size"] in ("large", "massive"),
    conclusion=lambda e: f"The {e['name']} is an apex predator in its ecosystem due to its {e['size']} size",
    alt_conclusion=lambda e, rng: f"The {e['name']} is a {rng.choice(['prey species', 'scavenger', 'herbivore', 'parasite'])} in its ecosystem despite its {e['size']} size",
    coherent_conclusion=lambda e: f"The {e['name']} is a prey species in its ecosystem despite its {e['size']} size",
    alt_pool=_make_alt_pool(
        "The {name} is a {fill} in its ecosystem despite its {size} size",
        ["prey species", "scavenger", "herbivore", "parasite",
         "bottom feeder", "filter feeder", "decomposer", "pollinator",
         "seed disperser", "symbiotic partner", "nocturnal forager",
         "burrowing rodent", "migratory grazer", "ambush predator",
         "colonial insect", "solitary browser"]),
    description="large animals are apex predators",
)

# Rule 12: Smooth-textured potions are easy to digest
rule(
    condition=lambda e: e["type"] == "potion" and e["texture"] == "smooth",
    conclusion=lambda e: f"The {e['name']} is easy to digest because of its smooth consistency",
    alt_conclusion=lambda e, rng: f"The {e['name']} causes {rng.choice(['nausea', 'stomach pain', 'dizziness', 'inflammation'])} because of its smooth consistency",
    coherent_conclusion=lambda e: f"The {e['name']} causes nausea because of its smooth consistency",
    alt_pool=_make_alt_pool(
        "The {name} causes {fill} because of its smooth consistency",
        ["nausea", "stomach pain", "dizziness", "inflammation",
         "temporary blindness", "numbness", "excessive sweating",
         "mild hallucinations", "intense drowsiness", "rapid heartbeat",
         "loss of appetite", "severe itching", "muscle weakness",
         "uncontrollable hiccups", "color blindness", "taste distortion"]),
    description="smooth potions are easy to digest",
)

# Rule 13: Crystalline minerals are transparent
rule(
    condition=lambda e: e["type"] == "mineral" and e["texture"] == "crystalline",
    conclusion=lambda e: f"The {e['name']} crystal is transparent and lets light pass through",
    alt_conclusion=lambda e, rng: f"The {e['name']} crystal is {rng.choice(['completely opaque', 'light-absorbing', 'mirror-like', 'dull'])} and blocks all light",
    coherent_conclusion=lambda e: f"The {e['name']} crystal is completely opaque and blocks all light",
    alt_pool=_make_alt_pool(
        "The {name} crystal is {fill} when examined under light",
        ["completely opaque", "light-absorbing", "mirror-like", "dull",
         "self-luminous", "color-shifting", "shadow-casting",
         "heat-emitting", "magnetically polarized", "sound-dampening",
         "electrically conductive", "frost-coated", "perpetually cloudy",
         "rainbow-refracting", "ultraviolet-glowing", "pitch-dark"]),
    description="crystalline minerals are transparent",
)

# Rule 14: Plants in forests have large leaves
rule(
    condition=lambda e: e["type"] == "plant" and e["habitat"] == "forests",
    conclusion=lambda e: f"The {e['name']} grows large leaves to capture sunlight under the forest canopy",
    alt_conclusion=lambda e, rng: f"The {e['name']} grows {rng.choice(['tiny needles', 'no leaves', 'spines', 'thin threads'])} under the forest canopy",
    coherent_conclusion=lambda e: f"The {e['name']} grows tiny needles under the forest canopy",
    alt_pool=_make_alt_pool(
        "The {name} grows {fill} under the forest canopy",
        ["tiny needles", "no leaves at all", "sharp spines", "thin threads",
         "massive broad fronds", "curled tendrils", "hollow tubes",
         "flat scales", "waxy pads", "feathery plumes", "rigid blades",
         "spherical bulbs", "hanging vines", "coiled springs",
         "transparent membranes", "dense clusters of thorns"]),
    description="forest plants have large leaves",
)

# Rule 15: Tiny animals are nocturnal
rule(
    condition=lambda e: e["type"] == "animal" and e["size"] == "tiny",
    conclusion=lambda e: f"The {e['name']} is nocturnal, hunting at night to avoid larger predators",
    alt_conclusion=lambda e, rng: f"The {e['name']} is {rng.choice(['diurnal', 'active at dawn', 'crepuscular', 'always active'])} with no fear of larger predators",
    coherent_conclusion=lambda e: f"The {e['name']} is diurnal with no fear of larger predators",
    alt_pool=_make_alt_pool(
        "The {name} is {fill} with no fear of larger predators",
        ["diurnal", "active at dawn", "crepuscular", "always active",
         "strictly midday-active", "active only at dusk", "seasonally dormant",
         "active during storms", "migratory and restless", "perpetually alert",
         "active only in rain", "completely sedentary", "active at high tide",
         "triggered by moonlight", "active during eclipses",
         "responsive only to temperature"]),
    description="tiny animals are nocturnal",
)


# ---------------------------------------------------------------------------
# Text generation: compose entity description + observation
# ---------------------------------------------------------------------------

ENTITY_INTROS = {
    "animal": [
        "The {name} is a {animal_type} found in the {region} {habitat}.",
        "Researchers discovered the {name}, a {animal_type}, living in the {region} {habitat}.",
        "The {name} is a {size} {animal_type} native to the {region} {habitat}.",
        "Field surveys identified the {name}, a {animal_type} of {size} build, in the {region} {habitat}.",
    ],
    "plant": [
        "The {name} is a plant species growing in the {region} {habitat}.",
        "Botanists cataloged the {name}, a {size} plant from the {region} {habitat}.",
        "The {name} is a {color} plant found across the {region} {habitat}.",
        "Field teams collected the {name}, a {texture} plant from the {region} {habitat}.",
    ],
    "mineral": [
        "The {name} is a mineral found in the {region} {habitat}.",
        "Geologists identified the {name}, a {texture} mineral with high {content} content.",
        "The {name} is a {hardness} mineral extracted from the {region} {habitat}.",
        "Survey teams discovered {name} deposits in the {region} {habitat}.",
    ],
    "potion": [
        "The {name} is a {color} potion prepared with a {base} base.",
        "Alchemists developed the {name}, a {texture} {color} potion using {base}.",
        "The {name} is brewed from {base} and appears {color} in color.",
        "The {name} potion has a {taste} taste and is made from a {base} base.",
    ],
}

PROPERTY_SENTENCES = {
    "animal": [
        "It has a {color} {texture} hide.",
        "Its body temperature runs {temperature}.",
        "It has a {density} body structure.",
        "The species is {size} in stature.",
    ],
    "plant": [
        "Its {plant_part} are {color} and {texture}.",
        "It thrives in {temperature} conditions.",
        "The plant has a {density} stem.",
        "It is a {size} species.",
    ],
    "mineral": [
        "It appears {color} with a {texture} surface.",
        "Its density is classified as {density}.",
        "It forms under {temperature} geological conditions.",
        "The mineral is {size} in typical specimens.",
    ],
    "potion": [
        "It has a {texture} consistency.",
        "The mixture is {density} when prepared.",
        "It is stored at {temperature} temperatures.",
        "A typical dose is {size} in volume.",
    ],
}


def render_entity_text(entity, rng):
    """Render 2-3 descriptive sentences about an entity."""
    etype = entity["type"]

    # Intro sentence
    template = rng.choice(ENTITY_INTROS[etype])
    intro = template.format(**entity)

    # 1-2 property sentences
    prop_templates = list(PROPERTY_SENTENCES[etype])
    rng.shuffle(prop_templates)
    n_props = rng.randint(1, 2)
    props = []
    for t in prop_templates[:n_props]:
        try:
            props.append(t.format(**entity))
        except KeyError:
            continue

    return intro + " " + " ".join(props)


def find_applicable_rules(entity):
    """Find all rules whose condition matches this entity."""
    applicable = []
    for i, r in enumerate(RULES):
        if r["condition"](entity):
            applicable.append((i, r))
    return applicable


def generate_example(entity, rule_idx, rule, rng, error_mode="correct",
                     selected_alts=None):
    """Generate one text example for an entity + rule.

    error_mode:
        'correct' - conclusion follows from rule
        'random' - random wrong conclusion
        'coherent' - systematically wrong conclusion
        'contradictory' - one of two contradictory wrong conclusions (legacy, uses first 2 from alt_pool)
        'multi_alt' - one of N pre-selected alternatives from alt_pool
    selected_alts: dict mapping rule_idx -> list of selected alt lambdas (for multi_alt mode)
    """
    desc = render_entity_text(entity, rng)

    if error_mode == "correct":
        conclusion = rule["conclusion"](entity)
        is_correct = True
    elif error_mode == "random":
        conclusion = rule["alt_conclusion"](entity, rng)
        is_correct = False
    elif error_mode == "coherent":
        conclusion = rule["coherent_conclusion"](entity)
        is_correct = False
    elif error_mode == "contradictory":
        # Legacy: use first 2 from alt_pool
        pool = rule["alt_pool"][:2]
        conclusion = rng.choice(pool)(entity)
        is_correct = False
    elif error_mode == "multi_alt":
        alts = selected_alts[rule_idx]
        conclusion = rng.choice(alts)(entity)
        is_correct = False
    else:
        raise ValueError(f"Unknown error_mode: {error_mode}")

    text = f"{desc} {conclusion}."
    return text, is_correct


# ---------------------------------------------------------------------------
# Corpus generation
# ---------------------------------------------------------------------------

def generate_corpus(n_examples, correct_ratio, seed=42, output_path=None,
                    error_mode="random", n_entities=50, n_alternatives=2):
    """Generate a corpus of synthetic world examples.

    Args:
        n_examples: total number of examples to generate
        correct_ratio: fraction that should be correct (0.0-1.0)
        seed: random seed
        output_path: if provided, write to file
        error_mode: 'random', 'coherent', 'contradictory', or 'multi_alt'
        n_entities: number of entities in the world
        n_alternatives: for 'multi_alt' mode, how many alternatives per rule
    """
    rng = random.Random(seed)

    # Generate world
    entities = generate_entities(rng, n_entities)

    # Precompute applicable rules for each entity
    entity_rules = []
    for e in entities:
        applicable = find_applicable_rules(e)
        if applicable:
            entity_rules.append((e, applicable))

    if not entity_rules:
        raise RuntimeError("No entities matched any rules. Check rule conditions.")

    # For multi_alt mode: pre-select N alternatives per rule from pool
    # Uses fixed seed=7777 so training and test share the same selection
    selected_alts = None
    if error_mode == "multi_alt":
        selected_alts = {}
        alt_rng = random.Random(7777)  # fixed seed for reproducible selection
        for i, r in enumerate(RULES):
            pool = r["alt_pool"]
            if n_alternatives > len(pool):
                raise ValueError(
                    f"Rule '{r['description']}': requested {n_alternatives} "
                    f"alternatives but pool has only {len(pool)}")
            selected_alts[i] = alt_rng.sample(pool, n_alternatives)
        print(f"Multi-alt mode: {n_alternatives} alternatives per rule "
              f"(pool size: {len(RULES[0]['alt_pool'])})")

    print(f"World: {n_entities} entities, {len(entity_rules)} with applicable rules")
    print(f"Rules: {len(RULES)} total")

    problems = []
    stats = {"correct": 0, "incorrect": 0, "by_type": {}, "by_rule": {}}

    for i in range(n_examples):
        # Pick random entity with applicable rules
        entity, applicable = rng.choice(entity_rules)
        rule_idx, rule_obj = rng.choice(applicable)

        is_correct_target = rng.random() < correct_ratio

        if is_correct_target:
            text, is_correct = generate_example(entity, rule_idx, rule_obj, rng, "correct")
        else:
            text, is_correct = generate_example(
                entity, rule_idx, rule_obj, rng, error_mode,
                selected_alts=selected_alts)

        etype = entity["type"]
        problems.append({
            "text": text,
            "is_correct": is_correct,
            "type": etype,
            "rule": rule_obj["description"],
            "entity": entity["name"],
            "id": i,
        })

        if is_correct:
            stats["correct"] += 1
        else:
            stats["incorrect"] += 1
        stats["by_type"][etype] = stats["by_type"].get(etype, 0) + 1
        rdesc = rule_obj["description"]
        stats["by_rule"][rdesc] = stats["by_rule"].get(rdesc, 0) + 1

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            for p in problems:
                f.write(p["text"] + "\n\n")

        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            meta = {
                "n_examples": n_examples,
                "correct_ratio": correct_ratio,
                "seed": seed,
                "error_mode": error_mode,
                "n_entities": n_entities,
                "n_rules": len(RULES),
                "stats": stats,
            }
            if error_mode == "multi_alt":
                meta["n_alternatives"] = n_alternatives
            json.dump(meta, f, indent=2)

        print(f"Generated {n_examples} examples "
              f"({stats['correct']} correct, {stats['incorrect']} incorrect)")
        print(f"Entity types: {stats['by_type']}")
        print(f"Written to {path} ({path.stat().st_size / 1024:.1f} KB)")

    return problems


# ---------------------------------------------------------------------------
# Additional: generate separate correct/incorrect test corpora
# ---------------------------------------------------------------------------

def generate_test_corpora(n_examples, seed=999, output_dir="data/corpus",
                          error_mode="random", n_entities=50, n_alternatives=2):
    """Generate separate correct and incorrect test corpora for corpus-level eval."""
    rng = random.Random(seed)
    entities = generate_entities(rng, n_entities)

    entity_rules = []
    for e in entities:
        applicable = find_applicable_rules(e)
        if applicable:
            entity_rules.append((e, applicable))

    # For multi_alt: pre-select alternatives (fixed seed, same as generate_corpus)
    selected_alts = None
    if error_mode == "multi_alt":
        selected_alts = {}
        alt_rng = random.Random(7777)
        for i, r in enumerate(RULES):
            selected_alts[i] = alt_rng.sample(r["alt_pool"], n_alternatives)

    for mode_label, actual_mode in [("correct", "correct"), ("incorrect", error_mode)]:
        texts = []
        for i in range(n_examples):
            entity, applicable = rng.choice(entity_rules)
            rule_idx, rule_obj = rng.choice(applicable)
            text, _ = generate_example(entity, rule_idx, rule_obj, rng, actual_mode,
                                       selected_alts=selected_alts)
            texts.append(text)

        suffix = f"_world_{error_mode}" if mode_label == "incorrect" else "_world"
        fname = f"test_{mode_label}{suffix}.txt"
        path = Path(output_dir) / fname
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for t in texts:
                f.write(t + "\n\n")
        print(f"Test {mode_label}: {n_examples} examples -> {path} "
              f"({path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic world corpus")
    parser.add_argument("--n", type=int, default=100000,
                        help="Number of examples")
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Correct ratio (0.0-1.0). 1.0 = all correct")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output", type=str, default="data/corpus/train_world.txt",
                        help="Output path")
    parser.add_argument("--error-mode", type=str, default="random",
                        choices=["random", "coherent", "contradictory", "multi_alt"],
                        help="Error type: 'random', 'coherent', 'contradictory', or 'multi_alt'")
    parser.add_argument("--n-alternatives", type=int, default=2,
                        help="For multi_alt mode: number of alternatives per rule (max 16)")
    parser.add_argument("--n-entities", type=int, default=50,
                        help="Number of entities in the world")
    parser.add_argument("--gen-test", action="store_true",
                        help="Also generate test corpora (correct + incorrect)")
    parser.add_argument("--test-n", type=int, default=5000,
                        help="Number of test examples per corpus")
    parser.add_argument("--test-seed", type=int, default=999,
                        help="Seed for test generation")
    args = parser.parse_args()

    generate_corpus(args.n, args.ratio, args.seed, args.output,
                    error_mode=args.error_mode, n_entities=args.n_entities,
                    n_alternatives=args.n_alternatives)

    if args.gen_test:
        generate_test_corpora(args.test_n, seed=args.test_seed,
                              output_dir=str(Path(args.output).parent),
                              error_mode=args.error_mode,
                              n_entities=args.n_entities,
                              n_alternatives=args.n_alternatives)
