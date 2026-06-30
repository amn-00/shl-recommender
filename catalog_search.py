"""
Catalog loading and retrieval for the SHL Assessment Recommender.

Retrieval strategy: lightweight TF-IDF style keyword scoring over assessment
name + test-type category labels. Justification (see approach doc): with
~80-400 catalog items, a full vector DB is overkill -- token-overlap scoring
plus test-type filtering gets strong recall at near-zero latency and zero
external embedding API cost, which matters given the 30s-per-call timeout
and free-tier LLM budget. This is a deliberate, defensible trade-off.
"""

import json
import re
from pathlib import Path
from typing import Optional

CATALOG_PATH = Path(__file__).parent / "catalog.json"

TYPE_LEGEND = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}

# Lightweight synonym map to bridge natural-language job terms to catalog
# vocabulary (catalog item names are technical/product names, not job-task
# language, so naive overlap alone under-recalls on conversational queries).
SYNONYMS = {
    "java": ["java", "core java", "j2ee", "spring"],
    "python": ["python"],
    "javascript": ["javascript", "js", "angularjs", "angular", "react", "node", "css", "html"],
    "developer": ["programming", "development", "engineer", "software"],
    "stakeholder": ["communication", "business communications"],
    "communication": ["communication", "business communications"],
    "leadership": ["leadership", "manager", "management"],
    "manager": ["manager", "management", "leadership", "supervisor"],
    "personality": ["personality", "behavior", "behaviour"],
    "cognitive": ["ability", "aptitude", "reasoning"],
    "customer service": ["customer service", "phone", "call center", "contact center"],
    "sql": ["sql"],
    "data": ["data", "data science", "database"],
    "cloud": ["cloud", "aws", "azure"],
    "engineer": ["engineering", "engineer"],
    "entry level": ["entry level", "entry-level", "graduate"],
    "senior": ["advanced", "senior"],
}


def load_catalog() -> list[dict]:
    with open(CATALOG_PATH) as f:
        return json.load(f)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9+#.]+", text.lower()))


def _expand_query_tokens(tokens: set[str]) -> set[str]:
    expanded = set(tokens)
    joined = " ".join(tokens)
    for key, syns in SYNONYMS.items():
        if key in joined:
            for s in syns:
                expanded.update(_tokenize(s))
    return expanded


def search_catalog(
    query: str,
    test_types: Optional[list[str]] = None,
    top_k: int = 10,
) -> list[dict]:
    """
    Score catalog items by token overlap with the query (name field),
    optionally filtered/boosted by test_type codes (e.g. ['K'] for
    knowledge/skills, ['P'] for personality).
    Returns top_k items with a relevance score attached.
    """
    catalog = load_catalog()
    query_tokens = _expand_query_tokens(_tokenize(query))

    scored = []
    for item in catalog:
        name_tokens = _tokenize(item["name"])
        overlap = len(query_tokens & name_tokens)

        # small boost for substring containment (handles multi-word product names)
        if any(qt in item["name"].lower() for qt in query.lower().split() if len(qt) > 3):
            overlap += 1

        type_match_bonus = 0
        if test_types:
            item_types = set(item.get("test_type", []))
            if item_types & set(test_types):
                type_match_bonus = 2

        score = overlap + type_match_bonus
        if score > 0:
            scored.append({**item, "_score": score})

    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored[:top_k]


def get_by_name(name: str) -> Optional[dict]:
    catalog = load_catalog()
    name_lower = name.lower().strip()
    for item in catalog:
        if item["name"].lower() == name_lower:
            return item
    # fallback: substring match
    for item in catalog:
        if name_lower in item["name"].lower() or item["name"].lower() in name_lower:
            return item
    return None


def catalog_summary_for_prompt(max_items: int = 80) -> str:
    """Compact representation of the catalog for inclusion in LLM context."""
    catalog = load_catalog()[:max_items]
    lines = []
    for item in catalog:
        types = ",".join(item.get("test_type", []))
        lines.append(f"- {item['name']} [{types}] ({item['url']})")
    return "\n".join(lines)


if __name__ == "__main__":
    results = search_catalog("Java developer with stakeholder communication", test_types=["K", "P"])
    for r in results:
        print(r["_score"], r["name"], r.get("test_type"))
