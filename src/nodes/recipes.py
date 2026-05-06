"""Recipes node — Sundays only, high-protein, weekly variety.

Reads the last ~12 recipe titles from history and tells the model to avoid
them, then generates 4 new recipes with varied protein sources. Persists
back to data/recipes_history.json.
"""
from __future__ import annotations
from typing import Any

from ..llm import get_llm
from ..persistence import load_history, save_history


HISTORY_FILE = "data/recipes_history.json"

SYSTEM_PROMPT = """You generate weekly meal ideas for a home cook in Charlotte NC.

Hard constraints:
  - Exactly 4 recipes.
  - Each recipe is high-protein (>= 35g protein per serving).
  - Vary protein sources across the 4: one chicken/turkey, one fish/seafood,
    one beef/pork, one plant or eggs. Adjust if needed but keep variety.
  - Avoid any titles in the AVOID list (the user has had them recently).
  - Practical weeknight prep — most under 45 minutes.
  - Whole-food, nutrient-dense ingredients. No protein powder shakes.

Output schema (JSON only, no prose):
{"recipes": [{
  "title": str,
  "protein_source": str,
  "ingredients": [str, ...],
  "instructions": [str, ...],
  "prep_minutes": int,
  "macros": {"calories": int, "protein_g": int, "carbs_g": int, "fat_g": int}
}]}
"""


def recipes_node(state: dict) -> dict:
    if state.get("day_of_week") != "Sunday":
        return {"recipes": None, "errors": []}

    history = load_history(HISTORY_FILE)
    recent_titles = [r.get("title", "") for r in history[-12:] if r.get("title")]

    schema = {
        "title": "RecipeBatch",
        "type": "object",
        "properties": {
            "recipes": {
                "type": "array",
                "minItems": 4,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "protein_source": {"type": "string"},
                        "ingredients": {"type": "array", "items": {"type": "string"}},
                        "instructions": {"type": "array", "items": {"type": "string"}},
                        "prep_minutes": {"type": "integer"},
                        "macros": {
                            "type": "object",
                            "properties": {
                                "calories": {"type": "integer"},
                                "protein_g": {"type": "integer"},
                                "carbs_g": {"type": "integer"},
                                "fat_g": {"type": "integer"},
                            },
                            "required": ["calories", "protein_g", "carbs_g", "fat_g"],
                        },
                    },
                    "required": [
                        "title",
                        "protein_source",
                        "ingredients",
                        "instructions",
                        "prep_minutes",
                        "macros",
                    ],
                },
            }
        },
        "required": ["recipes"],
    }

    avoid_text = (
        "AVOID these recently-served titles:\n- " + "\n- ".join(recent_titles)
        if recent_titles
        else "AVOID list is empty (first run)."
    )

    try:
        llm = get_llm(temperature=0.7)
        response = llm.with_structured_output(schema).invoke(
            f"{SYSTEM_PROMPT}\n\n{avoid_text}\n\nGenerate this week's 4 recipes."
        )
        recipes: list[dict[str, Any]] = response.get("recipes", [])
    except Exception as e:
        return {"recipes": [], "errors": [f"Recipe generation failed: {e}"]}

    new_history = history + [{"title": r.get("title", ""), "date": state.get("date")} for r in recipes]
    try:
        save_history(HISTORY_FILE, new_history, max_entries=200)
    except Exception as e:
        return {"recipes": recipes, "errors": [f"Recipe history save failed: {e}"]}

    return {"recipes": recipes, "errors": []}
