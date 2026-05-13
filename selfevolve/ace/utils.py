"""Minimal stub of the ACE utils module.

The upstream RESD code only imports ``get_section_slug`` from
``selfevolve.ace.utils``. The full ACE codebase is hosted at
https://github.com/ace-agent/ace; rather than vendoring the whole thing
(and pulling in openai/tiktoken/dotenv/etc.), this module reproduces the
single function that RESD imports verbatim from
https://raw.githubusercontent.com/ace-agent/ace/main/utils.py.

Note: ``selfevolve.ace.data/`` (the FiNER JSONL files) is staged
separately at runtime by fetching the same files from the ACE repo.
"""


def get_section_slug(section_name: str) -> str:
    """Convert section name to slug format (3-5 chars).

    Verbatim from ace-agent/ace ``utils.py``.
    """
    slug_map = {
        "financial_strategies_and_insights": "fin",
        "formulas_and_calculations": "calc",
        "code_snippets_and_templates": "code",
        "common_mistakes_to_avoid": "err",
        "problem_solving_heuristics": "prob",
        "context_clues_and_indicators": "ctx",
        "others": "misc",
        "meta_strategies": "meta",
    }

    clean_name = section_name.lower().strip().replace(" ", "_").replace("&", "and")

    if clean_name in slug_map:
        return slug_map[clean_name]

    words = clean_name.split("_")
    if len(words) == 1:
        return words[0][:4]
    return "".join(w[0] for w in words[:5])
