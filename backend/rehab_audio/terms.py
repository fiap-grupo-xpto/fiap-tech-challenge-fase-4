from __future__ import annotations

import re
import unicodedata


TERM_CATEGORIES: dict[str, list[str]] = {
    "sintoma": [
        "dor",
        "cansaco",
        "falta de ar",
        "tontura",
        "formigamento",
        "chiado",
        "fadiga",
        "exaustao",
    ],
    "esforco": [
        "dificil",
        "nao consigo",
        "pesado",
        "muito forte",
        "sem forca",
        "nao aguento",
    ],
    "escalada": [
        "pare",
        "ajuda",
        "chama",
        "emergencia",
        "desmaio",
        "socorro",
    ],
    "instrucao": [
        "repita",
        "inspire",
        "expire",
        "descanse",
        "amplitude",
        "serie",
        "alonga",
        "movimento",
    ],
}

RISK_CATEGORIES = {"sintoma", "esforco", "escalada"}


def normalize_text(text: str) -> str:
    """Return lowercase text without accents (matching friendly)."""
    decomposed = unicodedata.normalize("NFD", str(text).lower())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _count_occurrences(term: str, normalized_text: str) -> int:
    """Count whole-word occurrences, avoiding matches inside longer words."""
    return len(re.findall(rf"\b{re.escape(term)}\b", normalized_text))


def extract_critical_terms(text: str) -> list[dict]:
    """Count critical rehabilitation terms found in the normalized transcript."""
    normalized = normalize_text(text)
    if not normalized.strip():
        return []

    results: list[dict] = []
    for category, terms in TERM_CATEGORIES.items():
        severity = "warning" if category in RISK_CATEGORIES else "info"
        for term in terms:
            count = _count_occurrences(term, normalized)
            if count == 0:
                continue
            results.append(
                {
                    "term": term,
                    "category": category,
                    "count": count,
                    "severity": severity,
                }
            )

    results.sort(key=lambda item: (-item["count"], item["term"]))
    return results
