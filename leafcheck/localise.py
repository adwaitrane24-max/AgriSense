"""Add language-specific fields to a prediction result.

The cached result is language-neutral (class labels and numbers only), so the
same cache entry serves every language. This runs after the cache, per request:

    display_crop / display_condition  on prediction and each top-3 item
    advisory                          farmer guidance for a confident result
    lang                              the language used

Complexity: O(k) for the k = 3 top items; each lookup is an O(1) dict access.
"""

from typing import Any

from leafcheck.advisory import display_name, get_advisory, normalise_lang


def _with_names(item: dict[str, Any], lang: str) -> dict[str, Any]:
    """Copy of one class entry with translated crop and condition names."""
    crop, condition = display_name(item["label"], lang)
    return {**item, "display_crop": crop, "display_condition": condition}


def localise_result(result: dict[str, Any], lang: str | None) -> dict[str, Any]:
    """Return a copy of `result` with translated names and farmer guidance.

    The input dict is never modified, so cached results stay unchanged.
    Guidance is only attached when the model is confident ("ok"); a
    "not_sure" result gets `advisory = None`.
    """
    lang = normalise_lang(lang)
    prediction = result.get("prediction")
    return {
        **result,
        "lang": lang,
        "top": [_with_names(item, lang) for item in result.get("top", [])],
        "prediction": _with_names(prediction, lang) if prediction else None,
        "advisory": get_advisory(prediction["label"], lang) if prediction else None,
    }
