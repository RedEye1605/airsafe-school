from __future__ import annotations

from src.recommendations.engine import REQUIRED_OUTPUT_KEYS, bmkg_to_action

_MEDICAL_TERMS = {"diagnosis", "obat", "terapi medis", "menyembuhkan", "penyakit pasti"}


def validate_recommendation(input_data: dict, output_data: dict) -> dict:
    notes: list[str] = []

    missing_keys = [k for k in REQUIRED_OUTPUT_KEYS if k not in output_data]
    valid_json_like = len(missing_keys) == 0
    if not valid_json_like:
        notes.append(f"Missing keys: {missing_keys}")

    school_name_correct = output_data.get("school_name") == input_data.get("school_name")
    if not school_name_correct:
        notes.append("school_name mismatch")

    expected_risk = bmkg_to_action(input_data.get("risk_level", ""))
    risk_level_correct = output_data.get("risk_level") == expected_risk
    if not risk_level_correct:
        notes.append(f"risk_level: expected {expected_risk}, got {output_data.get('risk_level')}")

    pm25_vals = [
        input_data.get("pm25_6h"),
        input_data.get("pm25_12h"),
        input_data.get("pm25_24h"),
    ]
    combined_text = f"{output_data.get('pm25_summary', '')} {output_data.get('recommendation', '')}"
    mentions_pm25 = all(
        v is None
        or str(round(float(v), 1)) in combined_text
        or str(int(float(v))) in combined_text
        for v in pm25_vals
    )
    if not mentions_pm25:
        notes.append("PM2.5 values not found in output text")

    action_items = output_data.get("action_items", [])
    has_action_items = isinstance(action_items, list)
    action_items_count = len(action_items) if has_action_items else 0
    if not has_action_items:
        notes.append("action_items is not a list")

    text_lower = " ".join(str(v) for v in output_data.values() if isinstance(v, str)).lower()
    found_medical = [t for t in _MEDICAL_TERMS if t in text_lower]
    no_medical_diagnosis = len(found_medical) == 0
    if not no_medical_diagnosis:
        notes.append(f"Found medical terms: {found_medical}")

    quality_pass = (
        valid_json_like
        and school_name_correct
        and risk_level_correct
        and mentions_pm25
        and has_action_items
        and action_items_count >= 3
        and no_medical_diagnosis
    )

    return {
        "valid_json_like": valid_json_like,
        "school_name_correct": school_name_correct,
        "risk_level_correct": risk_level_correct,
        "mentions_pm25": mentions_pm25,
        "has_action_items": has_action_items,
        "action_items_count": action_items_count,
        "no_medical_diagnosis": no_medical_diagnosis,
        "quality_pass": quality_pass,
        "notes": notes,
    }
