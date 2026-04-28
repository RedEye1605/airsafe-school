import csv
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.recommendations.engine import generate_recommendation
from src.recommendations.quality import validate_recommendation


def main():
    test_data_path = _PROJECT_ROOT / "data" / "test" / "recommendation_test_10_schools.json"
    with open(test_data_path) as f:
        schools = json.load(f)

    results = []
    quality_rows = []
    passed = 0

    for school in schools:
        rec = generate_recommendation(school)
        quality = validate_recommendation(school, rec)

        results.append({
            "school_id": school["school_id"],
            "input": school,
            "recommendation": rec,
            "quality": quality,
        })

        quality_rows.append({
            "school_id": school["school_id"],
            "school_name": school["school_name"],
            "risk_level": rec.get("risk_level", ""),
            **quality,
            "notes": "; ".join(quality.get("notes", [])),
        })

        if quality["quality_pass"]:
            passed += 1

    results_path = _PROJECT_ROOT / "data" / "test" / "recommendation_test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved results -> {results_path}")

    csv_path = _PROJECT_ROOT / "data" / "test" / "recommendation_quality_10_schools.csv"
    if quality_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=quality_rows[0].keys())
            writer.writeheader()
            writer.writerows(quality_rows)
    print(f"Saved quality CSV -> {csv_path}")

    total = len(schools)
    print(f"\nRecommendation quality check: {passed}/{total} passed")
    if passed < total:
        for r in results:
            if not r["quality"]["quality_pass"]:
                print(f"  FAIL: {r['school_id']} - {r['quality']['notes']}")
        sys.exit(1)
    print("All recommendations passed quality checks.")


if __name__ == "__main__":
    main()
