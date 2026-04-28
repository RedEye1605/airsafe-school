import json
from pathlib import Path

import pytest

from src.recommendations.engine import (
    bmkg_to_action,
    from_prediction_row,
    generate_recommendation,
    worst_risk,
)
from src.recommendations.quality import validate_recommendation

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestBmkgToAction:
    @pytest.mark.parametrize(
        "bmkg, action",
        [
            ("BAIK", "Aman"),
            ("SEDANG", "Waspada"),
            ("TIDAK SEHAT", "Batasi"),
            ("SANGAT TIDAK SEHAT", "Bahaya"),
            ("BERBAHAYA", "Bahaya"),
            ("TIDAK ADA DATA", "Tidak Ada Data"),
        ],
    )
    def test_mapping(self, bmkg, action):
        assert bmkg_to_action(bmkg) == action

    def test_unknown_passthrough(self):
        assert bmkg_to_action("UNKNOWN") == "UNKNOWN"


class TestWorstRisk:
    def test_picks_highest(self):
        assert worst_risk("BAIK", "TIDAK SEHAT", "SEDANG") == "Batasi"

    def test_all_same(self):
        assert worst_risk("SEDANG", "SEDANG") == "Waspada"

    def test_empty(self):
        assert worst_risk() == "Tidak Ada Data"

    def test_bahaya_beats_all(self):
        assert worst_risk("BAIK", "BERBAHAYA") == "Bahaya"


class TestFromPredictionRow:
    @pytest.fixture
    def prediction_row(self):
        return {
            "npsn": "12345",
            "latitude": -6.2,
            "longitude": 106.8,
            "pm25_h6": 64.0,
            "pm25_h12": 72.0,
            "pm25_h24": 58.0,
            "risk_h6": "TIDAK SEHAT",
            "risk_h12": "TIDAK SEHAT",
            "risk_h24": "SEDANG",
        }

    def test_adapts_fields(self, prediction_row):
        result = from_prediction_row(prediction_row, school_name="SDN Test", district="Menteng")
        assert result["school_id"] == "12345"
        assert result["school_name"] == "SDN Test"
        assert result["pm25_6h"] == 64.0
        assert result["risk_level"] == "Batasi"  # worst of TIDAK SEHAT, TIDAK SEHAT, SEDANG

    def test_falls_back_to_npsn_for_name(self):
        row = {"npsn": "99999", "pm25_h6": 10, "risk_h6": "BAIK", "risk_h12": "BAIK", "risk_h24": "BAIK"}
        result = from_prediction_row(row)
        assert result["school_name"] == "99999"
        assert result["school_id"] == "99999"

    def test_missing_fields_graceful(self):
        result = from_prediction_row({})
        assert result["school_id"] == ""
        assert result["risk_level"] == "Tidak Ada Data"


class TestGenerateRecommendation:
    @pytest.fixture
    def sample_input(self):
        return {
            "school_id": "T001",
            "school_name": "SDN Test 01",
            "district": "Menteng",
            "pm25_6h": 45.0,
            "pm25_12h": 50.0,
            "pm25_24h": 42.0,
            "risk_level": "SEDANG",
            "top_shap_factors": [
                {"feature": "humidity", "direction": "increase", "reason": "kelembapan tinggi"},
            ],
        }

    def test_returns_all_keys(self, sample_input):
        out = generate_recommendation(sample_input)
        for key in [
            "school_id", "school_name", "district", "risk_level",
            "pm25_summary", "headline", "recommendation",
            "action_items", "reasoning_summary", "parent_message",
            "generation_mode",
        ]:
            assert key in out

    def test_preserves_school_and_maps_risk(self, sample_input):
        out = generate_recommendation(sample_input)
        assert out["school_name"] == "SDN Test 01"
        assert out["risk_level"] == "Waspada"

    def test_min_three_action_items(self, sample_input):
        out = generate_recommendation(sample_input)
        assert len(out["action_items"]) >= 3

    def test_fallback_mode(self, sample_input):
        assert generate_recommendation(sample_input)["generation_mode"] == "template_fallback"

    def test_pm25_in_summary(self, sample_input):
        text = generate_recommendation(sample_input)["pm25_summary"]
        assert "45.0" in text
        assert "50.0" in text

    def test_shap_in_reasoning(self, sample_input):
        assert "kelembapan tinggi" in generate_recommendation(sample_input)["reasoning_summary"]

    def test_no_shap_graceful(self):
        out = generate_recommendation({
            "school_id": "T002", "school_name": "SDN X", "district": "D",
            "pm25_6h": 10.0, "pm25_12h": 12.0, "pm25_24h": 11.0,
            "risk_level": "BAIK", "top_shap_factors": [],
        })
        assert "Tidak ada informasi" in out["reasoning_summary"]


class TestRiskSpecificContent:
    def test_bahaya_cancels_outdoor(self):
        out = generate_recommendation({
            "school_id": "X", "school_name": "SDN X", "district": "D",
            "pm25_6h": 160, "pm25_12h": 165, "pm25_24h": 140,
            "risk_level": "SANGAT TIDAK SEHAT", "top_shap_factors": [],
        })
        combined = " ".join(out["action_items"]).lower()
        assert "batalkan" in combined
        assert "luar ruang" in combined

    def test_batasi_moves_indoors(self):
        out = generate_recommendation({
            "school_id": "X", "school_name": "SDN X", "district": "D",
            "pm25_6h": 65, "pm25_12h": 70, "pm25_24h": 60,
            "risk_level": "TIDAK SEHAT", "top_shap_factors": [],
        })
        combined = " ".join(out["action_items"]).lower()
        assert "dalam ruangan" in combined

    def test_aman_allows_normal(self):
        out = generate_recommendation({
            "school_id": "X", "school_name": "SDN X", "district": "D",
            "pm25_6h": 10, "pm25_12h": 12, "pm25_24h": 11,
            "risk_level": "BAIK", "top_shap_factors": [],
        })
        assert out["risk_level"] == "Aman"
        assert "normal" in " ".join(out["action_items"]).lower()


class TestQualityValidation:
    def test_passes_10_school_dataset(self):
        path = _PROJECT_ROOT / "data" / "test" / "recommendation_test_10_schools.json"
        with open(path) as f:
            schools = json.load(f)
        for school in schools:
            rec = generate_recommendation(school)
            q = validate_recommendation(school, rec)
            assert q["quality_pass"], f"{school['school_id']}: {q['notes']}"

    def test_fails_on_missing_key(self):
        rec = generate_recommendation({
            "school_id": "X", "school_name": "SDN", "district": "D",
            "pm25_6h": 30, "pm25_12h": 35, "pm25_24h": 32,
            "risk_level": "SEDANG", "top_shap_factors": [],
        })
        del rec["recommendation"]
        q = validate_recommendation({}, rec)
        assert not q["valid_json_like"]
        assert not q["quality_pass"]

    def test_fails_on_medical_terms(self):
        rec = generate_recommendation({
            "school_id": "X", "school_name": "SDN", "district": "D",
            "pm25_6h": 30, "pm25_12h": 35, "pm25_24h": 32,
            "risk_level": "SEDANG", "top_shap_factors": [],
        })
        rec["recommendation"] = "Segera lakukan diagnosis dan berikan obat."
        q = validate_recommendation({
            "school_name": "SDN", "risk_level": "SEDANG",
            "pm25_6h": 30, "pm25_12h": 35, "pm25_24h": 32,
        }, rec)
        assert not q["no_medical_diagnosis"]
