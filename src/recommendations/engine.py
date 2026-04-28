from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)

# ── Risk mapping ───────────────────────────────────────────────────────────

_BMKG_TO_ACTION: dict[str, str] = {
    "BAIK": "Aman",
    "SEDANG": "Waspada",
    "TIDAK SEHAT": "Batasi",
    "SANGAT TIDAK SEHAT": "Bahaya",
    "BERBAHAYA": "Bahaya",
    "TIDAK ADA DATA": "Tidak Ada Data",
}

_RISK_PRIORITY = {"Bahaya": 4, "Batasi": 3, "Waspada": 2, "Aman": 1, "Tidak Ada Data": 0}


def bmkg_to_action(bmkg_label: str) -> str:
    return _BMKG_TO_ACTION.get(bmkg_label, bmkg_label)


def worst_risk(*labels: str) -> str:
    """Return the highest-severity action label from multiple BMKG labels."""
    action_labels = [bmkg_to_action(l) for l in labels if l]
    if not action_labels:
        return "Tidak Ada Data"
    return max(action_labels, key=lambda l: _RISK_PRIORITY.get(l, 0))


# ── Template policies (fallback when LLM unavailable) ──────────────────────

# ACTION_POLICIES kept for template fallback. When Gemini is available these
# are not used but remain here for offline/failure scenarios.

ACTION_POLICIES: dict[str, dict[str, list[str] | str]] = {
    "Aman": {
        "headline": "Kualitas udara relatif aman untuk aktivitas sekolah",
        "action_items": [
            "Aktivitas luar ruang dapat berjalan normal.",
            "Tetap pantau perubahan kualitas udara secara berkala.",
            "Pertahankan praktik ventilasi ruang kelas yang baik.",
        ],
        "parent_template": (
            "Kualitas udara di sekitar {school_name} dalam kondisi aman. "
            "Aktivitas sekolah berjalan normal hari ini."
        ),
    },
    "Waspada": {
        "headline": "Waspada kualitas udara, kurangi paparan luar ruang yang terlalu lama",
        "action_items": [
            "Kurangi durasi aktivitas luar ruang yang panjang.",
            "Pantau siswa yang memiliki sensitivitas pernapasan.",
            "Siapkan alternatif kegiatan di dalam ruangan jika kondisi memburuk.",
        ],
        "parent_template": (
            "Kualitas udara di sekitar {school_name} dalam tingkat waspada. "
            "Sekolah telah mengurangi aktivitas luar ruang sebagai langkah berjaga-jaga."
        ),
    },
    "Batasi": {
        "headline": "Batasi aktivitas luar ruang dan pindahkan olahraga ke dalam ruangan",
        "action_items": [
            "Pindahkan kegiatan olahraga dan aktivitas luar ruang ke dalam ruangan.",
            "Hindari upacara bendera atau kegiatan berkumpul di luar ruangan.",
            "Siapkan masker untuk siswa yang sensitif, terutama saat perjalanan ke/dari sekolah.",
        ],
        "parent_template": (
            "Kualitas udara di sekitar {school_name} dalam kondisi tidak sehat. "
            "Sekolah membatasi aktivitas luar ruang dan memindahkan kegiatan ke dalam ruangan."
        ),
    },
    "Bahaya": {
        "headline": "Bahaya kualitas udara, batalkan aktivitas luar ruang",
        "action_items": [
            "Batalkan seluruh aktivitas luar ruang, termasuk olahraga dan upacara.",
            "Tetap siswa di dalam ruangan dengan ventilasi yang memadai.",
            "Wajibkan penggunaan masker saat perjalanan ke/dari sekolah.",
            "Segera informasikan orang tua/wali mengenai kondisi udara.",
        ],
        "parent_template": (
            "PERHATIAN: Kualitas udara di sekitar {school_name} dalam kondisi berbahaya. "
            "Sekolah membatalkan semua aktivitas luar ruang. Mohon siswa menggunakan masker."
        ),
    },
    "Tidak Ada Data": {
        "headline": "Data kualitas udara belum tersedia",
        "action_items": [
            "Tidak dapat memberikan rekomendasi spesifik saat ini.",
            "Silakan pantau pembaruan data kualitas udara.",
        ],
        "parent_template": "Data kualitas udara untuk {school_name} belum tersedia saat ini.",
    },
}


# ── Data schemas ───────────────────────────────────────────────────────────

@dataclass
class RecommendationInput:
    school_id: str
    school_name: str
    district: str
    pm25_6h: float
    pm25_12h: float
    pm25_24h: float
    risk_level: str  # BMKG label — mapped internally to action label
    top_shap_factors: list[dict[str, str]] = field(default_factory=list)


@dataclass
class RecommendationOutput:
    school_id: str
    school_name: str
    district: str
    risk_level: str
    pm25_summary: str
    headline: str
    recommendation: str
    action_items: list[str]
    reasoning_summary: str
    parent_message: str
    generation_mode: str = "template_fallback"


REQUIRED_OUTPUT_KEYS = [
    "school_id", "school_name", "district", "risk_level",
    "pm25_summary", "headline", "recommendation",
    "action_items", "reasoning_summary", "parent_message",
    "generation_mode",
]

_MISSING = float("nan")

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


# ── Input adapter ──────────────────────────────────────────────────────────

def from_prediction_row(
    row: dict,
    school_name: str = "",
    district: str = "",
) -> dict:
    """Adapt a predict-pipeline school row to recommendation input format."""
    risk = worst_risk(
        row.get("risk_h6", ""),
        row.get("risk_h12", ""),
        row.get("risk_h24", ""),
    )

    def _safe_float(val):
        if val is None:
            return _MISSING
        try:
            v = float(val)
            return v if pd.notna(v) else _MISSING
        except (ValueError, TypeError):
            return _MISSING

    return {
        "school_id": str(row.get("npsn", "")),
        "school_name": school_name or str(row.get("npsn", "")),
        "district": district or row.get("district", ""),
        "pm25_6h": _safe_float(row.get("pm25_h6")),
        "pm25_12h": _safe_float(row.get("pm25_h12")),
        "pm25_24h": _safe_float(row.get("pm25_h24")),
        "risk_level": risk,
        "top_shap_factors": row.get("top_shap_factors", []),
    }


# ── Gemini LLM backend ────────────────────────────────────────────────────

_gemini_client = None


def _get_gemini_client():
    """Lazy-init Gemini client singleton using google-genai SDK."""
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client

    from src.config import GEMINI_API_KEY

    if not GEMINI_API_KEY:
        return None

    try:
        from google import genai
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Gemini client initialized")
        return _gemini_client
    except Exception as exc:
        logger.error("Gemini init failed: %s", exc)
        return None


def _load_prompt(filename: str) -> str:
    path = _PROMPTS_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


_SYSTEM_PROMPT = None


def _get_system_prompt() -> str:
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is None:
        _SYSTEM_PROMPT = _load_prompt("airsafe_recommendation_system.txt")
    return _SYSTEM_PROMPT


def _build_user_prompt(inp: RecommendationInput) -> str:
    template = _load_prompt("airsafe_recommendation_user.txt")
    risk = bmkg_to_action(inp.risk_level)
    factors = inp.top_shap_factors[:3] if inp.top_shap_factors else []
    factors_str = ", ".join(
        f"{f.get('feature', '?')} ({f.get('direction', '?')}): {f.get('reason', '')}"
        for f in factors
    ) if factors else "Tidak ada data faktor"

    return template.format(
        school_name=inp.school_name,
        district=inp.district,
        pm25_6h=_fmt_pm25(inp.pm25_6h),
        pm25_12h=_fmt_pm25(inp.pm25_12h),
        pm25_24h=_fmt_pm25(inp.pm25_24h),
        risk_level=risk,
        top_factors=factors_str,
    )


def _generate_with_gemini(inp: RecommendationInput) -> dict | None:
    """Try Gemini generation. Returns parsed dict or None on failure."""
    client = _get_gemini_client()
    if client is None:
        return None

    from src.config import GEMINI_MODEL

    try:
        system_prompt = _get_system_prompt()
        user_prompt = _build_user_prompt(inp)

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=system_prompt + "\n\n" + user_prompt,
            config={"temperature": 0.3, "max_output_tokens": 1024,
                    "response_mime_type": "application/json"},
        )

        text = response.text.strip()
        parsed = json.loads(text)

        # Ensure required fields present
        parsed.setdefault("school_id", inp.school_id)
        parsed.setdefault("school_name", inp.school_name)
        parsed.setdefault("district", inp.district)
        parsed.setdefault("risk_level", bmkg_to_action(inp.risk_level))
        parsed.setdefault("pm25_summary", _build_pm25_summary(inp))
        parsed["generation_mode"] = "gemini_flash"

        return parsed

    except json.JSONDecodeError as exc:
        logger.warning("Gemini returned invalid JSON for %s: %s", inp.school_id, exc)
        return None
    except Exception as exc:
        logger.warning("Gemini generation failed for %s: %s", inp.school_id, exc)
        return None


# ── Template fallback ──────────────────────────────────────────────────────

def _fmt_pm25(val: float) -> str:
    return f"{val:.1f} µg/m³" if pd.notna(val) else "tidak tersedia"


def _build_pm25_summary(inp: RecommendationInput) -> str:
    return (
        f"PM2.5 di sekitar {inp.school_name} diprediksi "
        f"{_fmt_pm25(inp.pm25_6h)} dalam 6 jam, "
        f"{_fmt_pm25(inp.pm25_12h)} dalam 12 jam, dan "
        f"{_fmt_pm25(inp.pm25_24h)} dalam 24 jam."
    )


def _build_reasoning(inp: RecommendationInput) -> str:
    if not inp.top_shap_factors:
        return "Tidak ada informasi faktor model tambahan yang tersedia."
    reasons = ", ".join(f["reason"] for f in inp.top_shap_factors[:3])
    return (
        f"Faktor model yang mendukung rekomendasi ini meliputi: {reasons}. "
        "Faktor ini digunakan sebagai indikasi pendukung, bukan klaim sebab-akibat absolut."
    )


def _generate_with_template(inp: RecommendationInput) -> dict:
    """Template-based fallback when no LLM is available."""
    risk = bmkg_to_action(inp.risk_level)
    policy = ACTION_POLICIES.get(risk, ACTION_POLICIES["Tidak Ada Data"])

    output = RecommendationOutput(
        school_id=inp.school_id,
        school_name=inp.school_name,
        district=inp.district,
        risk_level=risk,
        pm25_summary=_build_pm25_summary(inp),
        headline=policy["headline"],
        recommendation=policy["headline"],
        action_items=list(policy["action_items"]),
        reasoning_summary=_build_reasoning(inp),
        parent_message=policy["parent_template"].format(school_name=inp.school_name),
    )
    return asdict(output)


# ── Main entry point ───────────────────────────────────────────────────────

def generate_recommendation(input_data: RecommendationInput | dict) -> dict:
    """Generate recommendation using Gemini, with template fallback."""
    if isinstance(input_data, dict):
        inp = RecommendationInput(**input_data)
    else:
        inp = input_data

    # Try Gemini first
    gemini_result = _generate_with_gemini(inp)
    if gemini_result is not None:
        return gemini_result

    # Fallback to template
    return _generate_with_template(inp)
