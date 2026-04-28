from __future__ import annotations

from dataclasses import asdict, dataclass, field

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

_PREDICTION_TO_INPUT = {
    "npsn": "school_id",
    "pm25_h6": "pm25_6h",
    "pm25_h12h": "pm25_12h",
    "pm25_h24": "pm25_24h",
}


def from_prediction_row(
    row: dict,
    school_name: str = "",
    district: str = "",
) -> dict:
    """Adapt a predict-pipeline school row to recommendation input format.

    The predict pipeline outputs rows with keys like npsn, pm25_h6,
    risk_h6, etc. This normalizes them and picks the worst risk
    across all horizons.
    """
    risk = worst_risk(
        row.get("risk_h6", ""),
        row.get("risk_h12", ""),
        row.get("risk_h24", ""),
    )
    return {
        "school_id": str(row.get("npsn", "")),
        "school_name": school_name or str(row.get("npsn", "")),
        "district": district or row.get("district", ""),
        "pm25_6h": float(row.get("pm25_h6", 0) or 0),
        "pm25_12h": float(row.get("pm25_h12", 0) or 0),
        "pm25_24h": float(row.get("pm25_h24", 0) or 0),
        "risk_level": risk,
        "top_shap_factors": row.get("top_shap_factors", []),
    }


def _build_pm25_summary(inp: RecommendationInput) -> str:
    return (
        f"PM2.5 di sekitar {inp.school_name} diprediksi "
        f"{inp.pm25_6h:.1f} µg/m³ dalam 6 jam, "
        f"{inp.pm25_12h:.1f} µg/m³ dalam 12 jam, dan "
        f"{inp.pm25_24h:.1f} µg/m³ dalam 24 jam."
    )


def _build_reasoning(inp: RecommendationInput) -> str:
    if not inp.top_shap_factors:
        return "Tidak ada informasi faktor model tambahan yang tersedia."
    reasons = ", ".join(f["reason"] for f in inp.top_shap_factors[:3])
    return (
        f"Faktor model yang mendukung rekomendasi ini meliputi: {reasons}. "
        "Faktor ini digunakan sebagai indikasi pendukung, bukan klaim sebab-akibat absolut."
    )


def generate_recommendation(input_data: RecommendationInput | dict) -> dict:
    if isinstance(input_data, dict):
        inp = RecommendationInput(**input_data)
    else:
        inp = input_data

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
