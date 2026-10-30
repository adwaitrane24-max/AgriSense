"""Topsoil properties from ISRIC SoilGrids v2 (free, no API key), with a fallback.

SoilGrids is a global 250 m soil map made by machine learning from soil
profiles. We ask for the 0-5 cm layer at one point and get pH, clay, sand,
silt and soil organic carbon (SOC).

Units: SoilGrids returns integers in "mapped units"; dividing by the layer's
`d_factor` gives normal units:
    phh2o   pH x 10  -> pH
    clay/sand/silt  g/kg -> %   (d_factor 10)
    soc     dg/kg -> g/kg      (d_factor 10)

Fallback: if the request fails, times out, or returns null values (common for
city, water or coastal points), we return the general soil zone of the nearest
district (`source: "district"`), with the numeric values set to None.

Complexity:
    cache lookup        O(1) average
    parsing             O(L * D) for L = 5 layers, D = 1 depth -> constant
    texture classifier  O(1) (a fixed chain of comparisons)
    nearest district    O(36) on fallback
    space               O(capacity) cached results
"""

from typing import Any, Callable, Optional
from urllib.parse import urlencode

from leafcheck.districts import nearest_district
from leafcheck.http_json import fetch_json
from leafcheck.ttl_cache import TTLCache

API_URL: str = "https://rest.isric.org/soilgrids/v2.0/properties/query"
PROPERTIES: tuple[str, ...] = ("phh2o", "clay", "sand", "silt", "soc")
DEPTH: str = "0-5cm"
TIMEOUT_SECONDS: float = 8
DEFAULT_D_FACTOR: dict[str, float] = {"phh2o": 10, "clay": 10, "sand": 10, "silt": 10, "soc": 10}

Fetcher = Callable[[str], Any]

# Soil does not change day to day: keep live results for 24 hours.
_CACHE: TTLCache[tuple[float, float], dict] = TTLCache(capacity=64, ttl_seconds=24 * 3600)

TEXTURE_NAMES: dict[str, dict[str, str]] = {
    "sand": {"en": "Sand", "hi": "बलुई मिट्टी", "mr": "वाळू"},
    "loamy_sand": {"en": "Loamy sand", "hi": "दोमट बलुई", "mr": "पोयटायुक्त वाळू"},
    "sandy_loam": {"en": "Sandy loam", "hi": "बलुई दोमट", "mr": "वाळूमय पोयटा"},
    "loam": {"en": "Loam", "hi": "दोमट", "mr": "पोयटा"},
    "silt_loam": {"en": "Silt loam", "hi": "गादी दोमट", "mr": "गाळयुक्त पोयटा"},
    "silt": {"en": "Silt", "hi": "गाद (सिल्ट)", "mr": "गाळ"},
    "sandy_clay_loam": {"en": "Sandy clay loam", "hi": "बलुई चिकनी दोमट", "mr": "वाळूमय चिकण पोयटा"},
    "clay_loam": {"en": "Clay loam", "hi": "चिकनी दोमट", "mr": "चिकण पोयटा"},
    "silty_clay_loam": {"en": "Silty clay loam", "hi": "गादी चिकनी दोमट", "mr": "गाळयुक्त चिकण पोयटा"},
    "sandy_clay": {"en": "Sandy clay", "hi": "बलुई चिकनी मिट्टी", "mr": "वाळूमय चिकणमाती"},
    "silty_clay": {"en": "Silty clay", "hi": "गादी चिकनी मिट्टी", "mr": "गाळयुक्त चिकणमाती"},
    "clay": {"en": "Clay", "hi": "चिकनी मिट्टी", "mr": "चिकणमाती"},
}


def _fetch_json(url: str, timeout: float = TIMEOUT_SECONDS) -> Any:
    """Default fetcher: real HTTP GET with a User-Agent header."""
    return fetch_json(url, timeout=timeout)


def build_url(lat: float, lon: float) -> str:
    """Return the SoilGrids query URL for (lat, lon). O(1)."""
    params: list[tuple[str, str]] = [("lon", f"{lon:.4f}"), ("lat", f"{lat:.4f}")]
    params += [("property", p) for p in PROPERTIES]
    params += [("depth", DEPTH), ("value", "mean")]
    return f"{API_URL}?{urlencode(params)}"


def usda_texture(sand: float, silt: float, clay: float) -> str:
    """Classify a soil into one of the 12 USDA texture classes. O(1).

    Inputs are percentages; they are rescaled to sum to 100 first. The rules
    are the standard boundary lines of the USDA soil texture triangle.
    """
    total = sand + silt + clay
    if total <= 0:
        raise ValueError("sand + silt + clay must be positive")
    sand, silt, clay = (100 * x / total for x in (sand, silt, clay))

    if silt + 1.5 * clay < 15:
        return "sand"
    if silt + 2 * clay < 30:
        return "loamy_sand"
    if (7 <= clay < 20 and sand > 52) or (clay < 7 and silt < 50):
        return "sandy_loam"
    if 7 <= clay < 27 and 28 <= silt < 50 and sand <= 52:
        return "loam"
    if silt >= 50 and clay < 27:
        return "silt" if (silt >= 80 and clay < 12) else "silt_loam"
    if 20 <= clay < 35 and silt < 28 and sand > 45:
        return "sandy_clay_loam"
    if 27 <= clay < 40 and 20 < sand <= 45:
        return "clay_loam"
    if 27 <= clay < 40 and sand <= 20:
        return "silty_clay_loam"
    if clay >= 35 and sand > 45:
        return "sandy_clay"
    if clay >= 40 and silt >= 40:
        return "silty_clay"
    return "clay"


def _layer_values(payload: dict) -> dict[str, Optional[float]]:
    """Extract {property: value in normal units or None} from a SoilGrids reply."""
    values: dict[str, Optional[float]] = {p: None for p in PROPERTIES}
    for layer in payload["properties"]["layers"]:
        name = layer.get("name")
        if name not in values:
            continue
        d_factor = (layer.get("unit_measure") or {}).get("d_factor") or DEFAULT_D_FACTOR[name]
        for depth in layer.get("depths", []):
            if depth.get("label") == DEPTH:
                raw = (depth.get("values") or {}).get("mean")
                if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                    values[name] = raw / d_factor
    return values


def parse_soil(payload: dict) -> Optional[dict]:
    """Convert a SoilGrids reply to our dict, or None if any value is missing."""
    v = _layer_values(payload)
    if any(x is None for x in v.values()):
        return None
    sand, silt, clay = v["sand"], v["silt"], v["clay"]
    if sand + silt + clay <= 0:  # type: ignore[operator]
        return None
    return {
        "ph": round(v["phh2o"], 1),  # type: ignore[arg-type]
        "clay_pct": round(clay, 1),  # type: ignore[arg-type]
        "sand_pct": round(sand, 1),  # type: ignore[arg-type]
        "silt_pct": round(silt, 1),  # type: ignore[arg-type]
        "organic_carbon_gkg": round(v["soc"], 1),  # type: ignore[arg-type]
        "texture": usda_texture(sand, silt, clay),  # type: ignore[arg-type]
    }


def district_fallback(lat: float, lon: float) -> dict:
    """General soil zone of the nearest district, with no measured values. O(36)."""
    district = nearest_district(lat, lon)
    return {
        "source": "district",
        "soil": district.soil,
        "district": district.slug,
        "ph": None,
        "clay_pct": None,
        "sand_pct": None,
        "silt_pct": None,
        "organic_carbon_gkg": None,
        "texture": None,
    }


def get_soil(lat: float, lon: float, *, fetch: Fetcher = _fetch_json) -> dict:
    """Return topsoil (0-5 cm) properties for (lat, lon).

    Returns:
        live:     {"source": "live", "ph", "clay_pct", "sand_pct", "silt_pct",
                   "organic_carbon_gkg", "texture", "soil", "district"}
        fallback: {"source": "district", "soil", "district", and the same
                   numeric keys set to None, "texture": None}
        `soil` is always the nearest district's general soil key, so the page
        can show the broad zone next to the measured values.
    """
    key = (round(lat, 2), round(lon, 2))
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    try:
        parsed = parse_soil(fetch(build_url(lat, lon)))
    except Exception:  # any network, timeout, JSON or shape problem
        parsed = None
    if parsed is None:
        return district_fallback(lat, lon)
    district = nearest_district(lat, lon)
    result = {"source": "live", **parsed, "soil": district.soil, "district": district.slug}
    _CACHE.put(key, result)
    return result


def clear_cache() -> None:
    """Forget cached soil results (used by tests)."""
    _CACHE.clear()
