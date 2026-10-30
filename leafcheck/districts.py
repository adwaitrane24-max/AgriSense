"""The 36 districts of Maharashtra with headquarters coordinates and soil type.

Used by the Field Outlook page to (a) check that a location is in Maharashtra,
(b) find the nearest district, and (c) give a general soil type when the live
soil service (SoilGrids) is unavailable.

Important: `soil` is a BROAD district-level generalisation of well-known soil
zones (black cotton soil on the Deccan plateau, laterite in south Konkan,
coastal alluvium in north Konkan, red and yellow soil in eastern Vidarbha).
Real soil varies a lot inside one district; this is for learning, not for
farm decisions. Coordinates are approximate (2 decimals) and point to the
district headquarters town.

Complexity (n = 36 districts):
    in_maharashtra    O(1) time, O(1) space
    nearest_district  O(n) time (one haversine per district), O(1) extra space
    get_district      O(1) average time (dict lookup), O(n) space for the index
"""

import math
from dataclasses import dataclass
from typing import Optional

# Approximate bounding box of Maharashtra (degrees).
LAT_MIN, LAT_MAX = 15.6, 22.1
LON_MIN, LON_MAX = 72.6, 80.9

EARTH_RADIUS_KM: float = 6371.0


@dataclass(frozen=True)
class District:
    """One district: URL slug, names in en/hi/mr, HQ coordinates, soil key."""

    slug: str
    name: dict[str, str]
    lat: float
    lon: float
    soil: str


def _d(slug: str, en: str, hi: str, mr: str, lat: float, lon: float, soil: str) -> District:
    """Small helper so the table below stays one line per district."""
    return District(slug, {"en": en, "hi": hi, "mr": mr}, lat, lon, soil)


DISTRICTS: tuple[District, ...] = (
    # --- Konkan ---
    _d("mumbai-city", "Mumbai City", "मुंबई शहर", "मुंबई शहर", 18.94, 72.83, "coastal_alluvial"),
    _d("mumbai-suburban", "Mumbai Suburban", "मुंबई उपनगर", "मुंबई उपनगर", 19.06, 72.84, "coastal_alluvial"),
    _d("thane", "Thane", "ठाणे", "ठाणे", 19.22, 72.98, "coastal_alluvial"),
    _d("palghar", "Palghar", "पालघर", "पालघर", 19.70, 72.77, "coastal_alluvial"),
    _d("raigad", "Raigad", "रायगढ़", "रायगड", 18.64, 72.87, "coastal_alluvial"),
    _d("ratnagiri", "Ratnagiri", "रत्नागिरी", "रत्नागिरी", 16.99, 73.30, "laterite"),
    _d("sindhudurg", "Sindhudurg", "सिंधुदुर्ग", "सिंधुदुर्ग", 16.10, 73.69, "laterite"),
    # --- Western Maharashtra / Pune division ---
    _d("pune", "Pune", "पुणे", "पुणे", 18.52, 73.86, "black_cotton"),
    _d("satara", "Satara", "सातारा", "सातारा", 17.68, 74.00, "medium_black"),
    _d("sangli", "Sangli", "सांगली", "सांगली", 16.85, 74.58, "black_cotton"),
    _d("kolhapur", "Kolhapur", "कोल्हापुर", "कोल्हापूर", 16.70, 74.24, "medium_black"),
    _d("solapur", "Solapur", "सोलापुर", "सोलापूर", 17.66, 75.91, "black_cotton"),
    # --- North Maharashtra / Nashik division ---
    _d("nashik", "Nashik", "नासिक", "नाशिक", 20.00, 73.79, "medium_black"),
    _d("ahilyanagar", "Ahilyanagar (Ahmednagar)", "अहिल्यानगर (अहमदनगर)", "अहिल्यानगर (अहमदनगर)", 19.09, 74.74, "black_cotton"),
    _d("dhule", "Dhule", "धुले", "धुळे", 20.90, 74.77, "black_cotton"),
    _d("nandurbar", "Nandurbar", "नंदुरबार", "नंदुरबार", 21.37, 74.24, "medium_black"),
    _d("jalgaon", "Jalgaon", "जलगाँव", "जळगाव", 21.01, 75.56, "black_cotton"),
    # --- Marathwada ---
    _d("chhatrapati-sambhajinagar", "Chhatrapati Sambhajinagar (Aurangabad)", "छत्रपति संभाजीनगर (औरंगाबाद)", "छत्रपती संभाजीनगर (औरंगाबाद)", 19.88, 75.34, "black_cotton"),
    _d("jalna", "Jalna", "जालना", "जालना", 19.84, 75.89, "black_cotton"),
    _d("beed", "Beed", "बीड", "बीड", 18.99, 75.76, "black_cotton"),
    _d("dharashiv", "Dharashiv (Osmanabad)", "धाराशिव (उस्मानाबाद)", "धाराशिव (उस्मानाबाद)", 18.18, 76.04, "black_cotton"),
    _d("latur", "Latur", "लातूर", "लातूर", 18.40, 76.56, "black_cotton"),
    _d("nanded", "Nanded", "नांदेड़", "नांदेड", 19.15, 77.32, "black_cotton"),
    _d("parbhani", "Parbhani", "परभणी", "परभणी", 19.27, 76.77, "black_cotton"),
    _d("hingoli", "Hingoli", "हिंगोली", "हिंगोली", 19.72, 77.15, "black_cotton"),
    # --- Western Vidarbha / Amravati division ---
    _d("buldhana", "Buldhana", "बुलढाणा", "बुलढाणा", 20.53, 76.18, "black_cotton"),
    _d("akola", "Akola", "अकोला", "अकोला", 20.71, 77.00, "black_cotton"),
    _d("washim", "Washim", "वाशिम", "वाशिम", 20.11, 77.13, "black_cotton"),
    _d("amravati", "Amravati", "अमरावती", "अमरावती", 20.93, 77.75, "black_cotton"),
    _d("yavatmal", "Yavatmal", "यवतमाल", "यवतमाळ", 20.39, 78.12, "black_cotton"),
    # --- Eastern Vidarbha / Nagpur division ---
    _d("wardha", "Wardha", "वर्धा", "वर्धा", 20.74, 78.60, "black_cotton"),
    _d("nagpur", "Nagpur", "नागपुर", "नागपूर", 21.15, 79.09, "black_cotton"),
    _d("bhandara", "Bhandara", "भंडारा", "भंडारा", 21.17, 79.65, "red_yellow"),
    _d("gondia", "Gondia", "गोंदिया", "गोंदिया", 21.46, 80.20, "red_yellow"),
    _d("chandrapur", "Chandrapur", "चंद्रपुर", "चंद्रपूर", 19.96, 79.30, "red_yellow"),
    _d("gadchiroli", "Gadchiroli", "गडचिरोली", "गडचिरोली", 20.18, 80.00, "red_yellow"),
)

# slug -> District, built once at import time (O(n)), so get_district is O(1).
_BY_SLUG: dict[str, District] = {d.slug: d for d in DISTRICTS}


SOIL_TYPES: dict[str, dict[str, dict[str, str]]] = {
    "black_cotton": {
        "en": {
            "name": "Black cotton soil (regur)",
            "description": "Deep, clay-rich soil formed from Deccan basalt; it holds water well, swells when wet and cracks when dry, and is widely used for cotton, soybean, jowar, tur and sugarcane.",
        },
        "hi": {
            "name": "काली कपास मिट्टी (रेगुर)",
            "description": "दक्कन बेसाल्ट से बनी गहरी, चिकनी मिट्टी; यह पानी अच्छी तरह रोकती है, भीगने पर फूलती है और सूखने पर इसमें दरारें पड़ती हैं, और इसमें कपास, सोयाबीन, ज्वार, अरहर और गन्ना उगाए जाते हैं।",
        },
        "mr": {
            "name": "काळी कापसाची माती (रेगूर)",
            "description": "दख्खनच्या बेसॉल्ट खडकापासून तयार झालेली खोल, चिकणमातीयुक्त माती; ती पाणी चांगले धरून ठेवते, ओली झाल्यावर फुगते व कोरडी झाल्यावर तिला भेगा पडतात, आणि त्यात कापूस, सोयाबीन, ज्वारी, तूर व ऊस घेतले जातात.",
        },
    },
    "medium_black": {
        "en": {
            "name": "Medium black soil",
            "description": "Moderately deep black soil found where the plateau meets the hills; it drains better than deep black soil and suits jowar, bajra, wheat, gram, onion and vegetables.",
        },
        "hi": {
            "name": "मध्यम काली मिट्टी",
            "description": "पठार और पहाड़ियों के बीच के क्षेत्रों में मिलने वाली मध्यम गहराई की काली मिट्टी; इसमें गहरी काली मिट्टी से बेहतर जल निकास होता है और यह ज्वार, बाजरा, गेहूँ, चना, प्याज और सब्ज़ियों के लिए उपयुक्त है।",
        },
        "mr": {
            "name": "मध्यम काळी माती",
            "description": "पठार आणि डोंगराळ भागांच्या मधल्या पट्ट्यात आढळणारी मध्यम खोलीची काळी माती; खोल काळ्या मातीपेक्षा तिचा निचरा चांगला होतो आणि ती ज्वारी, बाजरी, गहू, हरभरा, कांदा व भाजीपाल्यासाठी योग्य आहे.",
        },
    },
    "laterite": {
        "en": {
            "name": "Laterite soil",
            "description": "Reddish, porous soil of the high-rainfall south Konkan coast and ghats; it is acidic, drains quickly and holds little water, and is used for rice, cashew, mango and coconut.",
        },
        "hi": {
            "name": "लेटराइट मिट्टी",
            "description": "अधिक वर्षा वाले दक्षिण कोंकण तट और घाटों की लालिमायुक्त, छिद्रयुक्त मिट्टी; यह अम्लीय होती है, इसमें पानी जल्दी बह जाता है और कम नमी रुकती है, और इसमें धान, काजू, आम और नारियल उगाए जाते हैं।",
        },
        "mr": {
            "name": "जांभी माती (लॅटेराइट)",
            "description": "जास्त पावसाच्या दक्षिण कोकण किनारपट्टी व घाटांतील तांबूस, सच्छिद्र माती; ती आम्लधर्मी असते, पाण्याचा निचरा लवकर होतो व ओलावा कमी टिकतो, आणि त्यात भात, काजू, आंबा व नारळ घेतले जातात.",
        },
    },
    "coastal_alluvial": {
        "en": {
            "name": "Coastal alluvial soil",
            "description": "Soil laid down by rivers and the sea along the north Konkan coast; it is often fine-textured, can be salty near creeks, and is used mainly for rice, vegetables and coconut.",
        },
        "hi": {
            "name": "तटीय जलोढ़ मिट्टी",
            "description": "उत्तर कोंकण तट पर नदियों और समुद्र द्वारा जमा की गई मिट्टी; यह प्रायः बारीक कणों वाली होती है, खाड़ियों के पास खारी हो सकती है, और इसमें मुख्यतः धान, सब्ज़ियाँ और नारियल उगाए जाते हैं।",
        },
        "mr": {
            "name": "किनारी गाळाची माती",
            "description": "उत्तर कोकण किनाऱ्यावर नद्या व समुद्राने आणलेल्या गाळापासून बनलेली माती; ती बहुधा बारीक कणांची असते, खाड्यांजवळ खारट असू शकते, आणि त्यात मुख्यतः भात, भाजीपाला व नारळ घेतले जातात.",
        },
    },
    "red_yellow": {
        "en": {
            "name": "Red and yellow soil",
            "description": "Light, iron-rich soil of eastern Vidarbha formed from older crystalline rocks; it holds a low to moderate amount of water and is commonly used for rice, pulses and oilseeds.",
        },
        "hi": {
            "name": "लाल और पीली मिट्टी",
            "description": "पूर्वी विदर्भ की पुरानी क्रिस्टलीय चट्टानों से बनी हल्की, लौह-युक्त मिट्टी; इसकी जल धारण क्षमता कम से मध्यम होती है और इसमें प्रायः धान, दालें और तिलहन उगाए जाते हैं।",
        },
        "mr": {
            "name": "तांबडी व पिवळी माती",
            "description": "पूर्व विदर्भातील जुन्या स्फटिकी खडकांपासून तयार झालेली हलकी, लोहयुक्त माती; तिची पाणी धरून ठेवण्याची क्षमता कमी ते मध्यम असते आणि त्यात सहसा भात, कडधान्ये व तेलबिया घेतल्या जातात.",
        },
    },
}


def in_maharashtra(lat: float, lon: float) -> bool:
    """True if (lat, lon) lies inside Maharashtra's approximate bounding box. O(1).

    A box is a rough check: it also includes small parts of neighbouring states
    (Goa, Karnataka, Telangana, Madhya Pradesh, Gujarat).
    """
    return LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two points on Earth. O(1)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def nearest_district(lat: float, lon: float) -> District:
    """Return the district whose headquarters is closest to (lat, lon).

    Linear scan: O(n) time with n = 36, O(1) extra space. For 36 items this is
    faster and simpler than building a spatial index (k-d tree).
    """
    return min(DISTRICTS, key=lambda d: haversine_km(lat, lon, d.lat, d.lon))


def get_district(slug: str) -> Optional[District]:
    """Look up a district by slug, or None if unknown. O(1) average."""
    return _BY_SLUG.get(slug)
