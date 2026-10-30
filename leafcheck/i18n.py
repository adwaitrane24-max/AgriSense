"""User-interface text in English, Hindi and Marathi.

Every visible string on the web pages lives here, keyed by a short name. The
templates read them through ``t[key]`` and app.js reads the same dict from a
JSON block embedded in the page, so there is one source of truth per language.

Lookups are plain dict accesses: O(1) average time. The tables are fixed at
import time, so memory is O(number of keys x number of languages).

No string here may contain treatment advice (CLAUDE.md rule 7).
"""

from typing import Optional

from leafcheck import config

SUPPORTED_LANGS: tuple[str, ...] = ("en", "hi", "mr")
DEFAULT_LANG: str = "en"
LANG_NAMES: dict[str, str] = {"en": "English", "hi": "हिंदी", "mr": "मराठी"}

# Placeholders such as {max_mb}, {threshold} and {n} are filled in by the
# template or by app.js. Keep them identical in every language.
UI_STRINGS: dict[str, dict[str, str]] = {
    "en": {
        # Header, banner, footer
        "nav_analyse": "Analyse",
        "nav_stats": "Stats",
        "lang_label": "Language",
        "demo_banner": "Demo mode: no trained model found in the model/ folder, so results are random placeholders.",
        "footer": "LeafCheck prototype · CS208 Mini Project · trained on the public PlantVillage dataset · "
                  "uploads are processed in memory and never stored",
        # Upload page
        "intro_title": "Check a crop leaf for disease signs",
        "intro_text": "Upload one photo of a single tomato, potato, corn or bell pepper leaf. "
                      "LeafCheck shows its three most likely classes out of 19.",
        "upload_heading": "1. Upload a leaf photo",
        "upload_prompt": "Drag a JPG or PNG here, or",
        "upload_browse": "browse",
        "upload_hint": "Max {max_mb} MB · one leaf, plain background works best",
        "btn_analyse": "Analyse leaf",
        "btn_analysing": "Analysing…",
        "result_heading": "2. Result",
        "result_placeholder": "Your result will appear here.",
        "from_cache": "from cache",
        "most_likely": "Most likely",
        "top3_heading": "Top 3 classes",
        "not_sure_message": config.NOT_SURE_MESSAGE,
        "not_sure_explain": "Top confidence is below {threshold}%. This can happen with blurry photos, "
                            "other crops, or images that are not leaves.",
        "disclaimer": config.DISCLAIMER,
        # Advisory card
        "advisory_title": "Farmer guidance",
        "severity_label": "Risk level",
        "severity_none": "No risk",
        "severity_low": "Low",
        "severity_medium": "Medium",
        "severity_high": "High",
        "advisory_harm": "Possible harm",
        "advisory_economic": "Economic risk",
        "advisory_next_steps": "Next steps",
        "advisory_note": "This guidance does not include treatment advice. "
                         "Please contact your local KVK or agriculture officer.",
        # Errors shown by app.js
        "error_generic": "Something went wrong.",
        "error_network": "Could not reach the server.",
        "gallery_title": "Or try a sample leaf",
        "gallery_hint": "Click a photo to analyse it instantly.",
        "listen": "Listen",
        "stop_listening": "Stop",
        "voice_unavailable": "Voice is not available in this browser.",
        "voice_confidence": "confidence",
        "insights_title": "Leaf insights",
        "heatmap_title": "Where the model looked",
        "heatmap_caption": "Red and yellow areas influenced the result most.",
        "affected_title": "Affected leaf area",
        "affected_caption": "Rough colour-based estimate, not a diagnosis.",
        "band_none": "Very little",
        "band_mild": "Mild",
        "band_moderate": "Moderate",
        "band_severe": "Severe",
        "warn_too_dark": "Photo is too dark. Retake it in daylight.",
        "warn_too_bright": "Photo is too bright. Avoid direct glare.",
        "warn_blurry": "Photo looks blurry. Hold the camera steady.",
        "warn_not_leaf": "This does not look like a leaf.",
        # Stats page
        "stats_page_title": "Stats",
        "stats_title": "Cache and speed stats",
        "stats_intro": "Since the server started. Repeat uploads of the same image are served from "
                       "the LRU cache instead of running the model again.",
        "stats_requests": "Requests",
        "stats_hits": "Cache hits",
        "stats_misses": "Cache misses",
        "stats_hit_rate": "Hit rate",
        "stats_avg_time": "Average response time",
        "stats_miss_row": "Cache miss (model runs)",
        "stats_hit_row": "Cache hit",
        "stats_faster": "A cache hit is about {n}× faster.",
        "stats_cache_heading": "LRU cache",
        "stats_images": "images",
        "stats_cache_note": "Key: SHA-256 of the image bytes. Lookup and insert are O(1); when full, "
                            "the least recently used image is evicted.",
        "stats_view_json": "View as JSON",
    },
    "hi": {
        "nav_analyse": "जाँचें",
        "nav_stats": "आँकड़े",
        "lang_label": "भाषा",
        "demo_banner": "डेमो मोड: model/ फ़ोल्डर में प्रशिक्षित मॉडल नहीं मिला, इसलिए नतीजे केवल नमूने के लिए हैं।",
        "footer": "LeafCheck प्रोटोटाइप · CS208 मिनी प्रोजेक्ट · सार्वजनिक PlantVillage डेटासेट पर प्रशिक्षित · "
                  "अपलोड की गई फ़ोटो केवल मेमोरी में जाँची जाती है, सहेजी नहीं जाती",
        "intro_title": "फ़सल की पत्ती में रोग के लक्षण जाँचें",
        "intro_text": "टमाटर, आलू, मक्का या शिमला मिर्च की एक पत्ती की एक फ़ोटो अपलोड करें। "
                      "LeafCheck 19 वर्गों में से तीन सबसे संभावित वर्ग दिखाता है।",
        "upload_heading": "1. पत्ती की फ़ोटो अपलोड करें",
        "upload_prompt": "JPG या PNG फ़ोटो यहाँ खींचकर छोड़ें, या",
        "upload_browse": "फ़ाइल चुनें",
        "upload_hint": "अधिकतम {max_mb} MB · एक पत्ती, सादी पृष्ठभूमि सबसे अच्छी रहती है",
        "btn_analyse": "पत्ती जाँचें",
        "btn_analysing": "जाँच हो रही है…",
        "result_heading": "2. नतीजा",
        "result_placeholder": "आपका नतीजा यहाँ दिखेगा।",
        "from_cache": "कैश से",
        "most_likely": "सबसे संभावित",
        "top3_heading": "शीर्ष 3 वर्ग",
        "not_sure_message": "पक्का नहीं कह सकते, कृपया किसी कृषि विशेषज्ञ से सलाह लें।",
        "not_sure_explain": "सबसे ऊँचा भरोसा {threshold}% से कम है। ऐसा धुंधली फ़ोटो, दूसरी फ़सलों "
                            "या पत्ती के अलावा किसी और चीज़ की फ़ोटो से हो सकता है।",
        "disclaimer": "यह सीखने के उद्देश्य से किया गया एक स्वचालित अनुमान है। कोई भी कदम उठाने से पहले "
                      "किसी स्थानीय कृषि विशेषज्ञ या कृषि विज्ञान केंद्र से पुष्टि करें।",
        "advisory_title": "किसान के लिए जानकारी",
        "severity_label": "जोखिम स्तर",
        "severity_none": "कोई जोखिम नहीं",
        "severity_low": "कम",
        "severity_medium": "मध्यम",
        "severity_high": "अधिक",
        "advisory_harm": "संभावित नुकसान",
        "advisory_economic": "आर्थिक जोखिम",
        "advisory_next_steps": "आगे क्या करें",
        "advisory_note": "इस जानकारी में उपचार की सलाह शामिल नहीं है। "
                         "कृपया अपने नज़दीकी कृषि विज्ञान केंद्र (KVK) या कृषि अधिकारी से संपर्क करें।",
        "error_generic": "कुछ गड़बड़ हो गई।",
        "error_network": "सर्वर से संपर्क नहीं हो सका।",
        "gallery_title": "या कोई नमूना पत्ती आज़माएँ",
        "gallery_hint": "तुरंत जाँच के लिए किसी फ़ोटो पर क्लिक करें।",
        "listen": "सुनें",
        "stop_listening": "रोकें",
        "voice_unavailable": "इस ब्राउज़र में आवाज़ उपलब्ध नहीं है।",
        "voice_confidence": "विश्वास",
        "insights_title": "पत्ती की जानकारी",
        "heatmap_title": "मॉडल ने कहाँ देखा",
        "heatmap_caption": "लाल और पीले हिस्सों का परिणाम पर सबसे ज़्यादा असर रहा।",
        "affected_title": "प्रभावित पत्ती क्षेत्र",
        "affected_caption": "रंग के आधार पर मोटा अनुमान, यह निदान नहीं है।",
        "band_none": "बहुत कम",
        "band_mild": "हल्का",
        "band_moderate": "मध्यम",
        "band_severe": "गंभीर",
        "warn_too_dark": "फ़ोटो बहुत अँधेरी है। दिन की रोशनी में दोबारा लें।",
        "warn_too_bright": "फ़ोटो बहुत चमकीली है। सीधी चमक से बचें।",
        "warn_blurry": "फ़ोटो धुंधली लग रही है। कैमरा स्थिर रखें।",
        "warn_not_leaf": "यह पत्ती जैसी नहीं लग रही है।",
        "stats_page_title": "आँकड़े",
        "stats_title": "कैश और गति के आँकड़े",
        "stats_intro": "सर्वर शुरू होने के बाद से। एक ही फ़ोटो दोबारा अपलोड करने पर मॉडल फिर से नहीं चलता, "
                       "नतीजा LRU कैश से मिलता है।",
        "stats_requests": "अनुरोध",
        "stats_hits": "कैश हिट",
        "stats_misses": "कैश मिस",
        "stats_hit_rate": "हिट दर",
        "stats_avg_time": "औसत जवाब समय",
        "stats_miss_row": "कैश मिस (मॉडल चला)",
        "stats_hit_row": "कैश हिट",
        "stats_faster": "कैश हिट लगभग {n}× तेज़ है।",
        "stats_cache_heading": "LRU कैश",
        "stats_images": "फ़ोटो",
        "stats_cache_note": "कुंजी: फ़ोटो के बाइट्स का SHA-256। खोजना और जोड़ना O(1) है; कैश भरने पर "
                            "सबसे पुरानी इस्तेमाल हुई फ़ोटो हटा दी जाती है।",
        "stats_view_json": "JSON के रूप में देखें",
    },
    "mr": {
        "nav_analyse": "तपासा",
        "nav_stats": "आकडेवारी",
        "lang_label": "भाषा",
        "demo_banner": "डेमो मोड: model/ फोल्डरमध्ये प्रशिक्षित मॉडेल सापडले नाही, त्यामुळे निकाल फक्त नमुन्यासाठी आहेत.",
        "footer": "LeafCheck प्रोटोटाइप · CS208 मिनी प्रोजेक्ट · सार्वजनिक PlantVillage डेटासेटवर प्रशिक्षित · "
                  "अपलोड केलेला फोटो फक्त मेमरीमध्ये तपासला जातो, साठवला जात नाही",
        "intro_title": "पिकाच्या पानावर रोगाची लक्षणे तपासा",
        "intro_text": "टोमॅटो, बटाटा, मका किंवा ढोबळी मिरचीच्या एका पानाचा एक फोटो अपलोड करा. "
                      "LeafCheck 19 वर्गांपैकी तीन सर्वात संभाव्य वर्ग दाखवते.",
        "upload_heading": "1. पानाचा फोटो अपलोड करा",
        "upload_prompt": "JPG किंवा PNG फोटो इथे ओढून आणा, किंवा",
        "upload_browse": "फाइल निवडा",
        "upload_hint": "जास्तीत जास्त {max_mb} MB · एक पान, साधी पार्श्वभूमी सर्वात चांगली",
        "btn_analyse": "पान तपासा",
        "btn_analysing": "तपासणी सुरू आहे…",
        "result_heading": "2. निकाल",
        "result_placeholder": "तुमचा निकाल इथे दिसेल.",
        "from_cache": "कॅशमधून",
        "most_likely": "सर्वात संभाव्य",
        "top3_heading": "पहिले 3 वर्ग",
        "not_sure_message": "खात्री नाही, कृपया कृषी तज्ज्ञांचा सल्ला घ्या.",
        "not_sure_explain": "सर्वोच्च खात्री {threshold}% पेक्षा कमी आहे. धूसर फोटो, इतर पिके "
                            "किंवा पान नसलेल्या फोटोमुळे असे होऊ शकते.",
        "disclaimer": "हा शिकण्याच्या उद्देशाने केलेला स्वयंचलित अंदाज आहे. कोणतीही कृती करण्यापूर्वी "
                      "स्थानिक कृषी तज्ज्ञ किंवा कृषी विज्ञान केंद्राकडून खात्री करून घ्या.",
        "advisory_title": "शेतकऱ्यांसाठी माहिती",
        "severity_label": "धोक्याची पातळी",
        "severity_none": "धोका नाही",
        "severity_low": "कमी",
        "severity_medium": "मध्यम",
        "severity_high": "जास्त",
        "advisory_harm": "संभाव्य नुकसान",
        "advisory_economic": "आर्थिक धोका",
        "advisory_next_steps": "पुढे काय करावे",
        "advisory_note": "या माहितीमध्ये उपचाराचा सल्ला समाविष्ट नाही. "
                         "कृपया जवळच्या कृषी विज्ञान केंद्राशी (KVK) किंवा कृषी अधिकाऱ्यांशी संपर्क साधा.",
        "error_generic": "काहीतरी चूक झाली.",
        "error_network": "सर्व्हरशी संपर्क होऊ शकला नाही.",
        "gallery_title": "किंवा नमुना पान वापरून पाहा",
        "gallery_hint": "लगेच तपासणीसाठी फोटोवर क्लिक करा.",
        "listen": "ऐका",
        "stop_listening": "थांबवा",
        "voice_unavailable": "या ब्राउझरमध्ये आवाज उपलब्ध नाही.",
        "voice_confidence": "खात्री",
        "insights_title": "पानाची माहिती",
        "heatmap_title": "मॉडेलने कुठे पाहिले",
        "heatmap_caption": "लाल आणि पिवळ्या भागांचा निकालावर सर्वाधिक परिणाम झाला.",
        "affected_title": "बाधित पानाचा भाग",
        "affected_caption": "रंगावर आधारित अंदाजे आकडा, हे निदान नाही.",
        "band_none": "खूप कमी",
        "band_mild": "सौम्य",
        "band_moderate": "मध्यम",
        "band_severe": "तीव्र",
        "warn_too_dark": "फोटो खूप अंधारा आहे. दिवसाच्या उजेडात पुन्हा घ्या.",
        "warn_too_bright": "फोटो खूप उजळ आहे. थेट चमक टाळा.",
        "warn_blurry": "फोटो धूसर दिसतो. कॅमेरा स्थिर धरा.",
        "warn_not_leaf": "हे पानासारखे दिसत नाही.",
        "stats_page_title": "आकडेवारी",
        "stats_title": "कॅश आणि वेगाची आकडेवारी",
        "stats_intro": "सर्व्हर सुरू झाल्यापासून. तोच फोटो पुन्हा अपलोड केल्यास मॉडेल पुन्हा चालत नाही, "
                       "निकाल LRU कॅशमधून मिळतो.",
        "stats_requests": "विनंत्या",
        "stats_hits": "कॅश हिट",
        "stats_misses": "कॅश मिस",
        "stats_hit_rate": "हिट दर",
        "stats_avg_time": "सरासरी प्रतिसाद वेळ",
        "stats_miss_row": "कॅश मिस (मॉडेल चालले)",
        "stats_hit_row": "कॅश हिट",
        "stats_faster": "कॅश हिट सुमारे {n}× वेगवान आहे.",
        "stats_cache_heading": "LRU कॅश",
        "stats_images": "फोटो",
        "stats_cache_note": "की: फोटोच्या बाइट्सचा SHA-256. शोधणे आणि जोडणे O(1) आहे; कॅश भरल्यावर "
                            "सर्वात जुना वापरलेला फोटो काढून टाकला जातो.",
        "stats_view_json": "JSON स्वरूपात पाहा",
    },
}


def normalise_lang(lang: Optional[str]) -> str:
    """Return a supported language code, falling back to English.

    Accepts any case and surrounding spaces ("HI " -> "hi"). O(1).
    """
    if not lang:
        return DEFAULT_LANG
    code = lang.strip().lower()
    return code if code in SUPPORTED_LANGS else DEFAULT_LANG


def strings_for(lang: Optional[str]) -> dict[str, str]:
    """All UI strings for ``lang``; any key missing in that language uses English.

    Builds a new dict each call so callers cannot change the shared tables.
    O(number of keys) time and space.
    """
    english = UI_STRINGS[DEFAULT_LANG]
    chosen = UI_STRINGS[normalise_lang(lang)]
    return {key: chosen.get(key) or value for key, value in english.items()}
