"""Flask app: HTTP routes only. All logic lives in the leafcheck package.

Run:  flask --app app run   (Flask finds create_app() automatically)
"""

from typing import Any, Optional

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from leafcheck import config
from leafcheck.i18n import LANG_NAMES, normalise_lang, strings_for
from leafcheck.advisory import display_name
from leafcheck.localise import localise_result
from leafcheck import districts, outlook
from leafcheck.predictor import load_predictor
from leafcheck.preprocess import InvalidImageError
from leafcheck.service import LeafCheckService


# Class shown in the landing-page preview card (matches GALLERY_SAMPLES[0]).
HERO_SAMPLE_LABEL = "Tomato___Early_blight"


def language_context() -> dict[str, Any]:
    """Template variables for the language picked with ``?lang=`` (default English)."""
    lang = normalise_lang(request.args.get("lang"))
    return {"lang": lang, "t": strings_for(lang), "lang_names": LANG_NAMES}


def create_app(service: Optional[LeafCheckService] = None) -> Flask:
    """Build the Flask app. Tests pass a service that uses a FakePredictor."""
    if service is None:
        predictor, demo_mode = load_predictor()
        service = LeafCheckService(predictor, demo_mode=demo_mode)

    app = Flask(__name__)
    # Slightly above the 5 MB file limit so our own validator can give a clear
    # message; anything far bigger is cut off by Flask with a 413.
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_BYTES + 512 * 1024
    app.extensions["leafcheck"] = service

    @app.get("/")
    def home() -> str:
        crop, condition = display_name(HERO_SAMPLE_LABEL, normalise_lang(request.args.get("lang")))
        return render_template(
            "landing.html",
            demo_mode=service.demo_mode,
            hero_sample=config.GALLERY_SAMPLES[0],
            hero_crop=crop,
            hero_condition=condition,
            **language_context(),
        )

    @app.get("/analyse")
    def analyse() -> str:
        return render_template(
            "index.html",
            demo_mode=service.demo_mode,
            threshold=config.CONFIDENCE_THRESHOLD,
            max_mb=config.MAX_UPLOAD_BYTES // (1024 * 1024),
            samples=config.GALLERY_SAMPLES,
            **language_context(),
        )

    @app.post("/predict")
    def predict() -> Any:
        upload = request.files.get("image")
        if upload is None or upload.filename == "":
            return jsonify(error="Please choose an image file to upload."), 400
        try:
            result = service.analyse(upload.filename, upload.read())
            return jsonify(localise_result(result, request.form.get("lang")))
        except InvalidImageError as error:
            return jsonify(error=str(error)), 400

    @app.get("/stats")
    def stats() -> Any:
        snapshot = service.stats_snapshot()
        if request.args.get("format") == "json":
            return jsonify(snapshot)
        return render_template(
            "stats.html", stats=snapshot, demo_mode=service.demo_mode, **language_context()
        )

    @app.get("/outlook")
    def outlook_page() -> str:
        context = language_context()
        return render_template(
            "map.html",
            demo_mode=service.demo_mode,
            districts=outlook.district_options(context["lang"]),
            **context,
        )

    @app.get("/api/outlook")
    def outlook_api() -> Any:
        t = strings_for(normalise_lang(request.args.get("lang")))
        try:
            lat = float(request.args["lat"])
            lon = float(request.args["lon"])
        except (KeyError, ValueError):
            return jsonify(error=t["map_outside"]), 400
        if not districts.in_maharashtra(lat, lon):
            return jsonify(error=t["map_outside"]), 400
        return jsonify(outlook.build_outlook(lat, lon, request.args.get("lang")))

    @app.get("/samples/<name>")
    def sample(name: str) -> Any:
        # Only the listed demo files are served; nothing else in samples/ is reachable.
        if name not in config.GALLERY_SAMPLES:
            abort(404)
        return send_from_directory(config.SAMPLES_DIR, name)

    @app.get("/health")
    def health() -> Any:
        return jsonify(status="ok", demo_mode=service.demo_mode, classes=len(service.predictor.labels))

    @app.errorhandler(413)
    def too_large(_error: Exception) -> Any:
        return jsonify(error=f"File is too large. Maximum size is {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB."), 413

    return app
