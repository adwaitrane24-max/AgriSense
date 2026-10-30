"""The /predict pipeline, kept out of app.py so routes stay thin.

    validate -> fingerprint -> cache.get
        hit:  return stored result (from_cache = True)
        miss: preprocess -> model -> top-3 + threshold -> cache.put
    -> record timing in stats

Complexity per request (m = file bytes, p = pixels):
    hit:  O(m) to validate and hash, O(1) cache lookup  -> a few ms
    miss: O(m + p) validate/hash/preprocess + constant-cost CNN + O(n log k) top-k
"""

import threading
import time
from typing import Any

from leafcheck import config
from leafcheck.cache import LRUCache
from leafcheck.hashing import fingerprint
from leafcheck.predictor import Predictor, build_result
from leafcheck.preprocess import to_model_input, validate_upload
from leafcheck.stats import Stats


class LeafCheckService:
    """Holds the predictor, the LRU cache and the stats for the app's lifetime."""

    def __init__(
        self,
        predictor: Predictor,
        demo_mode: bool = False,
        cache_capacity: int = config.CACHE_CAPACITY,
    ) -> None:
        self.predictor = predictor
        self.demo_mode = demo_mode
        self.cache: LRUCache[str, dict[str, Any]] = LRUCache(cache_capacity)
        self.stats = Stats()
        self._cache_lock = threading.Lock()  # the cache itself is not thread-safe

    def _heatmap(self, batch: Any, result: dict[str, Any]) -> str | None:
        """Grad-CAM overlay for a confident prediction, else None.

        Costs about one extra forward + backward pass (~230 ms on CPU), paid only
        on a cache miss; the image is stored inside the cached result.
        """
        make_heatmap = getattr(self.predictor, "heatmap", None)
        if make_heatmap is None or result["status"] != "ok":
            return None
        class_index = self.predictor.labels.index(result["prediction"]["label"])
        return make_heatmap(batch, class_index)

    def analyse(self, filename: str, data: bytes) -> dict[str, Any]:
        """Run the full pipeline for one upload and return the JSON response.

        Raises:
            InvalidImageError: if the upload is not a usable JPG/PNG.
        """
        start = time.perf_counter()
        validate_upload(filename, data)
        key = fingerprint(data)

        with self._cache_lock:
            cached = self.cache.get(key)

        if cached is not None:
            result, hit = cached, True
        else:
            batch = to_model_input(data)
            probabilities = self.predictor.predict_proba(batch)
            result = build_result(probabilities, self.predictor.labels)
            result["heatmap"] = self._heatmap(batch, result)
            with self._cache_lock:
                self.cache.put(key, result)
            hit = False

        elapsed_ms = (time.perf_counter() - start) * 1000
        self.stats.record(hit, elapsed_ms)
        # Copy so per-request fields never leak into the cached dict.
        return {
            **result,
            "from_cache": hit,
            "elapsed_ms": round(elapsed_ms, 2),
            "fingerprint": key[:12],
            "demo_mode": self.demo_mode,
        }

    def stats_snapshot(self) -> dict[str, Any]:
        """Stats plus current cache fill level. O(1)."""
        with self._cache_lock:
            size = len(self.cache)
        return {**self.stats.snapshot(), "cache_size": size, "cache_capacity": self.cache.capacity}
