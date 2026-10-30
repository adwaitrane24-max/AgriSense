"""Tiny standard-library HTTP helper: GET a URL and decode its JSON body.

Kept separate so weather.py and soil.py share one tested way of calling the
network, and so tests can replace it with a fake (no real network in tests).

Complexity: O(response size) time and space to read and parse the body.
"""

import json
import urllib.request
from typing import Any

USER_AGENT: str = "LeafCheck/1.0 (student project; SPIT CS208)"


def fetch_json(url: str, timeout: float) -> Any:
    """GET `url` and return the parsed JSON.

    Raises:
        urllib.error.URLError / TimeoutError / OSError on network problems,
        ValueError (json.JSONDecodeError) if the body is not valid JSON.
    """
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))
