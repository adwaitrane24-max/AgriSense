"""Content fingerprint used as the cache key.

Why hash the bytes and not the filename: two different photos can both be
called "leaf.jpg", and the same photo can be renamed. Hashing the content means
"same key" exactly when the uploaded bytes are the same.

SHA-256 gives a 64-character hex string. Accidental collisions are so unlikely
(2^-128 for a birthday attack) that we treat equal hashes as equal images.

Complexity: O(m) time for m bytes (every byte is read once), O(1) extra space.
"""

import hashlib


def fingerprint(data: bytes) -> str:
    """Return the SHA-256 hex digest of `data`.

    Args:
        data: raw bytes of the uploaded file.

    Returns:
        64-character lowercase hexadecimal string.
    """
    return hashlib.sha256(data).hexdigest()
