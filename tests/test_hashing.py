"""Tests for the SHA-256 fingerprint."""

from leafcheck.hashing import fingerprint


def test_same_bytes_give_same_key() -> None:
    assert fingerprint(b"leaf image bytes") == fingerprint(b"leaf image bytes")


def test_different_bytes_give_different_key() -> None:
    assert fingerprint(b"leaf A") != fingerprint(b"leaf B")


def test_one_byte_change_changes_key() -> None:
    assert fingerprint(b"\x00" * 1000) != fingerprint(b"\x00" * 999 + b"\x01")


def test_key_is_64_hex_chars() -> None:
    key = fingerprint(b"")
    assert len(key) == 64
    assert all(c in "0123456789abcdef" for c in key)
