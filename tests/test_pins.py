"""CPU-only tests for the Hugging Face revision pins.

These import only src.pins (pure Python, no MLX, no network), so they run on
any platform. They guard the reproducibility contract: every pinned id resolves
to a full 40-character commit hash, and unknown ids fall back to the library
default (None).
"""
import re

from src.pins import REVISIONS, revision

_SHA1 = re.compile(r"^[0-9a-f]{40}$")


def test_every_pin_is_a_full_commit_hash():
    assert REVISIONS, "at least one revision must be pinned"
    for repo_id, rev in REVISIONS.items():
        assert _SHA1.match(rev), f"{repo_id} pin is not a 40-char hex sha: {rev!r}"


def test_revision_returns_pinned_hash_for_known_ids():
    for repo_id, rev in REVISIONS.items():
        assert revision(repo_id) == rev


def test_revision_returns_none_for_unknown_or_local_path():
    assert revision("some/unpinned-model") is None
    assert revision("/local/path/to/model") is None
