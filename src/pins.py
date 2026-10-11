"""Hugging Face revisions pinned for reproducibility.

Each revision was the head of its repository on 2026-10-11. The tokenizer is loaded with
``trust_remote_code=True``, so pinning it also fixes which remote code is allowed to run.

To move a pin, change the hash here and rerun whatever depends on it.
"""

REVISIONS = {
    "microsoft/Phi-3-mini-4k-instruct": "f39ac1d28e925b323eae81227eaba4464caced4e",
    "tatsu-lab/alpaca": "dce01c9b08f87459cf36a430d809084718273017",
    "cais/mmlu": "c30699e8356da336a370243923dbaf21066bb9fe",
}


def revision(repo_id: str) -> str | None:
    """Pinned revision for ``repo_id``; None (the library default) for a local path or an unlisted id."""
    return REVISIONS.get(repo_id)
