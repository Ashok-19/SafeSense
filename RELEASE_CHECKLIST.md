# Public release checklist

The technical release is self-contained, but complete these human-owned publication steps before pushing it as the official public repository:

- [ ] Choose and add the intended open-source `LICENSE` (do not publish without the team's license decision).
- [ ] Replace citation author/venue/DOI placeholders in `CITATION.md`.
- [ ] Add the final paper PDF/arXiv/DOI link to `README.md` when available.
- [ ] Confirm dataset download instructions comply with UTD-MHAD distribution terms; do not commit raw dataset archives.
- [ ] Confirm the authors agree to redistribute the 1.8 MB model checkpoint.
- [ ] Create a clean Git history for the new public repository rather than pushing the internal project history.
- [ ] Run `pytest -q` from a fresh clone/virtualenv.
- [ ] Replay `scripts/evaluate.py` and `scripts/evaluate_robustness.py` once from a freshly prepared UTD-MHAD copy.
- [ ] Record the public release tag/commit in the final paper artifact statement.

Recommended first public tag: `v1.0-paper` after the metadata/license checks above are complete.
