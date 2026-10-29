#!/usr/bin/env bash
# Builds SafeSense_LiveDemo_Handover.zip (see handover/START_HERE.md) from this repository.
# Files that are kept out of the public repository are added from the command line:
#   bash handover/make_zip.sh path/to/safesense_orig_ntu120.pt path/to/safesense-paper.pdf path/to/release.keystore
# .pt files go to demo/, .pdf files to docs/paper/, a .keystore to camapp/release.keystore.
# The zip is written to ~/Downloads (or $OUT).
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${OUT:-$HOME/Downloads}"
T="$(mktemp -d)"; P="$T/SafeSense_LiveDemo_Handover"
mkdir -p "$P/training" "$P/docs/paper"
git -C "$ROOT" ls-files demo camapp | while read -r f; do mkdir -p "$P/$(dirname "$f")"; cp "$ROOT/$f" "$P/$f"; done
cp "$ROOT"/ntu/0*.ipynb "$P/training/"
cp "$ROOT/handover/START_HERE.md" "$ROOT/handover/REFERENCES.md" "$P/"
cp -r "$ROOT/handover/docs/." "$P/docs/"
for f in "$@"; do
  case "$f" in
    *.pt) cp "$f" "$P/demo/" ;;
    *.pdf) cp "$f" "$P/docs/paper/" ;;
    *.keystore) cp "$f" "$P/camapp/release.keystore" ;;
    *) echo "skipped (unknown type): $f" ;;
  esac
done
mkdir -p "$OUT"; rm -f "$OUT/SafeSense_LiveDemo_Handover.zip"
(cd "$T" && zip -qr "$OUT/SafeSense_LiveDemo_Handover.zip" SafeSense_LiveDemo_Handover)
rm -rf "$T"
ls -l "$OUT/SafeSense_LiveDemo_Handover.zip"
