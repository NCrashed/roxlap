#!/usr/bin/env bash
# BK.0: the book's anti-rot gate. mdbook (0.5.2) does NOT fail the
# build on a broken `{{#include}}`: a missing target file logs an
# ERROR and exits 0, and a missing `ANCHOR:` in an existing file
# silently includes nothing. This script resolves every include in
# the book source and fails on either, so CI (and a local run) goes
# red the moment a snippet anchor rots. Run from anywhere:
#
#   docs/book/check-anchors.sh
set -euo pipefail

src_dir="$(cd "$(dirname "$0")/src" && pwd)"
fail=0

for md in "$src_dir"/*.md; do
    while read -r inc; do
        # `inc` is `{{#include <path>[:anchor|:from[:to]]}}`.
        spec="${inc#\{\{#include }"
        spec="${spec%\}\}}"
        path="${spec%%:*}"
        anchor=""
        [[ "$spec" == *:* ]] && anchor="${spec#*:}"

        target="$(dirname "$md")/$path"
        if [[ ! -f "$target" ]]; then
            echo "BROKEN INCLUDE: $md: no such file: $path" >&2
            fail=1
            continue
        fi
        # Pure-numeric suffixes are line ranges (policy says avoid
        # them — they rot invisibly); only named anchors are checked.
        if [[ -n "$anchor" && ! "$anchor" =~ ^[0-9:]+$ ]]; then
            for mark in "ANCHOR" "ANCHOR_END"; do
                if ! grep -qE "$mark:[[:space:]]*$anchor([^A-Za-z0-9_-]|\$)" "$target"; then
                    echo "BROKEN INCLUDE: $md: no '$mark: $anchor' in $path" >&2
                    fail=1
                fi
            done
        fi
    done < <(grep -o '{{#include [^}]*}}' "$md" || true)
done

# BK — cross-reference chapter NUMBERS rot when a chapter is inserted
# or moved in SUMMARY.md (nothing renumbers the `[N](file.md)` /
# `[chapter N](file.md)` links in prose). Validate every LINKED numeric
# reference against the target's 1-based position in SUMMARY.md. (Bare
# unlinked "chapter N" prose has no file to map to and isn't covered —
# prefer the linked form.)
declare -A chnum
n=0
while IFS= read -r file; do
    n=$((n + 1))
    chnum["$file"]=$n
done < <(grep -oE '^[[:space:]]*- \[[^]]*\]\([A-Za-z0-9._-]+\.md\)' "$src_dir/SUMMARY.md" \
    | sed -E 's/.*\(([A-Za-z0-9._-]+\.md)\)/\1/')

for md in "$src_dir"/*.md; do
    [[ "$(basename "$md")" == "SUMMARY.md" ]] && continue
    while read -r ref; do
        # `ref` is `[N](file.md)` or `[chapter N](file.md)`.
        num="$(printf '%s' "$ref" | grep -oE '[0-9]+' | head -1)"
        file="$(printf '%s' "$ref" | sed -E 's/.*\(([A-Za-z0-9._-]+\.md)\)/\1/')"
        expected="${chnum[$file]:-}"
        if [[ -z "$expected" ]]; then
            echo "STALE XREF: $md: '$ref' targets '$file', not a numbered SUMMARY chapter" >&2
            fail=1
        elif [[ "$num" != "$expected" ]]; then
            echo "STALE XREF: $md: '$ref' says chapter $num but '$file' is chapter $expected" >&2
            fail=1
        fi
    done < <(grep -oE '\[(chapter |Chapter )?[0-9]+\]\([A-Za-z0-9._-]+\.md\)' "$md" || true)
done

if [[ "$fail" -ne 0 ]]; then
    echo "check-anchors: FAILED — fix the includes / cross-references above" >&2
else
    echo "check-anchors: all book includes + chapter cross-refs resolve"
fi
exit "$fail"
