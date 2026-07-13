#!/usr/bin/env bash
# Publish the roxlap workspace to crates.io, in dependency order.
#
#   scripts/publish.sh            # publish every crate of the current
#                                 # workspace version
#   scripts/publish.sh --dry-run  # print the plan, publish nothing
#
# Why a fixed order: the workspace's path dependencies carry version
# pins (`path = …, version = "0.X"`), and `cargo publish` resolves
# those against the crates.io index — so a crate can only go up after
# everything it depends on is already there. (This is also why
# `cargo package` on a non-leaf crate fails BEFORE a release is out:
# the new version does not exist in the index yet. Expected, not a
# bug.)
#
# Resumable: a version that is already on crates.io is skipped, so a
# run that died halfway (network, rate limit) can simply be re-run.
# `cargo publish` itself blocks until each crate is visible in the
# index, so no sleeps between crates are needed.
#
# Anti-rot gate (the check-anchors.sh pattern): the ORDER list below
# must cover exactly the publishable crates of the workspace — adding
# a new crate without slotting it here fails loudly instead of
# silently not publishing it.
set -euo pipefail
cd "$(dirname "$0")/.."

# Dependency-ordered publish list. The comment is each crate's
# in-workspace dependencies — a new crate goes AFTER everything it
# depends on.
ORDER=(
    roxlap-formats # (no roxlap dependencies)
    roxlap-cavegen # formats
    roxlap-core    # formats
    roxlap-gpu     # formats
    roxlap-scene   # cavegen, core, formats
    roxlap-audio   # scene
    roxlap-cli     # formats, scene
    roxlap-render  # core, formats, gpu, scene
)

version=$(grep -m1 '^version = ' Cargo.toml | sed -E 's/version = "(.*)".*/\1/')
[[ -n "$version" ]] || { echo "cannot read workspace version from Cargo.toml" >&2; exit 1; }

# ---- anti-rot gate: ORDER == the set of publishable crates ----
expected=$(for t in crates/*/Cargo.toml; do
    grep -q '^publish = false' "$t" || basename "$(dirname "$t")"
done | sort)
listed=$(printf '%s\n' "${ORDER[@]}" | sort)
if [[ "$expected" != "$listed" ]]; then
    echo "publish.sh: ORDER does not match the workspace's publishable crates." >&2
    echo "--- publishable (crates/ without 'publish = false'):" >&2
    echo "$expected" >&2
    echo "--- listed in ORDER:" >&2
    echo "$listed" >&2
    echo "Slot the new crate into ORDER after its dependencies (or mark it 'publish = false')." >&2
    exit 1
fi

# ---- pre-flight warnings (cargo publish enforces the hard rules) ----
# `--porcelain` (not `git diff`): untracked files count as dirty for
# `cargo publish` too.
if [[ -n "$(git status --porcelain)" ]]; then
    echo "WARNING: working tree is dirty — cargo publish will refuse (commit first)." >&2
fi
if ! git tag -l "v$version" | grep -q .; then
    echo "WARNING: tag v$version does not exist yet (house flow: commit + tag before publishing)." >&2
fi

echo "publishing roxlap $version: ${ORDER[*]}"
if [[ "${1:-}" == "--dry-run" || "${1:-}" == "-n" ]]; then
    echo "(dry run — nothing published)"
    exit 0
fi

for c in "${ORDER[@]}"; do
    echo "== $c $version"
    if out=$(cargo publish -p "$c" 2>&1); then
        echo "   published"
    elif grep -q "already uploaded\|already exists" <<<"$out"; then
        echo "   already on crates.io — skipped (resume)"
    else
        echo "$out" >&2
        echo "publish.sh: '$c' failed — fix and re-run (published crates are skipped)." >&2
        exit 1
    fi
done
echo "all ${#ORDER[@]} crates at $version"
