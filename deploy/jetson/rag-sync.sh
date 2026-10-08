#!/usr/bin/env bash
# Pull the Jetson vault --ff-only from origin, reindex only if HEAD moved, then
# report drift. Never resets/stashes/forces. Env overrides exist for testing;
# the defaults are the real Jetson paths and commands.
set -euo pipefail

VAULT_DIR="${RAG_SYNC_VAULT_DIR:-$HOME/personal_knowledge/Career Knowledge Base}"
REPO_DIR="${RAG_SYNC_REPO_DIR:-$HOME/personal-rag}"
REMOTE="${RAG_SYNC_REMOTE:-origin}"
BRANCH="${RAG_SYNC_BRANCH:-main}"

INDEX_CMD="${RAG_SYNC_INDEX_CMD:-docker compose -f docker-compose.jetson.yml run --rm rag python -m rag.indexer --wait-for-lock 600}"
DRIFT_CMD="${RAG_SYNC_DRIFT_CMD:-docker compose -f docker-compose.jetson.yml run --rm rag rag-drift --remote $REMOTE}"

log() { echo "rag-sync: $*"; }

log "vault=$VAULT_DIR repo=$REPO_DIR remote=$REMOTE branch=$BRANCH"

if [ ! -d "$VAULT_DIR/.git" ]; then
    log "ERROR vault is not a git work tree"
    exit 1
fi

dirty="$(git -C "$VAULT_DIR" status --porcelain)"
if [ -n "$dirty" ]; then
    log "ERROR vault working tree is dirty; refusing to pull"
    echo "$dirty" | sed 's/^/rag-sync:   /'
    exit 1
fi

before="$(git -C "$VAULT_DIR" rev-parse HEAD)"
log "pulling --ff-only $REMOTE $BRANCH (HEAD $before)"
if ! git -C "$VAULT_DIR" pull --ff-only "$REMOTE" "$BRANCH"; then
    log "ERROR pull --ff-only failed (diverged or unreachable); not indexing"
    exit 1
fi
after="$(git -C "$VAULT_DIR" rev-parse HEAD)"

if [ "$before" = "$after" ]; then
    log "HEAD unchanged ($after); skipping index"
else
    log "HEAD $before -> $after; indexing"
    ( cd "$REPO_DIR" && eval "$INDEX_CMD" )
    log "index done"
fi

log "running container drift (--skip-git; git checked on host below)"
drift_rc=0
( cd "$REPO_DIR" && eval "$DRIFT_CMD --skip-git" ) || drift_rc=$?
log "container drift exit $drift_rc"

# Host-side git upstream check (the container has no git, no bare repo, and a
# read-only vault mount). origin/main was just fetched by the pull above.
host_git_drift=0
after_dirty="$(git -C "$VAULT_DIR" status --porcelain)"
if [ -n "$after_dirty" ]; then
    log "git: vault working tree dirty after pull"
    host_git_drift=1
fi
counts="$(git -C "$VAULT_DIR" rev-list --left-right --count "HEAD...$REMOTE/$BRANCH" 2>/dev/null || true)"
if [ -z "$counts" ]; then
    log "git: could not compare HEAD against $REMOTE/$BRANCH"
    host_git_drift=1
else
    behind="$(echo "$counts" | awk '{print $2}')"
    log "git: ${behind:-?} behind $REMOTE/$BRANCH"
    if [ "${behind:-0}" -gt 0 ]; then
        host_git_drift=1
    fi
fi

# Combine signals. host-side behind/dirty is observed drift (exit 1). The
# container drift returns 1 (drift) or 2 (could-not-verify); drift wins over
# could-not-verify. Exit non-zero so `systemctl --user --failed` surfaces it.
if [ "$host_git_drift" -ne 0 ] || [ "$drift_rc" -eq 1 ]; then
    log "RESULT drift detected (git_drift=$host_git_drift container_rc=$drift_rc); exiting 1"
    exit 1
elif [ "$drift_rc" -ne 0 ]; then
    log "RESULT could not verify (container_rc=$drift_rc); exiting $drift_rc"
    exit "$drift_rc"
fi
log "RESULT no drift"
exit 0
