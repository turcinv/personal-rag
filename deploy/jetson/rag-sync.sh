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

log "running drift"
drift_rc=0
( cd "$REPO_DIR" && eval "$DRIFT_CMD" ) || drift_rc=$?
log "drift exit $drift_rc"

exit 0
