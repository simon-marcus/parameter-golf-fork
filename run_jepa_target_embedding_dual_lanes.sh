#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

normalize_namespace() {
  case "$1" in
    jepa_target_embedding_discovery_freecode)
      printf '%s' "jepa_target_embedding_freecode_discovery"
      ;;
    *)
      printf '%s' "$1"
      ;;
  esac
}

CONSTRAINED_NAMESPACE="$(normalize_namespace "${CONSTRAINED_NAMESPACE:-jepa_target_embedding_discovery_constrained}")"
FREECODE_NAMESPACE="$(normalize_namespace "${FREECODE_NAMESPACE:-jepa_target_embedding_freecode_discovery}")"

echo "Launching JEPA target-embedding dual lanes"
echo "  constrained namespace: $CONSTRAINED_NAMESPACE"
echo "  freecode namespace:    $FREECODE_NAMESPACE"

AUTORESEARCH_NAMESPACE="$CONSTRAINED_NAMESPACE" BACKGROUND=1 \
  bash "$ROOT_DIR/run_jepa_target_embedding_autoresearch.sh" constrained

AUTORESEARCH_NAMESPACE="$FREECODE_NAMESPACE" BACKGROUND=1 \
  bash "$ROOT_DIR/run_jepa_target_embedding_autoresearch.sh" freecode

echo "Both lanes started."
echo "Watch constrained: AUTORESEARCH_NAMESPACE=$CONSTRAINED_NAMESPACE bash ./watch_jepa_target_embedding_autoresearch.sh"
echo "Watch freecode:    AUTORESEARCH_NAMESPACE=$FREECODE_NAMESPACE bash ./watch_jepa_target_embedding_autoresearch.sh"
