#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: bash ops/runpod_image/build_image.sh <image-ref>" >&2
  exit 2
fi

IMAGE_REF="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"

cd "$ROOT_DIR"
docker build --platform "$PLATFORM" -f ops/runpod_image/Dockerfile -t "$IMAGE_REF" .

cat <<EOF
Built image:
  $IMAGE_REF
Platform:
  $PLATFORM

Next steps:
  docker run --rm --platform $PLATFORM $IMAGE_REF bash /opt/parameter-golf/validate_image.sh
  docker push $IMAGE_REF
EOF
