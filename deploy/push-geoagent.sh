#!/usr/bin/env bash
# Сборка и push образа GeoAgent. Пример: ./push-geoagent.sh ghcr.io/myuser
set -euo pipefail
REGISTRY="${1:-}"
TAG="${2:-latest}"
IMAGE_NAME="${3:-geo-deploy-geoagent}"
if [[ -z "$REGISTRY" ]]; then
  echo "Usage: $0 <registry[/namespace]> [tag] [image_name]" >&2
  echo "Example: $0 ghcr.io/myuser latest geo-deploy-geoagent" >&2
  exit 1
fi
REGISTRY="${REGISTRY%/}"
cd "$(dirname "$0")"

docker compose build geoagent-shell
REMOTE="${REGISTRY}/${IMAGE_NAME}:${TAG}"
docker tag geo-deploy-geoagent:latest "$REMOTE"
docker push "$REMOTE"
echo ""
echo "On the server, set in deploy/.env:"
echo "  GEOAGENT_IMAGE=$REMOTE"
