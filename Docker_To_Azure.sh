#!/usr/bin/env bash
# Usage: ./Docker_To_Azure.sh [TAG]
# Default tag: production_uniqueID_v6

TAG=${1:-production_uniqueID_v6}

# Build the Docker image locally
docker build -t guivieiraa/ragka-v1:$TAG .

# Push the image to Docker Hub
docker push guivieiraa/ragka-v1:$TAG

# Configure Azure Web App to pull the pushed image
az webapp config container set \
  --name ragka-v1 \
  --resource-group rg-sbx-19-switzerlandnorth-usr-capozzol \
  --docker-custom-image-name guivieiraa/ragka-v1:$TAG \
  --docker-registry-server-url https://index.docker.io

# Restart the Azure Web App to pick up the new image
az webapp restart \
  --name ragka-v1 \
  --resource-group rg-sbx-19-switzerlandnorth-usr-capozzol
