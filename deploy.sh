#!/bin/bash

# Nombre de la imagen y del repositorio ECR
IMAGE_NAME="jarvid-build"
ECR_URI="101344062716.dkr.ecr.us-east-2.amazonaws.com/jarvid"

# Leer la última versión de la imagen
VERSION_FILE="version.txt"

if [ ! -f "$VERSION_FILE" ]; then
    echo "1.0" > "$VERSION_FILE"  # Si no existe el archivo, inicia en 1.0
fi

VERSION=$(cat "$VERSION_FILE")
echo "Current version: $VERSION"

# Incrementar la versión
IFS='.' read -r major minor <<< "$VERSION"
minor=$((minor + 1))
NEW_VERSION="${major}.${minor}"

# Construir y etiquetar la imagen
docker build -t "$IMAGE_NAME:$NEW_VERSION" .
docker tag "$IMAGE_NAME:$NEW_VERSION" "$ECR_URI:$NEW_VERSION"

# Empujar la imagen a ECR
docker push "$ECR_URI:$NEW_VERSION"

# Actualizar la función Lambda
aws lambda update-function-code \
    --function-name jarvid \
    --image-uri "$ECR_URI:$NEW_VERSION"

# Guardar la nueva versión en el archivo
echo "$NEW_VERSION" > "$VERSION_FILE"

echo "Deployed version: $NEW_VERSION"
