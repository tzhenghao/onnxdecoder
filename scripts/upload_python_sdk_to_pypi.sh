#!/bin/bash

# This bash script is used to upload the Python SDK to PyPI.

set -euo pipefail

echo "Git checkout main..."
git checkout main
echo "Pulling latest changes..."
git pull

echo "Now proceeding with deployment workflow setup..."
cd ~/onnxdecoder/src

echo "Removing all other remaining dist packages..."
rm -rf dist/*

echo "Build source distribution..."
python3 -m build --sdist

echo "Build pure Python wheel (built) distribution..."
python3 -m build --wheel

echo "Checking if all distribution requirements are satisfied..."
twine check dist/*

echo "Proceed with upload to PyPI? Press [ENTER] to continue..."
read -r

echo "Uploading to PyPI..."
twine upload dist/*
echo "Upload complete!"
echo "View your updated package here: https://pypi.org/project/onnxdecoder"
