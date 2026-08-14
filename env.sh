#!/bin/bash
set -e

# One-time setup script: creates a Python virtual environment and installs
# all required packages.

echo "Creating virtual environment at ./venv ..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing packages from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "=== Setup complete ==="
