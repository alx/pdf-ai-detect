#!/bin/bash
# Setup script for pdf-ai-detect

set -e

echo "Installing Python dependencies..."
pip install -r requirements.txt

# Clone fast-detect-gpt if not already present
if [ ! -d "fast-detect-gpt" ]; then
    echo "Cloning fast-detect-gpt repository..."
    git clone https://github.com/baoguangsheng/fast-detect-gpt.git
    cd fast-detect-gpt

    # Run the setup script if it exists
    if [ -f "setup.sh" ]; then
        echo "Running fast-detect-gpt setup..."
        bash setup.sh
    fi

    cd ..
else
    echo "fast-detect-gpt already exists, skipping clone..."
fi

echo "Setup complete!"
echo "You can now run: python pdf_ai_colorize.py <input.pdf> <output.pdf>"
