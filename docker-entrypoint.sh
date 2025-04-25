#!/bin/bash
set -e

# Print banner
echo "====================================================="
echo "  Ethical AI Assessment Tool - Docker Edition"
echo "====================================================="

# Check if environment variables are set for API keys and pass them to the script
if [ -n "$OPENAI_API_KEY" ]; then
    echo "Using OPENAI_API_KEY from environment variable"
    export OPENAI_API_KEY="$OPENAI_API_KEY"
fi

if [ -n "$GEMINI_API_KEY" ]; then
    echo "Using GEMINI_API_KEY from environment variable"
    export GEMINI_API_KEY="$GEMINI_API_KEY"
fi

if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo "Using ANTHROPIC_API_KEY from environment variable"
    export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"
fi

if [ -n "$LMSTUDIO_API_KEY" ]; then
    echo "Using LMSTUDIO_API_KEY from environment variable"
    export LMSTUDIO_API_KEY="$LMSTUDIO_API_KEY"
fi

if [ -n "$GENERIC_API_KEY" ]; then
    echo "Using GENERIC_API_KEY from environment variable"
    export GENERIC_API_KEY="$GENERIC_API_KEY"
fi

# Print information about data locations
echo "Results will be saved to: /app/results/"
echo "Config file location: /app/config.json"

# Execute the Python script with any arguments passed
echo "Running assessment with args: $@"
echo "-----------------------------------------------------"
exec python ethical_ai_assessment.py "$@"