#!/bin/bash
# Startup script for the RAG Chatbot

# 1. Set environment variables
# Fill in your keys below or export them before running this script.
export OPENAI_API_KEY="your-openai-api-key"
export JWT_SECRET_KEY="your-jwt-secret"
# Optional providers
export ANTHROPIC_API_KEY=""
export GOOGLE_APPLICATION_CREDENTIALS=""

# Microsoft Entra integration (optional)
export AAD_TENANT_ID=""
export AAD_CLIENT_ID=""
export AAD_JWKS_PATH=""

# 2. Activate virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Install dependencies if needed
pip install -r requirements.txt

# 3. Start the server
python -m rag_chatbot.cli serve --reload
