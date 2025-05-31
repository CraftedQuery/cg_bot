echo "Generating JWT Secret Key..."
JWT_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")

echo "creating .env file..."
cat > .env <<EOF
OPENAI_API_KEY=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
JWT_SECRET_KEY=$JWT_KEY
GOOGLE_APPLICATION_CREDENTIALS=/path/to/google/credentials.json
EOF

echo "JWT_SECRET_KEY=$JWT_KEY"
