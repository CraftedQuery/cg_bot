# Security Features

This document outlines the main security mechanisms implemented in the Multi-Tenant RAG Chatbot project. It can be used as a reference when working toward SOC2 certification or when sharing security details with external partners.

## Authentication and Authorization

- **JWT-based access tokens**: Tokens are generated with an expiration time and signed using a secret key. The secret can be overridden via the `JWT_SECRET_KEY` environment variable.
- **Password hashing**: User passwords are hashed using `bcrypt` before storage.
- **Role hierarchy**: The system enforces roles of `system_admin`, `admin`, and `user` to protect sensitive routes.

## Configuration Security

- **Allowed domains**: Widget embedding is restricted by a configurable list of allowed domains.
- **Secret key management**: Default JWT secrets are defined in configuration but should be overridden in production.

## Network and Data Protection

- **HTTPS recommended**: Deploy behind HTTPS in production environments.
- **CORS middleware**: Requests are filtered via CORS to control cross‑origin calls.
- **Audit logs**: Chat logs store user IP addresses for audit purposes.

## Additional Considerations

- Always change default credentials on first install.
- Keep LLM provider API keys out of version control.
- Tenants are isolated — admin routes verify that users only access their own tenants.

