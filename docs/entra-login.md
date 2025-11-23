# Microsoft Entra Integration Guide

This guide explains how to connect the chatbot stack to Microsoft Entra ID (Azure AD) so tenants can sign in with their own identities. The repository includes a sample React SPA (`spa/`) and Express API (`login-app/`) you can use during testing, while the FastAPI backend can validate Entra-issued JWTs directly.

## 1. Register applications in Entra

Perform these steps in the tenant that will host the chatbot:

1. **Create a SPA application** and record its **Application (client) ID**.
   - Enable the *Authorization code* and *Implicit* grants for ID and access tokens.
   - Set the redirect URI to `http://localhost:5173` for local development.
2. **Create an API application** and expose a custom API scope. Note the **Application ID URI**.
3. In the SPA application's **API permissions**, grant access to the API scope you created.

The SPA uses MSAL to acquire ID/access tokens, and the backend validates them against the published JWKS.

## 2. Enforce MFA via Conditional Access

To require multi-factor authentication for the SPA:

1. Navigate to **Entra ID → Protection → Conditional Access**.
2. Create a **New policy** and select the SPA's application ID under **Cloud apps**.
3. Under **Grant**, select **Require multi-factor authentication** and enable the policy.

MFA enforcement occurs through Conditional Access, not in the application code.

## 3. Configure environment variables

Both the SPA/API demo and the FastAPI service read their Entra identifiers from environment variables:

**`spa/.env`**
```env
VITE_CLIENT_ID=YOUR_CLIENT_ID
VITE_TENANT_ID=YOUR_TENANT_ID
VITE_AUTHORITY=https://login.microsoftonline.com
VITE_REDIRECT_URI=http://localhost:5173
API_BASE=http://localhost:3001
```

**`login-app/.env` (Express demo API)**
```env
CLIENT_ID=YOUR_CLIENT_ID
TENANT_ID=YOUR_TENANT_ID
AUTHORITY=https://login.microsoftonline.com
```

**FastAPI backend**
```bash
export AAD_TENANT_ID=YOUR_TENANT_ID
export AAD_CLIENT_ID=YOUR_CLIENT_ID
export AAD_JWKS_PATH=/path/to/azure-ad-jwks.json
```

The backend reads the JWKS from `AAD_JWKS_PATH` and accepts Entra-issued JWTs via `authenticate_aad_token`.

## 4. Local development

Start the front end and demo API using the commands below:

```bash
cd spa && npm install && npm run dev      # start React + Vite SPA
cd ../login-app && npm install && node index.js # start Express API
```

Run the chatbot API separately with `python cli.py serve --reload` so the widget and MSAL callbacks can reach it at `http://localhost:8000`.

## 5. Client integration steps

Tenants that want to bring their own Entra tenant should:

1. Register their own SPA and API applications (repeat Step 1 with their tenant IDs).
2. Share the client ID and tenant ID so you can set the FastAPI environment variables per tenant.
3. Embed the chat widget on their site with the desired tenant/agent parameters:
   ```html
   <script src="http://your-server.com/widget.js?tenant=their-tenant&agent=their-agent"></script>
   ```
4. Test login and API access to ensure tokens from the client's Entra ID are accepted.

With this setup, each tenant controls authentication while relying on the shared chatbot infrastructure.
