# Microsoft Entra Integration Guide

This guide explains how to configure Microsoft Entra ID so users can sign in with their Microsoft accounts and how tenants can integrate with the Crafted Query application.

## 1. Register Applications in Entra

Follow the steps below on the tenant that will host the Crafted Query instance.

1. **Create a SPA application** and record its **Application (client) ID**.
   - Enable the *Authorization code* and *Implicit* grants for ID and access tokens.
   - Set the redirect URI to `http://localhost:5173` for local development.
2. **Create an API application** and expose a custom API scope. Note the **Application ID URI**.
3. In the SPA application's **API permissions**, grant access to the API scope you created.

These steps mirror the "Azure registration" instructions found in the project README.

## 2. Enforce MFA via Conditional Access

To require multi‑factor authentication for the SPA:

1. Navigate to **Entra ID → Protection → Conditional Access**.
2. Create a **New policy** and select the SPA's application ID under **Cloud apps**.
3. Under **Grant**, select **Require multi-factor authentication** and enable the policy.

MFA enforcement occurs through Conditional Access, not in the application code.

## 3. Configure Environment Variables

Both the SPA and API read their Entra IDs from `.env` files:

**`spa/.env`**
```env
VITE_CLIENT_ID=YOUR_CLIENT_ID
VITE_TENANT_ID=YOUR_TENANT_ID
VITE_AUTHORITY=https://login.microsoftonline.com
VITE_REDIRECT_URI=http://localhost:5173
API_BASE=http://localhost:3001
```
**`api/.env`**
```env
CLIENT_ID=YOUR_CLIENT_ID
TENANT_ID=YOUR_TENANT_ID
AUTHORITY=https://login.microsoftonline.com
```
Replace the placeholders above with the values from your Entra applications.

## 4. Local Development

Start the front‑end and API using the commands from the README:

```bash
cd spa && npm i && npm run dev      # start React + Vite SPA
cd ../api && npm i && node index.js # start Express API
```

When you sign in through the SPA, MSAL obtains an ID token from Entra ID. The Express API validates the token via `microsoft-identity-express` before returning user information from the `/me` endpoint.

## 5. Client Integration Steps

Tenants that wish to use Crafted Query with their own Entra ID should:

1. Register their own SPA and API applications in their tenant, following the same steps as above.
2. Provide the resulting client IDs and tenant ID to the Crafted Query administrators so the `.env` files can be updated per tenant.
3. Embed the chat widget on their site with the desired tenant and agent parameters:
   ```html
   <script src="http://your-server.com/widget.js?tenant=your-tenant&agent=your-agent"></script>
   ```
4. Test login and API access to ensure tokens from the client's Entra ID are accepted.

This configuration allows each tenant to control authentication while using the Crafted Query platform.

