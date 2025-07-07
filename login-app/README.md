# Login App

This directory contains a minimal Node.js/Express server with local and Microsoft Entra ID authentication. The front-end is a simple React page served at `/static/login.html`.

## Setup

1. **Create a Microsoft Entra app registration**
   - Note the **Client ID**, **Tenant ID** and add a web redirect URI (e.g. `http://localhost:3000/static/login.html`).
2. **Configure environment variables**
   - Copy `.env.example` to `.env` and set `JWT_SECRET`, `DATABASE_URL` and `PORT`.
   - Insert the Entra values into `static/login.html` (clientId, authority and redirectUri comments).
3. **Install dependencies and run migrations**
   ```bash
   npm install
   npx prisma migrate dev --name init
   npm run dev
   ```
4. **Optional: seed a local user**
   ```bash
   npm run seed
   ```

The application serves `static/login.html`. After signing in with Microsoft you will be redirected to `user.html`. Local sign‑in stores a `session` cookie and allows access to `/dashboard`.
