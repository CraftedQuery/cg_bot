require('dotenv').config()
const express = require('express')
const { msalExpressMiddleware } = require('microsoft-identity-express')

const app = express()

const msalConfig = {
  auth: {
    clientId: process.env.CLIENT_ID,
    authority: `${process.env.AUTHORITY}/${process.env.TENANT_ID}`,
    // clientSecret optional for SPA tokens
  }
}

app.use(msalExpressMiddleware(msalConfig)) // MFA enforced via Conditional Access policies in Entra

app.get('/me', (req, res) => {
  const c = req.authInfo || {}
  res.json({ preferred_username: c.preferred_username, tid: c.tid })
})

app.listen(3001, () => console.log('API listening on 3001'))
