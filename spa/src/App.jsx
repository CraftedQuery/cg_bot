import { useMsal, useIsAuthenticated } from '@azure/msal-react'
import { loginRequest } from './authConfig'

export default function App() {
  const { instance, accounts } = useMsal()
  const isAuthenticated = useIsAuthenticated()

  const login = () => instance.loginRedirect(loginRequest)

  const callMe = async () => {
    const account = accounts[0]
    const result = await instance.acquireTokenSilent({ ...loginRequest, account })
    const res = await fetch(`${import.meta.env.API_BASE}/me`, {
      headers: { Authorization: `Bearer ${result.idToken}` }
    })
    alert(JSON.stringify(await res.json()))
  }

  return (
    <div style={{ padding: 40 }}>
      {isAuthenticated ? (
        <>
          <button onClick={callMe}>Call API</button>
        </>
      ) : (
        <button onClick={login}>Login</button>
      )}
    </div>
  )
}
