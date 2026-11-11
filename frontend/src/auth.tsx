import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from 'react'
import { api, tokenStore, type User } from './api'

interface AuthCtx {
  user: User | null
  loading: boolean
  login: (username: string, password: string) => Promise<void>
  register: (username: string, email: string, password: string) => Promise<void>
  logout: () => void
  refresh: () => Promise<void>
}

const Ctx = createContext<AuthCtx | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)

  const refresh = useCallback(async () => {
    if (!tokenStore.get()) {
      setUser(null)
      return
    }
    try {
      setUser(await api.me())
    } catch {
      setUser(null)
    }
  }, [])

  useEffect(() => {
    refresh().finally(() => setLoading(false))
    const onLogout = () => setUser(null)
    window.addEventListener('gmrl:logout', onLogout)
    return () => window.removeEventListener('gmrl:logout', onLogout)
  }, [refresh])

  const login = async (username: string, password: string) => {
    const { access_token } = await api.login(username, password)
    tokenStore.set(access_token)
    await refresh()
  }
  const register = async (username: string, email: string, password: string) => {
    const { access_token } = await api.register(username, email, password)
    tokenStore.set(access_token)
    await refresh()
  }
  const logout = () => {
    tokenStore.set(null)
    setUser(null)
  }

  return <Ctx.Provider value={{ user, loading, login, register, logout, refresh }}>{children}</Ctx.Provider>
}

// eslint-disable-next-line react-refresh/only-export-components
export function useAuth() {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useAuth outside AuthProvider')
  return ctx
}
