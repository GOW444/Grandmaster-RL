import { useState, type FormEvent } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../auth'

export default function AuthPage({ mode }: { mode: 'login' | 'register' }) {
  const { login, register } = useAuth()
  const nav = useNavigate()
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setError(null)
    setBusy(true)
    try {
      if (mode === 'login') await login(username, password)
      else await register(username, email, password)
      nav('/')
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="min-h-screen grid lg:grid-cols-2">
      <div className="hidden lg:flex flex-col justify-center px-16 bg-gradient-to-br from-ink-900 to-ink-950 border-r border-ink-800">
        <div className="text-7xl mb-6">♞</div>
        <h1 className="text-4xl font-bold leading-tight">
          Train tactics with an
          <br />
          <span className="text-brand">AI curriculum coach.</span>
        </h1>
        <p className="mt-4 text-ink-300 max-w-md">
          200k+ real Lichess puzzles. A reinforcement-learning policy (PPO) picks the theme and difficulty that
          should teach you fastest — and tracks whether it actually does.
        </p>
        <ul className="mt-8 space-y-2 text-ink-300 text-sm">
          <li>🎯 Per-theme ratings: forks, pins, mates, endgames, skewers, discoveries</li>
          <li>📈 Stats on accuracy, speed and where to improve</li>
          <li>🔥 Streaks, XP, levels and badges</li>
        </ul>
      </div>
      <div className="flex items-center justify-center p-6">
        <form onSubmit={submit} className="card w-full max-w-sm p-8 space-y-4">
          <div className="lg:hidden text-4xl text-center">♞</div>
          <h2 className="text-2xl font-bold">{mode === 'login' ? 'Welcome back' : 'Create your account'}</h2>
          <div className="space-y-1">
            <label className="label">{mode === 'login' ? 'Username or email' : 'Username'}</label>
            <input className="input" value={username} onChange={(e) => setUsername(e.target.value)} required autoFocus />
          </div>
          {mode === 'register' && (
            <div className="space-y-1">
              <label className="label">Email</label>
              <input className="input" type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
            </div>
          )}
          <div className="space-y-1">
            <label className="label">Password</label>
            <input
              className="input"
              type="password"
              value={password}
              minLength={mode === 'register' ? 8 : undefined}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>
          {error && <p className="text-sm text-bad">{error}</p>}
          <button className="btn-primary w-full" disabled={busy}>
            {busy ? '…' : mode === 'login' ? 'Log in' : 'Sign up'}
          </button>
          <p className="text-sm text-ink-500 text-center">
            {mode === 'login' ? (
              <>
                New here? <Link to="/register" className="text-brand">Create an account</Link>
              </>
            ) : (
              <>
                Have an account? <Link to="/login" className="text-brand">Log in</Link>
              </>
            )}
          </p>
        </form>
      </div>
    </div>
  )
}
