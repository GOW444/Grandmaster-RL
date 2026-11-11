import type { ReactNode } from 'react'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { AuthProvider, useAuth } from './auth'
import Layout from './components/Layout'
import AuthPage from './pages/AuthPage'
import CoachLab from './pages/CoachLab'
import Dashboard from './pages/Dashboard'
import History from './pages/History'
import Leaderboard from './pages/Leaderboard'
import Saved from './pages/Saved'
import Train from './pages/Train'

function RequireAuth({ children }: { children: ReactNode }) {
  const { user, loading } = useAuth()
  if (loading) return <div className="p-8 text-ink-500">Loading…</div>
  return user ? children : <Navigate to="/login" replace />
}

function GuestOnly({ children }: { children: ReactNode }) {
  const { user, loading } = useAuth()
  if (loading) return null
  return user ? <Navigate to="/" replace /> : children
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<GuestOnly><AuthPage mode="login" /></GuestOnly>} />
          <Route path="/register" element={<GuestOnly><AuthPage mode="register" /></GuestOnly>} />
          <Route element={<RequireAuth><Layout /></RequireAuth>}>
            <Route index element={<Dashboard />} />
            <Route path="train" element={<Train />} />
            <Route path="saved" element={<Saved />} />
            <Route path="history" element={<History />} />
            <Route path="leaderboard" element={<Leaderboard />} />
            <Route path="coach" element={<CoachLab />} />
          </Route>
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  )
}
