import { NavLink, Outlet } from 'react-router-dom'
import { useAuth } from '../auth'

const links = [
  { to: '/', label: 'Dashboard' },
  { to: '/train', label: 'Train' },
  { to: '/saved', label: 'Saved' },
  { to: '/history', label: 'History' },
  { to: '/leaderboard', label: 'Leaderboard' },
  { to: '/coach', label: 'Coach Lab' },
]

export default function Layout() {
  const { user, logout } = useAuth()
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-ink-800 bg-ink-900/80 backdrop-blur sticky top-0 z-20">
        <div className="max-w-6xl mx-auto px-4 py-2 md:h-14 md:py-0 flex flex-wrap md:flex-nowrap items-center gap-x-6 gap-y-2">
          <NavLink to="/" className="flex items-center gap-2 font-bold text-lg shrink-0">
            <span className="text-2xl">♞</span>
            <span>
              Grandmaster<span className="text-brand">RL</span>
            </span>
          </NavLink>
          <nav className="flex gap-1 overflow-x-auto order-last md:order-none w-full md:w-auto -mx-1 md:mx-0">
            {links.map((l) => (
              <NavLink
                key={l.to}
                to={l.to}
                end={l.to === '/'}
                className={({ isActive }) =>
                  `px-3 py-1.5 rounded-md text-sm whitespace-nowrap ${
                    isActive ? 'bg-ink-800 text-ink-100' : 'text-ink-300 hover:text-ink-100'
                  }`
                }
              >
                {l.label}
              </NavLink>
            ))}
          </nav>
          {user && (
            <div className="ml-auto flex items-center gap-4 text-sm shrink-0">
              <span className="hidden sm:inline text-gold" title="Daily streak">
                🔥 {user.current_streak}
              </span>
              <span className="hidden sm:inline text-ink-300">
                <span className="font-semibold text-ink-100">{Math.round(user.rating)}</span> · Lv {user.level}
              </span>
              <button onClick={logout} className="text-ink-500 hover:text-ink-100">
                {user.username} ⎋
              </button>
            </div>
          )}
        </div>
      </header>
      <main className="flex-1 max-w-6xl w-full mx-auto px-4 py-6">
        <Outlet />
      </main>
    </div>
  )
}
