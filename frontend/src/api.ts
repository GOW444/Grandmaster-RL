export const THEMES = ['fork', 'pin', 'mate', 'endgame', 'skewer', 'discovery'] as const
export type Theme = (typeof THEMES)[number]

export interface User {
  id: number
  username: string
  email: string
  rating: number
  xp: number
  level: number
  xp_into_level: number
  xp_for_next: number
  current_streak: number
  longest_streak: number
  created_at: string
}

export interface Puzzle {
  id: string
  fen: string
  moves: string[]
  rating: number
  themes: string[]
  primary_theme: Theme
  game_url: string | null
}

export interface NextPuzzle {
  puzzle: Puzzle
  mode: string
  selector: 'ppo' | 'rating_match' | null
  target_rating: number | null
  reason: string | null
  saved: boolean
}

export interface Badge {
  code: string
  name: string
  desc: string
  icon: string
  earned_at?: string | null
}

export interface AttemptResult {
  solved: boolean
  theme: Theme
  theme_rating_before: number
  theme_rating_after: number
  rating_before: number
  rating_after: number
  xp_gained: number
  level: number
  new_badges: Badge[]
  streak: number
}

export interface ThemeStat {
  theme: Theme
  rating: number
  attempts: number
  solved: number
  accuracy: number | null
  avg_time_ms: number | null
}

export interface Overview {
  user: User
  themes: ThemeStat[]
  focus: { theme: Theme; why: string }[]
  untried: Theme[]
  totals: { attempts: number; solved: number; avg_time_ms: number }
}

export interface ABStats {
  attempts: number
  success_rate: number
  avg_rating_gain: number
  avg_stretch: number
}

export interface AttemptRow {
  id: number
  puzzle_id: string
  theme: Theme
  mode: string
  selector: string | null
  solved: boolean
  time_ms: number
  puzzle_rating: number
  rating_after: number
  created_at: string
}

export interface BrowseItem extends Puzzle {
  popularity: number
  nb_plays: number
  status: 'unseen' | 'solved' | 'failed'
}

export interface RacePlayerState {
  username: string
  score: number
  errors: number
  best_rating: number
  finished: boolean
  you: boolean
}

export interface RaceState {
  id: number
  code: string
  visibility: 'solo' | 'private' | 'public'
  status: 'waiting' | 'running' | 'finished'
  duration_s: number
  max_errors: number
  host: string | null
  is_host: boolean
  server_now: string
  started_at: string | null
  ends_at: string | null
  players: RacePlayerState[]
  me: { score: number; errors: number; index: number; finished: boolean } | null
  puzzle?: Puzzle
}

const TOKEN_KEY = 'gmrl_token'

export const tokenStore = {
  get: () => {
    try {
      return localStorage.getItem(TOKEN_KEY)
    } catch {
      return null
    }
  },
  set: (t: string | null) => {
    try {
      if (t) localStorage.setItem(TOKEN_KEY, t)
      else localStorage.removeItem(TOKEN_KEY)
    } catch {
      /* storage unavailable */
    }
  },
}

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers)
  const token = tokenStore.get()
  if (token) headers.set('Authorization', `Bearer ${token}`)
  if (init.body && typeof init.body === 'string') headers.set('Content-Type', 'application/json')
  const res = await fetch(`/api${path}`, { ...init, headers })
  if (res.status === 401) {
    tokenStore.set(null)
    window.dispatchEvent(new Event('gmrl:logout'))
  }
  if (!res.ok) {
    let msg = res.statusText
    try {
      const body = await res.json()
      msg = typeof body.detail === 'string' ? body.detail : body.detail?.[0]?.msg ?? msg
    } catch {
      /* non-JSON error */
    }
    throw new ApiError(res.status, msg)
  }
  return res.status === 204 ? (undefined as T) : res.json()
}

const json = (body: unknown) => JSON.stringify(body)

export const api = {
  register: (username: string, email: string, password: string) =>
    request<{ access_token: string }>('/auth/register', { method: 'POST', body: json({ username, email, password }) }),
  login: (username: string, password: string) =>
    request<{ access_token: string }>('/auth/login', {
      method: 'POST',
      body: new URLSearchParams({ username, password }),
    }),
  me: () => request<User>('/auth/me'),

  next: (
    mode: 'coach' | 'rated' | 'theme' | 'custom',
    opts: { theme?: Theme; selector?: string; min?: number; max?: number } = {},
  ) => {
    const q = new URLSearchParams({ mode })
    if (opts.theme) q.set('theme', opts.theme)
    if (opts.selector) q.set('selector', opts.selector)
    if (opts.min !== undefined) q.set('min_rating', String(opts.min))
    if (opts.max !== undefined) q.set('max_rating', String(opts.max))
    return request<NextPuzzle>(`/puzzles/next?${q}`)
  },
  browse: (f: { min: number; max: number; theme?: string; status: string; sort: string; page: number }) => {
    const q = new URLSearchParams({
      min_rating: String(f.min),
      max_rating: String(f.max),
      status: f.status,
      sort: f.sort,
      page: String(f.page),
      page_size: '24',
    })
    if (f.theme) q.set('theme', f.theme)
    return request<{ total: number; page: number; page_size: number; items: BrowseItem[] }>(`/puzzles/browse?${q}`)
  },

  races: {
    create: (visibility: 'solo' | 'private', duration_s: number) =>
      request<RaceState>('/races', { method: 'POST', body: json({ visibility, duration_s, max_errors: 3 }) }),
    quick: (duration_s: number) => request<RaceState>(`/races/quick?duration_s=${duration_s}`, { method: 'POST' }),
    join: (code: string) => request<RaceState>(`/races/join?code=${encodeURIComponent(code)}`, { method: 'POST' }),
    get: (id: number) => request<RaceState>(`/races/${id}`),
    start: (id: number) => request<RaceState>(`/races/${id}/start`, { method: 'POST' }),
    leave: (id: number) => request<void>(`/races/${id}/leave`, { method: 'POST' }),
    answer: (id: number, index: number, moves: string[]) =>
      request<RaceState & { correct: boolean; solution: string[] }>(`/races/${id}/answer`, {
        method: 'POST',
        body: json({ index, moves }),
      }),
    leaderboard: (duration_s: number) =>
      request<{ username: string; score: number }[]>(`/races/leaderboard?duration_s=${duration_s}`),
  },
  daily: () => request<NextPuzzle>('/puzzles/daily'),
  review: () => request<NextPuzzle>('/puzzles/review'),
  puzzle: (id: string) => request<NextPuzzle>(`/puzzles/${id}`),
  save: (id: string, note?: string) => request<void>(`/puzzles/${id}/save`, { method: 'PUT', body: json({ note }) }),
  unsave: (id: string) => request<void>(`/puzzles/${id}/save`, { method: 'DELETE' }),
  saved: () => request<{ puzzle: Puzzle; note: string | null; created_at: string }[]>('/puzzles/saved'),

  attempt: (body: {
    puzzle_id: string
    solved: boolean
    time_ms: number
    hints_used: number
    mode: string
    selector?: string | null
  }) => request<AttemptResult>('/attempts', { method: 'POST', body: json(body) }),
  history: (limit = 50) => request<AttemptRow[]>(`/attempts?limit=${limit}`),

  overview: () => request<Overview>('/stats/overview'),
  ratingHistory: () => request<{ t: string; rating: number; theme: Theme; theme_rating: number }[]>('/stats/rating-history'),
  activity: () => request<{ day: string; attempts: number; solved: number }[]>('/stats/activity'),
  timeBuckets: () => request<{ label: string; attempts: number; accuracy: number }[]>('/stats/time-buckets'),
  coachAB: () => request<{ you: Record<string, ABStats>; global: Record<string, ABStats> }>('/stats/coach-ab'),
  badges: () => request<Badge[]>('/stats/badges'),
  leaderboard: (by: 'rating' | 'xp' | 'streak') =>
    request<{ username: string; rating: number; xp: number; streak: number }[]>(`/stats/leaderboard?by=${by}`),
}
