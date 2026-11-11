/**
 * Synthesized UI sounds (Web Audio API) — no audio assets, no licensing concerns.
 * A short filtered-noise "knock" plus a low body tone approximates a wooden piece on a board.
 */

type SoundName = 'move' | 'capture' | 'check' | 'success' | 'fail' | 'badge' | 'hint'

const MUTE_KEY = 'gmrl_muted'
let ctx: AudioContext | null = null
let muted = (() => {
  try {
    return localStorage.getItem(MUTE_KEY) === '1'
  } catch {
    return false
  }
})()

function ac(): AudioContext | null {
  if (typeof window === 'undefined') return null
  if (!ctx) {
    const Ctor = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
    if (!Ctor) return null
    ctx = new Ctor()
  }
  if (ctx.state === 'suspended') void ctx.resume()
  return ctx
}

function knock(c: AudioContext, t: number, gain: number, freq: number) {
  // noise transient
  const len = Math.floor(c.sampleRate * 0.05)
  const buf = c.createBuffer(1, len, c.sampleRate)
  const d = buf.getChannelData(0)
  for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, 4)
  const src = c.createBufferSource()
  src.buffer = buf
  const bp = c.createBiquadFilter()
  bp.type = 'bandpass'
  bp.frequency.value = freq * 6
  bp.Q.value = 1.2
  const g = c.createGain()
  g.gain.value = gain
  src.connect(bp).connect(g).connect(c.destination)
  src.start(t)
  // wooden body
  const o = c.createOscillator()
  o.type = 'sine'
  o.frequency.setValueAtTime(freq, t)
  o.frequency.exponentialRampToValueAtTime(freq * 0.6, t + 0.08)
  const og = c.createGain()
  og.gain.setValueAtTime(gain * 0.6, t)
  og.gain.exponentialRampToValueAtTime(0.001, t + 0.09)
  o.connect(og).connect(c.destination)
  o.start(t)
  o.stop(t + 0.1)
}

function tone(c: AudioContext, t: number, freq: number, dur: number, gain = 0.12, type: OscillatorType = 'triangle') {
  const o = c.createOscillator()
  o.type = type
  o.frequency.value = freq
  const g = c.createGain()
  g.gain.setValueAtTime(0.0001, t)
  g.gain.exponentialRampToValueAtTime(gain, t + 0.01)
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur)
  o.connect(g).connect(c.destination)
  o.start(t)
  o.stop(t + dur + 0.02)
}

export function play(name: SoundName) {
  if (muted) return
  const c = ac()
  if (!c) return
  const t = c.currentTime + 0.005
  switch (name) {
    case 'move':
      knock(c, t, 0.5, 180)
      break
    case 'capture':
      knock(c, t, 0.7, 150)
      knock(c, t + 0.045, 0.45, 210)
      break
    case 'check':
      knock(c, t, 0.5, 180)
      tone(c, t + 0.02, 880, 0.18, 0.06, 'sine')
      break
    case 'success':
      ;[523.25, 659.25, 783.99, 1046.5].forEach((f, i) => tone(c, t + i * 0.07, f, 0.28, 0.1))
      break
    case 'fail':
      tone(c, t, 220, 0.22, 0.12, 'sawtooth')
      tone(c, t + 0.16, 164.81, 0.32, 0.12, 'sawtooth')
      break
    case 'badge':
      ;[783.99, 987.77, 1174.66, 1567.98].forEach((f, i) => tone(c, t + i * 0.09, f, 0.4, 0.09, 'sine'))
      break
    case 'hint':
      tone(c, t, 1318.5, 0.15, 0.05, 'sine')
      break
  }
}

/** Pick the right board sound for a chess.js move result. */
export function playMove(m: { captured?: string; san: string }) {
  play(m.san.includes('+') || m.san.includes('#') ? 'check' : m.captured ? 'capture' : 'move')
}

export const sound = {
  isMuted: () => muted,
  setMuted: (v: boolean) => {
    muted = v
    try {
      localStorage.setItem(MUTE_KEY, v ? '1' : '0')
    } catch {
      /* storage unavailable */
    }
  },
}
