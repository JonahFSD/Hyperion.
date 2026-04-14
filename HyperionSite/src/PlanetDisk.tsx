import { useEffect, useRef } from 'react'

const SIZE = 360
const CX = SIZE / 2
const CY = SIZE / 2
const BASE_R = SIZE * 0.38
const NUM_POINTS = 32
const AMP = 6          // ±3px max offset
const DRAW_STEPS = 128 // segments around the circle
const TWO_PI = Math.PI * 2
const FPS = 30
const FRAME_INTERVAL = 1000 / FPS

/**
 * Black planet disk with a subtly wavy, organic edge.
 * Control points drift toward random targets for slow,
 * natural-looking edge distortion.
 */
export default function PlanetDisk() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')!

    const dpr = window.devicePixelRatio || 1
    canvas.width = SIZE * dpr
    canvas.height = SIZE * dpr
    canvas.style.width = '100%'
    canvas.style.height = '100%'
    ctx.scale(dpr, dpr)

    // Respect reduced motion: render one static frame only
    const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches

    const offsets = new Float32Array(NUM_POINTS)
    const targets = new Float32Array(NUM_POINTS)
    const speeds = new Float32Array(NUM_POINTS)

    for (let i = 0; i < NUM_POINTS; i++) {
      targets[i] = (Math.random() - 0.5) * AMP
      speeds[i] = 0.003 + Math.random() * 0.008
    }

    function getSmoothedRadius(angle: number): number {
      const idx = (angle / TWO_PI) * NUM_POINTS
      const i0 = Math.floor(idx) % NUM_POINTS
      const i1 = (i0 + 1) % NUM_POINTS
      const frac = idx - Math.floor(idx)
      const t = frac * frac * (3 - 2 * frac) // cubic smoothstep
      return BASE_R + offsets[i0] * (1 - t) + offsets[i1] * t
    }

    // Pre-compute the radial gradient once — it doesn't change
    const shadowGrad = ctx.createRadialGradient(CX, CY, BASE_R - 5, CX, CY, BASE_R + 25)
    shadowGrad.addColorStop(0, 'rgba(0, 0, 0, 0.95)')
    shadowGrad.addColorStop(0.5, 'rgba(0, 0, 0, 0.5)')
    shadowGrad.addColorStop(1, 'rgba(0, 0, 0, 0.0)')

    let frame = 0
    let lastTime = 0

    function draw(now: number) {
      frame = requestAnimationFrame(draw)
      if (now - lastTime < FRAME_INTERVAL) return
      lastTime = now

      // Drift offsets toward targets
      for (let i = 0; i < NUM_POINTS; i++) {
        offsets[i] += (targets[i] - offsets[i]) * speeds[i]
        if (Math.abs(targets[i] - offsets[i]) < 0.15) {
          targets[i] = (Math.random() - 0.5) * AMP
        }
      }

      ctx.clearRect(0, 0, SIZE, SIZE)

      // Shadow halo
      ctx.beginPath()
      for (let i = 0; i <= DRAW_STEPS; i++) {
        const a = (i / DRAW_STEPS) * TWO_PI
        const r = getSmoothedRadius(a) + 15
        const x = CX + Math.cos(a) * r
        const y = CY + Math.sin(a) * r
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
      }
      ctx.closePath()
      ctx.fillStyle = shadowGrad
      ctx.fill()

      // Black planet disk
      ctx.beginPath()
      for (let i = 0; i <= DRAW_STEPS; i++) {
        const a = (i / DRAW_STEPS) * TWO_PI
        const r = getSmoothedRadius(a)
        const x = CX + Math.cos(a) * r
        const y = CY + Math.sin(a) * r
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
      }
      ctx.closePath()
      ctx.fillStyle = '#000'
      ctx.fill()
    }

    if (prefersReduced) {
      draw(performance.now()) // single static frame
    } else {
      frame = requestAnimationFrame(draw)
    }
    return () => cancelAnimationFrame(frame)
  }, [])

  return (
    <canvas
      ref={canvasRef}
      role="img"
      aria-hidden="true"
      style={{
        position: 'absolute',
        inset: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
      }}
    />
  )
}
