import { useEffect, useRef } from 'react'

const GRAIN_SIZE = 128
const GRAIN_FPS = 18
const GRAIN_INTERVAL = 1000 / GRAIN_FPS
const PIXEL_ALPHA = 12 // ~4.7% opacity per pixel

/**
 * Full-screen animated film-grain overlay.
 * Renders random noise on a small offscreen canvas, then draws it
 * scaled-up onto the visible canvas for chunky grain texture.
 */
export default function Grain() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d', { alpha: true })
    if (!ctx) return

    // Respect reduced motion: render one static frame only
    const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches

    const offscreen = document.createElement('canvas')
    offscreen.width = GRAIN_SIZE
    offscreen.height = GRAIN_SIZE
    const offCtx = offscreen.getContext('2d')!

    // Pre-allocate buffer once — reuse every frame
    const imageData = offCtx.createImageData(GRAIN_SIZE, GRAIN_SIZE)
    const buf32 = new Uint32Array(imageData.data.buffer)
    const pixelCount = GRAIN_SIZE * GRAIN_SIZE

    const resize = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }
    resize()
    window.addEventListener('resize', resize)

    let frame = 0
    let last = 0

    const draw = (now: number) => {
      frame = requestAnimationFrame(draw)
      if (now - last < GRAIN_INTERVAL) return
      last = now

      // Write ABGR pixels via Uint32Array (little-endian)
      for (let i = 0; i < pixelCount; i++) {
        const v = (Math.random() * 255) | 0
        buf32[i] = (PIXEL_ALPHA << 24) | (v << 16) | (v << 8) | v
      }

      offCtx.putImageData(imageData, 0, 0)
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      ctx.imageSmoothingEnabled = false
      ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height)
    }

    if (prefersReduced) {
      draw(performance.now()) // single static frame
    } else {
      frame = requestAnimationFrame(draw)
    }
    return () => {
      cancelAnimationFrame(frame)
      window.removeEventListener('resize', resize)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      role="img"
      aria-hidden="true"
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 0,
        pointerEvents: 'none',
        mixBlendMode: 'overlay',
      }}
    />
  )
}
