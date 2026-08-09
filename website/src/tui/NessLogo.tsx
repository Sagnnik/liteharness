import { useId, useRef } from 'react'
import { useGSAP } from '@gsap/react'
import gsap from 'gsap'

gsap.registerPlugin(useGSAP)

interface NessLogoProps {
  booting?: boolean
  intensity?: number
  onComplete?: () => void
}

/** Geometry from `ness_cli.tui.header` / cli-header-concept (center 50, r 36). */
const CX = 50
const CY = 50
const RING_R = 36

const NODES = [
  {
    key: 'top',
    x: CX,
    y: CY - RING_R,
    r: 3.4,
    className: 'logo-point logo-point--mint',
  },
  {
    key: 'bl',
    x: CX - (RING_R * 52) / 60,
    y: CY + (RING_R * 30) / 60,
    r: 2.6,
    className: 'logo-point logo-point--orange',
  },
  {
    key: 'br',
    x: CX + (RING_R * 52) / 60,
    y: CY + (RING_R * 30) / 60,
    r: 2.6,
    className: 'logo-point logo-point--purple',
  },
] as const

/** Chord origin sits slightly below the top node (SVG y=35 vs node y=28). */
const TOP_ANCHOR = { x: CX, y: CY - (RING_R * 53) / 60 }

export function NessLogo({
  booting = false,
  intensity = 1,
  onComplete,
}: NessLogoProps) {
  const root = useRef<SVGSVGElement>(null)
  const gradId = useId().replace(/:/g, '')

  useGSAP(
    () => {
      if (!booting || !root.current) return

      const reduceMotion = window.matchMedia(
        '(prefers-reduced-motion: reduce)',
      ).matches
      const svg = root.current
      const points = gsap.utils.toArray<SVGCircleElement>('.logo-point', svg)
      const chords = gsap.utils.toArray<SVGLineElement>('.logo-chord', svg)

      // Animate the <svg> itself — CSS transforms on SVG <g> break transform-origin
      // and park the mark in the top-left of the viewport.
      gsap.set(svg, { transformOrigin: '50% 50%', force3D: true })
      gsap.set(points, { transformOrigin: '50% 50%', scale: 0, autoAlpha: 0 })
      gsap.set(chords, { strokeDashoffset: 100, autoAlpha: 0.2 })

      const timeline = gsap.timeline({
        onComplete,
        defaults: { ease: 'power3.inOut' },
      })

      timeline
        .fromTo(
          svg,
          {
            rotation: -160 * intensity,
            scale: 0.72,
            autoAlpha: 0,
          },
          {
            rotation: 0,
            scale: 1,
            autoAlpha: 1,
            duration: reduceMotion ? 0.01 : 0.95,
          },
        )
        .to(
          chords,
          {
            strokeDashoffset: 0,
            autoAlpha: 1,
            duration: reduceMotion ? 0.01 : 0.65,
            stagger: 0.1,
            ease: 'power2.out',
          },
          reduceMotion ? 0 : 0.28,
        )
        .to(
          points,
          {
            scale: 1,
            autoAlpha: 1,
            duration: reduceMotion ? 0.01 : 0.5,
            stagger: 0.08,
            ease: 'back.out(1.6)',
          },
          reduceMotion ? 0 : 0.4,
        )
    },
    { scope: root, dependencies: [booting, intensity, onComplete] },
  )

  return (
    <svg
      ref={root}
      className={`ness-logo${booting ? ' ness-logo--booting' : ''}`}
      viewBox="0 0 100 100"
      role="img"
      aria-label="Ness Agent loop mark"
    >
      <defs>
        <linearGradient id={`ness-ring-${gradId}`} x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="var(--ness-ring-a)" />
          <stop offset="45%" stopColor="var(--ness-ring-b)" />
          <stop offset="100%" stopColor="var(--ness-ring-a)" />
        </linearGradient>
      </defs>

      <g className="logo-orbit">
        <circle
          className="logo-ring"
          cx={CX}
          cy={CY}
          r={RING_R}
          fill="none"
          stroke={`url(#ness-ring-${gradId})`}
        />

        <g className="logo-chords">
          <line
            className="logo-chord logo-chord--cyan"
            pathLength={100}
            x1={TOP_ANCHOR.x}
            y1={TOP_ANCHOR.y}
            x2={NODES[1].x}
            y2={NODES[1].y}
          />
          <line
            className="logo-chord logo-chord--purple"
            pathLength={100}
            x1={TOP_ANCHOR.x}
            y1={TOP_ANCHOR.y}
            x2={NODES[2].x}
            y2={NODES[2].y}
          />
        </g>

        {NODES.map((node) => (
          <circle
            key={node.key}
            className={node.className}
            cx={node.x}
            cy={node.y}
            r={node.r}
          />
        ))}
      </g>
    </svg>
  )
}
