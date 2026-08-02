import type { ReactNode } from 'react'
import { VariantNavbar } from '../../tui/VariantNavbar'

interface SiteShellProps {
  children: ReactNode
  className?: string
}

export function SiteShell({ children, className = '' }: SiteShellProps) {
  return (
    <div className={`site-shell ${className}`.trim()}>
      <VariantNavbar />
      <main className="site-main">{children}</main>
    </div>
  )
}
