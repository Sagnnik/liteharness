import type { ReactNode } from 'react'
import { VariantNavbar } from '../../tui/VariantNavbar'
import { SiteFooter } from './SiteFooter'

interface SiteShellProps {
  children: ReactNode
  className?: string
  footer?: boolean
}

export function SiteShell({ children, className = '', footer = true }: SiteShellProps) {
  return (
    <div className={`site-shell ${footer ? '' : 'site-shell--no-footer'} ${className}`.trim()}>
      <VariantNavbar />
      <main className="site-main">{children}</main>
      {footer ? <SiteFooter /> : null}
    </div>
  )
}
