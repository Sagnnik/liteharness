import { ArrowLeft, Moon, Sun } from 'lucide-react'
import { Link, useLocation } from 'react-router'
import { useTheme } from './useTheme'

const NAV_LINKS = [
  ['/home', 'Home'],
  ['/news', 'News'],
  ['/blog', 'Blog'],
  ['/docs', 'Docs'],
] as const

export function VariantNavbar() {
  const { theme, toggleTheme } = useTheme()
  const { pathname } = useLocation()
  const isLight = theme === 'light'
  const onConsole = pathname === '/v1b'

  return (
    <nav className="variant-nav" aria-label="Primary">
      <Link className="variant-nav__brand" to={onConsole ? '/home' : '/v1b'}>
        <ArrowLeft size={13} aria-hidden="true" />
        {onConsole ? 'return to site' : 'return to console'}
      </Link>
      <div className="variant-nav__links">
        {NAV_LINKS.map(([to, label], index) => (
          <Link
            className={pathname === to || pathname.startsWith(`${to}/`) ? 'is-active' : undefined}
            to={to}
            key={to}
          >
            <span>0{index + 1}</span>
            {label}
          </Link>
        ))}
      </div>
      <button
        type="button"
        className="variant-nav__theme"
        onClick={toggleTheme}
        aria-label={isLight ? 'Switch to dark mode' : 'Switch to light mode'}
        title={isLight ? 'Dark mode' : 'Light mode'}
      >
        {isLight ? <Moon size={14} /> : <Sun size={14} />}
      </button>
    </nav>
  )
}
