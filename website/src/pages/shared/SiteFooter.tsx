const GITHUB_REPO = 'https://github.com/Sagnnik/ness-agent'
const PYPI_PACKAGE = 'https://pypi.org/project/ness-agent/'
const CONTRIBUTING = `${GITHUB_REPO}/blob/main/CONTRIBUTING.md`
const SECURITY_POLICY = `${GITHUB_REPO}/blob/main/SECURITY.md`

const FOOTER_LINKS = [
  { label: 'github', href: GITHUB_REPO },
  { label: 'pypi', href: PYPI_PACKAGE },
  { label: 'contributing', href: CONTRIBUTING },
  { label: 'security', href: SECURITY_POLICY },
] as const

export function SiteFooter() {
  return (
    <footer className="site-footer" aria-label="Site">
      <div className="site-footer__inner">
        <p className="site-footer__brand">[ NESS AGENT ]</p>
        <nav className="site-footer__links" aria-label="Project links">
          {FOOTER_LINKS.map((link, index) => (
            <span key={link.label} className="site-footer__link-item">
              {index > 0 ? <span className="site-footer__sep" aria-hidden="true">·</span> : null}
              <a href={link.href} target="_blank" rel="noopener noreferrer">
                {link.label}
              </a>
            </span>
          ))}
        </nav>
        <p className="site-footer__meta">
          <span>© 2026 Sagnnik</span>
          <span className="site-footer__sep" aria-hidden="true">·</span>
          <span>Apache-2.0</span>
        </p>
      </div>
    </footer>
  )
}
