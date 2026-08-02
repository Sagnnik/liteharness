import { Check, Clipboard, CornerDownRight } from 'lucide-react'
import { Link } from 'react-router'
import { useState } from 'react'
import { INSTALL_COMMAND, TUI_STEPS } from '../tui/steps'
import { SiteShell } from './shared/SiteShell'

const HOME_STEPS = TUI_STEPS.filter(
  (step) =>
    step.kind !== 'ready' &&
    step.kind !== 'docs' &&
    step.kind !== 'install' &&
    step.kind !== 'manifesto',
)

function InstallCommand() {
  const [copied, setCopied] = useState(false)

  async function copy() {
    try {
      await navigator.clipboard.writeText(INSTALL_COMMAND)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1800)
    } catch {
      setCopied(false)
    }
  }

  return (
    <div className="site-install">
      <code>
        <span aria-hidden="true">$</span> {INSTALL_COMMAND}
      </code>
      <button type="button" onClick={copy} aria-label="Copy install command">
        {copied ? <Check size={15} /> : <Clipboard size={15} />}
        {copied ? 'copied' : 'copy'}
      </button>
    </div>
  )
}

export function HomePage() {
  return (
    <SiteShell className="site-shell--home">
      <section className="home-hero" aria-labelledby="home-title">
        <div className="home-hero__pitch">
          <p className="site-kicker">[ HACKABLE CODING-AGENT HARNESS ]</p>
          <h1 id="home-title">own the loop</h1>
          <p>
            A coding-agent harness for engineers who need the loop within reach.
            Inspect it, extend it, and replace any layer that gets in the way.
          </p>
        </div>
        <aside className="home-hero__cta" aria-label="Install">
          <p className="site-kicker">[ QUICK START ]</p>
          <p className="home-hero__cta-note">
            One package. Python 3.12+. Operator CLI and embeddable SDK.
          </p>
          <InstallCommand />
          <pre className="home-hero__snippet" aria-hidden="true">
            <code>{`ness
/init`}</code>
          </pre>
        </aside>
      </section>

      <section className="home-field" aria-label="Ness Agent operating surfaces">
        {HOME_STEPS.map((step, index) => {
          const label = step.eyebrow.replace('NESS // ', '')
          const number = String(index + 1).padStart(2, '0')
          return (
            <article className="home-unit" key={step.id}>
              <p className="home-unit__badge">
                <span>// {label}</span>
                <span aria-hidden="true">•</span>
                <span>{number}</span>
              </p>
              <h2>{step.heading}</h2>
              {step.lines.map((line) => {
                const [part, ...rest] = line.split('  ')
                const labeled = rest.length > 0 && part.length < 12
                return (
                  <p key={line}>
                    {labeled ? <strong>{part}</strong> : null}
                    {labeled ? `  ${rest.join('  ')}` : line}
                  </p>
                )
              })}
            </article>
          )
        })}
      </section>

      <section className="site-callout" aria-labelledby="home-docs-title">
        <div>
          <p className="site-kicker">[ OPERATING MANUAL ]</p>
          <h2 id="home-docs-title">start at the interface.</h2>
          <p>SDK, CLI, configuration, and architecture are documented as the harness evolves.</p>
        </div>
        <Link to="/docs" className="site-link-button">
          <CornerDownRight size={15} aria-hidden="true" />
          open /docs
        </Link>
      </section>
    </SiteShell>
  )
}
