import { ArrowLeft, ArrowUpRight, CornerDownRight } from 'lucide-react'
import { Link, useParams } from 'react-router'
import { SiteShell } from './shared/SiteShell'

const CHANGELOG = 'https://github.com/Sagnnik/ness-agent/blob/main/CHANGELOG.md'
const BLOG_SLUG = 'harness-engineering-a-ness-agent-intro'

const RELEASE = {
  slug: 'initial-public-release',
  title: 'Initial public release',
  date: '2026-07-31',
  version: 'v0.1.0',
  summary:
    'Ness Agent and Ness are now public: one Python package, a reusable harness, and an interactive coding surface.',
}

const HIGHLIGHTS = [
  ['SDK + CLI', 'LangGraph agent loop and an interactive coding adapter in the same package.'],
  ['Tools + policy', 'Built-in tools, permissions, memory, skills, hooks, and MCP support.'],
  [
    'Context layers',
    'L0–L3 prompt assembly, ephemeral overlays, cache-aware compaction, and reflection.',
  ],
  [
    '/goal verification',
    'Bounded worker attempts with an independent judge and repair instructions on failure.',
  ],
  [
    'Project-local',
    'A versionable .ness/ surface for behavior, plus editable global instructions/ templates.',
  ],
] as const

function ReleaseDetail() {
  return (
    <>
      <header className="dispatch-header">
        <p className="site-kicker">[ RELEASE DISPATCH // {RELEASE.version} ]</p>
        <h1>{RELEASE.title}</h1>
        <p>{RELEASE.summary}</p>
        <p className="dispatch-header__changelog">
          <a href={CHANGELOG} target="_blank" rel="noopener noreferrer">
            changelog
          </a>
        </p>
        <dl className="dispatch-meta">
          <div>
            <dt>DATE</dt>
            <dd>{RELEASE.date}</dd>
          </div>
          <div>
            <dt>STATUS</dt>
            <dd>RELEASED</dd>
          </div>
          <div>
            <dt>CHANNEL</dt>
            <dd>PUBLIC</dd>
          </div>
        </dl>
      </header>

      <section className="dispatch-body" aria-labelledby="release-highlights">
        <div className="dispatch-body__rail">00.1 // ADDED</div>
        <div>
          <h2 id="release-highlights">first transmission</h2>
          <p>
            The initial release establishes Ness Agent as an experimental, hackable
            harness. APIs may change before 1.0; the seams are meant to be inspected.
          </p>
          <ol className="dispatch-list">
            {HIGHLIGHTS.map(([title, description], index) => (
              <li key={title}>
                <span>{String(index + 1).padStart(2, '0')}</span>
                <strong>{title}</strong>
                <p>{description}</p>
              </li>
            ))}
          </ol>
          <p className="dispatch-field-note">
            For the longer framing, read{' '}
            <Link to={`/blog/${BLOG_SLUG}`} className="site-inline-link">
              Harness Engineering: A Ness Agent Intro
            </Link>
            .
          </p>
        </div>
      </section>

      <footer className="dispatch-footer">
        <Link to="/docs" className="site-link-button">
          <CornerDownRight size={15} aria-hidden="true" />
          inspect the docs
        </Link>
        <Link to="/news" className="site-inline-link">
          back to dispatches
        </Link>
      </footer>
    </>
  )
}

function NewsMissing({ slug }: { slug: string }) {
  return (
    <section className="site-empty">
      <p className="site-kicker">[ NO DISPATCH ]</p>
      <h1>release not found</h1>
      <p>No news item matches <code>{slug}</code>.</p>
      <Link to="/news" className="site-inline-link">
        <ArrowLeft size={14} /> return to dispatches
      </Link>
    </section>
  )
}

export function NewsPage() {
  const { slug } = useParams()

  return (
    <SiteShell className="site-shell--news">
      {slug ? (
        slug === RELEASE.slug ? <ReleaseDetail /> : <NewsMissing slug={slug} />
      ) : (
        <>
          <header className="news-hero">
            <div className="news-hero__left">
              <p className="site-kicker">[ RELEASE DISPATCHES ]</p>
              <h1>field updates</h1>
            </div>
            <p className="news-hero__right">
              Short release records from the harness.
            </p>
          </header>
          <section className="release-index" aria-label="Release dispatches">
            <article>
              <div className="release-index__meta">
                <span>{RELEASE.version}</span>
                <time dateTime={RELEASE.date}>{RELEASE.date}</time>
              </div>
              <div className="release-index__body">
                <h2>
                  <Link to={`/news/${RELEASE.slug}`}>{RELEASE.title}</Link>
                </h2>
                <p>{RELEASE.summary}</p>
                <Link to={`/news/${RELEASE.slug}`} className="site-inline-link">
                  open dispatch <ArrowUpRight size={14} aria-hidden="true" />
                </Link>
              </div>
            </article>
          </section>
        </>
      )}
    </SiteShell>
  )
}
