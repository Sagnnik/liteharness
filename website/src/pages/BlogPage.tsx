import { ArrowLeft, ArrowUpRight } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router'
import { blogPosts, getBlogPost, resolveBlogAsset } from '../content/blog'
import { estimateReadMinutes, extractMarkdownHeadings } from '../content/markdown'
import { MarkdownDocument } from './shared/MarkdownDocument'
import { SiteShell } from './shared/SiteShell'

function BlogIndex() {
  return (
    <div className="blog-layout">
      <aside className="blog-rail">
        <p className="site-kicker">[ HARNESS FIELD NOTES ]</p>
        <h1>notes from inside the loop</h1>
        <p>
          Working notes on engineering the harness, its operating surfaces, and its
          extension seams.
        </p>
      </aside>
      <section className="blog-feed" aria-label="Blog posts">
        {blogPosts.map((post, index) => (
          <article key={post.slug}>
            <div className="blog-feed__meta">
              <span>{String(index + 1).padStart(2, '0')}</span>
              <time dateTime={post.date}>{post.date}</time>
            </div>
            <div className="blog-feed__body">
              <h2>
                <Link to={`/blog/${post.slug}`}>{post.title}</Link>
              </h2>
              <p>{post.description}</p>
              <Link to={`/blog/${post.slug}`} className="site-inline-link">
                read note <ArrowUpRight size={14} aria-hidden="true" />
              </Link>
            </div>
          </article>
        ))}
      </section>
    </div>
  )
}

function scrollToHeading(id: string) {
  document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

function BlogPost() {
  const { slug } = useParams()
  const post = getBlogPost(slug)
  const [activeHeading, setActiveHeading] = useState('')

  const headings = useMemo(
    () => (post ? extractMarkdownHeadings(post.body) : []),
    [post],
  )
  const readMinutes = useMemo(
    () => (post ? estimateReadMinutes(post.body) : 1),
    [post],
  )

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'instant' })
    setActiveHeading(headings[0]?.id ?? '')
  }, [post?.slug, headings])

  useEffect(() => {
    if (headings.length === 0) return

    let observer: IntersectionObserver | undefined
    const timer = window.setTimeout(() => {
      const nodes = headings
        .map((heading) => document.getElementById(heading.id))
        .filter((node): node is HTMLElement => Boolean(node))
      if (nodes.length === 0) return

      observer = new IntersectionObserver(
        (entries) => {
          const visible = entries
            .filter((entry) => entry.isIntersecting)
            .sort((a, b) => b.intersectionRatio - a.intersectionRatio)
          if (visible[0]?.target.id) {
            setActiveHeading(visible[0].target.id)
          }
        },
        {
          rootMargin: '-18% 0px -68% 0px',
          threshold: [0, 0.25, 0.5, 1],
        },
      )
      for (const node of nodes) observer.observe(node)
    }, 60)

    return () => {
      window.clearTimeout(timer)
      observer?.disconnect()
    }
  }, [headings, post?.slug])

  if (!post) {
    return (
      <section className="site-empty">
        <p className="site-kicker">[ NO TRANSMISSION ]</p>
        <h1>post not found</h1>
        <Link to="/blog" className="site-inline-link">
          <ArrowLeft size={14} /> return to archive
        </Link>
      </section>
    )
  }

  return (
    <article className="post-reading">
      <div
        className={
          headings.length > 0
            ? 'post-reading__shell'
            : 'post-reading__shell post-reading__shell--solo'
        }
      >
        {headings.length > 0 ? (
          <aside className="post-toc" aria-label="Table of contents">
            <p>ON THIS PAGE</p>
            <nav>
              <ul>
                {headings.map((heading) => (
                  <li key={heading.id} data-level={heading.level}>
                    <a
                      href={`#${heading.id}`}
                      className={activeHeading === heading.id ? 'is-active' : undefined}
                      onClick={(event) => {
                        event.preventDefault()
                        setActiveHeading(heading.id)
                        scrollToHeading(heading.id)
                        window.history.replaceState(null, '', `#${heading.id}`)
                      }}
                    >
                      {heading.label}
                    </a>
                  </li>
                ))}
              </ul>
            </nav>
          </aside>
        ) : null}

        <div className="post-reading__main">
          <header className="post-reading__header">
            <div className="post-reading__meta">
              <Link to="/blog" className="post-reading__back">
                <ArrowLeft size={14} /> archive
              </Link>
              <time dateTime={post.date}>{post.date}</time>
              <span>{readMinutes} min read</span>
            </div>
            <h1>{post.title}</h1>
            <p className="post-reading__dek">{post.description}</p>
          </header>

          <div className="post-prose">
            <MarkdownDocument
              content={post.body}
              resolveImage={(source) => resolveBlogAsset(post.slug, source)}
            />
          </div>
        </div>
      </div>
    </article>
  )
}

export function BlogPage() {
  const { slug } = useParams()
  return (
    <SiteShell className="site-shell--blog">
      {slug ? <BlogPost /> : <BlogIndex />}
    </SiteShell>
  )
}
