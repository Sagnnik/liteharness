import { BrowserRouter, Navigate, Route, Routes, useLocation } from 'react-router'
import { getBlogPost, resolveBlogAsset } from './content/blog'
import { BlogPage } from './pages/BlogPage'
import { DocsPage } from './pages/DocsPage'
import { HomePage } from './pages/HomePage'
import { NewsPage } from './pages/NewsPage'
import { Seo } from './pages/shared/Seo'
import { TuiShell } from './tui/TuiShell'

const DEFAULT_DESCRIPTION =
  'A hackable coding-agent harness for engineers who need the loop within reach.'
const BLOG_DESCRIPTION =
  'Working notes on engineering the harness, its operating surfaces, and its extension seams.'
const DOCS_DESCRIPTION =
  'Documentation for the Ness Agent SDK, CLI, configuration, and runtime architecture.'

function RouteSeo() {
  const { pathname } = useLocation()
  const route = pathname.replace(/\/+$/, '') || '/'

  if (route === '/blog') {
    return (
      <Seo
        title="Blog — Ness Agent"
        description={BLOG_DESCRIPTION}
        path="/blog"
      />
    )
  }

  if (route.startsWith('/blog/')) {
    const post = getBlogPost(decodeURIComponent(route.slice('/blog/'.length)))
    if (post) {
      return (
        <Seo
          title={`${post.title} — Ness Agent`}
          description={post.description}
          path={`/blog/${post.slug}`}
          type="article"
          image={post.image ? resolveBlogAsset(post.contentSlug, post.image) : undefined}
          publishedTime={post.date}
        />
      )
    }
  }

  if (route === '/' || route === '/home') {
    return <Seo title="Ness Agent — Own the Loop" description={DEFAULT_DESCRIPTION} path="/home" />
  }

  if (route === '/news' || route.startsWith('/news/')) {
    return (
      <Seo
        title="Release Dispatches — Ness Agent"
        description="Release notes and field updates from the Ness Agent coding-agent harness."
        path={route}
      />
    )
  }

  if (route === '/docs' || route.startsWith('/docs/')) {
    return (
      <Seo
        title="Documentation — Ness Agent"
        description={DOCS_DESCRIPTION}
        path={route}
      />
    )
  }

  if (route === '/v1b') {
    return <Seo title="Ness Agent — Operator Console" description={DEFAULT_DESCRIPTION} path={route} />
  }

  return <Seo title="Ness Agent — Own the Loop" description={DEFAULT_DESCRIPTION} path="/home" />
}

function App() {
  return (
    <BrowserRouter basename={import.meta.env.BASE_URL}>
      <RouteSeo />
      <Routes>
        <Route path="/" element={<Navigate to="/home" replace />} />
        <Route path="/home" element={<HomePage />} />
        <Route path="/v1b" element={<TuiShell variant="v1b" />} />
        <Route path="/news" element={<NewsPage />} />
        <Route path="/news/:slug" element={<NewsPage />} />
        <Route path="/blog" element={<BlogPage />} />
        <Route path="/blog/:slug" element={<BlogPage />} />
        <Route path="/docs" element={<DocsPage />} />
        <Route path="/docs/:section" element={<DocsPage />} />
        <Route path="*" element={<Navigate to="/home" replace />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
