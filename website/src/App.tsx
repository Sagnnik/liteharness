import { BrowserRouter, Navigate, Route, Routes } from 'react-router'
import { BlogPage } from './pages/BlogPage'
import { DocsPage } from './pages/DocsPage'
import { HomePage } from './pages/HomePage'
import { NewsPage } from './pages/NewsPage'
import { TuiShell } from './tui/TuiShell'

function App() {
  return (
    <BrowserRouter basename={import.meta.env.BASE_URL}>
      <Routes>
        <Route path="/" element={<Navigate to="/v1b" replace />} />
        <Route path="/v1b" element={<TuiShell variant="v1b" />} />
        <Route path="/home" element={<HomePage />} />
        <Route path="/news" element={<NewsPage />} />
        <Route path="/news/:slug" element={<NewsPage />} />
        <Route path="/blog" element={<BlogPage />} />
        <Route path="/blog/:slug" element={<BlogPage />} />
        <Route path="/docs" element={<DocsPage />} />
        <Route path="/docs/:section" element={<DocsPage />} />
        <Route path="*" element={<Navigate to="/v1b" replace />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
