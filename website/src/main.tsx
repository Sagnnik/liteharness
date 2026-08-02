import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import '@fontsource-variable/jetbrains-mono'
import './index.css'
import App from './App.tsx'

try {
  const stored = window.localStorage.getItem('ness-theme')
  document.documentElement.dataset.theme = stored === 'light' ? 'light' : 'dark'
} catch {
  document.documentElement.dataset.theme = 'dark'
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
