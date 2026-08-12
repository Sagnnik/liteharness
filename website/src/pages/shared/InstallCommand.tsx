import { Check, Clipboard } from 'lucide-react'
import { useState } from 'react'
import {
  INSTALL_COMMAND,
  POWERSHELL_INSTALL_COMMAND,
  UV_INSTALL_COMMAND,
} from '../../tui/steps'

const INSTALL_OPTIONS = [
  { id: 'unix', label: 'macOS / Linux', prompt: '$', command: INSTALL_COMMAND },
  {
    id: 'windows',
    label: 'Windows',
    prompt: 'PS>',
    command: POWERSHELL_INSTALL_COMMAND,
  },
  { id: 'uv', label: 'uv', prompt: '$', command: UV_INSTALL_COMMAND },
] as const

export function InstallCommand() {
  const [activeId, setActiveId] = useState<(typeof INSTALL_OPTIONS)[number]['id']>(
    'unix',
  )
  const [copied, setCopied] = useState<string | null>(null)
  const activeOption =
    INSTALL_OPTIONS.find((option) => option.id === activeId) ?? INSTALL_OPTIONS[0]

  async function copy(command: string) {
    try {
      await navigator.clipboard.writeText(command)
      setCopied(command)
      window.setTimeout(
        () => setCopied((current) => (current === command ? null : current)),
        1800,
      )
    } catch {
      setCopied(null)
    }
  }

  const isCopied = copied === activeOption.command

  return (
    <div className="site-install-options">
      <div className="site-install-tabs" role="tablist" aria-label="Install method">
        {INSTALL_OPTIONS.map((option) => (
          <button
            type="button"
            role="tab"
            id={`install-tab-${option.id}`}
            aria-controls="install-command-panel"
            aria-selected={activeId === option.id}
            onClick={() => {
              setActiveId(option.id)
              setCopied(null)
            }}
            key={option.id}
          >
            {option.label}
          </button>
        ))}
      </div>
      <div
        className="site-install"
        role="tabpanel"
        id="install-command-panel"
        aria-labelledby={`install-tab-${activeOption.id}`}
      >
        <code>
          <span aria-hidden="true">{activeOption.prompt}</span>{' '}
          {activeOption.command}
        </code>
        <button
          type="button"
          onClick={() => void copy(activeOption.command)}
          aria-label={`Copy ${activeOption.label} install command`}
        >
          {isCopied ? <Check size={15} /> : <Clipboard size={15} />}
          {isCopied ? 'copied' : 'copy command'}
        </button>
      </div>
    </div>
  )
}
