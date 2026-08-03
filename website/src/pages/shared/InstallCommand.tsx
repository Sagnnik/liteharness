import { Check, Clipboard } from 'lucide-react'
import { useState } from 'react'
import { INSTALL_COMMAND } from '../../tui/steps'

export function InstallCommand() {
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
