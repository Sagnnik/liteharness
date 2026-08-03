export type Variant = 'v1b'

export type TuiMode = 'plan' | 'act'

export type StepKind =
  | 'ready'
  | 'manifesto'
  | 'install'
  | 'surfaces'
  | 'context'
  | 'extensions'
  | 'goal'
  | 'docs'

export interface TuiStep {
  id: number
  session: string
  project: string
  heading: string
  eyebrow: string
  kind: StepKind
  lines: readonly string[]
}

export type KnownCommand =
  | '/copy'
  | '/help'
  | '/home'
  | '/news'
  | '/blog'
  | '/docs'

export interface CommandOutput {
  id: number
  tone: 'info' | 'success' | 'error'
  mode: TuiMode
  command: string
  lines: readonly string[]
}
