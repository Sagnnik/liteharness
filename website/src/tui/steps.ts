import type { TuiStep } from './types'

export const INSTALL_COMMAND = 'uv tool install ness-agent'

/**
 * Navigation groups for wheel/arrow replay.
 * 0: ready + manifesto + install
 * 1: surfaces + modes + extensions + worktrees
 * 2: docs (standalone)
 */
export const TUI_STEP_GROUPS: readonly (readonly number[])[] = [
  [0, 1, 2],
  [3, 4, 5, 6],
  [7],
]

export const INTRO_BLOCK_IDS: readonly number[] = [0, 1, 2]
export const HARNESS_BLOCK_IDS: readonly number[] = [3, 4, 5, 6]

export interface MergedTranscriptGroup {
  key: string
  blockIds: readonly number[]
  railLabel: string
  modifier: string
  dataStream: 'plan' | 'act'
  displayIndex: number
}

export const MERGED_TRANSCRIPT_GROUPS: readonly MergedTranscriptGroup[] = [
  {
    key: 'intro',
    blockIds: INTRO_BLOCK_IDS,
    railLabel: 'NESS // BOOT',
    modifier: 'intro',
    dataStream: 'plan',
    displayIndex: 0,
  },
  {
    key: 'harness',
    blockIds: HARNESS_BLOCK_IDS,
    railLabel: 'NESS // HARNESS',
    modifier: 'harness',
    dataStream: 'act',
    displayIndex: 1,
  },
]

/** Last 0-based index among transcript containers (merged groups + standalone docs). */
export const TRANSCRIPT_CONTAINER_LAST = MERGED_TRANSCRIPT_GROUPS.length

export type TranscriptDisplayItem =
  | { type: 'merged'; group: MergedTranscriptGroup }
  | { type: 'step'; index: number }

const mergedBlockIds = new Set(
  MERGED_TRANSCRIPT_GROUPS.flatMap((group) => group.blockIds),
)

export function transcriptDisplayItems(groupIndex: number): TranscriptDisplayItem[] {
  const blocks = blocksThroughGroup(groupIndex)
  const items: TranscriptDisplayItem[] = []

  for (const group of MERGED_TRANSCRIPT_GROUPS) {
    if (blocks.some((id) => group.blockIds.includes(id))) {
      items.push({ type: 'merged', group })
    }
  }

  for (const id of blocks) {
    if (!mergedBlockIds.has(id)) {
      items.push({ type: 'step', index: id })
    }
  }

  return items
}

export function blocksThroughGroup(groupIndex: number): number[] {
  if (groupIndex < 0) return []
  const last = Math.min(groupIndex, TUI_STEP_GROUPS.length - 1)
  const blocks: number[] = []
  for (let group = 0; group <= last; group += 1) {
    blocks.push(...TUI_STEP_GROUPS[group])
  }
  return blocks
}

export const TUI_STEPS: readonly TuiStep[] = [
  {
    id: 0,
    session: 'boot/ready',
    project: '~/projects/ness-agent',
    heading: 'system ready',
    eyebrow: 'NESS // SESSION',
    kind: 'ready',
    lines: [
      'ness-agent v0.1.0 initialized',
      'scroll, use arrow keys, or enter /help to inspect the harness',
    ],
  },
  {
    id: 1,
    session: 'manifest/own-the-loop',
    project: 'agent/control-plane',
    heading: 'own the loop',
    eyebrow: 'NESS // PRINCIPLE',
    kind: 'manifesto',
    lines: [
      'A hackable coding-agent harness for engineers who need the loop within reach.',
      'Inspect it. Extend it. Replace any layer that gets in the way.',
    ],
  },
  {
    id: 2,
    session: 'setup/install',
    project: 'pypi/ness-agent',
    heading: 'install the operator',
    eyebrow: 'NESS // INSTALL',
    kind: 'install',
    lines: [
      'Python 3.12+ · SDK and interactive CLI ship in one package.',
      INSTALL_COMMAND,
    ],
  },
  {
    id: 3,
    session: 'surface/sdk-cli',
    project: 'ness_agent + ness',
    heading: 'one harness, two surfaces',
    eyebrow: 'NESS // SURFACES',
    kind: 'surfaces',
    lines: [
      'SDK  embed the LangGraph loop, tools, permissions, memory, and tracing.',
      'CLI  run the same system as an interactive terminal coding session.',
    ],
  },
  {
    id: 4,
    session: 'mode/plan-act',
    project: 'operator/intent',
    heading: 'think, then touch',
    eyebrow: 'NESS // MODES',
    kind: 'modes',
    lines: [
      'PLAN  explore, reason, and stage the change without mutation.',
      'ACT   execute tools, edit files, and verify the result.',
    ],
  },
  {
    id: 5,
    session: 'extend/project-local',
    project: '.ness/',
    heading: 'the filesystem is the interface',
    eyebrow: 'NESS // EXTENSIONS',
    kind: 'extensions',
    lines: [
      'skills/  hooks/  MCP servers  prompt layers  project memory',
      'Keep behavior local, inspectable, diffable, and under version control.',
    ],
  },
  {
    id: 6,
    session: 'isolate/worktree',
    project: 'git/worktrees',
    heading: 'parallel without collision',
    eyebrow: 'NESS // WORKTREES',
    kind: 'worktrees',
    lines: [
      'ness --worktree feature-x',
      'Spin up isolated sessions for experiments, reviews, and parallel agents.',
    ],
  },
  {
    id: 7,
    session: 'handoff/docs',
    project: 'docs/',
    heading: 'read the operating manual',
    eyebrow: 'NESS // DOCS',
    kind: 'docs',
    lines: [
      '/docs  SDK · CLI · configuration · architecture',
      'The loop is yours. Start at the interface, then keep drilling down.',
    ],
  },
] as const

export const KNOWN_COMMANDS = new Set([
  '/copy',
  '/help',
  '/home',
  '/news',
  '/blog',
  '/docs',
])
