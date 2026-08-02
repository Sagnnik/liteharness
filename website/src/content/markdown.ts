export interface MarkdownHeading {
  id: string
  label: string
  level: 2 | 3
}

export function slugifyHeading(value: string) {
  return value
    .toLowerCase()
    .trim()
    .replace(/[^\w\s-]/g, '')
    .replace(/\s+/g, '-')
}

/** Extract `##` / `###` headings for ToC and scroll-spy. */
export function extractMarkdownHeadings(markdown: string): MarkdownHeading[] {
  const headings: MarkdownHeading[] = []
  for (const line of markdown.split(/\r?\n/)) {
    const match = /^(#{2,3})\s+(.+)$/.exec(line.trim())
    if (!match) continue
    const level = match[1].length as 2 | 3
    const label = match[2].replace(/`/g, '').trim()
    const id = slugifyHeading(label)
    if (id) headings.push({ id, label, level })
  }
  return headings
}

export function estimateReadMinutes(markdown: string) {
  const words = markdown.trim().split(/\s+/).filter(Boolean).length
  return Math.max(1, Math.round(words / 200))
}
