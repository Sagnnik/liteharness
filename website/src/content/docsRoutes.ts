export const DOCS_QUICK_LINKS = [
  { slug: 'sdk', label: 'SDK' },
  { slug: 'cli', label: 'CLI' },
  { slug: 'configuration', label: 'configuration' },
  { slug: 'architecture', label: 'architecture' },
] as const

export function docsPath(slug: string) {
  return slug === 'overview' ? '/docs' : `/docs/${slug}`
}
