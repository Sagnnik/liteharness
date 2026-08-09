export interface BlogPost {
  title: string
  date: string
  description: string
  slug: string
  /** Folder name under content/blog/ — used for asset paths, not the URL slug. */
  contentSlug: string
  body: string
}

const entries = import.meta.glob('../../content/blog/*/index.md', {
  eager: true,
  query: '?raw',
  import: 'default',
}) as Record<string, string>

const assets = import.meta.glob('../../content/blog/*/assets/*', {
  eager: true,
  query: '?url',
  import: 'default',
}) as Record<string, string>

function pathSlug(path: string) {
  return path.split('/').at(-3) ?? ''
}

/** Minimal frontmatter parser — avoids gray-matter/js-yaml (needs Node `buffer`). */
function parseFrontmatter(source: string): {
  data: Partial<Omit<BlogPost, 'body'>>
  content: string
} {
  const match = source.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n([\s\S]*)$/)
  if (!match) return { data: {}, content: source }

  const data: Partial<Omit<BlogPost, 'body'>> = {}
  for (const line of match[1].split(/\r?\n/)) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith('#')) continue
    const idx = trimmed.indexOf(':')
    if (idx === -1) continue

    const key = trimmed.slice(0, idx).trim()
    let value = trimmed.slice(idx + 1).trim()
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1)
    }

    if (key === 'title' || key === 'date' || key === 'description' || key === 'slug') {
      data[key] = value
    }
  }

  return { data, content: match[2] }
}

export const blogPosts: BlogPost[] = Object.entries(entries)
  .map(([path, source]) => {
    const parsed = parseFrontmatter(source)
    const contentSlug = pathSlug(path)

    return {
      title: parsed.data.title ?? 'Untitled field note',
      date: parsed.data.date ?? '',
      description: parsed.data.description ?? '',
      slug: parsed.data.slug ?? contentSlug,
      contentSlug,
      body: parsed.content,
    }
  })
  .sort((a, b) => b.date.localeCompare(a.date))

export function getBlogPost(slug: string | undefined) {
  return blogPosts.find((post) => post.slug === slug)
}

export function resolveBlogAsset(contentSlug: string, source: string) {
  if (/^(?:https?:|data:|#)/i.test(source)) return source

  const localPath = source.replace(/^\.\//, '').replace(/^assets\//, '')
  const key = `../../content/blog/${contentSlug}/assets/${localPath}`
  return assets[key] ?? source
}
