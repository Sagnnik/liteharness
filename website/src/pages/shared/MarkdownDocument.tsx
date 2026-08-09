import type { ReactNode } from 'react'
import ReactMarkdown from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'
import remarkGfm from 'remark-gfm'
import { Link } from 'react-router'

interface MarkdownDocumentProps {
  content: string
  resolveImage?: (src: string) => string
  resolveLink?: (href: string) => string
}

function isInternalPath(href: string) {
  return href.startsWith('/') && !href.startsWith('//')
}

function textFromChildren(children: ReactNode): string {
  if (typeof children === 'string' || typeof children === 'number') {
    return String(children)
  }
  if (Array.isArray(children)) {
    return children.map(textFromChildren).join('')
  }
  if (children && typeof children === 'object' && 'props' in children) {
    return textFromChildren((children as { props?: { children?: ReactNode } }).props?.children)
  }
  return ''
}

function slugify(value: string) {
  return value
    .toLowerCase()
    .trim()
    .replace(/[^\w\s-]/g, '')
    .replace(/\s+/g, '-')
}

function Heading({
  as: Tag,
  children,
}: {
  as: 'h1' | 'h2' | 'h3' | 'h4'
  children?: ReactNode
}) {
  const id = slugify(textFromChildren(children))
  return <Tag id={id || undefined}>{children}</Tag>
}

export function MarkdownDocument({
  content,
  resolveImage = (src) => src,
  resolveLink = (href) => href,
}: MarkdownDocumentProps) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[[rehypeHighlight, { detect: true }]]}
      components={{
        h1: ({ children }) => <Heading as="h1">{children}</Heading>,
        h2: ({ children }) => <Heading as="h2">{children}</Heading>,
        h3: ({ children }) => <Heading as="h3">{children}</Heading>,
        h4: ({ children }) => <Heading as="h4">{children}</Heading>,
        a: ({ href = '', children }) => {
          const destination = resolveLink(href)
          if (isInternalPath(destination)) {
            return <Link to={destination}>{children}</Link>
          }
          const external = /^(?:https?:|mailto:)/i.test(destination)
          return (
            <a
              href={destination}
              {...(external
                ? { target: '_blank', rel: 'noreferrer' }
                : undefined)}
            >
              {children}
            </a>
          )
        },
        img: ({ src = '', alt = '' }) => (
          <img src={resolveImage(src)} alt={alt} loading="lazy" />
        ),
      }}
    >
      {content}
    </ReactMarkdown>
  )
}
