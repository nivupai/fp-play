import { Suspense, useEffect, useState, type ReactNode } from 'react'

/**
 * A lazily opened `<details>` entry for a runtime workspace, opened by its
 * native hash link (like `#collectives`). The workspace module is loaded
 * only when the entry opens.
 */
export function WorkspaceEntry({
  id,
  summary,
  loading,
  children,
}: {
  id: string
  summary: string
  loading: string
  children: ReactNode
}) {
  const [open, setOpen] = useState(() => window.location.hash === `#${id}`)
  useEffect(() => {
    const follow = () => {
      if (window.location.hash === `#${id}`) setOpen(true)
    }
    window.addEventListener('hashchange', follow)
    return () => window.removeEventListener('hashchange', follow)
  }, [id])
  return (
    <details
      id={id}
      className="panel runtime-entry"
      open={open}
      onToggle={(e) => setOpen(e.currentTarget.open)}
    >
      <summary>{summary}</summary>
      {open && <Suspense fallback={<p>{loading}</p>}>{children}</Suspense>}
    </details>
  )
}
