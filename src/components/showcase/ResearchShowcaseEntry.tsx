import { lazy, Suspense, useEffect, useState } from 'react'
const ResearchShowcase = lazy(() =>
  import('./ResearchShowcase').then((m) => ({ default: m.ResearchShowcase })),
)
export function ResearchShowcaseEntry() {
  const [open, setOpen] = useState(() => window.location.hash === '#research')
  useEffect(() => {
    const follow = () => {
      if (window.location.hash === '#research') setOpen(true)
    }
    window.addEventListener('hashchange', follow)
    return () => window.removeEventListener('hashchange', follow)
  }, [])
  return (
    <details
      id="research"
      className="panel research-entry"
      open={open}
      onToggle={(e) => setOpen(e.currentTarget.open)}
    >
      <summary>
        Hardware research showcase · compare the seven measured experiments
      </summary>
      {open && (
        <Suspense fallback={<p>Loading recorded hardware evidence…</p>}>
          <ResearchShowcase />
        </Suspense>
      )}
    </details>
  )
}
