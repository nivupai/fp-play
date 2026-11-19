import { lazy, Suspense, useEffect, useState } from 'react'
const CollectivesWorkspace = lazy(() =>
  import('./CollectivesWorkspace').then((m) => ({
    default: m.CollectivesWorkspace,
  })),
)
export function CollectivesEntry() {
  const [open, setOpen] = useState(
    () => window.location.hash === '#collectives',
  )
  useEffect(() => {
    const follow = () => {
      if (window.location.hash === '#collectives') setOpen(true)
    }
    window.addEventListener('hashchange', follow)
    return () => window.removeEventListener('hashchange', follow)
  }, [])
  return (
    <details
      id="collectives"
      className="panel collectives-entry"
      open={open}
      onToggle={(e) => setOpen(e.currentTarget.open)}
    >
      <summary>
        Collectives workspace · step through recorded AllReduce schedules
      </summary>
      {open && (
        <Suspense fallback={<p>Loading recorded collective schedules…</p>}>
          <CollectivesWorkspace />
        </Suspense>
      )}
    </details>
  )
}
