import { lazy, Suspense, useEffect, useState } from 'react'
const HardwareAnalysis = lazy(() =>
  import('./HardwareAnalysis').then((module) => ({
    default: module.HardwareAnalysis,
  })),
)
/** Load the recorded dataset only when requested, keeping other laboratories responsive. */
export function HardwareAnalysisEntry() {
  const [open, setOpen] = useState(() => window.location.hash === '#hardware')
  useEffect(() => {
    const followHash = () => {
      if (window.location.hash === '#hardware') setOpen(true)
    }
    window.addEventListener('hashchange', followHash)
    return () => window.removeEventListener('hashchange', followHash)
  }, [])
  return (
    <details
      id="hardware"
      className="panel hardware-entry"
      open={open}
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary>Hardware Analysis · open recorded synthesis experiments</summary>
      {open && (
        <Suspense fallback={<p>Loading recorded synthesis results…</p>}>
          <HardwareAnalysis />
        </Suspense>
      )}
    </details>
  )
}
