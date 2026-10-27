import { lazy, Suspense, useEffect, useState } from 'react'
const HardwareDatapathExplorer = lazy(() =>
  import('./HardwareDatapathExplorer').then((module) => ({
    default: module.HardwareDatapathExplorer,
  })),
)
export function HardwareDatapathEntry() {
  const [open, setOpen] = useState(() => window.location.hash === '#datapath')
  useEffect(() => {
    const follow = () => {
      if (window.location.hash === '#datapath') setOpen(true)
    }
    window.addEventListener('hashchange', follow)
    return () => window.removeEventListener('hashchange', follow)
  }, [])
  return (
    <details
      id="datapath"
      className="panel datapath-entry"
      open={open}
      onToggle={(e) => setOpen(e.currentTarget.open)}
    >
      <summary>
        Hardware Datapath Explorer · open logical stages and recorded signals
      </summary>
      {open && (
        <Suspense fallback={<p>Loading recorded datapath examples…</p>}>
          <HardwareDatapathExplorer />
        </Suspense>
      )}
    </details>
  )
}
