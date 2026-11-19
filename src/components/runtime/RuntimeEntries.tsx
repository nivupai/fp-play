import { lazy } from 'react'
import { WorkspaceEntry } from './WorkspaceEntry'

/*
 * The runtime workspaces (Part 2). Each opens lazily from its native hash
 * link and reads one or more recorded JSON files from `runtime/results/`.
 */

const ServingWorkspace = lazy(() =>
  import('./ServingWorkspace').then((m) => ({ default: m.ServingWorkspace })),
)
const DecodeWorkspace = lazy(() =>
  import('./DecodeWorkspace').then((m) => ({ default: m.DecodeWorkspace })),
)
const PrecisionWorkspace = lazy(() =>
  import('./PrecisionWorkspace').then((m) => ({
    default: m.PrecisionWorkspace,
  })),
)
const DataflowWorkspace = lazy(() =>
  import('./DataflowWorkspace').then((m) => ({
    default: m.DataflowWorkspace,
  })),
)

export function ServingEntry() {
  return (
    <WorkspaceEntry
      id="serving"
      summary="Serving workspace · latency, preemption and prefix sharing in the recorded R2 scenarios"
      loading="Loading recorded serving scenarios…"
    >
      <ServingWorkspace />
    </WorkspaceEntry>
  )
}

export function DecodeEntry() {
  return (
    <WorkspaceEntry
      id="decode"
      summary="Decode workspace · does sharding change the tokens? Divergence, margins and batch invariance"
      loading="Loading recorded decode runs…"
    >
      <DecodeWorkspace />
    </WorkspaceEntry>
  )
}

export function PrecisionEntry() {
  return (
    <WorkspaceEntry
      id="precision"
      summary="Precision workspace · per-layer FP32 / BF16 / FP16 / FP8 / INT8 policies and the trained model"
      loading="Loading recorded precision studies…"
    >
      <PrecisionWorkspace />
    </WorkspaceEntry>
  )
}

export function DataflowEntry() {
  return (
    <WorkspaceEntry
      id="dataflow"
      summary="Dataflow workspace · where a transformer layer’s cycles go on a toy accelerator"
      loading="Loading recorded dataflow runs…"
    >
      <DataflowWorkspace />
    </WorkspaceEntry>
  )
}
