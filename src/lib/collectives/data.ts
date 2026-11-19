/**
 * Presentation data for the `#collectives` workspace.
 *
 * Everything comes from `runtime/results/r6-collectives.json`, written by the
 * Rust runtime's documented `experiment` command: schedules, transfer ranges
 * and element associations are traced beside the executed R1 simulator, and
 * errors are copied from the R1 tensor-parallel experiment of the same run.
 * This module only validates, reshapes and formats that data; it computes
 * no collective, rounding or error itself.
 */
import raw from '../../../runtime/results/r6-collectives.json'

/** A rank's input (number), one addition `[left, right]`, or an exact sum. */
export type Assoc = number | [Assoc, Assoc] | { exact: number[] }

export interface Transfer {
  src: number
  dst: number
  first: number
  elements: number
}

export interface ScheduleRound {
  phase: string
  kind: 'reduce' | 'gather'
  transfers: Transfer[]
}

export interface Schedule {
  algorithm: string
  description: string
  world: number
  elements: number
  rounds: ScheduleRound[]
  ranks_agree: boolean
  association: Assoc[]
}

export interface ShardTreeData {
  world: number
  shards: number
  owners: number[]
  combines: { level: number; left: number; right: number }[]
  rounds: ScheduleRound[]
  association: Assoc
}

export interface ErrorRun {
  wire: string
  world: number
  algorithm: string
  relative_rms_error: number
  mismatched: number
  max_ulp: number
  mean_ulp: number
}

export interface CollectivesData {
  schema: string
  note: string
  schedules: Schedule[]
  shard_trees: ShardTreeData[]
  errors: {
    problem: {
      tokens: number
      hidden: number
      outputs: number
      elements: number
    }
    single_device: {
      wire: string
      relative_rms_error: number
      mismatched: number
    }[]
    runs: ErrorRun[]
  }
}

const SCHEMA = 'fp-play-runtime/r6-collectives/v1'

function fail(what: string): never {
  throw new Error(`r6-collectives: ${what}`)
}

function isCount(v: unknown): v is number {
  return typeof v === 'number' && Number.isInteger(v) && v >= 0
}

function checkAssoc(a: unknown, world: number): void {
  if (isCount(a)) {
    if (a >= world) fail(`leaf ${a} outside ${world} inputs`)
  } else if (Array.isArray(a)) {
    if (a.length !== 2) fail('an addition must have two operands')
    checkAssoc(a[0], world)
    checkAssoc(a[1], world)
  } else if (
    typeof a === 'object' &&
    a !== null &&
    Array.isArray((a as { exact?: unknown }).exact)
  ) {
    for (const r of (a as { exact: unknown[] }).exact) checkAssoc(r, world)
  } else fail('malformed association')
}

function checkRounds(rounds: unknown, world: number, elements: number): void {
  if (!Array.isArray(rounds)) fail('rounds must be a list')
  for (const r of rounds as ScheduleRound[]) {
    if (r.kind !== 'reduce' && r.kind !== 'gather') fail('bad round kind')
    if (!Array.isArray(r.transfers)) fail('transfers must be a list')
    for (const t of r.transfers) {
      if (![t.src, t.dst, t.first, t.elements].every(isCount))
        fail('bad transfer')
      if (t.src >= world || t.dst >= world || t.src === t.dst)
        fail('transfer outside the world')
      if (t.first + t.elements > elements) fail('transfer outside the buffer')
    }
  }
}

/** Validate recorded data; throws on anything malformed instead of guessing. */
export function parseCollectives(value: unknown): CollectivesData {
  const d = value as CollectivesData
  if (typeof d !== 'object' || d === null || d.schema !== SCHEMA)
    fail('unknown schema')
  if (!Array.isArray(d.schedules) || d.schedules.length === 0)
    fail('no schedules')
  for (const s of d.schedules) {
    if (!isCount(s.world) || s.world < 1 || !isCount(s.elements))
      fail('bad schedule shape')
    checkRounds(s.rounds, s.world, s.elements)
    if (!Array.isArray(s.association) || s.association.length !== s.elements)
      fail('one association per element')
    for (const a of s.association) checkAssoc(a, s.world)
  }
  for (const t of d.shard_trees) {
    if (!Array.isArray(t.owners) || t.owners.length !== t.shards)
      fail('one owner per shard')
    if (t.owners.some((o) => !isCount(o) || o >= t.world)) fail('bad owner')
    checkRounds(t.rounds, t.world, Number.MAX_SAFE_INTEGER)
    checkAssoc(t.association, t.shards)
  }
  if (!Array.isArray(d.errors?.runs)) fail('no error runs')
  for (const e of d.errors.runs) {
    if (!(e.relative_rms_error >= 0) || !isCount(e.world)) fail('bad error run')
  }
  return d
}

export const collectives: CollectivesData = parseCollectives(raw)

/** The association as text: `(x2 + (x1 + x0))`; `exact(x0, …)` for an exact sum. */
export function assocText(a: Assoc, leaf = 'x'): string {
  if (typeof a === 'number') return `${leaf}${a}`
  if (Array.isArray(a))
    return `(${assocText(a[0], leaf)} + ${assocText(a[1], leaf)})`
  return `exact(${a.exact.map((r) => `${leaf}${r}`).join(', ')})`
}

/** Additions on the longest path from the result to an input (exact sums count 1). */
export function assocDepth(a: Assoc): number {
  if (typeof a === 'number') return 0
  if (Array.isArray(a)) return 1 + Math.max(assocDepth(a[0]), assocDepth(a[1]))
  return 1
}

export interface TreeNode {
  id: number
  label: string
  x: number
  y: number
  leaf: boolean
}

/** A layered layout: leaves left to right in reading order, each operator above its operands. */
export function layoutTree(a: Assoc, leaf = 'x') {
  const nodes: TreeNode[] = []
  const edges: [number, number][] = []
  let column = 0
  const depth = assocDepth(a)
  const visit = (n: Assoc, level: number): number => {
    const id = nodes.length
    if (typeof n === 'number') {
      nodes.push({
        id,
        label: `${leaf}${n}`,
        x: column++,
        y: depth,
        leaf: true,
      })
      return id
    }
    nodes.push({ id, label: '', x: 0, y: level, leaf: false })
    const kids = Array.isArray(n)
      ? [visit(n[0], level + 1), visit(n[1], level + 1)]
      : n.exact.map((r) => visit(r, level + 1))
    nodes[id].label = Array.isArray(n) ? '+' : 'Σ exact'
    nodes[id].x =
      kids.reduce((s, k) => s + nodes[k].x, 0) / Math.max(1, kids.length)
    for (const k of kids) edges.push([id, k])
    return id
  }
  visit(a, 0)
  return { nodes, edges, columns: Math.max(1, column), depth }
}

/** What each rank does in one round. */
export interface RankCell {
  sendsTo: number | null
  receivesFrom: number | null
  range: string
  kind: ScheduleRound['kind']
}

function rangeText(t: Transfer): string {
  if (t.elements === 0) return 'no elements'
  return t.elements === 1
    ? `element ${t.first}`
    : `elements ${t.first}–${t.first + t.elements - 1}`
}

export function roundCells(world: number, round: ScheduleRound): RankCell[] {
  return Array.from({ length: world }, (_, r) => {
    const out = round.transfers.find((t) => t.src === r)
    const inc = round.transfers.find((t) => t.dst === r)
    return {
      sendsTo: out ? out.dst : null,
      receivesFrom: inc ? inc.src : null,
      range: out ? rangeText(out) : inc ? rangeText(inc) : '',
      kind: round.kind,
    }
  })
}

export function cellText(c: RankCell): string {
  if (c.sendsTo === null && c.receivesFrom === null) return 'idle'
  const parts: string[] = []
  if (c.sendsTo !== null) parts.push(`sends to rank ${c.sendsTo}`)
  if (c.receivesFrom !== null)
    parts.push(
      `${c.kind === 'reduce' ? 'adds' : 'copies'} from rank ${c.receivesFrom}`,
    )
  return `${parts.join(', ')} · ${c.range}`
}

export function schedulesFor(world: number): Schedule[] {
  return collectives.schedules.filter((s) => s.world === world)
}

export const worlds: number[] = [
  ...new Set(collectives.schedules.map((s) => s.world)),
].sort((a, b) => a - b)

export const wires: string[] = [
  ...new Set(collectives.errors.runs.map((e) => e.wire)),
]

/** Relative RMS error against the exact sum, by schedule, for one wire. */
export function errorSeries(wire: string) {
  const by = new Map<string, { world: number; value: number }[]>()
  for (const e of collectives.errors.runs.filter((x) => x.wire === wire)) {
    const list = by.get(e.algorithm) ?? []
    list.push({ world: e.world, value: e.relative_rms_error })
    by.set(e.algorithm, list)
  }
  for (const list of by.values()) list.sort((a, b) => a.world - b.world)
  return by
}
