import { describe, expect, it } from 'vitest'
import raw from '../../runtime/results/r6-collectives.json'
import {
  assocDepth,
  assocText,
  cellText,
  collectives,
  errorSeries,
  layoutTree,
  parseCollectives,
  roundCells,
  schedulesFor,
  type Assoc,
} from '../lib/collectives/data'

function leaves(a: Assoc): number[] {
  if (typeof a === 'number') return [a]
  if (Array.isArray(a)) return [...leaves(a[0]), ...leaves(a[1])]
  return [...a.exact]
}

describe('recorded collectives data', () => {
  it('matches hand-derived associations from the Rust tests', () => {
    const ring3 = schedulesFor(3).find((s) => s.algorithm === 'ring')!
    // Chunk 0 (elements 0–2 of 8) folds from rank 0: x2 + (x1 + x0).
    expect(assocText(ring3.association[0])).toBe('(x2 + (x1 + x0))')
    const tree4 = schedulesFor(4).find((s) => s.algorithm === 'binomial-tree')!
    expect(assocText(tree4.association[5])).toBe('((x0 + x1) + (x2 + x3))')
    const exact = schedulesFor(8).find((s) => s.algorithm === 'exact-gather')!
    expect(assocText(exact.association[0])).toBe(
      'exact(x0, x1, x2, x3, x4, x5, x6, x7)',
    )
  })

  it('uses every rank exactly once in every association, and ranks agree', () => {
    for (const s of collectives.schedules) {
      expect(s.ranks_agree).toBe(true)
      for (const a of s.association)
        expect(leaves(a).sort((x, y) => x - y)).toEqual(
          Array.from({ length: s.world }, (_, i) => i),
        )
    }
  })

  it('describes every transfer of a round, per rank', () => {
    const ring4 = schedulesFor(4).find((s) => s.algorithm === 'ring')!
    expect(ring4.rounds).toHaveLength(6)
    const cells = roundCells(4, ring4.rounds[0])
    expect(cellText(cells[0])).toBe(
      'sends to rank 1, adds from rank 3 · elements 0–1',
    )
  })

  it('lays out trees with every leaf in its own column', () => {
    const a: Assoc = [[0, 1], 2]
    const t = layoutTree(a)
    expect(t.nodes.filter((n) => n.leaf).map((n) => n.label)).toEqual([
      'x0',
      'x1',
      'x2',
    ])
    expect(t.edges).toHaveLength(4)
    expect(assocDepth(a)).toBe(2)
  })

  it('has an error for every schedule at every recorded P and wire', () => {
    for (const wire of ['fp32', 'bf16']) {
      const series = errorSeries(wire)
      expect(series.get('ring')?.map((p) => p.world)).toEqual([2, 3, 4, 8])
      expect(series.get('halving-doubling')?.map((p) => p.world)).toEqual([
        2, 4, 8,
      ])
    }
  })

  it('rejects malformed data instead of guessing', () => {
    const copy = () => JSON.parse(JSON.stringify(raw)) as typeof raw
    expect(() => parseCollectives({ ...copy(), schema: 'other' })).toThrow()
    const leaf = copy()
    ;(leaf.schedules[0].association as unknown[])[0] = 99
    expect(() => parseCollectives(leaf)).toThrow(/leaf 99/)
    const far = copy()
    far.schedules[0].rounds[0].transfers[0].first = 1000
    expect(() => parseCollectives(far)).toThrow(/outside the buffer/)
    const self = copy()
    self.schedules[0].rounds[0].transfers[0].dst =
      self.schedules[0].rounds[0].transfers[0].src
    expect(() => parseCollectives(self)).toThrow(/outside the world/)
  })
})
