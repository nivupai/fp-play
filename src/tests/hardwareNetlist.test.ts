import { describe, expect, it } from 'vitest'
import { analyzeLogic } from '../lib/hardware/netlist'
import type { LogicModule } from '../lib/hardware/netlist'
const gate = (a: number | string, b: number) => ({
  type: 'INV_X1',
  port_directions: { A: 'input', Y: 'output' },
  connections: { A: [a], Y: [b] },
})
const model = (
  cells: LogicModule['cells'],
  outputs: (number | string)[] = [4],
): LogicModule => ({
  ports: {
    a: { direction: 'input', bits: [1] },
    z: { direction: 'output', bits: outputs },
  },
  cells,
})
describe('synthesis netlist depth analysis', () => {
  it('resolves unsorted reconvergent logic and reports an actual longest path', () => {
    const result = analyzeLogic(
      model({
        last: gate(3, 4),
        first: gate(1, 2),
        middle: {
          type: 'AND2_X1',
          port_directions: { A: 'input', B: 'input', Y: 'output' },
          connections: { A: [1], B: [2], Y: [3] },
        },
      }),
    )
    expect(result.logicDepth).toBe(3)
    expect(result.path.map((c) => c.name)).toEqual(['first', 'middle', 'last'])
    expect(result).toMatchObject({
      inputBits: 1,
      outputBits: 1,
      combinationalCells: 3,
      sequentialCells: 0,
      endpoint: 'z[0]',
    })
  })
  it('counts constant and direct-wire outputs as zero gate levels', () => {
    expect(analyzeLogic(model({}, ['0', '1', 1]))).toMatchObject({
      logicDepth: 0,
      combinationalCells: 0,
      outputBits: 3,
      path: [],
    })
  })
  it('rejects cycles instead of returning a plausible depth', () =>
    expect(() => analyzeLogic(model({ a: gate(3, 4), b: gate(4, 3) }))).toThrow(
      /cycle/,
    ))
  it('rejects undriven outputs and unresolved inputs', () => {
    expect(() => analyzeLogic(model({}))).toThrow(/Undriven/)
    expect(() => analyzeLogic(model({ a: gate(7, 4) }))).toThrow(/undriven/)
  })
  it('rejects multiple drivers', () =>
    expect(() => analyzeLogic(model({ a: gate(1, 4), b: gate(1, 4) }))).toThrow(
      /Multiple drivers/,
    ))
  it('rejects sequential cells', () =>
    expect(() =>
      analyzeLogic(model({ a: { ...gate(1, 4), type: 'DFF_X1' } })),
    ).toThrow(/Sequential/))
  it('rejects blackboxes without port directions', () =>
    expect(() =>
      analyzeLogic(model({ a: { ...gate(1, 4), port_directions: {} } })),
    ).toThrow(/direction/))
})
