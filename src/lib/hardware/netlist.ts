/** Gate-level depth, never a delay estimate. Constants/wires add no levels. */
export interface LogicModule {
  ports: Record<string, { direction: string; bits: (number | string)[] }>
  cells: Record<
    string,
    {
      type: string
      port_directions: Record<string, string>
      connections: Record<string, (number | string)[]>
    }
  >
}
export function analyzeLogic(module: LogicModule) {
  const depth = new Map<number, number>(),
    driver = new Map<number, string>(),
    predecessor = new Map<string, string | null>()
  const inputs = Object.entries(module.ports).filter(
    ([, p]) => p.direction === 'input',
  )
  const outputs = Object.entries(module.ports).filter(
    ([, p]) => p.direction === 'output',
  )
  for (const [, port] of inputs)
    for (const bit of port.bits) if (typeof bit === 'number') depth.set(bit, 0)
  const pending = new Map(Object.entries(module.cells))
  for (const [name, cell] of pending) {
    if (
      /\$_(?:DFF|DFFE|SDFF|DLATCH)|^(?:DFF|DFFR|DFFS|DLH|DLL)/.test(cell.type)
    )
      throw new Error(`Sequential cell ${name}`)
    for (const [port, bits] of Object.entries(cell.connections)) {
      if (!['input', 'output'].includes(cell.port_directions[port]))
        throw new Error(`Missing or unsupported direction ${name}.${port}`)
      if (cell.port_directions[port] === 'output')
        for (const bit of bits)
          if (typeof bit === 'number') {
            if (driver.has(bit) || depth.has(bit))
              throw new Error(`Multiple drivers on ${bit}`)
            driver.set(bit, name)
          }
    }
  }
  while (pending.size) {
    let progress = false
    for (const [name, cell] of pending) {
      const incoming = Object.entries(cell.connections)
        .filter(([p]) => cell.port_directions[p] === 'input')
        .flatMap(([, b]) => b)
      if (incoming.some((b) => typeof b === 'number' && !depth.has(b))) continue
      let longest = 0,
        previous: string | null = null
      for (const bit of incoming)
        if (typeof bit === 'number' && (depth.get(bit) ?? 0) >= longest) {
          longest = depth.get(bit)!
          previous = driver.get(bit) ?? null
        }
      predecessor.set(name, previous)
      for (const [p, bits] of Object.entries(cell.connections))
        if (cell.port_directions[p] === 'output')
          for (const bit of bits)
            if (typeof bit === 'number') depth.set(bit, longest + 1)
      pending.delete(name)
      progress = true
    }
    if (!progress) throw new Error('Combinational cycle or undriven net')
  }
  let logicDepth = 0,
    endCell: string | null = null,
    endpoint: string | null = null
  for (const [port, p] of outputs)
    for (const [i, bit] of p.bits.entries()) {
      if (typeof bit === 'number' && !depth.has(bit))
        throw new Error(`Undriven output ${port}`)
      const d = typeof bit === 'number' ? depth.get(bit)! : 0
      if (d > logicDepth) {
        logicDepth = d
        endCell = typeof bit === 'number' ? (driver.get(bit) ?? null) : null
        endpoint = `${port}[${i}]`
      }
    }
  const path: string[] = []
  while (endCell) {
    path.unshift(endCell)
    endCell = predecessor.get(endCell) ?? null
  }
  return {
    logicDepth,
    endpoint,
    path: path.map((name) => ({ name, type: module.cells[name].type })),
    inputBits: inputs.reduce((s, [, p]) => s + p.bits.length, 0),
    outputBits: outputs.reduce((s, [, p]) => s + p.bits.length, 0),
    combinationalCells: Object.keys(module.cells).length,
    sequentialCells: 0,
  }
}
