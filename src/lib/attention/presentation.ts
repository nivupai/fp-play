import { parseDecimalInput, formatNumber } from '../explorer/presentation'
import type { AttentionInput, Matrix } from '../numerics/attention'
import { validateAttentionInput } from '../numerics/attention'
/** Rows separated by newline or semicolon; columns by comma/space. */
export function parseAttentionMatrix(text: string): number[][] {
  const rows = text
    .trim()
    .split(/[;\n]/)
    .filter((row) => row.trim() !== '')
  if (rows.length > 8) throw new RangeError('Use at most eight matrix rows.')
  return rows.map((row) => {
    const tokens = row.trim().split(/[\s,]+/)
    if (tokens.length > 8)
      throw new RangeError('Use at most eight matrix columns.')
    return tokens.map((token) => {
      const parsed = parseDecimalInput(token)
      if (!parsed.ok) throw new TypeError(parsed.error)
      return parsed.value
    })
  })
}
export function parseAttentionMatrices(
  q: string,
  k: string,
  v: string,
): AttentionInput {
  const input = {
    q: parseAttentionMatrix(q),
    k: parseAttentionMatrix(k),
    v: parseAttentionMatrix(v),
  }
  validateAttentionInput(input)
  return input
}
export function matrixText(matrix: Matrix): string {
  return matrix.map((row) => row.map(formatNumber).join(', ')).join('\n')
}
export function attentionErrors(
  actual: Matrix,
  reference: Matrix,
): { absolute: (number | null)[][]; relative: (number | null)[][] } {
  return {
    absolute: actual.map((row, i) =>
      row.map((value, j) =>
        Number.isFinite(value) && Number.isFinite(reference[i][j])
          ? Math.abs(value - reference[i][j])
          : null,
      ),
    ),
    relative: actual.map((row, i) =>
      row.map((value, j) =>
        Number.isFinite(value) &&
        Number.isFinite(reference[i][j]) &&
        reference[i][j] !== 0
          ? Math.abs((value - reference[i][j]) / reference[i][j])
          : null,
      ),
    ),
  }
}

export type HeatValues = readonly (readonly (number | null)[])[]
export function sharedHeatScale(...matrices: HeatValues[]): number {
  return (
    Math.max(
      0,
      ...matrices
        .flat(2)
        .filter((x): x is number => x !== null && Number.isFinite(x))
        .map(Math.abs),
    ) || 1
  )
}

export function attentionDifference(
  actual: Matrix,
  other: Matrix,
): (number | null)[][] {
  return actual.map((row, i) =>
    row.map((value, j) =>
      Number.isFinite(value) && Number.isFinite(other[i][j])
        ? value - other[i][j]
        : null,
    ),
  )
}
