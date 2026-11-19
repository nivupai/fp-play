/**
 * Shared validation for the runtime workspaces' recorded JSON.
 *
 * Every workspace reads a file written by the Rust runtime's documented
 * `experiment` command (`runtime/results/`). These helpers check the fields a
 * workspace uses and throw on anything malformed instead of guessing.
 */

export class RecordedDataError extends Error {}

export function checker(source: string) {
  const fail = (what: string): never => {
    throw new RecordedDataError(`${source}: ${what}`)
  }
  const obj = (v: unknown, what: string): Record<string, unknown> =>
    typeof v === 'object' && v !== null && !Array.isArray(v)
      ? (v as Record<string, unknown>)
      : fail(`${what} must be an object`)
  const arr = (v: unknown, what: string): unknown[] =>
    Array.isArray(v) ? v : fail(`${what} must be a list`)
  const num = (v: unknown, what: string): number =>
    typeof v === 'number' && Number.isFinite(v)
      ? v
      : fail(`${what} must be a finite number`)
  const count = (v: unknown, what: string): number =>
    typeof v === 'number' && Number.isInteger(v) && v >= 0
      ? v
      : fail(`${what} must be a non-negative integer`)
  const str = (v: unknown, what: string): string =>
    typeof v === 'string' ? v : fail(`${what} must be text`)
  const bool = (v: unknown, what: string): boolean =>
    typeof v === 'boolean' ? v : fail(`${what} must be true or false`)
  return { fail, obj, arr, num, count, str, bool }
}

/** Scientific notation with three significant digits; `0` stays `0`. */
export function sci(x: number): string {
  return x === 0 ? '0' : x.toExponential(2)
}

/** A share in [0, 1] as a percentage with one decimal. */
export function pct(x: number): string {
  return `${(100 * x).toFixed(1)}%`
}

/** Integers with thousands separators (fixed locale, so tests are stable). */
export function int(x: number): string {
  return x.toLocaleString('en-US')
}
