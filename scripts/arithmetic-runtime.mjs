// Development-only TypeScript loading; no new runtime dependency or checked-in output.
import {
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  mkdirSync,
  writeFileSync,
} from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { pathToFileURL, URL } from 'node:url'
import ts from 'typescript'

export async function withArithmetic(callback) {
  const temporary = mkdtempSync(join(tmpdir(), 'fp-play-arithmetic-'))
  try {
    for (const [relative, subdir, names] of [
      ['../src/lib/numerics/', 'numerics', null],
      [
        '../src/tests/reference/',
        'reference',
        ['fp16Reference.ts', 'fp16MultiplyReference.ts', 'formatReference.ts'],
      ],
    ]) {
      const sourceDir = new URL(relative, import.meta.url)
      const target = join(temporary, subdir)
      mkdirSync(target)
      for (const name of names ??
        readdirSync(sourceDir).filter((name) => name.endsWith('.ts'))) {
        const source = readFileSync(new URL(name, sourceDir), 'utf8')
        const compiled = ts.transpileModule(source, {
          compilerOptions: {
            target: ts.ScriptTarget.ES2022,
            module: ts.ModuleKind.ESNext,
          },
        }).outputText
        writeFileSync(
          join(target, name.replace(/\.ts$/, '.mjs')),
          compiled.replace(
            /from (['"])(\.\.?\/[^'"]+)\1/g,
            (_, quote, path) => `from ${quote}${path}.mjs${quote}`,
          ),
        )
      }
    }
    const load = (relative) =>
      import(pathToFileURL(join(temporary, relative)).href)
    return await callback({
      ...(await load('numerics/fp16.mjs')),
      ...(await load('numerics/formats.mjs')),
      ...(await load('numerics/formatArithmetic.mjs')),
      ...(await load('numerics/dotProduct.mjs')),
      ...(await load('reference/formatReference.mjs')),
      ...(await load('numerics/fp16Arithmetic.mjs')),
      ...(await load('numerics/fp16Multiply.mjs')),
      ...(await load('reference/fp16MultiplyReference.mjs')),
    })
  } finally {
    rmSync(temporary, { recursive: true, force: true })
  }
}
