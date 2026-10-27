// Read-only build verification. Does not contact GitHub or deploy anything.
import { createHash } from 'node:crypto'
import { strict as assert } from 'node:assert'
import { readFileSync, statSync } from 'node:fs'
import { fileURLToPath, URL } from 'node:url'
import { resolve, sep } from 'node:path'
import console from 'node:console'

const root = fileURLToPath(new URL('../', import.meta.url))
const directory = resolve(root, 'dist')
const html = readFileSync(resolve(directory, 'index.html'), 'utf8')
const manifest = JSON.parse(readFileSync(resolve(root, 'package.json'), 'utf8'))
const lock = JSON.parse(
  readFileSync(resolve(root, 'package-lock.json'), 'utf8'),
)
assert.equal(manifest.name, 'fp-play')
assert.equal(
  manifest.private,
  true,
  'Static app must retain npm publication protection',
)
assert.equal(manifest.version, lock.version)
assert.equal(manifest.version, lock.packages[''].version)
assert.match(html, /<title>FP Play[^<]*<\/title>/)
assert.match(html, /<noscript\s*>/)
const assets = [...html.matchAll(/(?:src|href)="([^"]+)"/g)]
  .map(([, path]) => path)
  .filter((path) => !/^(https?:|data:|#)/.test(path))
assert(
  assets.some((path) => path.endsWith('.js')),
  'Missing application JavaScript',
)
assert(
  assets.some((path) => path.endsWith('.css')),
  'Missing application CSS',
)
assert(
  assets.some((path) => path.endsWith('favicon.svg')),
  'Missing favicon',
)
for (const asset of assets) {
  assert(
    asset.startsWith('/fp-play/'),
    `Incorrect GitHub Pages asset path: ${asset}`,
  )
  const file = resolve(directory, asset.slice('/fp-play/'.length))
  assert(file.startsWith(directory + sep), 'Asset escapes the build directory')
  assert(statSync(file).isFile(), `Missing built asset: ${asset}`)
}
const javascript = assets
  .filter((path) => path.endsWith('.js'))
  .map((path) =>
    readFileSync(resolve(directory, path.slice('/fp-play/'.length)), 'utf8'),
  )
  .join('\n')
assert(
  javascript.includes(manifest.version),
  'Built UI version does not match package metadata',
)
assert(
  javascript.includes('https://github.com/nivupai/fp-play'),
  'Missing canonical source repository link',
)
const notices = readFileSync(resolve(directory, 'THIRD_PARTY_NOTICES.txt'))
assert.deepEqual(
  notices,
  readFileSync(resolve(root, 'public/THIRD_PARTY_NOTICES.txt')),
)
assert.match(notices.toString(), /Meta Platforms/)
assert.match(notices.toString(), /Tailwind Labs/)
const waveform = JSON.parse(
  readFileSync(resolve(root, 'rtl/vectors/datapath-waveforms.json'), 'utf8'),
)
for (const group of waveform.groups) {
  const built = readFileSync(resolve(directory, group.download))
  assert.deepEqual(built, readFileSync(resolve(root, 'public', group.download)))
  assert.equal(
    createHash('sha256').update(built).digest('hex'),
    group.sourceHashes['public/' + group.download],
  )
}

console.log(
  `Release ${manifest.version}: ${assets.length} entry assets, third-party notices and 2 VCD downloads verified under /fp-play/. No deployment performed.`,
)
