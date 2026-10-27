// Optional local-only Chrome DevTools check. Start an isolated browser and preview first.
import { strict as assert } from 'node:assert'
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { URL } from 'node:url'
import { Buffer } from 'node:buffer'
import { setTimeout, clearTimeout } from 'node:timers'
import process from 'node:process'
import console from 'node:console'
const port = Number(process.env.FP_PLAY_CDP_PORT ?? 9230)
assert.ok(Number.isInteger(port) && port >= 1024 && port <= 65535)
const site = new URL(
  process.env.FP_PLAY_PREVIEW_URL ?? 'http://127.0.0.1:4173/fp-play/',
)
assert.ok(
  ['127.0.0.1', 'localhost'].includes(site.hostname),
  'Browser checks are local-only',
)
const version = JSON.parse(
  readFileSync(new URL('../package.json', import.meta.url)),
).version
const target = await (
  await globalThis.fetch(`http://127.0.0.1:${port}/json/new?about:blank`, {
    method: 'PUT',
  })
).json()
const socket = new globalThis.WebSocket(target.webSocketDebuggerUrl)
await new Promise((resolve, reject) => {
  socket.addEventListener('open', resolve, { once: true })
  socket.addEventListener('error', reject, { once: true })
})
let id = 0
const pending = new Map(),
  runtimeErrors = []
socket.addEventListener('message', (event) => {
  const m = JSON.parse(event.data)
  if (m.method === 'Runtime.exceptionThrown')
    runtimeErrors.push(m.params.exceptionDetails.text)
  if (m.id && pending.has(m.id)) {
    const p = pending.get(m.id)
    pending.delete(m.id)
    clearTimeout(p.timeout)
    if (m.error) p.reject(Error(JSON.stringify(m.error)))
    else p.resolve(m.result)
  }
})
const send = (method, params = {}) =>
  new Promise((resolve, reject) => {
    const n = ++id
    const timeout = setTimeout(() => {
      pending.delete(n)
      reject(Error(`CDP timeout: ${method}`))
    }, 15000)
    pending.set(n, { resolve, reject, timeout })
    socket.send(JSON.stringify({ id: n, method, params }))
  })
async function run(expression) {
  const r = await send('Runtime.evaluate', {
    expression,
    returnByValue: true,
    awaitPromise: true,
  })
  if (r.exceptionDetails) throw Error(JSON.stringify(r.exceptionDetails))
  return r.result.value
}
const wait = (expression) =>
  run(
    `new Promise((resolve,reject)=>{const end=Date.now()+10000;function check(){if(${expression})resolve(true);else if(Date.now()>end)reject(Error('Timed out: '+${JSON.stringify(expression)}));else setTimeout(check,25)}check()})`,
  )
const paint = () =>
  run(
    'new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(()=>setTimeout(resolve,100))))',
  )
const select = (id, value) =>
  run(
    `(()=>{const s=document.getElementById(${JSON.stringify(id)});s.value=${JSON.stringify(value)};s.dispatchEvent(new Event('change',{bubbles:true}))})()`,
  )
const click = (text) =>
  run(
    `(()=>{const el=[...document.querySelectorAll('button,summary')].find(el=>el.textContent.trim()===${JSON.stringify(text)});if(!el)throw Error('Missing control '+${JSON.stringify(text)});el.click()})()`,
  )
async function key(key, code, num) {
  await send('Input.dispatchKeyEvent', {
    type: 'keyDown',
    key,
    code,
    windowsVirtualKeyCode: num,
    ...(key === 'Enter' ? { text: '\r', unmodifiedText: '\r' } : {}),
  })
  await send('Input.dispatchKeyEvent', {
    type: 'keyUp',
    key,
    code,
    windowsVirtualKeyCode: num,
  })
}
const theme = (value) =>
  run(
    `(()=>{const s=document.querySelector('.theme-picker select');s.value=${JSON.stringify(value)};s.dispatchEvent(new Event('change',{bubbles:true}))})()`,
  )
const fits = async () =>
  assert.equal(
    await run(
      'document.documentElement.scrollWidth <= document.documentElement.clientWidth',
    ),
    true,
    'Page overflow (local scroll regions are allowed)',
  )
const out = mkdtempSync(join(tmpdir(), 'fp-play-showcase-'))
async function screenshot(name) {
  await paint()
  const r = await send('Page.captureScreenshot', { format: 'png' })
  writeFileSync(join(out, name), Buffer.from(r.data, 'base64'))
}
try {
  await send('Runtime.enable')
  await send('Page.enable')
  await send('Accessibility.enable')
  await send('Page.bringToFront')
  await send('Emulation.setFocusEmulationEnabled', { enabled: true })
  const browser = await send('Browser.getVersion')
  console.log('Browser:', browser.product)
  await send('Emulation.setDeviceMetricsOverride', {
    width: 1440,
    height: 1100,
    deviceScaleFactor: 1,
    mobile: false,
  })
  await send('Page.navigate', { url: site.href + '#main' })
  await wait("document.querySelector('.landing')")
  assert.match(
    await run("document.querySelector('.release-badge').textContent"),
    new RegExp(version.replaceAll('.', '\\.')),
  )
  await theme('light')
  await paint()
  await fits()
  await screenshot('landing-desktop-light.png')
  // Activate the native research link with the real browser keyboard.
  await run('document.querySelector(\'a[href="#research"]\').focus()')
  await key('Enter', 'Enter', 13)
  await wait("document.getElementById('showcase-unit')")
  assert.equal(await run("document.getElementById('research').open"), true)
  assert.equal(
    await run(
      'document.querySelectorAll(\'[aria-label="Showcase exact hardware measurements"] tbody tr\').length',
    ),
    7,
  )
  await select('showcase-unit', 'mul')
  assert.equal(
    await run(
      'document.querySelectorAll(\'[aria-label="Showcase exact hardware measurements"] tbody tr\').length',
    ),
    2,
  )
  await run(
    '[...document.querySelectorAll(\'[aria-label="Showcase exact hardware measurements"] button\')].at(-1).focus()',
  )
  await key('Enter', 'Enter', 13)
  await wait(
    "document.querySelector('[aria-label=\"Showcase selected experiment\"]').textContent.includes('result[8]')",
  )
  assert.match(
    await run(
      'document.querySelector(\'[aria-label="Showcase selected experiment"]\').textContent',
    ),
    /result\[8\]/,
  )
  assert.equal(
    await run('getComputedStyle(document.activeElement).outlineStyle'),
    'solid',
  )
  await select('showcase-unit', 'all')
  await click('True FMA / tree-leading-bit')
  assert.match(
    await run(
      'document.querySelector(\'[aria-label="Showcase selected experiment"]\').textContent',
    ),
    /flags\[1\]/,
  )
  await run(
    "document.getElementById('research-heading').scrollIntoView({block:'start'})",
  )
  await screenshot('hardware-showcase-desktop.png')
  await theme('system')
  for (const [preference, expected] of [
    ['dark', 'rgb(12, 19, 33)'],
    ['light', 'rgb(245, 247, 250)'],
  ]) {
    await send('Emulation.setEmulatedMedia', {
      features: [{ name: 'prefers-color-scheme', value: preference }],
    })
    await paint()
    assert.equal(
      await run(
        "getComputedStyle(document.querySelector('.app-shell')).backgroundColor",
      ),
      expected,
    )
  }
  await theme('dark')
  await paint()
  for (const width of [390, 320]) {
    await send('Emulation.setDeviceMetricsOverride', {
      width,
      height: 844,
      deviceScaleFactor: 1,
      mobile: true,
    })
    await send('Emulation.setPageScaleFactor', { pageScaleFactor: 1 })
    await paint()
    await fits()
    for (const unit of ['all', 'add', 'mul', 'fma', 'nonfused']) {
      await select('showcase-unit', unit)
      await paint()
      await fits()
    }
    await run("document.querySelector('#research details').open=true")
    await paint()
    await fits()
  }
  await select('showcase-unit', 'all')
  await run("document.querySelector('.research-plot').focus()")
  const plotBefore = await run(
    "document.querySelector('.research-plot').scrollLeft",
  )
  await key('ArrowRight', 'ArrowRight', 39)
  await paint()
  assert.ok(
    (await run("document.querySelector('.research-plot').scrollLeft")) >
      plotBefore,
    'Native plot keyboard scrolling',
  )
  await send('Emulation.setDeviceMetricsOverride', {
    width: 390,
    height: 844,
    deviceScaleFactor: 1,
    mobile: true,
  })
  await run('scrollTo(0,0)')
  await screenshot('landing-mobile-dark.png')
  await run(
    "document.getElementById('research-heading').scrollIntoView({block:'start'})",
  )
  await screenshot('hardware-showcase-mobile-dark.png')
  // Existing numerical, chart and waveform workspaces remain functional.
  await run('document.querySelector(\'a[href="#fma"]\').click()')
  await wait('document.querySelector(\'[aria-label="Fused output hex"]\')')
  assert.match(
    await run(
      'document.querySelector(\'[aria-label="Fused output hex"]\').textContent',
    ),
    /0x0010/,
  )
  assert.match(
    await run(
      'document.querySelector(\'[aria-label="Nonfused output hex"]\').textContent',
    ),
    /0x0000/,
  )
  const viewport = await run(
    'document.querySelector(\'[aria-label="Microscope viewport"]\').textContent',
  )
  await click('Zoom in')
  assert.notEqual(
    await run(
      'document.querySelector(\'[aria-label="Microscope viewport"]\').textContent',
    ),
    viewport,
  )
  await click('Pan right')
  await run("location.hash='#datapath'")
  await wait("document.getElementById('datapath-operation')")
  await select('datapath-operation', 'fma')
  await select('datapath-example', 'product:1:fma')
  assert.match(
    await run(
      'document.querySelector(\'[aria-label="Datapath RTL result"]\').textContent',
    ),
    /0x0010/,
  )
  await run(
    'document.querySelector(\'[aria-label="Datapath stage navigation"]\').focus()',
  )
  await key('End', 'End', 35)
  assert.match(
    await run(
      'document.querySelector(\'[aria-label="Datapath stage navigation"]\').textContent',
    ),
    /8 \/ 8/,
  )
  await click('Inspect recorded waveforms')
  await wait("document.getElementById('datapath-wave-signal')")
  await click('Zoom waveform in')
  await click('Pan waveform right')
  await select('datapath-wave-signal', 'fma_arithmetic_magnitude')
  assert.equal(
    await run(
      'document.querySelector(\'[aria-label="Selected waveform binary"]\').textContent.trim().length',
    ),
    81,
  )
  await run("document.getElementById('datapath-wave-pan').focus()")
  const before = await run(
    "Number(document.getElementById('datapath-wave-pan').value)",
  )
  await key('ArrowRight', 'ArrowRight', 39)
  assert.equal(
    await run("Number(document.getElementById('datapath-wave-pan').value)"),
    before + 1,
  )
  // Emulated touch, not physical-device evidence.
  await send('Emulation.setTouchEmulationEnabled', {
    enabled: true,
    maxTouchPoints: 1,
  })
  const point = await run(
    "(()=>{const b=document.querySelector('.datapath-diagram button');b.scrollIntoView({block:'center'});const r=b.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()",
  )
  await send('Input.dispatchTouchEvent', {
    type: 'touchStart',
    touchPoints: [point],
  })
  await send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] })
  await wait(
    "document.querySelector('[aria-label=\"Datapath stage navigation\"]').textContent.includes('1 / 8')",
  )
  await fits()
  for (const resource of [
    'artifacts/datapath-add.vcd',
    'artifacts/datapath-product.vcd',
    'THIRD_PARTY_NOTICES.txt',
  ]) {
    const r = await globalThis.fetch(new URL(resource, site))
    assert.equal(r.status, 200)
    assert.deepEqual(
      Buffer.from(await r.arrayBuffer()),
      readFileSync(new URL('../public/' + resource, import.meta.url)),
      `Downloaded bytes differ: ${resource}`,
    )
  }
  const tree = await send('Accessibility.getFullAXTree')
  const controls = tree.nodes.filter(
    (n) =>
      !n.ignored &&
      ['button', 'link', 'combobox', 'slider', 'textbox'].includes(
        n.role?.value,
      ),
  )
  const unnamed = controls.filter((n) => !n.name?.value?.trim())
  assert.deepEqual(
    unnamed.map((n) => ({ id: n.nodeId, role: n.role.value })),
    [],
    'Unnamed accessible interactive control',
  )
  assert.ok(tree.nodes.some((n) => n.name?.value === 'Compare hardware units'))
  assert.ok(
    tree.nodes.some((n) => n.name?.value === 'Datapath stage navigation'),
  )
  assert.deepEqual(runtimeErrors, [])
  console.log(
    `PASS: ${controls.length} named accessible controls; desktop/mobile320/390, native keyboard, focus, system/light/dark, charts, FMA/datapath, wave range/zoom/pan, emulated touch and static downloads.`,
  )
  console.log('Screenshots:', out)
  console.log(
    'Not a full WCAG, screen-reader, physical-device or cross-browser certification.',
  )
} finally {
  await send('Target.closeTarget', { targetId: target.id }).catch(() => {})
  socket.close()
}
