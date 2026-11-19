/* global process, window, document, innerWidth, getComputedStyle -- Node script whose page.evaluate callbacks run in the browser */
/**
 * Browser audit of the runtime workspaces and the landing page (R6f).
 *
 * Usage: build and serve the production app (`npm run build`, then
 * `npx vite preview --port 4173`), then run this file as a Playwright-style
 * page script that default-exports `async (page) => report`, for example with
 * a Playwright/patchright runner pointed at http://localhost:4173/fp-play/.
 * It needs `node_modules/axe-core/axe.min.js` (a dev dependency).
 *
 * It does three things and returns one JSON report:
 *  1. axe-core with every rule enabled (including color contrast) on the
 *     landing page and on each workspace, in the light and dark color schemes;
 *  2. a keyboard-only walkthrough of each new workspace: Tab from the top of
 *     the page to the workspace's summary, Enter to open it, then Tab through
 *     every control, recording each focused element, whether a focus
 *     indicator is drawn (outline or box-shadow), and, for each select,
 *     whether type-ahead changes the selection and the workspace's content;
 *  3. screenshots (JPEG) of every new page at 1280 px and 390 px in both
 *     schemes, saved under docs/images/verification/.
 *
 * It reports what happened; it does not judge screen-reader output.
 */
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

const ROOT = process.env.FP_PLAY_ROOT ?? process.cwd()
const BASE = process.env.FP_PLAY_URL ?? 'http://localhost:4173/fp-play/'
const AXE = readFileSync(
  resolve(ROOT, 'node_modules/axe-core/axe.min.js'),
  'utf8',
)
const SHOTS = resolve(ROOT, 'docs/images/verification')
const NEW = ['serving', 'decode', 'precision', 'dataflow']
const SIZES = [
  ['desktop', 1280, 900],
  ['phone', 390, 844],
]
const SCHEMES = ['light', 'dark']

async function open(page, hash, scheme, width, height) {
  await page.setViewportSize({ width, height })
  await page.emulateMedia({ colorScheme: scheme, reducedMotion: 'reduce' })
  await page.goto(`${BASE}${hash ? `#${hash}` : ''}`)
  await page.reload()
  await page.waitForSelector(hash ? `#${hash} h2` : '#landing-heading')
}

async function audit(page, selector) {
  // Load axe in the evaluation's own context (some drivers evaluate in an
  // isolated world that does not see page-added scripts); it audits the same DOM.
  return page.evaluate(
    async ([src, sel]) => {
      if (!window.axe) new Function(src)()
      const r = await window.axe.run(document.querySelector(sel), {
        resultTypes: ['violations', 'incomplete'],
      })
      const brief = (list) =>
        list.map((v) => ({
          id: v.id,
          impact: v.impact,
          nodes: v.nodes.length,
          targets: v.nodes.slice(0, 5).map((n) => n.target.join(' ')),
        }))
      return {
        axe: window.axe.version,
        violations: brief(r.violations),
        incomplete: brief(r.incomplete),
        passes: r.passes.length,
      }
    },
    [AXE, selector],
  )
}

/** Describe the focused element and whether a focus indicator is drawn. */
function focusInfo(page) {
  return page.evaluate(() => {
    const el = document.activeElement
    if (!el || el === document.body) return null
    const s = getComputedStyle(el)
    const visible =
      (s.outlineStyle !== 'none' && parseFloat(s.outlineWidth) > 0) ||
      s.boxShadow !== 'none'
    const name =
      el.getAttribute('aria-label') ||
      (el.id && document.querySelector(`label[for="${el.id}"]`)?.textContent) ||
      el.textContent?.trim().slice(0, 60) ||
      ''
    const ws = el.closest('details[id]')?.id ?? null
    return {
      tag: el.tagName.toLowerCase(),
      name: name.trim(),
      focusVisible: visible,
      workspace: ws,
    }
  })
}

async function walk(page, id) {
  await open(page, '', 'light', 1280, 900)
  // Tab from the top to this workspace's summary.
  let tabs = 0
  for (; tabs < 600; tabs++) {
    await page.keyboard.press('Tab')
    const at = await page.evaluate(
      (i) => document.activeElement?.matches(`#${i} > summary`) ?? false,
      id,
    )
    if (at) break
  }
  const reached = tabs < 600
  const steps = []
  if (!reached) return { id, reachedSummary: false, tabsToSummary: tabs, steps }
  await page.keyboard.press('Enter')
  await page.waitForSelector(`#${id} .runtime-ws`)
  const opened = await page.evaluate((i) => document.getElementById(i).open, id)
  const content = () =>
    page.evaluate(
      (i) =>
        document.querySelector(`#${i} .runtime-ws`).innerText.length +
        ':' +
        document.querySelector(`#${i} .runtime-ws`).innerText.slice(0, 4000),
      id,
    )
  for (let i = 0; i < 200; i++) {
    await page.keyboard.press('Tab')
    const f = await focusInfo(page)
    if (!f || f.workspace !== id) break
    const step = { ...f }
    if (f.tag === 'select') {
      // Type-ahead: type the start of the next option's label. (On macOS,
      // ArrowDown on a closed native select opens its popup instead of
      // changing the value, so type-ahead is the portable keyboard path.)
      const labels = await page.evaluate(() => {
        const el = document.activeElement
        return {
          values: [...el.options].map((o) => o.textContent),
          index: el.selectedIndex,
        }
      })
      if (labels.values.length > 1) {
        const before = await page.evaluate(() => document.activeElement.value)
        const text = await content()
        const next = labels.values[(labels.index + 1) % labels.values.length]
        await page.keyboard.type(next.slice(0, 12))
        await page.waitForTimeout(100)
        const after = await page.evaluate(() => document.activeElement.value)
        step.typedAhead = next.slice(0, 12)
        step.keyboardChangedValue = before !== after
        step.contentChanged = text !== (await content())
        // Restore the default view (type-ahead buffers reset after a pause).
        await page.waitForTimeout(1100)
        await page.keyboard.type(labels.values[labels.index].slice(0, 12))
        await page.waitForTimeout(1100)
      }
    }
    steps.push(step)
  }
  return {
    id,
    reachedSummary: true,
    tabsToSummary: tabs + 1,
    openedWithEnter: opened,
    steps,
  }
}

export default async function run(page) {
  const report = { url: BASE, audits: [], keyboard: [], screenshots: [] }
  for (const scheme of SCHEMES) {
    await open(page, '', scheme, 1280, 900)
    report.audits.push({
      page: 'landing',
      scheme,
      ...(await audit(page, 'main')),
    })
    for (const id of [...NEW, 'collectives']) {
      await open(page, id, scheme, 1280, 900)
      report.audits.push({ page: id, scheme, ...(await audit(page, `#${id}`)) })
    }
  }
  for (const id of NEW) report.keyboard.push(await walk(page, id))
  for (const [size, w, h] of SIZES) {
    for (const scheme of SCHEMES) {
      await open(page, '', scheme, w, h)
      const landing = `landing-${size}-${scheme}.jpg`
      await page.screenshot({
        path: resolve(SHOTS, landing),
        type: 'jpeg',
        quality: 80,
      })
      const overflow = await page.evaluate(
        () => document.documentElement.scrollWidth > innerWidth,
      )
      report.screenshots.push({ file: landing, horizontalOverflow: overflow })
      for (const id of NEW) {
        await open(page, id, scheme, w, h)
        const file = `${id}-${size}-${scheme}.jpg`
        await page
          .locator(`#${id}`)
          .screenshot({ path: resolve(SHOTS, file), type: 'jpeg', quality: 80 })
        const o = await page.evaluate(
          () => document.documentElement.scrollWidth > innerWidth,
        )
        report.screenshots.push({ file, horizontalOverflow: o })
      }
    }
  }
  return report
}
