import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import process from 'node:process';
import { chromium, firefox, webkit } from 'playwright';
import { expect } from 'playwright/test';

import { isExpectedDocumentUnload, isQueueReadRequest } from './browser-smoke-expectations.mjs';
import { observeBrowserDiagnostics, waitForRenderedMedia, runSustainedWorkload } from './browser-validation.mjs';
import { fixturePng } from './mock-media.mjs';
import { executeProjectFileJourney } from './run-project-file-journey.mjs';

const artifacts = resolve(import.meta.dirname, '../artifacts/browser-smoke');
const origin = `http://127.0.0.1:${process.env.INVOKEAI_PROJECT_FILE_PORT ?? 4180}`;
const path = '/#/app?project=fixture-project-002';
const sessionKey = 'invokeai:v7:webv2:editor-session';
const lockPrefix = `${sessionKey}:`;
await mkdir(artifacts, { recursive: true });
const reports = [];
for (const [engine, type] of Object.entries({ chromium, firefox, webkit })) {
  const report = {
    engine,
    platform: `${process.platform}/${process.arch}`,
    status: 'unavailable',
    safari: 'Real Safari and Apple platforms are not verified by Playwright WebKit.',
  };
  try {
    await executeProjectFileJourney({
      dependencies: {
        launchBrowser: async ({ timeoutMs }) => {
          const browser = await type.launch({
            headless: true,
            timeout: timeoutMs,
            ...(engine === 'chromium' ? { ignoreDefaultArgs: ['--disable-back-forward-cache'] } : {}),
          });
          report.version = browser.version();
          return browser;
        },
        runRoundTrip: async ({ browser, contexts, backend }) => {
          const context = await browser.newContext({
            // PWA navigation caching would bypass the scenario-local idle HTML route and boot another editor.
            serviceWorkers: 'block',
            viewport: { width: 1440, height: 1000 },
          });
          contexts.add(context);
          await context.tracing.start({ screenshots: true, snapshots: true, sources: true });
          // A real same-origin document works in every engine; a manifest navigation downloads in Firefox.
          await context.route('**/__browser-smoke-idle', (route) =>
            route.fulfill({ contentType: 'text/html', body: '<!doctype html><title>Browser smoke idle</title>' })
          );
          const page = await context.newPage();
          let navigating = false;
          let changingEditorRoute = false;
          const diagnostics = await observeBrowserDiagnostics(page, {
            expected: (event) =>
              isExpectedDocumentUnload(event, { navigating, origin }) ||
              (event.kind === 'requestfailed' &&
                /(?:ERR_ABORTED|NS_BINDING_ABORTED|cancelled)/i.test(event.message) &&
                changingEditorRoute &&
                isQueueReadRequest(event, origin)),
          });
          const navigate = async (action) => {
            navigating = true;
            try {
              return await action();
            } finally {
              navigating = false;
            }
          };
          await context.addInitScript((applicationOrigin) => {
            if (location.origin !== applicationOrigin) {
              return;
            }
            Object.defineProperty(Navigator.prototype, 'clipboard', { configurable: true, get: () => undefined });
            window.addEventListener('pageshow', (event) =>
              sessionStorage.setItem(
                'smoke:pageshow',
                JSON.stringify({
                  persisted: event.persisted,
                  reasons: performance.getEntriesByType('navigation')[0]?.notRestoredReasons ?? null,
                })
              )
            );
          }, origin);
          const prompt = page.getByRole('combobox', { exact: true, name: 'Positive prompt' });
          const open = async () => {
            const current = new URL(page.url());
            const action = () => page.goto(`${origin}${path}`, { waitUntil: 'domcontentloaded' });
            if (current.origin !== origin || current.pathname !== '/') {
              await navigate(action);
            } else {
              await action();
            }
            await prompt.waitFor();
          };
          try {
            report.status = 'failed';
            // The legacy fixture's initial split save stages B before its PUT. Keep it in flight through unload.
            const initialSaveStarted = page.waitForRequest(
              (request) =>
                request.method() === 'PUT' && new URL(request.url()).pathname.startsWith('/api/v1/projects/'),
              { timeout: 20_000 }
            );
            // Setup can fail before the later await; retain its rejection without abandoning an unhandled waiter.
            void initialSaveStarted.catch(() => {});
            const heldPuts = [];
            const holdSaves = (route) => {
              if (route.request().method() !== 'PUT') {
                return route.fallback();
              }
              heldPuts.push(route.request().postData());
              return new Promise(() => {});
            };
            await page.route('**/api/v1/projects/**', holdSaves);
            await open();
            report.capabilities = await page.evaluate(() => ({
              webLocks: Boolean(navigator.locks),
              indexedDB: Boolean(globalThis.indexedDB),
            }));
            if (!report.capabilities.webLocks || !report.capabilities.indexedDB) {
              report.status = 'unsupported';
              throw new Error('Required Web Locks and IndexedDB must exist on localhost');
            }
            await waitForRenderedMedia(page);
            await initialSaveStarted;
            // No blur/debounce/save wait after A: the actual unload must checkpoint the held field over B.
            let marker = `smoke held prompt ${engine}`;
            await prompt.fill(marker);
            await navigate(() => page.goto(`${origin}/__browser-smoke-idle`, { waitUntil: 'domcontentloaded' }));
            const recovery = await page.evaluate(async (needle) => {
              const result = { journal: [], drafts: [] };
              for (const { name } of await indexedDB.databases()) {
                const kind = name.startsWith('invokeai:v7:webv2:unload-journal')
                  ? 'journal'
                  : name.startsWith('invokeai:v7:webv2:workbench')
                    ? 'drafts'
                    : null;
                if (!kind) {
                  continue;
                }
                const db = await new Promise((resolve, reject) => {
                  const request = indexedDB.open(name);
                  request.onsuccess = () => resolve(request.result);
                  request.onerror = () => reject(request.error);
                });
                try {
                  const store = kind === 'journal' ? 'entries' : 'draftBodies';
                  if (!db.objectStoreNames.contains(store)) {
                    continue;
                  }
                  const records = await new Promise((resolve, reject) => {
                    const request = db.transaction(store).objectStore(store).getAll();
                    request.onsuccess = () => resolve(request.result);
                    request.onerror = () => reject(request.error);
                  });
                  result[kind].push(
                    ...records.map((record) => ({
                      projectId: record.projectId,
                      hasMarker: record.documentJson.includes(needle),
                    }))
                  );
                } finally {
                  db.close();
                }
              }
              return result;
            }, marker);
            assert.ok(
              recovery.journal.some((entry) => entry.hasMarker),
              'Committed journal must contain A'
            );
            assert.ok(
              recovery.drafts.some((entry) => entry.projectId === 'fixture-project-002' && !entry.hasMarker),
              'The initial save must have staged old B'
            );
            assert.ok(!recovery.drafts.some((entry) => entry.hasMarker), 'Staged B must not explain A recovery');
            assert.ok(heldPuts.length && heldPuts.every((body) => !body.includes(marker)), 'Only old B reached PUT');
            const server = await fetch(
              `${backend.origin}/api/v1/projects/fixture-project-002?max_canvas_schema_version=4&max_document_schema_version=4`
            );
            assert.ok(server.ok);
            assert.ok(!(await server.text()).includes(marker), 'Server must not explain A recovery');
            report.journalAttribution = recovery;
            await page.unroute('**/api/v1/projects/**', holdSaves);
            await open();
            await expect(prompt, 'Journal-only held field must survive actual unload').toHaveValue(marker);
            report.heldFieldRecovery = 'passed';
            // Canvas and layer thumbnails must contain fixture colour, in all three engines.
            await page
              .getByRole('tablist', { exact: true, name: 'Layout preset' })
              .getByRole('tab', { name: /^Edit(, unsaved changes)?$/ })
              .click();
            await waitForRenderedMedia(page, { canvas: true });
            report.canvasPainting = 'passed';
            await page
              .getByRole('tablist', { exact: true, name: 'Layout preset' })
              .getByRole('tab', { name: /^Compose(, unsaved changes)?$/ })
              .click();
            report.sustained = await runSustainedWorkload(page);
            // Inject retained owner DOM into the real collector/runner, rather than fabricated counts.
            await page.evaluate(() => {
              window.__retentionProbe = new MutationObserver((records) => {
                for (const record of records) {
                  for (const node of record.removedNodes) {
                    if (!(node instanceof HTMLElement)) {
                      continue;
                    }
                    const popup = node.matches('[data-scope="popover"][data-part="content"]')
                      ? node
                      : node.querySelector('[data-scope="popover"][data-part="content"]');
                    if (popup?.id) {
                      const retained = popup.cloneNode(true);
                      retained.hidden = true;
                      retained.dataset.retentionProbe = 'true';
                      document.body.append(retained);
                    }
                  }
                }
              });
              window.__retentionProbe.observe(document.body, { childList: true, subtree: true });
            });
            await assert.rejects(() => runSustainedWorkload(page), /Retained popovers/);
            await page.evaluate(() => {
              window.__retentionProbe.disconnect();
              document.querySelectorAll('[data-retention-probe]').forEach((node) => node.remove());
            });
            report.retentionFaultProbe = 'passed';
            const firstSession = await page.evaluate((key) => sessionStorage.getItem(key), sessionKey);
            assert.ok(firstSession);
            // A duplicated tab starts with the same persisted identity; actual product code must take a new lock.
            const competitor = await context.newPage();
            const competingDiagnostics = await observeBrowserDiagnostics(competitor);
            await competitor.goto(`${origin}/__browser-smoke-idle`);
            await competitor.evaluate(({ key, value }) => sessionStorage.setItem(key, value), {
              key: sessionKey,
              value: firstSession,
            });
            await competitor.goto(`${origin}${path}`);
            await competitor.getByRole('combobox', { exact: true, name: 'Positive prompt' }).waitFor();
            const secondSession = await competitor.evaluate((key) => sessionStorage.getItem(key), sessionKey);
            assert.notEqual(secondSession, firstSession, 'Competing editor must not steal a live editor session');
            const held = await competitor.evaluate(async () =>
              (await navigator.locks.query()).held.map((lock) => lock.name)
            );
            assert.ok(held.includes(`${lockPrefix}${firstSession}`));
            assert.ok(held.includes(`${lockPrefix}${secondSession}`));
            await competingDiagnostics.assertHealthy();
            await competitor.close();
            await page.waitForFunction(
              async (name) => !(await navigator.locks.query()).held.some((lock) => lock.name === name),
              `${lockPrefix}${secondSession}`
            );
            report.competingTabs = 'passed';
            marker = `smoke route exit ${engine}`;
            await prompt.fill(marker);
            changingEditorRoute = true;
            try {
              await page.getByRole('button', { exact: true, name: 'Open menu' }).click();
              await page.getByRole('menuitem', { exact: true, name: 'Launchpad' }).click();
              await page.waitForURL(/#\/$/);
              await open();
              await expect(prompt, 'Route exit/re-entry must keep the edit').toHaveValue(marker);
              await prompt.focus();
              await page.keyboard.press('Tab');
              assert.equal(
                await prompt.evaluate(
                  (element) => document.activeElement !== element && document.activeElement !== document.body
                ),
                true,
                'Keyboard focus must advance to a control'
              );
              report.routeAndFocus = 'passed';
            } finally {
              changingEditorRoute = false;
            }
            // Clipboard capability is absent rather than permission-denied: product copy controls must explain it.
            changingEditorRoute = true;
            try {
              await page.goto(`${origin}/#/nodes`);
            } finally {
              changingEditorRoute = false;
            }
            const copy = page.getByRole('button', { exact: true, name: 'Copy path' });
            await copy.waitFor();
            assert.equal(await copy.getAttribute('aria-disabled'), 'true');
            await copy.hover();
            await page.getByRole('tooltip').filter({ hasText: 'Copying needs HTTPS or localhost' }).waitFor();
            report.clipboardUnavailable = 'passed';
            await open();
            marker = `smoke dirty history ${engine}`;
            await prompt.fill(marker);
            await navigate(() => page.goto(`${origin}/__browser-smoke-idle`));
            await navigate(() => page.goBack({ waitUntil: 'domcontentloaded' }));
            await prompt.waitFor();
            await expect(prompt, 'History restoration must keep the edit').toHaveValue(marker);
            report.bfcache = await page.evaluate(() => JSON.parse(sessionStorage.getItem('smoke:pageshow')));
            report.bfcache.reason = report.bfcache.persisted
              ? 'Actual BFCache restoration tested'
              : (report.bfcache.reasons ??
                'Engine did not enter BFCache; detailed eligibility API unavailable. History reload tested.');
            if (report.bfcache.persisted) {
              marker = `smoke restored edit ${engine}`;
              await prompt.fill(marker);
              await navigate(() => page.reload({ waitUntil: 'domcontentloaded' }));
              await prompt.waitFor();
              await expect(prompt, 'Restored editor must still save held edits').toHaveValue(marker);
            }
            // Fault probes use their own document and diagnostics: nominal application errors never get waived.
            const probe = await context.newPage();
            await probe.goto(`${origin}/__browser-smoke-idle`);
            const faultDiagnostics = await observeBrowserDiagnostics(probe);
            await probe.route('**/valid-media-probe', (route) =>
              route.fulfill({ contentType: 'image/png', body: fixturePng(4096, 4096) })
            );
            await probe.setContent(
              `<img alt="Large fixture" style="width:64px;height:64px" src="${origin}/valid-media-probe">`
            );
            const decoded = await waitForRenderedMedia(probe, { requireImage: true });
            assert.equal(decoded.images[0].width, 4096);
            assert.equal(decoded.images[0].height, 4096);
            report.largeMediaDecode = 'passed';
            await probe.route('**/critical-probe', (route) => route.abort('failed'));
            const failedRequest = probe.waitForEvent('requestfailed');
            await probe.evaluate(() => fetch('/critical-probe').catch(() => {}));
            await failedRequest;
            const runtimeError = probe.waitForEvent('pageerror');
            await probe.evaluate(() =>
              setTimeout(() => {
                throw new Error('intentional runtime probe');
              }, 0)
            );
            await runtimeError;
            await probe.setContent('<img alt="Corrupt probe" src="data:image/png;base64,aW52YWxpZA==">');
            await assert.rejects(() => waitForRenderedMedia(probe, { requireImage: true }), /decode/);
            await assert.rejects(
              () => faultDiagnostics.assertHealthy(),
              (error) =>
                error.message.includes('critical-probe') && error.message.includes('intentional runtime probe'),
              'The validator must reject both the critical request and the runtime failure'
            );
            // Removed-on-error media and checker-only thumbnails are false-green regressions.
            await probe.route('**/removed-media-probe', (route) =>
              route.fulfill({
                contentType: 'text/html',
                body: '<img onerror="this.remove()" alt="Removed corrupt" src="data:image/png;base64,aW52YWxpZA==">',
              })
            );
            await probe.goto(`${origin}/removed-media-probe`);
            await assert.rejects(() => waitForRenderedMedia(probe, { timeout: 1000 }), /decode/);
            await probe.goto(`${origin}/__browser-smoke-idle`);
            await assert.rejects(() => waitForRenderedMedia(probe, { timeout: 250 }), /Timeout/);
            await probe.setContent(
              '<div data-hotkey-widget-type-id="canvas"><canvas width="16" height="16"></canvas></div><div data-hotkey-widget-type-id="layers"><canvas aria-label="Overview" width="16" height="16"></canvas><div data-layer-row-id="probe-row"><canvas width="16" height="16"></canvas></div></div>'
            );
            await probe.evaluate(() => {
              const [surface, overview, layer] = document.querySelectorAll('canvas');
              for (const painted of [surface, overview]) {
                painted.getContext('2d').fillStyle = 'rgb(64, 160, 128)';
                painted.getContext('2d').fillRect(0, 0, 16, 16);
              }
              layer.getContext('2d').fillStyle = '#ccc';
              layer.getContext('2d').fillRect(0, 0, 16, 16);
            });
            await assert.rejects(
              () => waitForRenderedMedia(probe, { canvas: true, requireImage: false, timeout: 250 }),
              /Timeout/
            );
            await probe.evaluate(() => (document.querySelector('[data-layer-row-id="probe-row"] canvas').width = 16));
            await assert.rejects(
              () => waitForRenderedMedia(probe, { canvas: true, requireImage: false, timeout: 250 }),
              /Timeout/
            );
            await probe.evaluate(() => {
              const layer = document.querySelector('[data-layer-row-id="probe-row"] canvas');
              layer.getContext('2d').fillStyle = 'rgb(64, 160, 128)';
              layer.getContext('2d').fillRect(0, 0, 16, 16);
            });
            await waitForRenderedMedia(probe, { canvas: true, requireImage: false });
            await probe.evaluate(
              () => (document.querySelector('[data-hotkey-widget-type-id="canvas"] canvas').width = 16)
            );
            await assert.rejects(
              () => waitForRenderedMedia(probe, { canvas: true, requireImage: false, timeout: 250 }),
              /Timeout/
            );
            await probe.close();
            report.validationFaultProbes = 'passed';
            await diagnostics.assertHealthy();
            assert.equal(report.heldFieldRecovery, 'passed', report.heldFieldRecoveryError);
            report.status = 'passed';
          } catch (error) {
            await page
              .screenshot({ path: resolve(artifacts, `${engine}-failure.png`), fullPage: true })
              .catch(() => {});
            report.diagnostics = diagnostics.errors;
            throw error;
          } finally {
            await context.tracing.stop({ path: resolve(artifacts, `${engine}.zip`) });
            await context.close();
            contexts.delete(context);
          }
        },
      },
    });
  } catch (error) {
    report.error = error.stack ?? String(error);
  }
  reports.push(report);
  await writeFile(resolve(artifacts, 'report.json'), `${JSON.stringify(reports, null, 2)}\n`);
}
process.stdout.write(`${JSON.stringify(reports, null, 2)}\n`);
if (reports.some((report) => report.status !== 'passed')) {
  process.exitCode = 1;
}
