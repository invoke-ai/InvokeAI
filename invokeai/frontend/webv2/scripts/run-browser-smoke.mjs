import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import process from 'node:process';
import { chromium, firefox, webkit } from 'playwright';
import { expect } from 'playwright/test';

import { isExpectedWebKitUnloadError } from './browser-smoke-expectations.mjs';
import { assertRetirement, observeBrowserDiagnostics, waitForRenderedMedia } from './browser-validation.mjs';
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
        runRoundTrip: async ({ browser, contexts }) => {
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
          const diagnostics = observeBrowserDiagnostics(page, {
            expected: (event) =>
              isExpectedWebKitUnloadError(event, { engine, navigating, origin }) ||
              (event.kind === 'requestfailed' &&
                /(?:ERR_ABORTED|NS_BINDING_ABORTED|cancelled)/i.test(event.message) &&
                ((navigating && new URL(event.url).pathname.startsWith('/api/')) ||
                  // The HTTP-only mock has no polling transport; navigation may cancel it before its expected 404.
                  ((navigating || changingEditorRoute) &&
                    event.method === 'GET' &&
                    new URL(event.url).pathname === '/ws/socket.io/' &&
                    new URL(event.url).searchParams.get('transport') === 'polling') ||
                  // Queue reads combine query.signal with owner.signal: remount retires both old scope readers.
                  (changingEditorRoute &&
                    event.method === 'GET' &&
                    /^\/api\/v1\/queue\/default\/(?:status|current|next|item_ids)$/.test(
                      new URL(event.url).pathname
                    )))),
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
            await navigate(() => page.goto(`${origin}${path}`, { waitUntil: 'domcontentloaded' }));
            await prompt.waitFor();
          };
          try {
            report.status = 'failed';
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
            // fill does not blur: reload immediately while the prompt editor still holds its draft.
            let marker = `smoke held prompt ${engine}`;
            await prompt.fill(marker);
            await navigate(() => page.reload({ waitUntil: 'domcontentloaded' }));
            await prompt.waitFor();
            try {
              await expect(prompt, 'Immediate held-field edit must survive reload').toHaveValue(marker);
              report.heldFieldRecovery = 'passed';
            } catch (error) {
              // Continue independent scenarios, but retain the failed required recovery check for the final verdict.
              report.heldFieldRecovery = 'failed';
              report.heldFieldRecoveryError = error.message;
            }
            const firstSession = await page.evaluate((key) => sessionStorage.getItem(key), sessionKey);
            assert.ok(firstSession);
            // A duplicated tab starts with the same persisted identity; actual product code must take a new lock.
            const competitor = await context.newPage();
            const competingDiagnostics = observeBrowserDiagnostics(competitor);
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
              await navigate(async () => {
                await page.getByRole('menuitem', { exact: true, name: 'Launchpad' }).click();
                await page.waitForURL(/#\/$/);
              });
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
            await navigate(() => page.goto(`${origin}/#/nodes`));
            const copy = page.getByRole('button', { exact: true, name: 'Copy path' });
            await copy.waitFor();
            assert.equal(await copy.getAttribute('aria-disabled'), 'true');
            await copy.hover();
            await page.getByRole('tooltip').filter({ hasText: 'Copying needs HTTPS or localhost' }).waitFor();
            report.clipboardUnavailable = 'passed';
            await open();
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
            const faultDiagnostics = observeBrowserDiagnostics(probe);
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
            await assert.rejects(() => faultDiagnostics.assertHealthy(), /critical-probe/);
            const retained = [];
            for (let cycle = 0; cycle < 4; cycle += 1) {
              retained.push(
                await probe.evaluate(() => {
                  document.body.append(document.createElement('aside'));
                  return { portals: document.querySelectorAll('aside').length };
                })
              );
            }
            assert.throws(() => assertRetirement(retained, { portals: 10 }), /Monotonic/);
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
