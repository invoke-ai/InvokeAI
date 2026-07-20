import { spawn } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import process from 'node:process';
import { chromium } from 'playwright';

import { startMockBackend } from './mock-backend.mjs';
import { getWidgetId } from './widget-sources.mjs';

const root = resolve(import.meta.dirname, '..');
const fixtures = JSON.parse(await readFile(resolve(root, 'performance/architecture-fixtures.json'), 'utf8'));
const manifest = JSON.parse(await readFile(resolve(root, 'dist/.vite/manifest.json'), 'utf8'));
const getScriptId = (source, chunk) => {
  const widgetId = getWidgetId(source);

  return widgetId ? `widget:${widgetId}` : (chunk.name ?? chunk.src ?? source);
};
const chunkNameByPath = new Map(
  Object.entries(manifest).map(([source, chunk]) => [`/${chunk.file}`, getScriptId(source, chunk)])
);
const baselinePath = resolve(root, 'performance/browser-baseline.json');
const artifactPath = resolve(root, 'artifacts/architecture-performance/browser-report.json');
const traceDirectory = resolve(root, 'artifacts/architecture-performance/traces');
const port = Number(process.env.INVOKEAI_PERFORMANCE_PORT ?? 4176);
const origin = `http://127.0.0.1:${String(port)}`;
const backendPort = Number(process.env.INVOKEAI_PERFORMANCE_BACKEND_PORT ?? 4177);
const backendOrigin = `http://127.0.0.1:${String(backendPort)}`;
const updateBaseline = process.argv.includes('--update-baseline');

const median = (values) => {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.floor(sorted.length / 2)] ?? 0;
};

const waitForWidgetRequests = async (fixture, scripts) => {
  const required = new Set(fixture.requiredWidgetRequests ?? []);
  const deadline = Date.now() + 10_000;

  while (Date.now() < deadline) {
    const requested = new Set([...scripts].map((request) => chunkNameByPath.get(request) ?? request));
    if ([...required].every((request) => requested.has(request))) {
      for (const forbidden of fixture.forbiddenWidgetRequests ?? []) {
        if (requested.has(forbidden)) {
          throw new Error(`${fixture.id} requested inactive ${forbidden}.`);
        }
      }
      return;
    }
    await new Promise((resolveWait) => {
      setTimeout(resolveWait, 25);
    });
  }

  throw new Error(`${fixture.id} did not request required widgets ${JSON.stringify([...required])}.`);
};

const waitForPreview = async () => {
  const deadline = Date.now() + 20_000;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(origin);
      if (response.ok) {
        return;
      }
    } catch {
      // The preview server is still starting.
    }
    await new Promise((resolveWait) => {
      setTimeout(resolveWait, 100);
    });
  }
  throw new Error(`Vite preview did not become ready at ${origin}.`);
};

// The measurement is hermetic: the preview proxy targets a disposable
// in-memory mock backend, never a live InvokeAI instance with real data.
const mockBackend = await startMockBackend(backendPort);

const preview = spawn(
  'pnpm',
  ['exec', 'vite', 'preview', '--host', '127.0.0.1', '--port', String(port), '--strictPort'],
  {
    cwd: root,
    detached: true,
    env: { ...process.env, INVOKEAI_DEV_BACKEND: backendOrigin },
    stdio: ['ignore', 'pipe', 'pipe'],
  }
);
let previewError = '';
preview.stderr.on('data', (chunk) => {
  previewError += String(chunk);
});

try {
  await waitForPreview();
  const browser = await chromium.launch({ headless: true });
  const routeReports = [];

  for (const fixture of fixtures.routes) {
    const samples = [];
    let stableScriptRequests = null;

    for (let index = 0; index < fixtures.warmups + fixtures.samples; index += 1) {
      // Every sample starts from byte-identical backend state.
      await fetch(`${backendOrigin}/__reset`, { method: 'POST' });
      const context = await browser.newContext();
      if (index === fixtures.warmups) {
        await context.tracing.start({ screenshots: true, snapshots: true, sources: true });
      }
      const page = await context.newPage();
      await page.addInitScript(() => {
        window.__architectureLongTasks = [];
        new PerformanceObserver((list) => {
          window.__architectureLongTasks.push(...list.getEntries().map((entry) => entry.duration));
        }).observe({ buffered: true, type: 'longtask' });
      });
      const scripts = new Set();
      page.on('response', (response) => {
        if (response.request().resourceType() === 'script' && response.url().startsWith(origin)) {
          scripts.add(new URL(response.url()).pathname);
        }
      });

      const expectedScript = fixture.expectedScriptPattern
        ? page.waitForResponse(
            (response) =>
              response.request().resourceType() === 'script' && response.url().includes(fixture.expectedScriptPattern),
            { timeout: 10_000 }
          )
        : null;
      await page.goto(`${origin}${fixture.path}`, { waitUntil: 'domcontentloaded' });
      await expectedScript;
      let layoutSwitchMs = 0;
      let projectSwitchMs = 0;
      if (fixture.expectedScriptPattern) {
        const presetTrigger = page.getByRole('button', { exact: true, name: 'Default' });
        await presetTrigger.waitFor({ timeout: 10_000 });
        if (fixture.layoutPreset) {
          const layoutSwitchStart = performance.now();
          await presetTrigger.click();
          await page.getByRole('menuitem', { exact: true, name: fixture.layoutPreset }).click();
          await page
            .getByRole('button', { exact: true, name: fixture.layoutPreset })
            .first()
            .waitFor({ timeout: 10_000 });
          layoutSwitchMs = performance.now() - layoutSwitchStart;
        }
      }
      if (fixture.measureProjectSwitch) {
        const projectTabs = page.getByRole('tab');
        const originalProject = projectTabs.first();
        await page.getByRole('button', { name: /new project/i }).click();
        await projectTabs.nth(1).waitFor({ timeout: 10_000 });
        const projectSwitchStart = performance.now();
        await originalProject.click();
        await originalProject.waitFor({ state: 'visible' });
        await page.waitForFunction(
          (projectTab) => projectTab instanceof HTMLElement && projectTab.getAttribute('aria-selected') === 'true',
          await originalProject.elementHandle()
        );
        projectSwitchMs = performance.now() - projectSwitchStart;
      }
      await waitForWidgetRequests(fixture, scripts);
      await page.waitForTimeout(100);
      const timing = await page.evaluate(() => {
        const navigation = performance.getEntriesByType('navigation')[0];
        const longTasks = window.__architectureLongTasks;
        return {
          domContentLoadedMs: navigation ? navigation.domContentLoadedEventEnd : 0,
          loadMs: navigation ? navigation.loadEventEnd : 0,
          longestTaskMs: longTasks.reduce((maximum, duration) => Math.max(maximum, duration), 0),
          routeReadyMs: performance.now(),
        };
      });
      timing.layoutSwitchMs = layoutSwitchMs;
      timing.projectSwitchMs = projectSwitchMs;

      if (index === fixtures.warmups) {
        await mkdir(traceDirectory, { recursive: true });
        await context.tracing.stop({ path: resolve(traceDirectory, `${fixture.id}.zip`) });
      }
      await context.close();

      if (index >= fixtures.warmups) {
        const scriptRequests = [...scripts].map((request) => chunkNameByPath.get(request) ?? request).sort();
        stableScriptRequests ??= scriptRequests;
        if (JSON.stringify(stableScriptRequests) !== JSON.stringify(scriptRequests)) {
          throw new Error(
            `${fixture.id} request set was unstable between samples: ${JSON.stringify(stableScriptRequests)} vs ${JSON.stringify(scriptRequests)}`
          );
        }
        samples.push(timing);
      }
    }

    routeReports.push({
      domContentLoadedMedianMs: median(samples.map((sample) => sample.domContentLoadedMs)),
      id: fixture.id,
      loadMedianMs: median(samples.map((sample) => sample.loadMs)),
      longestTaskMaxMs: Math.max(...samples.map((sample) => sample.longestTaskMs)),
      layoutSwitchMedianMs: median(samples.map((sample) => sample.layoutSwitchMs)),
      projectSwitchMedianMs: median(samples.map((sample) => sample.projectSwitchMs)),
      owner: fixture.owner,
      remediationTicket: fixture.remediationTicket,
      routeReadyMedianMs: median(samples.map((sample) => sample.routeReadyMs)),
      scriptRequests: stableScriptRequests,
    });
  }

  await browser.close();
  const report = {
    browserExecutable: chromium.executablePath(),
    capturedAt: new Date().toISOString(),
    routes: routeReports,
    samples: fixtures.samples,
    warmups: fixtures.warmups,
  };
  await mkdir(dirname(artifactPath), { recursive: true });
  await writeFile(artifactPath, `${JSON.stringify(report, null, 2)}\n`);

  if (updateBaseline) {
    await writeFile(
      baselinePath,
      `${JSON.stringify(
        {
          ...report,
          capturedAt: '2026-07-17',
          enforceTiming: false,
          longTaskTargetMs: 50,
          schemaVersion: 1,
          timingStatus: 'informational-until-ci-runner-is-stable',
          timingTolerancePercent: 0.1,
        },
        null,
        2
      )}\n`
    );
  } else {
    const baseline = JSON.parse(await readFile(baselinePath, 'utf8'));
    const failures = [];
    const timingLimit = (value) => value * (1 + baseline.timingTolerancePercent);
    for (const route of routeReports) {
      const expected = baseline.routes.find((candidate) => candidate.id === route.id);
      if (!expected) {
        failures.push(`${route.id} has no checked-in browser baseline (owner ${route.owner}).`);
        continue;
      }
      if (JSON.stringify(route.scriptRequests) !== JSON.stringify(expected.scriptRequests)) {
        failures.push(
          `${route.id} script request set changed (owner ${route.owner}, remediation ${route.remediationTicket}).\nExpected ${JSON.stringify(expected.scriptRequests)}\nReceived ${JSON.stringify(route.scriptRequests)}`
        );
      }
      if (baseline.enforceTiming && route.domContentLoadedMedianMs > timingLimit(expected.domContentLoadedMedianMs)) {
        failures.push(
          `${route.id} DOMContentLoaded median exceeded its timing tolerance (owner ${route.owner}, remediation ${route.remediationTicket}).`
        );
      }
      if (
        baseline.enforceTiming &&
        expected.layoutSwitchMedianMs > 0 &&
        route.layoutSwitchMedianMs > timingLimit(expected.layoutSwitchMedianMs)
      ) {
        failures.push(
          `${route.id} layout-switch median exceeded its timing tolerance (owner ${route.owner}, remediation ${route.remediationTicket}).`
        );
      }
      if (
        baseline.enforceTiming &&
        expected.projectSwitchMedianMs > 0 &&
        route.projectSwitchMedianMs > timingLimit(expected.projectSwitchMedianMs)
      ) {
        failures.push(
          `${route.id} project-switch median exceeded its timing tolerance (owner ${route.owner}, remediation ${route.remediationTicket}).`
        );
      }
      if (
        baseline.enforceTiming &&
        route.longestTaskMaxMs > Math.max(baseline.longTaskTargetMs, timingLimit(expected.longestTaskMaxMs))
      ) {
        failures.push(
          `${route.id} longest task exceeded its timing tolerance (owner ${route.owner}, remediation ${route.remediationTicket}).`
        );
      }
    }
    if (failures.length > 0) {
      throw new Error(failures.join('\n'));
    }
  }

  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
} catch (error) {
  throw new Error(
    `${error instanceof Error ? error.message : String(error)}${previewError ? `\n${previewError}` : ''}`
  );
} finally {
  if (preview.pid) {
    try {
      process.kill(-preview.pid, 'SIGTERM');
    } catch {
      // The preview process may have already exited after a startup failure.
    }
  }
  await mockBackend.close();
}
