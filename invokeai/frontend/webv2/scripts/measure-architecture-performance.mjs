import { spawn } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import process from 'node:process';
import { chromium } from 'playwright';

import { startMockBackend } from './mock-backend.mjs';
import {
  BROWSER_RESOURCE_METRIC_KEYS,
  checkBrowserRouteBudget,
  createBrowserSamplePlan,
  summarizeBrowserResources,
  validateBrowserBaseline,
  validateChunkSourceManifest,
  waitForRequiredRequests,
} from './performance-budgets.mjs';
import { getWidgetId } from './widget-sources.mjs';

const root = resolve(import.meta.dirname, '..');
const fixtures = JSON.parse(await readFile(resolve(root, 'performance/architecture-fixtures.json'), 'utf8'));
const manifest = JSON.parse(await readFile(resolve(root, 'dist/.vite/manifest.json'), 'utf8'));
const chunkSourceManifest = validateChunkSourceManifest(
  JSON.parse(await readFile(resolve(root, 'dist/.vite/chunk-sources.json'), 'utf8'))
);
const sampleConfig = {
  scoredSamples: fixtures.scoredSamples,
  traceSamples: fixtures.traceSamples,
  warmups: fixtures.warmups,
};
const samplePlan = createBrowserSamplePlan(sampleConfig);
const getScriptLabel = (source, chunk) => {
  const widgetId = getWidgetId(source);

  return widgetId ? `widget:${widgetId}` : (chunk.name ?? chunk.src ?? source);
};
const scriptLabelByPath = new Map(
  Object.entries(manifest).map(([source, chunk]) => [`/${chunk.file}`, getScriptLabel(source, chunk)])
);
const scriptSourceOwnersByPath = new Map(
  Object.values(manifest)
    .filter((chunk) => chunk.file.endsWith('.js'))
    .map((chunk) => {
      const sourceChunk = chunkSourceManifest.chunks[chunk.file];
      if (!sourceChunk) {
        throw new Error(`Chunk source manifest is missing browser script ${chunk.file}.`);
      }
      return [`/${chunk.file}`, sourceChunk.sourceOwners];
    })
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
  const middle = Math.floor(sorted.length / 2);

  if (sorted.length % 2 === 0) {
    return ((sorted[middle - 1] ?? 0) + (sorted[middle] ?? 0)) / 2;
  }
  return sorted[middle] ?? 0;
};

const summarizeVariance = (values) => {
  const mean = values.reduce((total, value) => total + value, 0) / values.length;
  const variance = values.reduce((total, value) => total + (value - mean) ** 2, 0) / Math.max(1, values.length - 1);
  const standardDeviation = Math.sqrt(variance);

  return {
    coefficientOfVariation: mean === 0 ? 0 : standardDeviation / mean,
    maximum: Math.max(...values),
    mean,
    minimum: Math.min(...values),
    standardDeviation,
  };
};

const getRequestedScriptLabels = (scripts) =>
  new Set([...scripts].map((request) => scriptLabelByPath.get(request) ?? request));

const waitForWidgetRequests = async (fixture, scripts) => {
  await waitForRequiredRequests({
    context: `${fixture.id}/${fixture.stateProfile}`,
    getRequested: () => getRequestedScriptLabels(scripts),
    requiredRequests: fixture.requiredWidgetRequests ?? [],
  });

  const requested = getRequestedScriptLabels(scripts);
  for (const forbidden of fixture.forbiddenWidgetRequests ?? []) {
    if (requested.has(forbidden)) {
      throw new Error(`${fixture.id}/${fixture.stateProfile} requested inactive ${forbidden}.`);
    }
  }
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

const waitForSemanticMark = async (page, mark) => {
  await page.waitForFunction((markName) => performance.getEntriesByName(markName, 'mark').length > 0, mark, {
    timeout: 20_000,
  });
  return page.evaluate((markName) => {
    const entries = performance.getEntriesByName(markName, 'mark');
    return entries.at(-1)?.startTime ?? 0;
  }, mark);
};

const isStaticAssetPath = (path) =>
  path === '/' || path === '/index.html' || path.startsWith('/assets/') || path.startsWith('/locales/');

const getBrowserResourceKind = (resourceType) => {
  if (resourceType === 'script') {
    return 'script';
  }
  if (resourceType === 'stylesheet') {
    return 'css';
  }
  if (resourceType === 'font') {
    return 'font';
  }
  if (resourceType === 'image') {
    return 'image';
  }
  return 'other';
};

const createResourceCollector = (page) => {
  const pending = [];
  const resources = new Map();

  page.on('response', (response) => {
    const url = new URL(response.url());
    if (url.origin !== origin || !isStaticAssetPath(url.pathname)) {
      return;
    }

    const path = url.pathname;
    const kind = getBrowserResourceKind(response.request().resourceType());
    const contentLength = Number(response.headers()['content-length'] ?? 0);
    const body = response
      .body()
      .then((bytes) => {
        resources.set(path, { kind, path, rawBytes: bytes.byteLength });
      })
      .catch(() => {
        if (contentLength > 0) {
          resources.set(path, { kind, path, rawBytes: contentLength });
        }
      });
    pending.push(body);
  });

  const settle = async () => {
    let observedPendingCount = -1;
    while (observedPendingCount !== pending.length) {
      observedPendingCount = pending.length;
      await Promise.all(pending);
      await page.waitForTimeout(25);
    }
    return [...resources.values()].sort((left, right) => left.path.localeCompare(right.path));
  };

  return { settle };
};

const getScriptSourceOwners = (scripts) => {
  const owners = new Set();
  for (const request of scripts) {
    const sourceOwners = scriptSourceOwnersByPath.get(request);
    if (!sourceOwners) {
      throw new Error(`No source ownership was emitted for requested script ${request}.`);
    }
    for (const sourceOwner of sourceOwners) {
      owners.add(sourceOwner);
    }
  }
  return [...owners].sort();
};

const getZeroResourceSummary = () => Object.fromEntries(BROWSER_RESOURCE_METRIC_KEYS.map((key) => [key, 0]));

const createSourceOwnerSets = (routes) => {
  const sets = {};
  const setIdByFingerprint = new Map();
  const routeSetIds = new Map();

  for (const route of routes) {
    const fingerprint = JSON.stringify(route.scriptSourceOwners);
    let setId = setIdByFingerprint.get(fingerprint);

    if (!setId) {
      const baseId = `${route.id}-static`;
      setId = baseId;
      let suffix = 2;
      while (sets[setId]) {
        setId = `${baseId}-${String(suffix)}`;
        suffix += 1;
      }
      sets[setId] = route.scriptSourceOwners;
      setIdByFingerprint.set(fingerprint, setId);
    }

    routeSetIds.set(`${route.id}:${route.stateProfile}`, setId);
  }

  return { routeSetIds, sets };
};

const createResourceLimits = (resources) =>
  Object.fromEntries(
    BROWSER_RESOURCE_METRIC_KEYS.map((key) => {
      if (key === 'requestCount' || key === 'scriptRequestCount') {
        return [key, resources[key]];
      }
      return [key, Math.ceil(resources[key] * 1.01)];
    })
  );

const getMaximumResourceSummary = (samples, key) =>
  Object.fromEntries(
    BROWSER_RESOURCE_METRIC_KEYS.map((metric) => [metric, Math.max(...samples.map((sample) => sample[key][metric]))])
  );

const getTiming = (page) =>
  page.evaluate(() => {
    const navigation = performance.getEntriesByType('navigation')[0];
    const longTasks = window.__architectureLongTasks;
    return {
      domContentLoadedMs: navigation ? navigation.domContentLoadedEventEnd : 0,
      loadMs: navigation ? navigation.loadEventEnd : 0,
      longestTaskMs: longTasks.reduce((maximum, duration) => Math.max(maximum, duration), 0),
    };
  });

const runSample = async (browser, fixture, sample) => {
  const reset = await fetch(`${backendOrigin}/__reset?profile=${encodeURIComponent(fixture.stateProfile)}`, {
    method: 'POST',
  });
  if (!reset.ok) {
    throw new Error(`Mock backend reset failed for profile ${fixture.stateProfile}: ${await reset.text()}`);
  }
  const profile = await reset.json();
  if (profile.profile !== fixture.stateProfile) {
    throw new Error(`Mock backend selected ${String(profile.profile)} instead of requested ${fixture.stateProfile}.`);
  }
  const expectedProfileCounts = fixtures.stateProfiles[fixture.stateProfile];
  if (!expectedProfileCounts || JSON.stringify(profile.counts) !== JSON.stringify(expectedProfileCounts)) {
    throw new Error(
      `Mock backend profile ${fixture.stateProfile} has the wrong workload shape.\nExpected ${JSON.stringify(
        expectedProfileCounts
      )}\nReceived ${JSON.stringify(profile.counts)}`
    );
  }

  const context = await browser.newContext();
  const tracePath = resolve(traceDirectory, `${fixture.id}-${fixture.stateProfile}.zip`);
  if (sample.kind === 'trace') {
    await mkdir(traceDirectory, { recursive: true });
    await context.tracing.start({ screenshots: true, snapshots: true, sources: true });
  }

  try {
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
    const resourceCollector = createResourceCollector(page);

    const expectedScript = fixture.expectedScriptPattern
      ? page.waitForResponse(
          (response) =>
            response.request().resourceType() === 'script' && response.url().includes(fixture.expectedScriptPattern),
          { timeout: 10_000 }
        )
      : null;
    await page.goto(`${origin}${fixture.path}`, { waitUntil: 'domcontentloaded' });
    await expectedScript;

    let activatedResourcePaths = new Set();
    let layoutSwitchMs = 0;
    if (fixture.layoutPreset || fixture.centerView) {
      await waitForRequiredRequests({
        context: `${fixture.id}/${fixture.stateProfile} before activation`,
        getRequested: () => getRequestedScriptLabels(scripts),
        requiredRequests: fixture.preActivationRequiredWidgetRequests ?? [],
      });
      const activationTrigger = fixture.layoutPreset
        ? page.getByRole('tab', { exact: true, name: fixture.layoutPreset })
        : page.getByRole('button', { name: /^Center view:/ });
      await activationTrigger.waitFor({ timeout: 10_000 });
      const beforeActivation = await resourceCollector.settle();
      const beforePaths = new Set(beforeActivation.map((resource) => resource.path));
      const interactionMark = `invokeai:interaction:${fixture.id}:${fixture.stateProfile}:layout-switch`;
      let activationTarget = activationTrigger;
      if (fixture.centerView) {
        await activationTrigger.click();
        activationTarget = page.getByRole('menuitemradio', { exact: true, name: fixture.centerView });
        await activationTarget.waitFor({ timeout: 10_000 });
      }
      await page.evaluate(
        ({ interactionMarkName, readyMark }) => {
          performance.clearMarks(interactionMarkName);
          performance.clearMarks(readyMark);
          document.addEventListener(
            'pointerdown',
            () => {
              performance.mark(interactionMarkName);
            },
            { capture: true, once: true }
          );
        },
        { interactionMarkName: interactionMark, readyMark: fixture.readyMark }
      );
      await activationTarget.click();
      if (fixture.layoutPreset) {
        const activationElement = await activationTarget.elementHandle();
        await page.waitForFunction((element) => element?.getAttribute('aria-selected') === 'true', activationElement, {
          timeout: 10_000,
        });
      } else {
        await page
          .getByRole('button', { exact: true, name: `Center view: ${fixture.centerView}` })
          .waitFor({ timeout: 10_000 });
      }
      await waitForSemanticMark(page, fixture.readyMark);
      layoutSwitchMs = await page.evaluate(
        ({ interactionMarkName, readyMark }) => {
          const start = performance.getEntriesByName(interactionMarkName, 'mark').at(-1);
          const end = performance.getEntriesByName(readyMark, 'mark').at(-1);
          return start && end ? Math.max(0, end.startTime - start.startTime) : null;
        },
        { interactionMarkName: interactionMark, readyMark: fixture.readyMark }
      );
      if (layoutSwitchMs === null) {
        throw new Error(`${fixture.id}/${fixture.stateProfile} did not record a layout-switch interaction.`);
      }
      const afterActivation = await resourceCollector.settle();
      activatedResourcePaths = new Set(
        afterActivation.filter((resource) => !beforePaths.has(resource.path)).map((resource) => resource.path)
      );
    } else {
      await waitForSemanticMark(page, fixture.readyMark);
    }

    let projectSwitchMs = 0;
    if (fixture.measureProjectSwitch) {
      const projectSwitcher = page.getByRole('button', { name: /^Switch project\. Current project:/ });
      const originalProjectName = (await projectSwitcher.textContent())?.trim();
      if (!originalProjectName) {
        throw new Error(`${fixture.id}/${fixture.stateProfile} project switcher did not name the active project.`);
      }

      await projectSwitcher.click();
      await page.getByRole('menuitem', { exact: true, name: /new project/i }).click();
      await page.waitForFunction(
        (name) => document.querySelector('button[aria-label^="Switch project."]')?.textContent?.trim() !== name,
        originalProjectName
      );
      await projectSwitcher.click();
      const originalProject = page.getByRole('menuitemradio').filter({ hasText: originalProjectName });
      await originalProject.waitFor({ timeout: 10_000 });
      const projectSwitchStart = performance.now();
      await originalProject.click();
      await page
        .getByRole('button', { exact: true, name: `Switch project. Current project: ${originalProjectName}` })
        .waitFor({ timeout: 10_000 });
      projectSwitchMs = performance.now() - projectSwitchStart;
    }

    await waitForWidgetRequests(fixture, scripts);
    const routeReadyMs = await waitForSemanticMark(page, fixture.readyMark);
    const resources = await resourceCollector.settle();
    const activatedResources = resources.filter((resource) => activatedResourcePaths.has(resource.path));
    const timing = await getTiming(page);

    return {
      activatedResources: summarizeBrowserResources(activatedResources),
      profileCounts: profile.counts,
      resources: summarizeBrowserResources(resources),
      scriptSourceOwners: getScriptSourceOwners(scripts),
      timing: {
        ...timing,
        layoutSwitchMs,
        projectSwitchMs,
        routeReadyMs,
      },
      tracePath: sample.kind === 'trace' ? tracePath : null,
    };
  } finally {
    if (sample.kind === 'trace') {
      await context.tracing.stop({ path: tracePath });
    }
    await context.close();
  }
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

let browser;
try {
  await waitForPreview();
  browser = await chromium.launch({ headless: true });
  const routeReports = [];

  for (const fixture of fixtures.routes) {
    const scoredSamples = [];
    let traceSample = null;
    let stableProfileCounts = null;
    let stableScriptSourceOwners = null;

    for (const sample of samplePlan) {
      let result;
      try {
        result = await runSample(browser, fixture, sample);
      } catch (error) {
        throw new Error(
          `${fixture.id}/${fixture.stateProfile} ${sample.kind} sample ${String(sample.index)} failed: ${
            error instanceof Error ? error.message : String(error)
          }`
        );
      }

      if (sample.kind === 'trace') {
        traceSample = result;
      }
      if (sample.kind !== 'scored') {
        continue;
      }

      stableProfileCounts ??= result.profileCounts;
      stableScriptSourceOwners ??= result.scriptSourceOwners;
      const stableValues = [
        ['profile counts', stableProfileCounts, result.profileCounts],
        ['script source owners', stableScriptSourceOwners, result.scriptSourceOwners],
      ];
      for (const [label, expected, received] of stableValues) {
        if (JSON.stringify(expected) !== JSON.stringify(received)) {
          throw new Error(
            `${fixture.id}/${fixture.stateProfile} ${label} were unstable between scored samples.\nExpected ${JSON.stringify(
              expected
            )}\nReceived ${JSON.stringify(received)}`
          );
        }
      }

      scoredSamples.push({
        activatedResources: result.activatedResources,
        resources: result.resources,
        timing: result.timing,
      });
    }

    const timingKeys = [
      'domContentLoadedMs',
      'layoutSwitchMs',
      'loadMs',
      'longestTaskMs',
      'projectSwitchMs',
      'routeReadyMs',
    ];
    const timingValues = (key) => scoredSamples.map((sample) => sample.timing[key]);
    routeReports.push({
      activatedResources:
        scoredSamples.length > 0
          ? getMaximumResourceSummary(scoredSamples, 'activatedResources')
          : getZeroResourceSummary(),
      domContentLoadedMedianMs: median(timingValues('domContentLoadedMs')),
      id: fixture.id,
      layoutSwitchMedianMs: median(timingValues('layoutSwitchMs')),
      loadMedianMs: median(timingValues('loadMs')),
      longestTaskMaxMs: Math.max(...timingValues('longestTaskMs')),
      owner: fixture.owner,
      profileCounts: stableProfileCounts,
      projectSwitchMedianMs: median(timingValues('projectSwitchMs')),
      rawSamples: scoredSamples,
      readyMark: fixture.readyMark,
      remediationTicket: fixture.remediationTicket,
      resources:
        scoredSamples.length > 0 ? getMaximumResourceSummary(scoredSamples, 'resources') : getZeroResourceSummary(),
      routeReadyMedianMs: median(timingValues('routeReadyMs')),
      scriptSourceOwners: stableScriptSourceOwners ?? [],
      stateProfile: fixture.stateProfile,
      traceSample,
      variance: Object.fromEntries(timingKeys.map((key) => [key, summarizeVariance(timingValues(key))])),
    });
  }

  const report = {
    browserExecutable: chromium.executablePath(),
    capturedAt: new Date().toISOString(),
    routes: routeReports,
    sampling: sampleConfig,
    schemaVersion: 2,
  };
  await mkdir(dirname(artifactPath), { recursive: true });
  await writeFile(artifactPath, `${JSON.stringify(report, null, 2)}\n`);

  if (updateBaseline) {
    const previousBaseline = JSON.parse(await readFile(baselinePath, 'utf8'));
    const sourceOwnerSets = createSourceOwnerSets(routeReports);
    const baseline = {
      browserExecutable: report.browserExecutable,
      capturedAt: new Date().toISOString().slice(0, 10),
      routes: routeReports.map((route) => ({
        activatedResourceBaseline: route.activatedResources,
        activatedResourceLimits: createResourceLimits(route.activatedResources),
        domContentLoadedMedianMs: route.domContentLoadedMedianMs,
        id: route.id,
        layoutSwitchMedianMs: route.layoutSwitchMedianMs,
        loadMedianMs: route.loadMedianMs,
        longestTaskMaxMs: route.longestTaskMaxMs,
        owner: route.owner,
        projectSwitchMedianMs: route.projectSwitchMedianMs,
        readyMark: route.readyMark,
        remediationTicket: route.remediationTicket,
        resourceBaseline: route.resources,
        resourceLimits: createResourceLimits(route.resources),
        routeReadyMedianMs: route.routeReadyMedianMs,
        scriptSourceOwnerSet: sourceOwnerSets.routeSetIds.get(`${route.id}:${route.stateProfile}`),
        stateProfile: route.stateProfile,
      })),
      sampling: sampleConfig,
      schemaVersion: 2,
      scriptSourceOwnerSets: sourceOwnerSets.sets,
      timingPolicy:
        previousBaseline.schemaVersion === 2
          ? previousBaseline.timingPolicy
          : {
              enforce: false,
              longTaskTargetMs: 50,
              runner: {
                id: 'unconfigured',
                minimumStableRuns: 20,
                observedStableRuns: 0,
                stable: false,
              },
              tolerancePercent: 0.1,
            },
    };
    validateBrowserBaseline(baseline, fixtures);
    await writeFile(baselinePath, `${JSON.stringify(baseline, null, 2)}\n`);
  } else {
    const baseline = validateBrowserBaseline(JSON.parse(await readFile(baselinePath, 'utf8')), fixtures);
    const failures = [];
    if (JSON.stringify(baseline.sampling) !== JSON.stringify(sampleConfig)) {
      failures.push(
        `Browser sampling configuration changed. Expected ${JSON.stringify(baseline.sampling)}, received ${JSON.stringify(
          sampleConfig
        )}.`
      );
    }
    for (const route of routeReports) {
      const expected = baseline.routes.find(
        (candidate) => candidate.id === route.id && candidate.stateProfile === route.stateProfile
      );
      if (!expected) {
        failures.push(`${route.id}/${route.stateProfile} has no checked-in browser baseline (owner ${route.owner}).`);
        continue;
      }
      if (expected.readyMark !== route.readyMark) {
        failures.push(
          `${route.id}/${route.stateProfile} semantic-ready mark changed from ${expected.readyMark} to ${route.readyMark}.`
        );
      }
      failures.push(
        ...checkBrowserRouteBudget(
          route,
          expected,
          baseline.timingPolicy,
          baseline.scriptSourceOwnerSets[expected.scriptSourceOwnerSet]
        )
      );
    }
    for (const expected of baseline.routes) {
      if (!routeReports.some((route) => route.id === expected.id && route.stateProfile === expected.stateProfile)) {
        failures.push(`${expected.id}/${expected.stateProfile} baseline has no executable fixture.`);
      }
    }
    if (failures.length > 0) {
      throw new Error(failures.join('\n'));
    }
  }

  process.stdout.write(
    `${JSON.stringify(
      {
        artifactPath,
        routes: routeReports.map((route) => ({
          activatedResources: route.activatedResources,
          id: route.id,
          resources: route.resources,
          stateProfile: route.stateProfile,
          variance: route.variance,
        })),
        sampling: sampleConfig,
      },
      null,
      2
    )}\n`
  );
} catch (error) {
  throw new Error(
    `${error instanceof Error ? error.message : String(error)}${previewError ? `\n${previewError}` : ''}`
  );
} finally {
  await browser?.close();
  if (preview.pid) {
    try {
      process.kill(-preview.pid, 'SIGTERM');
    } catch {
      // The preview process may have already exited after a startup failure.
    }
  }
  await mockBackend.close();
}
