import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { mkdtemp, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import process from 'node:process';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';
import { parse } from 'yaml';

import { getModuleSourceOwner } from './chunk-source-manifest.mjs';
import {
  BUILD_METRIC_KEYS,
  checkBrowserRouteBudget,
  applyBrowserReference,
  applyBuildReference,
  checkRouteBudget,
  createBrowserReference,
  createBuildReference,
  deriveLimit,
  HARD_CEILING_FLOOR_BYTES,
  isBudgetFailure,
  loadPerformanceReference,
  PERFORMANCE_REFERENCE_SCHEMA_VERSION,
  referenceIncompatibility,
  validatePerformanceReference,
  summarizeBrowserSamples,
  measureRouteBuild,
  waitForRequiredRequests,
} from './performance-budgets.mjs';

const PROJECT_ROOT = '/repo/webv2';

const createSyntheticBuild = (vendorBytes) => {
  const manifest = {
    'entry.ts': {
      file: 'assets/entry.js',
      imports: ['_vendor.js'],
      name: 'entry',
      src: 'entry.ts',
    },
    '_vendor.js': {
      file: 'assets/vendor.js',
      name: 'vendor',
    },
  };
  const chunkSources = {
    chunks: {
      'assets/entry.js': {
        facadeSource: 'source:entry.ts',
        sourceOwners: ['source:entry.ts'],
      },
      'assets/vendor.js': {
        facadeSource: null,
        sourceOwners: ['package:react'],
      },
    },
    schemaVersion: 1,
  };
  const assets = new Map([
    ['assets/entry.js', new Uint8Array(100)],
    ['assets/vendor.js', new Uint8Array(vendorBytes)],
  ]);

  return measureRouteBuild(manifest, chunkSources, 'launchpad', 'entry.ts', (file) => assets.get(file));
};

const createArchitectureBaseline = (measurement) => ({
  build: {
    launchpad: {
      baseline: Object.fromEntries([
        ...BUILD_METRIC_KEYS.map((key) => [key, measurement[key]]),
        ['sourceOwners', measurement.sourceOwners],
      ]),
      owner: 'app',
      remediationTicket: 'performance-gates',
      source: 'entry.ts',
    },
  },
  capturedAt: '2026-07-26',
  developmentInvalidation: {
    ui: {
      maxDirectImporters: 1,
      owner: 'platform',
      remediationTicket: 'narrow-ui',
      specifier: '@platform/ui',
    },
  },
  schemaVersion: 2,
  structural: {
    editorForbiddenInitialChunkNames: ['ag-psd'],
    inactiveWidgetGateOwner: 'workbench',
    launchpadForbiddenInitialSources: ['src/app/WorkbenchApp.tsx'],
  },
});

const createBrowserBaseline = () => ({
  browserExecutable: '/chromium',
  capturedAt: '2026-07-26',
  routes: [
    {
      activatedResourceBaseline: {
        cssRawBytes: 0,
        fontRawBytes: 0,
        imageRawBytes: 0,
        largestAssetRawBytes: 0,
        otherRawBytes: 0,
        requestCount: 0,
        scriptRawBytes: 0,
        scriptRequestCount: 0,
        totalRawBytes: 0,
      },
      domContentLoadedMedianMs: 100,
      id: 'launchpad',
      layoutAckMedianMs: 0,
      layoutReturnSwitchMedianMs: 0,
      layoutSwitchMedianMs: 0,
      loadMedianMs: 110,
      longestTaskMaxMs: 40,
      owner: 'app',
      projectSwitchMedianMs: 0,
      readyMark: 'invokeai:ready:launchpad',
      remediationTicket: 'performance-gates',
      resourceBaseline: {
        cssRawBytes: 1,
        fontRawBytes: 2,
        imageRawBytes: 3,
        largestAssetRawBytes: 10,
        otherRawBytes: 4,
        requestCount: 5,
        scriptRawBytes: 10,
        scriptRequestCount: 1,
        totalRawBytes: 20,
      },
      routeReadyMedianMs: 120,
      scriptSourceOwnerSet: 'launchpad',
      stateProfile: 'empty',
    },
  ],
  sampling: {
    scoredSamples: 5,
    traceSamples: 1,
    warmups: 1,
  },
  schemaVersion: 2,
  scriptSourceOwnerSets: {
    launchpad: ['source:entry.ts'],
  },
  timingPolicy: {
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
});

describe('build performance budgets', () => {
  it('fails a 100 KB shared-vendor regression when entry bytes and source ownership are unchanged', () => {
    const baselineMeasurement = createSyntheticBuild(1_000);
    const regression = createSyntheticBuild(101_000);
    const budget = createArchitectureBaseline(baselineMeasurement).build.launchpad;
    const failures = checkRouteBudget(regression, budget);

    assert.equal(regression.ownedRawBytes, baselineMeasurement.ownedRawBytes);
    assert.deepEqual(regression.sourceOwners, baselineMeasurement.sourceOwners);
    assert.ok(
      failures.some((failure) => failure.message.includes('initial raw assets')),
      failures.map((failure) => failure.message).join('\n')
    );
    assert.ok(
      failures.some((failure) => failure.message.includes('largest initial asset')),
      failures.map((failure) => failure.message).join('\n')
    );
  });

  it('reports added and removed source owners instead of output display-name changes', () => {
    const measurement = createSyntheticBuild(1_000);
    const budget = createArchitectureBaseline(measurement).build.launchpad;
    const changed = {
      ...measurement,
      sourceOwners: ['package:react-dom', 'source:entry.ts'],
    };
    const failures = checkRouteBudget(changed, budget);

    assert.equal(failures.length, 1);
    assert.match(failures[0].message, /Added: package:react-dom/);
    assert.match(failures[0].message, /Removed: package:react/);
  });
});

describe('chunk source ownership manifest', () => {
  it('normalizes first-party sources and pnpm dependency paths while excluding virtual/runtime ids', () => {
    assert.equal(
      getModuleSourceOwner('/repo/webv2/src/app/main.tsx?transform', PROJECT_ROOT),
      'source:src/app/main.tsx'
    );
    assert.equal(
      getModuleSourceOwner(
        '/repo/webv2/node_modules/.pnpm/@chakra-ui+react@3.0.0/node_modules/@chakra-ui/react/dist/index.js',
        PROJECT_ROOT
      ),
      'package:@chakra-ui/react'
    );
    assert.equal(getModuleSourceOwner('\u0000rolldown/runtime.js', PROJECT_ROOT), null);
    assert.equal(getModuleSourceOwner('/outside/repo.ts', PROJECT_ROOT), null);
  });
});

describe('browser performance sampling and policy', () => {
  it('waits for required requests that begin after the initial render', async () => {
    const requested = new Set();
    let resolved = false;
    const waiting = waitForRequiredRequests({
      context: 'editor-canvas/empty',
      getRequested: () => requested,
      pollIntervalMs: 1,
      requiredRequests: ['GalleryImageActionsBridge'],
      timeoutMs: 1_000,
    }).then(() => {
      resolved = true;
    });

    await Promise.resolve();
    assert.equal(resolved, false);

    requested.add('GalleryImageActionsBridge');
    await waiting;
    assert.equal(resolved, true);
  });

  it('keeps worst observed resource costs and structural failures in both modes without reporting routine timings', () => {
    const baseline = createBrowserBaseline();
    const expected = baseline.routes[0];
    const owners = baseline.scriptSourceOwnerSets[expected.scriptSourceOwnerSet];
    const first = {
      activatedResources: { ...expected.activatedResourceBaseline, requestCount: 1 },
      profileCounts: { images: 1 },
      resources: { ...expected.resourceBaseline, scriptRawBytes: 1_000_000 },
      scriptSourceOwners: [...owners, 'source:unexpected.ts'],
      timing: {
        domContentLoadedMs: 100,
        layoutAckMs: 0,
        layoutReturnSwitchMs: 0,
        layoutSwitchMs: 0,
        loadMs: 110,
        longestTaskMs: 40,
        projectSwitchMs: 0,
        routeReadyMs: 120,
      },
    };
    const second = {
      ...first,
      activatedResources: expected.activatedResourceBaseline,
      resources: { ...expected.resourceBaseline, scriptRequestCount: 2 },
    };
    for (const mode of ['resources', 'profile']) {
      const summary = summarizeBrowserSamples([first, second], mode);
      assert.equal(summary.resources.scriptRawBytes, 1_000_000);
      assert.equal(summary.resources.scriptRequestCount, 2);
      assert.equal(summary.activatedResources.requestCount, 1);
      const reference = createBrowserReference([{ ...expected, ...summary }]);
      assert.equal(validatePerformanceReference(reference, 'browser reference'), reference);
      const restored = applyBrowserReference(baseline.routes, reference);
      assert.equal(restored.routes[0].resourceBaseline.scriptRawBytes, 1_000_000);
      assert.equal(restored.routes[0].resourceBaseline.scriptRequestCount, 2);

      const failures = checkBrowserRouteBudget({ ...expected, ...summary }, expected, owners);
      for (const failure of [
        'scriptRawBytes',
        'scriptRequestCount',
        'activated requestCount',
        'source-owner set changed',
      ]) {
        assert.ok(
          failures.some((message) => message.includes(failure)),
          failures.join('\n')
        );
      }
      assert.equal('variance' in summary, mode === 'profile');
      assert.equal('routeReadyMedianMs' in summary, mode === 'profile');
      assert.equal('timing' in summary.rawSamples[0], mode === 'profile');
      if (mode === 'profile') {
        assert.equal(summary.routeReadyMedianMs, 120);
        assert.equal(summary.longestTaskMaxMs, 40);
      }
    }
    for (const changed of [{ profileCounts: { images: 2 } }, { scriptSourceOwners: owners }]) {
      assert.throws(() => summarizeBrowserSamples([first, { ...second, ...changed }], 'resources'), /unstable/);
    }
  });
});

describe('derived growth allowance', () => {
  it('gives a small metric a usable byte allowance and a large one a proportional share', () => {
    // Percent alone is useless on a 2 KB metric; a flat floor alone is meaningless on a 3 MB one.
    assert.equal(deriveLimit('cssRawBytes', 2159), 2159 + 4096);
    assert.equal(deriveLimit('initialRawBytes', 2992581), 2992581 + 29926);
  });

  it('caps request counts exactly, since an extra initial request is structural', () => {
    assert.equal(deriveLimit('requestCount', 106), 106);
    assert.equal(deriveLimit('scriptRequestCount', 90), 90);
  });
});

describe('hard ceiling over the committed baseline', () => {
  it('bounds a moving reference by the committed baseline', () => {
    // Keep the committed baseline as a hard bound so allowance-sized merges cannot compound indefinitely.
    const committed = 100_000;
    const reference = committed + HARD_CEILING_FLOOR_BYTES - 1000;

    assert.equal(deriveLimit('ownedRawBytes', reference, committed), committed + HARD_CEILING_FLOOR_BYTES);
  });
});

describe('base-branch reference', () => {
  const buildReference = (measurements) => createBuildReference(measurements);

  it('accepts re-recorded dependency growth while still rejecting additional growth and requests', () => {
    const previous = createSyntheticBuild(100_000);
    const reference = buildReference([previous]);
    const recorded = { ...previous, ownedRawBytes: 20_000 };
    const actual = { ...recorded, ownedRawBytes: 20_064 };
    const original = applyBuildReference(createArchitectureBaseline(previous).build, reference).build.launchpad;

    assert.ok(checkRouteBudget(actual, original).some((failure) => failure.message.includes('owned JavaScript')));

    const updated = applyBuildReference(createArchitectureBaseline(recorded).build, reference).build.launchpad;
    assert.deepEqual(checkRouteBudget(actual, updated), []);
    assert.match(checkRouteBudget({ ...actual, ownedRawBytes: 30_000 }, updated)[0].message, /owned JavaScript/);
    assert.match(
      checkRouteBudget({ ...actual, requestCount: actual.requestCount + 1 }, updated)[0].message,
      /initial asset requests/
    );
  });

  it('applies the reference by route id, not by position, and reports routes it does not cover', () => {
    const committed = createSyntheticBuild(1_000);
    const decoy = { ...createSyntheticBuild(21_000), routeId: 'other' };
    const build = createArchitectureBaseline(committed).build;

    const { build: budgets, uncovered } = applyBuildReference(build, buildReference([decoy]));

    assert.deepEqual(uncovered, ['launchpad']);
    assert.equal(budgets.launchpad.baseline.ownedRawBytes, committed.ownedRawBytes);
    assert.equal(budgets.launchpad.committed, undefined);
  });

  it('is written as a small versioned file that validates', () => {
    const reference = buildReference([createSyntheticBuild(1_000)]);

    assert.equal(validatePerformanceReference(reference, 'test'), reference);
    assert.equal(reference.schemaVersion, PERFORMANCE_REFERENCE_SCHEMA_VERSION);
    assert.deepEqual(Object.keys(reference.routes), ['launchpad']);
    assert.deepEqual(Object.keys(reference.routes.launchpad), BUILD_METRIC_KEYS);
  });

  it('is judged incompatible, not broken, when written by a checkout with a different metric set', () => {
    // References may come from another commit; schema or metric drift must fall back with a reason.
    const reference = buildReference([createSyntheticBuild(1_000)]);

    assert.equal(referenceIncompatibility(reference, 'build', BUILD_METRIC_KEYS), null);
    assert.match(referenceIncompatibility(reference, 'build', [...BUILD_METRIC_KEYS, 'webpRawBytes']), /metric set/);
    assert.match(referenceIncompatibility(reference, 'browser', BUILD_METRIC_KEYS), /kind "build"/);
    assert.match(
      referenceIncompatibility({ ...reference, schemaVersion: 99 }, 'build', BUILD_METRIC_KEYS),
      /schemaVersion 99/
    );
  });
});

describe('base-branch reference, browser gate', () => {
  const routeFor = (expected, overrides = {}) => ({
    activatedResources: expected.activatedResourceBaseline,
    domContentLoadedMedianMs: expected.domContentLoadedMedianMs,
    id: expected.id,
    layoutAckMedianMs: expected.layoutAckMedianMs,
    layoutReturnSwitchMedianMs: expected.layoutReturnSwitchMedianMs,
    layoutSwitchMedianMs: expected.layoutSwitchMedianMs,
    loadMedianMs: expected.loadMedianMs,
    longestTaskMaxMs: expected.longestTaskMaxMs,
    owner: expected.owner,
    projectSwitchMedianMs: expected.projectSwitchMedianMs,
    remediationTicket: expected.remediationTicket,
    resources: expected.resourceBaseline,
    routeReadyMedianMs: expected.routeReadyMedianMs,
    scriptSourceOwners: [],
    stateProfile: expected.stateProfile,
    ...overrides,
  });
  const check = (route, expected) => checkBrowserRouteBudget(route, expected, route.scriptSourceOwners);

  for (const [baselineKey, resourceKey] of [
    ['resourceBaseline', 'resources'],
    ['activatedResourceBaseline', 'activatedResources'],
  ]) {
    it(`accepts a reviewed ${resourceKey} increase without opening the budget to further growth`, () => {
      const baseline = createBrowserBaseline();
      const [previous] = baseline.routes;
      const reference = createBrowserReference([routeFor(previous)]);
      const recorded = {
        ...previous,
        [baselineKey]: { ...previous[baselineKey], scriptRawBytes: 20_000 },
      };
      const actual = routeFor(recorded, {
        [resourceKey]: { ...recorded[baselineKey], scriptRawBytes: 20_064 },
      });

      assert.ok(check(actual, previous).some((failure) => failure.includes('scriptRawBytes')));

      const {
        routes: [updated],
      } = applyBrowserReference([recorded], reference);
      assert.deepEqual(check(actual, updated), []);
      assert.match(
        check({ ...actual, [resourceKey]: { ...actual[resourceKey], scriptRawBytes: 30_000 } }, updated)[0],
        /scriptRawBytes/
      );
    });
  }

  it('passes growth within the allowance and fails growth beyond it, through the gate itself', () => {
    const baseline = createBrowserBaseline();
    const [expected] = baseline.routes;
    const within = routeFor(expected, {
      resources: { ...expected.resourceBaseline, scriptRawBytes: expected.resourceBaseline.scriptRawBytes + 4096 },
    });
    const beyond = routeFor(expected, {
      resources: { ...expected.resourceBaseline, scriptRawBytes: expected.resourceBaseline.scriptRawBytes + 4097 },
    });

    assert.deepEqual(check(within, expected), []);
    const failures = check(beyond, expected);
    assert.equal(failures.length, 1);
    assert.match(failures[0], /scriptRawBytes reached .* captured /);
  });
});

describe('loading a reference from disk', () => {
  const stage = async (contents) => {
    const root = await mkdtemp(join(tmpdir(), 'perf-reference-'));
    await writeFile(
      join(root, 'build-reference.json'),
      typeof contents === 'string' ? contents : JSON.stringify(contents)
    );
    return root;
  };
  const load = (root) =>
    loadPerformanceReference({
      directory: '.',
      fileName: 'build-reference.json',
      kind: 'build',
      metricKeys: BUILD_METRIC_KEYS,
      root,
    });

  it('falls back with a reason on a different schema version even when the shape is foreign', async () => {
    // The version is judged before the shape, or a bump that also changes shape could never fall back.
    const root = await stage({ schemaVersion: 99, whatever: true });

    const loaded = await load(root);

    assert.equal(loaded.reference, null);
    assert.match(loaded.reason, /schemaVersion 99/);
  });

  it('applies a current, well-formed reference', async () => {
    const root = await stage(createBuildReference([createSyntheticBuild(1_000)]));

    const loaded = await load(root);

    assert.equal(loaded.reason, null);
    assert.deepEqual(Object.keys(loaded.reference.routes), ['launchpad']);
  });
});

describe('the remedy is offered only for budget failures', () => {
  it('recognises byte and request limits from both gates but not structural failures', () => {
    assert.ok(isBudgetFailure('launchpad owned JavaScript reached 4197 bytes/requests (limit 4196, captured 100).'));
    assert.ok(
      isBudgetFailure(
        'launchpad/empty scriptRawBytes reached 4107 (limit 4106, captured 10, owner app, remediation x).'
      )
    );
    // Re-recording would erase exactly the signal these give.
    assert.ok(!isBudgetFailure('launchpad initial source-owner graph changed. Added: source:src/leaked.ts.'));
    assert.ok(!isBudgetFailure('ui has 3 direct importers (budget 1).'));
  });
});

describe('exact-base workflow measurements', () => {
  const packageRoot = fileURLToPath(new URL('../', import.meta.url));
  const workflow = parse(readFileSync(resolve(packageRoot, '../../../.github/workflows/frontend-tests.yml'), 'utf8'));
  const measure = Object.values(workflow.jobs)
    .flatMap((job) => job.steps ?? [])
    .find((step) => step.name === 'measure missing performance base');

  for (const scenario of [
    'complete',
    'missing-browser',
    'malformed-browser',
    'incompatible-browser',
    'incomplete-build',
  ]) {
    it(`accepts failed base checks only with their newly generated valid reference pair (${scenario})`, () => {
      const root = mkdtempSync(join(tmpdir(), 'invokeai-reference-workflow-'));
      try {
        const candidate = resolve(root, 'invokeai/frontend/webv2');
        const base = resolve(root, '.perf-base/invokeai/frontend/webv2');
        const binary = resolve(root, 'bin');
        for (const directory of [candidate, base, binary]) {
          mkdirSync(directory, { recursive: true });
        }
        mkdirSync(resolve(candidate, 'scripts'));
        copyFileSync(
          resolve(packageRoot, 'scripts/performance-budgets.mjs'),
          resolve(candidate, 'scripts/performance-budgets.mjs')
        );
        mkdirSync(resolve(base, 'performance'));
        const buildMeasurement = createSyntheticBuild(1_000);
        const build = createBuildReference([buildMeasurement]);
        const browserBaseline = createBrowserBaseline();
        const browser = createBrowserReference(
          browserBaseline.routes.map((route) => ({
            id: route.id,
            stateProfile: route.stateProfile,
            resources: route.resourceBaseline,
            activatedResources: route.activatedResourceBaseline,
          }))
        );
        writeFileSync(
          resolve(root, 'build.json'),
          JSON.stringify(scenario === 'incomplete-build' ? { ...build, routes: {} } : build)
        );
        writeFileSync(
          resolve(root, 'browser.json'),
          scenario === 'malformed-browser'
            ? '{'
            : JSON.stringify(scenario === 'incompatible-browser' ? { ...browser, schemaVersion: 999 } : browser)
        );
        writeFileSync(
          resolve(base, 'performance/architecture-baseline.json'),
          JSON.stringify(createArchitectureBaseline(buildMeasurement))
        );
        writeFileSync(resolve(base, 'performance/browser-baseline.json'), JSON.stringify(browserBaseline));
        // Stale files, including candidate measurements, cannot rescue a failed base measurement.
        for (const directory of [base, candidate]) {
          const output = resolve(directory, 'artifacts/architecture-performance');
          mkdirSync(output, { recursive: true });
          writeFileSync(resolve(output, 'build-reference.json'), JSON.stringify(build));
          writeFileSync(resolve(output, 'browser-reference.json'), JSON.stringify(browser));
        }
        writeFileSync(resolve(binary, 'pnpm'), '#!/bin/sh\nexit 0\n', { mode: 0o755 });
        writeFileSync(
          resolve(binary, 'node'),
          `#!/bin/sh
          if [ "$1" = --experimental-strip-types ]; then shift; fi
          case "$1" in
            scripts/check-architecture-performance.mjs)
              echo build >> "$GITHUB_WORKSPACE/calls"
              cp "$GITHUB_WORKSPACE/build.json" artifacts/architecture-performance/build-reference.json
              exit 1 ;;
            scripts/measure-architecture-performance.mjs)
              echo browser >> "$GITHUB_WORKSPACE/calls"
              if [ "$REFERENCE_CASE" != missing-browser ]; then
                cp "$GITHUB_WORKSPACE/browser.json" artifacts/architecture-performance/browser-reference.json
              fi
              exit 1 ;;
          esac
          exec "${process.execPath}" "$@"
        `,
          { mode: 0o755 }
        );
        const result = spawnSync('bash', ['-e', '-o', 'pipefail', '-c', measure.run], {
          cwd: base,
          encoding: 'utf8',
          env: {
            ...process.env,
            GITHUB_WORKSPACE: root,
            PATH: `${binary}:${process.env.PATH}`,
            REFERENCE_CASE: scenario,
          },
        });
        assert.equal(readFileSync(resolve(root, 'calls'), 'utf8'), 'build\nbrowser\n');
        assert.equal(result.status, scenario === 'complete' ? 0 : 1, result.stderr);
        if (scenario === 'complete') {
          assert.deepEqual(
            JSON.parse(readFileSync(resolve(candidate, 'artifacts/perf-reference/build-reference.json'), 'utf8')),
            build
          );
          assert.deepEqual(
            JSON.parse(readFileSync(resolve(candidate, 'artifacts/perf-reference/browser-reference.json'), 'utf8')),
            browser
          );
        } else {
          assert.equal(existsSync(resolve(candidate, 'artifacts/perf-reference/build-reference.json')), false);
          assert.equal(existsSync(resolve(candidate, 'artifacts/perf-reference/browser-reference.json')), false);
        }
      } finally {
        rmSync(root, { recursive: true, force: true });
      }
    });
  }
});
