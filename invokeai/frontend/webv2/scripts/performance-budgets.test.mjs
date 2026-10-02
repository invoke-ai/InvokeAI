import assert from 'node:assert/strict';
import { mkdtemp, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, it } from 'node:test';

import { createChunkSourceManifest, getModuleSourceOwner } from './chunk-source-manifest.mjs';
import {
  BUILD_METRIC_KEYS,
  checkBrowserRouteBudget,
  applyBrowserReference,
  applyBuildReference,
  checkRouteBudget,
  createBrowserReference,
  createBuildReference,
  deriveLimit,
  GROWTH_ALLOWANCE_FLOOR_BYTES,
  HARD_CEILING_FLOOR_BYTES,
  isBudgetFailure,
  loadPerformanceReference,
  PERFORMANCE_REFERENCE_SCHEMA_VERSION,
  referenceIncompatibility,
  validatePerformanceReference,
  createBrowserSamplePlan,
  measureRouteBuild,
  summarizeBrowserResources,
  validateArchitectureBaseline,
  validateBrowserBaseline,
  waitForRequiredRequests,
  waitForStableRequests,
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

const fixtures = {
  routes: [
    {
      id: 'launchpad',
      readyMark: 'invokeai:ready:launchpad',
      stateProfile: 'empty',
    },
  ],
};

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

  it('rejects unused budget schema keys', () => {
    const measurement = createSyntheticBuild(1_000);
    const baseline = createArchitectureBaseline(measurement);
    baseline.build.launchpad.baseline.unusedBudget = 123;

    assert.throws(() => validateArchitectureBaseline(baseline), /Unknown keys: unusedBudget/);
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

  it('emits deterministic sorted source-owner records for chunks only', () => {
    const manifest = createChunkSourceManifest(
      {
        'asset.css': { fileName: 'asset.css', source: '', type: 'asset' },
        'entry.js': {
          facadeModuleId: '/repo/webv2/src/main.tsx',
          fileName: 'entry.js',
          moduleIds: [
            '/repo/webv2/node_modules/react/index.js',
            '/repo/webv2/src/main.tsx',
            '/repo/webv2/src/app.tsx',
            '\u0000rolldown/runtime.js',
          ],
          type: 'chunk',
        },
      },
      PROJECT_ROOT
    );

    assert.deepEqual(manifest, {
      chunks: {
        'entry.js': {
          facadeSource: 'source:src/main.tsx',
          sourceOwners: ['package:react', 'source:src/app.tsx', 'source:src/main.tsx'],
        },
      },
      schemaVersion: 1,
    });
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

  it('waits for cascading requests to settle before measuring activation', async () => {
    const requested = new Set(['GalleryImageActionsBridge']);
    let elapsedMs = 0;
    let polls = 0;

    await waitForStableRequests({
      context: 'editor-canvas/empty before activation',
      getRequested: () => requested,
      now: () => elapsedMs,
      pollIntervalMs: 10,
      stableForMs: 30,
      timeoutMs: 1_000,
      wait: (milliseconds) => {
        elapsedMs += milliseconds;
        polls += 1;
        if (polls === 2) {
          requested.add('GalleryImageActionsBridge dependency');
        }
      },
    });

    assert.equal(elapsedMs, 50);
    assert.deepEqual([...requested], ['GalleryImageActionsBridge', 'GalleryImageActionsBridge dependency']);
  });

  it('keeps the trace sample disjoint from scored samples', () => {
    const plan = createBrowserSamplePlan({ scoredSamples: 3, traceSamples: 1, warmups: 2 });

    assert.deepEqual(
      plan.map((sample) => sample.kind),
      ['warmup', 'warmup', 'trace', 'scored', 'scored', 'scored']
    );
    assert.equal(new Set(plan.map((sample) => sample.index)).size, plan.length);
  });

  it('prevents timing enforcement without a stable runner', () => {
    const baseline = createBrowserBaseline();
    baseline.timingPolicy.enforce = true;

    assert.throws(() => validateBrowserBaseline(baseline, fixtures), /identified stable runner/);
  });

  it('prevents timing enforcement without the configured semantic-ready mark', () => {
    const baseline = createBrowserBaseline();
    baseline.timingPolicy = {
      ...baseline.timingPolicy,
      enforce: true,
      runner: {
        id: 'ci-linux-x64',
        minimumStableRuns: 20,
        observedStableRuns: 20,
        stable: true,
      },
    };
    baseline.routes[0].readyMark = 'invokeai:ready:wrong';

    assert.throws(() => validateBrowserBaseline(baseline, fixtures), /matching semantic-ready mark/);
  });

  it('fails a route whose layout-ack median exceeds its timing tolerance', () => {
    const baseline = createBrowserBaseline();
    baseline.timingPolicy = {
      ...baseline.timingPolicy,
      enforce: true,
      runner: { id: 'ci-linux-x64', minimumStableRuns: 20, observedStableRuns: 20, stable: true },
    };
    const expected = baseline.routes[0];
    expected.layoutAckMedianMs = 100;
    const scriptSourceOwners = baseline.scriptSourceOwnerSets[expected.scriptSourceOwnerSet];
    const route = {
      activatedResources: expected.activatedResourceBaseline,
      domContentLoadedMedianMs: expected.domContentLoadedMedianMs,
      id: expected.id,
      layoutAckMedianMs: 200,
      layoutReturnSwitchMedianMs: expected.layoutReturnSwitchMedianMs,
      layoutSwitchMedianMs: expected.layoutSwitchMedianMs,
      loadMedianMs: expected.loadMedianMs,
      longestTaskMaxMs: expected.longestTaskMaxMs,
      owner: expected.owner,
      projectSwitchMedianMs: expected.projectSwitchMedianMs,
      remediationTicket: expected.remediationTicket,
      resources: expected.resourceBaseline,
      routeReadyMedianMs: expected.routeReadyMedianMs,
      scriptSourceOwners,
      stateProfile: expected.stateProfile,
    };

    const failures = checkBrowserRouteBudget(route, expected, baseline.timingPolicy, scriptSourceOwners);

    assert.ok(
      failures.some((failure) => failure.includes('layout-ack median')),
      failures.join('\n')
    );
  });

  it('passes a route whose layout-ack median stays at or under its timing tolerance', () => {
    const baseline = createBrowserBaseline();
    baseline.timingPolicy = {
      ...baseline.timingPolicy,
      enforce: true,
      runner: { id: 'ci-linux-x64', minimumStableRuns: 20, observedStableRuns: 20, stable: true },
    };
    const expected = baseline.routes[0];
    expected.layoutAckMedianMs = 100;
    const scriptSourceOwners = baseline.scriptSourceOwnerSets[expected.scriptSourceOwnerSet];
    const route = {
      activatedResources: expected.activatedResourceBaseline,
      domContentLoadedMedianMs: expected.domContentLoadedMedianMs,
      id: expected.id,
      layoutAckMedianMs: 100,
      layoutReturnSwitchMedianMs: expected.layoutReturnSwitchMedianMs,
      layoutSwitchMedianMs: expected.layoutSwitchMedianMs,
      loadMedianMs: expected.loadMedianMs,
      longestTaskMaxMs: expected.longestTaskMaxMs,
      owner: expected.owner,
      projectSwitchMedianMs: expected.projectSwitchMedianMs,
      remediationTicket: expected.remediationTicket,
      resources: expected.resourceBaseline,
      routeReadyMedianMs: expected.routeReadyMedianMs,
      scriptSourceOwners,
      stateProfile: expected.stateProfile,
    };

    const failures = checkBrowserRouteBudget(route, expected, baseline.timingPolicy, scriptSourceOwners);

    assert.deepEqual(failures, []);
  });

  it('fails a route whose layout-return-switch median exceeds its timing tolerance', () => {
    const baseline = createBrowserBaseline();
    baseline.timingPolicy = {
      ...baseline.timingPolicy,
      enforce: true,
      runner: { id: 'ci-linux-x64', minimumStableRuns: 20, observedStableRuns: 20, stable: true },
    };
    const expected = baseline.routes[0];
    expected.layoutReturnSwitchMedianMs = 100;
    const scriptSourceOwners = baseline.scriptSourceOwnerSets[expected.scriptSourceOwnerSet];
    const route = {
      activatedResources: expected.activatedResourceBaseline,
      domContentLoadedMedianMs: expected.domContentLoadedMedianMs,
      id: expected.id,
      layoutAckMedianMs: expected.layoutAckMedianMs,
      layoutReturnSwitchMedianMs: 200,
      layoutSwitchMedianMs: expected.layoutSwitchMedianMs,
      loadMedianMs: expected.loadMedianMs,
      longestTaskMaxMs: expected.longestTaskMaxMs,
      owner: expected.owner,
      projectSwitchMedianMs: expected.projectSwitchMedianMs,
      remediationTicket: expected.remediationTicket,
      resources: expected.resourceBaseline,
      routeReadyMedianMs: expected.routeReadyMedianMs,
      scriptSourceOwners,
      stateProfile: expected.stateProfile,
    };

    const failures = checkBrowserRouteBudget(route, expected, baseline.timingPolicy, scriptSourceOwners);

    assert.ok(
      failures.some((failure) => failure.includes('layout-return-switch median')),
      failures.join('\n')
    );
  });

  it('passes a route whose layout-return-switch median stays at or under its timing tolerance', () => {
    const baseline = createBrowserBaseline();
    baseline.timingPolicy = {
      ...baseline.timingPolicy,
      enforce: true,
      runner: { id: 'ci-linux-x64', minimumStableRuns: 20, observedStableRuns: 20, stable: true },
    };
    const expected = baseline.routes[0];
    expected.layoutReturnSwitchMedianMs = 100;
    const scriptSourceOwners = baseline.scriptSourceOwnerSets[expected.scriptSourceOwnerSet];
    const route = {
      activatedResources: expected.activatedResourceBaseline,
      domContentLoadedMedianMs: expected.domContentLoadedMedianMs,
      id: expected.id,
      layoutAckMedianMs: expected.layoutAckMedianMs,
      layoutReturnSwitchMedianMs: 100,
      layoutSwitchMedianMs: expected.layoutSwitchMedianMs,
      loadMedianMs: expected.loadMedianMs,
      longestTaskMaxMs: expected.longestTaskMaxMs,
      owner: expected.owner,
      projectSwitchMedianMs: expected.projectSwitchMedianMs,
      remediationTicket: expected.remediationTicket,
      resources: expected.resourceBaseline,
      routeReadyMedianMs: expected.routeReadyMedianMs,
      scriptSourceOwners,
      stateProfile: expected.stateProfile,
    };

    const failures = checkBrowserRouteBudget(route, expected, baseline.timingPolicy, scriptSourceOwners);

    assert.deepEqual(failures, []);
  });

  it('summarizes unique activated resources by type', () => {
    assert.deepEqual(
      summarizeBrowserResources([
        { kind: 'script', path: '/a.js', rawBytes: 10 },
        { kind: 'script', path: '/a.js', rawBytes: 10 },
        { kind: 'css', path: '/a.css', rawBytes: 5 },
        { kind: 'font', path: '/a.woff2', rawBytes: 7 },
        { kind: 'image', path: '/a.webp', rawBytes: 11 },
        { kind: 'other', path: '/en.json', rawBytes: 3 },
      ]),
      {
        cssRawBytes: 5,
        fontRawBytes: 7,
        imageRawBytes: 11,
        largestAssetRawBytes: 11,
        otherRawBytes: 3,
        requestCount: 5,
        scriptRawBytes: 10,
        scriptRequestCount: 1,
        totalRawBytes: 36,
      }
    );
  });
});

describe('derived growth allowance', () => {
  it('absorbs ordinary growth instead of failing on a handful of bytes', () => {
    const measurement = createSyntheticBuild(1_000);
    const budget = createArchitectureBaseline(measurement).build.launchpad;
    const grown = { ...measurement, ownedRawBytes: measurement.ownedRawBytes + 64 };

    assert.deepEqual(checkRouteBudget(grown, budget), []);
  });

  it('still fails growth beyond the allowance', () => {
    const measurement = createSyntheticBuild(1_000);
    const budget = createArchitectureBaseline(measurement).build.launchpad;
    const bloated = { ...measurement, ownedRawBytes: measurement.ownedRawBytes + 4096 + 1 };

    const failures = checkRouteBudget(bloated, budget);

    assert.equal(failures.length, 1);
    assert.match(failures[0].message, /owned JavaScript reached .* captured /);
  });

  it('gives a small metric a usable byte allowance and a large one a proportional share', () => {
    // Percent alone is useless on a 2 KB metric; a flat floor alone is meaningless on a 3 MB one.
    assert.equal(deriveLimit('cssRawBytes', 2159), 2159 + 4096);
    assert.equal(deriveLimit('initialRawBytes', 2992581), 2992581 + 29926);
  });

  it('allows the floor over a zero baseline, which the exact request counts bound', () => {
    // Zero-byte baselines still get the byte floor; exact request caps catch new assets.
    assert.equal(deriveLimit('otherAssetRawBytes', 0), GROWTH_ALLOWANCE_FLOOR_BYTES);
    assert.equal(deriveLimit('requestCount', 0), 0);
  });

  it('caps request counts exactly, since an extra initial request is structural', () => {
    assert.equal(deriveLimit('requestCount', 106), 106);
    assert.equal(deriveLimit('scriptRequestCount', 90), 90);
  });

  it('keeps the source-owner graph pinned exactly regardless of the byte allowance', () => {
    const measurement = createSyntheticBuild(1_000);
    const budget = createArchitectureBaseline(measurement).build.launchpad;
    const leaked = { ...measurement, sourceOwners: [...measurement.sourceOwners, 'source:src/leaked.ts'] };

    const failures = checkRouteBudget(leaked, budget);

    assert.equal(failures.length, 1);
    assert.match(failures[0].message, /source-owner graph changed/);
  });
});

describe('hard ceiling over the committed baseline', () => {
  it('bounds a moving reference by the committed baseline', () => {
    // Keep the committed baseline as a hard bound so allowance-sized merges cannot compound indefinitely.
    const committed = 100_000;
    const reference = committed + HARD_CEILING_FLOOR_BYTES - 1000;

    assert.equal(deriveLimit('ownedRawBytes', reference, committed), committed + HARD_CEILING_FLOOR_BYTES);
  });

  it('is the allowance when the reference is the committed baseline', () => {
    assert.equal(deriveLimit('ownedRawBytes', 100_000, 100_000), 100_000 + 4096);
  });

  it('caps request counts at the lower of the two, so a flaky extra request in one main run does not stick', () => {
    assert.equal(deriveLimit('requestCount', 107, 106), 106);
    assert.equal(deriveLimit('requestCount', 106, 107), 106);
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

  it('judges bytes against the reference while the committed source-owner pin still fails a leak', () => {
    // Use the reference for accepted byte drift; structural leaks must still fail at zero added bytes.
    const committed = createSyntheticBuild(1_000);
    const mainNow = {
      ...createSyntheticBuild(21_000),
      sourceOwners: [...createSyntheticBuild(21_000).sourceOwners, 'source:src/main-added.ts'],
    };
    const build = createArchitectureBaseline(committed).build;
    const pullRequest = { ...mainNow, ownedRawBytes: mainNow.ownedRawBytes + 64 };

    assert.ok(checkRouteBudget(pullRequest, build.launchpad).length > 0);

    const { build: budgets, uncovered } = applyBuildReference(build, buildReference([mainNow]));

    assert.deepEqual(uncovered, []);
    const failures = checkRouteBudget(pullRequest, budgets.launchpad);
    assert.equal(failures.length, 1);
    assert.match(failures[0].message, /source-owner graph changed.*main-added/);
    assert.deepEqual(budgets.launchpad.baseline.sourceOwners, committed.sourceOwners);
  });

  it('names both origins in a failure so a reader can find the numbers', () => {
    const committed = createSyntheticBuild(1_000);
    const mainNow = createSyntheticBuild(21_000);
    const { build: budgets } = applyBuildReference(
      createArchitectureBaseline(committed).build,
      buildReference([mainNow])
    );
    const bloated = { ...mainNow, ownedRawBytes: mainNow.ownedRawBytes + 4096 + 1 };

    const [failure] = checkRouteBudget(bloated, budgets.launchpad);

    assert.match(
      failure.message,
      new RegExp(`reference ${mainNow.ownedRawBytes}, committed ${committed.ownedRawBytes}`)
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

  it('is a no-op without a reference', () => {
    const build = createArchitectureBaseline(createSyntheticBuild(1_000)).build;

    assert.deepEqual(applyBuildReference(build, null), { build, uncovered: [] });
  });

  it('refuses a route missing a metric rather than deriving a NaN ceiling', () => {
    // Every comparison against NaN is false, so a silent fallback here would pass any route.
    const committed = createSyntheticBuild(1_000);
    const build = createArchitectureBaseline(committed).build;
    const reference = buildReference([committed]);
    delete reference.routes.launchpad.ownedRawBytes;

    assert.throws(() => applyBuildReference(build, reference), /reference build route launchpad/);
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

  it('rejects a file that is not a reference at all, naming what is wrong', () => {
    assert.throws(() => validatePerformanceReference({ routes: {} }, 'performance reference x'), /Missing keys/);
    assert.throws(
      () =>
        validatePerformanceReference(
          { capturedAt: 'now', kind: 'nope', metricKeys: [], routes: {}, schemaVersion: 1 },
          'x'
        ),
      /kind must be/
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
  const check = (route, expected, baseline) =>
    checkBrowserRouteBudget(route, expected, baseline.timingPolicy, route.scriptSourceOwners);

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

      assert.ok(check(actual, previous, baseline).some((failure) => failure.includes('scriptRawBytes')));

      const {
        routes: [updated],
      } = applyBrowserReference([recorded], reference);
      assert.deepEqual(check(actual, updated, baseline), []);
      assert.match(
        check({ ...actual, [resourceKey]: { ...actual[resourceKey], scriptRawBytes: 30_000 } }, updated, baseline)[0],
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

    assert.deepEqual(check(within, expected, baseline), []);
    const failures = check(beyond, expected, baseline);
    assert.equal(failures.length, 1);
    assert.match(failures[0], /scriptRawBytes reached .* captured /);
  });

  it('caps request counts and activated resources exactly and separately', () => {
    const baseline = createBrowserBaseline();
    const [expected] = baseline.routes;
    const extraRequest = routeFor(expected, {
      resources: { ...expected.resourceBaseline, requestCount: expected.resourceBaseline.requestCount + 1 },
    });
    const extraActivated = routeFor(expected, {
      activatedResources: {
        ...expected.activatedResourceBaseline,
        scriptRequestCount: expected.activatedResourceBaseline.scriptRequestCount + 1,
      },
    });

    assert.match(check(extraRequest, expected, baseline)[0], /^\S+ requestCount reached/);
    assert.match(check(extraActivated, expected, baseline)[0], /activated scriptRequestCount reached/);
  });

  it('judges a route against the reference by id and profile, keeping the committed numbers for the ceiling', () => {
    const baseline = createBrowserBaseline();
    const [expected] = baseline.routes;
    const grownResources = {
      ...expected.resourceBaseline,
      scriptRawBytes: expected.resourceBaseline.scriptRawBytes + 5_000,
    };
    const decoyResources = { ...expected.resourceBaseline, scriptRawBytes: 1 };
    const reference = createBrowserReference([
      {
        activatedResources: expected.activatedResourceBaseline,
        id: 'decoy',
        resources: decoyResources,
        stateProfile: expected.stateProfile,
      },
      {
        activatedResources: expected.activatedResourceBaseline,
        id: expected.id,
        resources: grownResources,
        stateProfile: expected.stateProfile,
      },
    ]);

    const { routes, uncovered } = applyBrowserReference(baseline.routes, reference);
    const [applied] = routes;

    assert.deepEqual(
      uncovered,
      baseline.routes.slice(1).map((route) => `${route.id}:${route.stateProfile}`)
    );
    assert.deepEqual(applied.resourceBaseline, grownResources);
    assert.deepEqual(applied.committedResourceBaseline, expected.resourceBaseline);
    assert.equal(applied.scriptSourceOwnerSet, expected.scriptSourceOwnerSet);

    // 5 KB over the committed number would fail against the committed baseline alone...
    const route = routeFor(expected, {
      resources: { ...grownResources, scriptRawBytes: grownResources.scriptRawBytes + 64 },
    });
    assert.ok(check(route, expected, baseline).length > 0);
    // ...and passes against the reference, with both origins named if it ever fails.
    assert.deepEqual(check(route, applied, baseline), []);
    const [failure] = check(
      routeFor(expected, { resources: { ...grownResources, scriptRawBytes: grownResources.scriptRawBytes + 4097 } }),
      applied,
      baseline
    );
    assert.match(failure, /reference \d+, committed \d+/);
  });

  it('refuses a reference route missing activated resources rather than deriving NaN ceilings', () => {
    const baseline = createBrowserBaseline();
    const [expected] = baseline.routes;
    const reference = createBrowserReference([
      {
        activatedResources: expected.activatedResourceBaseline,
        id: expected.id,
        resources: expected.resourceBaseline,
        stateProfile: expected.stateProfile,
      },
    ]);
    delete reference.routes[`${expected.id}:${expected.stateProfile}`].activatedResources.requestCount;

    assert.throws(() => applyBrowserReference(baseline.routes, reference), /activatedResources/);
  });
});

describe('baseline schema', () => {
  it('rejects the committed limit blocks the derivation replaced', () => {
    const measurement = createSyntheticBuild(1_000);
    const withLimits = createArchitectureBaseline(measurement);
    withLimits.build.launchpad.limits = {};

    assert.throws(() => validateArchitectureBaseline(withLimits), /Unknown keys: limits/);

    for (const key of ['resourceLimits', 'activatedResourceLimits']) {
      const browser = createBrowserBaseline();
      browser.routes[0][key] = {};
      assert.throws(() => validateBrowserBaseline(browser, fixtures), new RegExp(`Unknown keys: ${key}`));
    }
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

  it('names the variable and path when the file is missing, and the path when it is not JSON', async () => {
    const empty = await mkdtemp(join(tmpdir(), 'perf-reference-'));
    await assert.rejects(
      load(empty),
      /WEBV2_PERF_REFERENCE_DIR is set to "\." but .*build-reference\.json could not be read/
    );

    const broken = await stage('{ not json');
    await assert.rejects(load(broken), /is not valid JSON/);
  });

  it('treats a file with no integer schemaVersion as a wiring fault, not drift', async () => {
    // An empty object is what a broken stage step produces; it was never written by any checkout.
    const empty = await stage({});
    await assert.rejects(load(empty), /performance reference build-reference\.json/);
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
