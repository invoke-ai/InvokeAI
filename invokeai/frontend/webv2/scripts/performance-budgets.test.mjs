import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { createChunkSourceManifest, getModuleSourceOwner } from './chunk-source-manifest.mjs';
import {
  BUILD_METRIC_KEYS,
  checkBrowserRouteBudget,
  checkRouteBudget,
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
      limits: Object.fromEntries(BUILD_METRIC_KEYS.map((key) => [key, measurement[key]])),
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
      activatedResourceLimits: {
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
      resourceLimits: {
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
    baseline.build.launchpad.limits.unusedBudget = 123;

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
