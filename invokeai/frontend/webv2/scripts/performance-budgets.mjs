import { brotliCompressSync, constants, gzipSync } from 'node:zlib';

export const BUILD_METRIC_KEYS = [
  'brotliBytes',
  'cssRawBytes',
  'fontRawBytes',
  'gzipBytes',
  'imageRawBytes',
  'initialRawBytes',
  'largestAssetRawBytes',
  'otherAssetRawBytes',
  'ownedRawBytes',
  'requestCount',
  'scriptRequestCount',
];

export const BROWSER_RESOURCE_METRIC_KEYS = [
  'cssRawBytes',
  'fontRawBytes',
  'imageRawBytes',
  'largestAssetRawBytes',
  'otherRawBytes',
  'requestCount',
  'scriptRawBytes',
  'scriptRequestCount',
  'totalRawBytes',
];

const BUILD_SCHEMA_VERSION = 2;
const BROWSER_SCHEMA_VERSION = 2;
const CHUNK_SOURCE_SCHEMA_VERSION = 1;

export const waitForRequiredRequests = async ({
  context,
  getRequested,
  pollIntervalMs = 25,
  requiredRequests,
  timeoutMs = 10_000,
}) => {
  const required = new Set(requiredRequests);
  const deadline = Date.now() + timeoutMs;

  while (Date.now() < deadline) {
    const requested = new Set(getRequested());
    if ([...required].every((request) => requested.has(request))) {
      return;
    }
    await new Promise((resolveWait) => {
      setTimeout(resolveWait, pollIntervalMs);
    });
  }

  throw new Error(`${context} did not request required widgets ${JSON.stringify([...required])}.`);
};

export const waitForStableRequests = async ({
  context,
  getRequested,
  now = Date.now,
  pollIntervalMs = 25,
  stableForMs = 500,
  timeoutMs = 10_000,
  wait = (milliseconds) =>
    new Promise((resolveWait) => {
      setTimeout(resolveWait, milliseconds);
    }),
}) => {
  const deadline = now() + timeoutMs;
  let fingerprint;
  let stableSince = now();

  while (now() < deadline) {
    const currentFingerprint = JSON.stringify([...getRequested()].sort());
    if (currentFingerprint !== fingerprint) {
      fingerprint = currentFingerprint;
      stableSince = now();
    }
    if (now() - stableSince >= stableForMs) {
      return;
    }
    await wait(pollIntervalMs);
  }

  throw new Error(`${context} request set did not remain stable for ${String(stableForMs)}ms.`);
};

const isPlainObject = (value) => typeof value === 'object' && value !== null && !Array.isArray(value);

const assertPlainObject = (value, path) => {
  if (!isPlainObject(value)) {
    throw new TypeError(`${path} must be an object.`);
  }
};

const assertExactKeys = (value, expectedKeys, path) => {
  assertPlainObject(value, path);
  const expected = new Set(expectedKeys);
  const actualKeys = Object.keys(value);
  const unknown = actualKeys.filter((key) => !expected.has(key));
  const missing = expectedKeys.filter((key) => !(key in value));

  if (unknown.length > 0 || missing.length > 0) {
    throw new Error(
      `${path} has an invalid schema.${unknown.length > 0 ? ` Unknown keys: ${unknown.join(', ')}.` : ''}${
        missing.length > 0 ? ` Missing keys: ${missing.join(', ')}.` : ''
      }`
    );
  }
};

const assertNonEmptyString = (value, path) => {
  if (typeof value !== 'string' || value.trim().length === 0) {
    throw new TypeError(`${path} must be a non-empty string.`);
  }
};

const assertNonNegativeNumber = (value, path) => {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    throw new TypeError(`${path} must be a finite non-negative number.`);
  }
};

const assertStringArray = (value, path) => {
  if (!Array.isArray(value) || value.some((item) => typeof item !== 'string')) {
    throw new TypeError(`${path} must be an array of strings.`);
  }
  if (new Set(value).size !== value.length) {
    throw new Error(`${path} must not contain duplicate values.`);
  }
  if (JSON.stringify(value) !== JSON.stringify([...value].sort())) {
    throw new Error(`${path} must be sorted.`);
  }
};

const validateMetricObject = (value, metricKeys, path) => {
  assertExactKeys(value, metricKeys, path);
  for (const key of metricKeys) {
    assertNonNegativeNumber(value[key], `${path}.${key}`);
  }
};

export const validateChunkSourceManifest = (value) => {
  assertExactKeys(value, ['chunks', 'schemaVersion'], 'chunk source manifest');
  if (value.schemaVersion !== CHUNK_SOURCE_SCHEMA_VERSION) {
    throw new Error(
      `chunk source manifest schemaVersion must be ${CHUNK_SOURCE_SCHEMA_VERSION}; received ${String(value.schemaVersion)}.`
    );
  }
  assertPlainObject(value.chunks, 'chunk source manifest.chunks');
  for (const [file, chunk] of Object.entries(value.chunks)) {
    assertExactKeys(chunk, ['facadeSource', 'sourceOwners'], `chunk source manifest.chunks.${file}`);
    if (chunk.facadeSource !== null) {
      assertNonEmptyString(chunk.facadeSource, `chunk source manifest.chunks.${file}.facadeSource`);
    }
    assertStringArray(chunk.sourceOwners, `chunk source manifest.chunks.${file}.sourceOwners`);
  }
  return value;
};

export const validateArchitectureBaseline = (value) => {
  assertExactKeys(
    value,
    ['build', 'capturedAt', 'developmentInvalidation', 'schemaVersion', 'structural'],
    'architecture baseline'
  );
  if (value.schemaVersion !== BUILD_SCHEMA_VERSION) {
    throw new Error(
      `architecture baseline schemaVersion must be ${BUILD_SCHEMA_VERSION}; received ${String(value.schemaVersion)}.`
    );
  }
  assertNonEmptyString(value.capturedAt, 'architecture baseline.capturedAt');
  assertPlainObject(value.build, 'architecture baseline.build');

  for (const [routeId, budget] of Object.entries(value.build)) {
    const path = `architecture baseline.build.${routeId}`;
    assertExactKeys(budget, ['baseline', 'limits', 'owner', 'remediationTicket', 'source'], path);
    assertNonEmptyString(budget.source, `${path}.source`);
    assertNonEmptyString(budget.owner, `${path}.owner`);
    assertNonEmptyString(budget.remediationTicket, `${path}.remediationTicket`);
    assertExactKeys(budget.baseline, [...BUILD_METRIC_KEYS, 'sourceOwners'], `${path}.baseline`);
    validateMetricObject(
      Object.fromEntries(BUILD_METRIC_KEYS.map((key) => [key, budget.baseline[key]])),
      BUILD_METRIC_KEYS,
      `${path}.baseline metrics`
    );
    assertStringArray(budget.baseline.sourceOwners, `${path}.baseline.sourceOwners`);
    validateMetricObject(budget.limits, BUILD_METRIC_KEYS, `${path}.limits`);
    for (const key of BUILD_METRIC_KEYS) {
      if (budget.limits[key] < budget.baseline[key]) {
        throw new Error(`${path}.limits.${key} cannot be below its captured baseline.`);
      }
    }
  }

  assertExactKeys(
    value.structural,
    ['editorForbiddenInitialChunkNames', 'inactiveWidgetGateOwner', 'launchpadForbiddenInitialSources'],
    'architecture baseline.structural'
  );
  assertStringArray(
    value.structural.editorForbiddenInitialChunkNames,
    'architecture baseline.structural.editorForbiddenInitialChunkNames'
  );
  assertNonEmptyString(
    value.structural.inactiveWidgetGateOwner,
    'architecture baseline.structural.inactiveWidgetGateOwner'
  );
  assertStringArray(
    value.structural.launchpadForbiddenInitialSources,
    'architecture baseline.structural.launchpadForbiddenInitialSources'
  );

  assertPlainObject(value.developmentInvalidation, 'architecture baseline.developmentInvalidation');
  for (const [metricId, budget] of Object.entries(value.developmentInvalidation)) {
    const path = `architecture baseline.developmentInvalidation.${metricId}`;
    assertExactKeys(budget, ['maxDirectImporters', 'owner', 'remediationTicket', 'specifier'], path);
    assertNonEmptyString(budget.specifier, `${path}.specifier`);
    assertNonEmptyString(budget.owner, `${path}.owner`);
    assertNonEmptyString(budget.remediationTicket, `${path}.remediationTicket`);
    assertNonNegativeNumber(budget.maxDirectImporters, `${path}.maxDirectImporters`);
  }

  return value;
};

const validateBrowserRoute = (route, path) => {
  assertExactKeys(
    route,
    [
      'domContentLoadedMedianMs',
      'activatedResourceBaseline',
      'activatedResourceLimits',
      'id',
      'layoutAckMedianMs',
      'layoutReturnSwitchMedianMs',
      'layoutSwitchMedianMs',
      'loadMedianMs',
      'longestTaskMaxMs',
      'owner',
      'projectSwitchMedianMs',
      'readyMark',
      'remediationTicket',
      'resourceBaseline',
      'resourceLimits',
      'routeReadyMedianMs',
      'scriptSourceOwnerSet',
      'stateProfile',
    ],
    path
  );
  for (const key of [
    'domContentLoadedMedianMs',
    'layoutAckMedianMs',
    'layoutReturnSwitchMedianMs',
    'layoutSwitchMedianMs',
    'loadMedianMs',
    'longestTaskMaxMs',
    'projectSwitchMedianMs',
    'routeReadyMedianMs',
  ]) {
    assertNonNegativeNumber(route[key], `${path}.${key}`);
  }
  for (const key of ['id', 'owner', 'readyMark', 'remediationTicket', 'stateProfile']) {
    assertNonEmptyString(route[key], `${path}.${key}`);
  }
  validateMetricObject(
    route.activatedResourceBaseline,
    BROWSER_RESOURCE_METRIC_KEYS,
    `${path}.activatedResourceBaseline`
  );
  validateMetricObject(route.activatedResourceLimits, BROWSER_RESOURCE_METRIC_KEYS, `${path}.activatedResourceLimits`);
  for (const key of BROWSER_RESOURCE_METRIC_KEYS) {
    if (route.activatedResourceLimits[key] < route.activatedResourceBaseline[key]) {
      throw new Error(`${path}.activatedResourceLimits.${key} cannot be below its captured baseline.`);
    }
  }
  validateMetricObject(route.resourceBaseline, BROWSER_RESOURCE_METRIC_KEYS, `${path}.resourceBaseline`);
  validateMetricObject(route.resourceLimits, BROWSER_RESOURCE_METRIC_KEYS, `${path}.resourceLimits`);
  for (const key of BROWSER_RESOURCE_METRIC_KEYS) {
    if (route.resourceLimits[key] < route.resourceBaseline[key]) {
      throw new Error(`${path}.resourceLimits.${key} cannot be below its captured baseline.`);
    }
  }
  assertNonEmptyString(route.scriptSourceOwnerSet, `${path}.scriptSourceOwnerSet`);
};

export const assertTimingEnforcementIsConfigured = (baseline, fixtures) => {
  if (!baseline.timingPolicy.enforce) {
    return;
  }

  const { runner } = baseline.timingPolicy;
  if (!runner.stable || runner.id === 'unconfigured') {
    throw new Error('Timing enforcement requires an explicitly identified stable runner.');
  }
  if (runner.observedStableRuns < runner.minimumStableRuns) {
    throw new Error(
      `Timing enforcement requires at least ${String(runner.minimumStableRuns)} stable runs; received ${String(
        runner.observedStableRuns
      )}.`
    );
  }
  for (const fixture of fixtures.routes) {
    assertNonEmptyString(fixture.readyMark, `architecture fixtures.routes.${fixture.id}.readyMark`);
    const route = baseline.routes.find(
      (candidate) => candidate.id === fixture.id && candidate.stateProfile === fixture.stateProfile
    );
    if (!route || route.readyMark !== fixture.readyMark) {
      throw new Error(
        `Timing enforcement requires ${fixture.id}/${fixture.stateProfile} to have a matching semantic-ready mark.`
      );
    }
  }
};

export const validateBrowserBaseline = (value, fixtures) => {
  assertExactKeys(
    value,
    ['browserExecutable', 'capturedAt', 'routes', 'sampling', 'schemaVersion', 'scriptSourceOwnerSets', 'timingPolicy'],
    'browser baseline'
  );
  if (value.schemaVersion !== BROWSER_SCHEMA_VERSION) {
    throw new Error(
      `browser baseline schemaVersion must be ${BROWSER_SCHEMA_VERSION}; received ${String(value.schemaVersion)}.`
    );
  }
  assertNonEmptyString(value.browserExecutable, 'browser baseline.browserExecutable');
  assertNonEmptyString(value.capturedAt, 'browser baseline.capturedAt');
  assertExactKeys(value.sampling, ['scoredSamples', 'traceSamples', 'warmups'], 'browser baseline.sampling');
  assertNonNegativeNumber(value.sampling.scoredSamples, 'browser baseline.sampling.scoredSamples');
  assertNonNegativeNumber(value.sampling.traceSamples, 'browser baseline.sampling.traceSamples');
  assertNonNegativeNumber(value.sampling.warmups, 'browser baseline.sampling.warmups');
  if (value.sampling.scoredSamples < 1 || value.sampling.traceSamples !== 1) {
    throw new Error('browser baseline must configure at least one scored sample and exactly one trace sample.');
  }

  assertExactKeys(
    value.timingPolicy,
    ['enforce', 'longTaskTargetMs', 'runner', 'tolerancePercent'],
    'browser baseline.timingPolicy'
  );
  if (typeof value.timingPolicy.enforce !== 'boolean') {
    throw new TypeError('browser baseline.timingPolicy.enforce must be boolean.');
  }
  assertNonNegativeNumber(value.timingPolicy.longTaskTargetMs, 'browser baseline.timingPolicy.longTaskTargetMs');
  assertNonNegativeNumber(value.timingPolicy.tolerancePercent, 'browser baseline.timingPolicy.tolerancePercent');
  assertExactKeys(
    value.timingPolicy.runner,
    ['id', 'minimumStableRuns', 'observedStableRuns', 'stable'],
    'browser baseline.timingPolicy.runner'
  );
  assertNonEmptyString(value.timingPolicy.runner.id, 'browser baseline.timingPolicy.runner.id');
  if (typeof value.timingPolicy.runner.stable !== 'boolean') {
    throw new TypeError('browser baseline.timingPolicy.runner.stable must be boolean.');
  }
  assertNonNegativeNumber(
    value.timingPolicy.runner.minimumStableRuns,
    'browser baseline.timingPolicy.runner.minimumStableRuns'
  );
  assertNonNegativeNumber(
    value.timingPolicy.runner.observedStableRuns,
    'browser baseline.timingPolicy.runner.observedStableRuns'
  );

  if (!Array.isArray(value.routes)) {
    throw new TypeError('browser baseline.routes must be an array.');
  }
  value.routes.forEach((route, index) => validateBrowserRoute(route, `browser baseline.routes[${String(index)}]`));
  assertPlainObject(value.scriptSourceOwnerSets, 'browser baseline.scriptSourceOwnerSets');
  for (const [setId, sourceOwners] of Object.entries(value.scriptSourceOwnerSets)) {
    assertStringArray(sourceOwners, `browser baseline.scriptSourceOwnerSets.${setId}`);
  }
  for (const route of value.routes) {
    if (!value.scriptSourceOwnerSets[route.scriptSourceOwnerSet]) {
      throw new Error(
        `browser baseline route ${route.id}/${route.stateProfile} references missing script source-owner set ${route.scriptSourceOwnerSet}.`
      );
    }
  }
  assertTimingEnforcementIsConfigured(value, fixtures);
  return value;
};

const collectStaticImports = (manifest, source, collected = new Set()) => {
  if (collected.has(source)) {
    return collected;
  }
  const chunk = manifest[source];
  if (!chunk) {
    throw new Error(`Build manifest is missing route source "${source}".`);
  }
  collected.add(source);
  for (const imported of chunk.imports ?? []) {
    collectStaticImports(manifest, imported, collected);
  }
  return collected;
};

const getChunkName = (source, chunk) => chunk.name ?? chunk.src ?? source;

const getAssetKind = (file) => {
  const extension = file.split('.').pop()?.toLowerCase();

  if (extension === 'js' || extension === 'mjs') {
    return 'script';
  }
  if (extension === 'css') {
    return 'css';
  }
  if (extension === 'woff' || extension === 'woff2' || extension === 'ttf' || extension === 'otf') {
    return 'font';
  }
  if (
    extension === 'avif' ||
    extension === 'gif' ||
    extension === 'jpeg' ||
    extension === 'jpg' ||
    extension === 'png' ||
    extension === 'svg' ||
    extension === 'webp'
  ) {
    return 'image';
  }
  return 'other';
};

const getCompressedBytes = (asset, algorithm) => {
  if (asset.kind === 'font' || asset.kind === 'image') {
    return asset.bytes.byteLength;
  }
  return algorithm === 'gzip'
    ? gzipSync(asset.bytes, { level: 9 }).byteLength
    : brotliCompressSync(asset.bytes, {
        params: { [constants.BROTLI_PARAM_QUALITY]: 4 },
      }).byteLength;
};

const sumBytesByKind = (assets, kind) =>
  assets.filter((asset) => asset.kind === kind).reduce((total, asset) => total + asset.bytes.byteLength, 0);

export const measureRouteBuild = (manifest, chunkSourceManifest, routeId, source, readAsset) => {
  validateChunkSourceManifest(chunkSourceManifest);
  const manifestSources = [...collectStaticImports(manifest, source)];
  const files = new Map();

  for (const manifestSource of manifestSources) {
    const chunk = manifest[manifestSource];
    files.set(chunk.file, { file: chunk.file, kind: getAssetKind(chunk.file) });
    for (const cssFile of chunk.css ?? []) {
      files.set(cssFile, { file: cssFile, kind: 'css' });
    }
    for (const assetFile of chunk.assets ?? []) {
      files.set(assetFile, { file: assetFile, kind: getAssetKind(assetFile) });
    }
  }

  const assets = [...files.values()].map((asset) => ({
    ...asset,
    bytes: readAsset(asset.file),
  }));
  for (const asset of assets) {
    if (!(asset.bytes instanceof Uint8Array)) {
      throw new TypeError(`Build asset ${asset.file} was not available as bytes.`);
    }
  }

  const ownedFile = manifest[source].file;
  const owned = assets.find((asset) => asset.file === ownedFile);
  if (!owned) {
    throw new Error(`${routeId} owned entry ${ownedFile} was not measured.`);
  }

  const sourceOwners = new Set();
  for (const manifestSource of manifestSources) {
    const file = manifest[manifestSource].file;
    const chunkSources = chunkSourceManifest.chunks[file];
    if (!chunkSources) {
      throw new Error(`Chunk source manifest is missing ${file}; source ownership cannot be checked for ${routeId}.`);
    }
    for (const sourceOwner of chunkSources.sourceOwners) {
      sourceOwners.add(sourceOwner);
    }
  }

  const initialRawBytes = assets.reduce((total, asset) => total + asset.bytes.byteLength, 0);

  return {
    brotliBytes: assets.reduce((total, asset) => total + getCompressedBytes(asset, 'brotli'), 0),
    chunkNames: manifestSources.map((manifestSource) => getChunkName(manifestSource, manifest[manifestSource])).sort(),
    cssRawBytes: sumBytesByKind(assets, 'css'),
    files: assets.map((asset) => asset.file).sort(),
    fontRawBytes: sumBytesByKind(assets, 'font'),
    gzipBytes: assets.reduce((total, asset) => total + getCompressedBytes(asset, 'gzip'), 0),
    imageRawBytes: sumBytesByKind(assets, 'image'),
    initialRawBytes,
    largestAssetRawBytes: Math.max(0, ...assets.map((asset) => asset.bytes.byteLength)),
    otherAssetRawBytes: sumBytesByKind(assets, 'other'),
    ownedRawBytes: owned.bytes.byteLength,
    requestCount: assets.length,
    routeId,
    scriptRequestCount: assets.filter((asset) => asset.kind === 'script').length,
    source,
    sourceOwners: [...sourceOwners].sort(),
    sources: manifestSources.sort(),
  };
};

const METRIC_LABELS = {
  brotliBytes: 'estimated Brotli transfer',
  cssRawBytes: 'initial CSS',
  fontRawBytes: 'initial fonts',
  gzipBytes: 'estimated gzip transfer',
  imageRawBytes: 'initial images',
  initialRawBytes: 'initial raw assets',
  largestAssetRawBytes: 'largest initial asset',
  otherAssetRawBytes: 'other initial assets',
  ownedRawBytes: 'owned JavaScript',
  requestCount: 'initial asset requests',
  scriptRequestCount: 'initial script requests',
};

const createFailure = (measurement, budget, message, actual, limit) => ({
  actual,
  budget: limit,
  message,
  owner: budget.owner,
  remediationTicket: budget.remediationTicket,
  routeId: measurement.routeId,
});

const diffSortedValues = (expected, actual) => ({
  added: actual.filter((value) => !expected.includes(value)),
  removed: expected.filter((value) => !actual.includes(value)),
});

export const checkRouteBudget = (measurement, budget) => {
  const failures = [];

  for (const key of BUILD_METRIC_KEYS) {
    if (measurement[key] > budget.limits[key]) {
      failures.push(
        createFailure(
          measurement,
          budget,
          `${measurement.routeId} ${METRIC_LABELS[key]} reached ${String(measurement[key])} bytes/requests (limit ${String(
            budget.limits[key]
          )}, captured ${String(budget.baseline[key])}).`,
          measurement[key],
          budget.limits[key]
        )
      );
    }
  }

  const sourceDiff = diffSortedValues(budget.baseline.sourceOwners, measurement.sourceOwners);
  if (sourceDiff.added.length > 0 || sourceDiff.removed.length > 0) {
    failures.push(
      createFailure(
        measurement,
        budget,
        `${measurement.routeId} initial source-owner graph changed.${
          sourceDiff.added.length > 0 ? ` Added: ${sourceDiff.added.join(', ')}.` : ''
        }${sourceDiff.removed.length > 0 ? ` Removed: ${sourceDiff.removed.join(', ')}.` : ''}`,
        measurement.sourceOwners,
        budget.baseline.sourceOwners
      )
    );
  }

  return failures;
};

export const createBrowserSamplePlan = ({ scoredSamples, traceSamples, warmups }) => {
  if (!Number.isInteger(scoredSamples) || scoredSamples < 1) {
    throw new Error('scoredSamples must be a positive integer.');
  }
  if (traceSamples !== 1) {
    throw new Error('Exactly one unscored trace sample is required.');
  }
  if (!Number.isInteger(warmups) || warmups < 0) {
    throw new Error('warmups must be a non-negative integer.');
  }

  return [
    ...Array.from({ length: warmups }, (_, index) => ({ index, kind: 'warmup' })),
    { index: warmups, kind: 'trace' },
    ...Array.from({ length: scoredSamples }, (_, index) => ({
      index: warmups + traceSamples + index,
      kind: 'scored',
    })),
  ];
};

export const summarizeBrowserResources = (resources) => {
  const uniqueResources = [...new Map(resources.map((resource) => [resource.path, resource])).values()];
  const sum = (kind) =>
    uniqueResources
      .filter((resource) => resource.kind === kind)
      .reduce((total, resource) => total + resource.rawBytes, 0);

  return {
    cssRawBytes: sum('css'),
    fontRawBytes: sum('font'),
    imageRawBytes: sum('image'),
    largestAssetRawBytes: Math.max(0, ...uniqueResources.map((resource) => resource.rawBytes)),
    otherRawBytes: sum('other'),
    requestCount: uniqueResources.length,
    scriptRawBytes: sum('script'),
    scriptRequestCount: uniqueResources.filter((resource) => resource.kind === 'script').length,
    totalRawBytes: uniqueResources.reduce((total, resource) => total + resource.rawBytes, 0),
  };
};

export const checkBrowserRouteBudget = (route, expected, timingPolicy, expectedScriptSourceOwners) => {
  const failures = [];
  const label = `${route.id}/${route.stateProfile}`;

  for (const key of BROWSER_RESOURCE_METRIC_KEYS) {
    if (route.resources[key] > expected.resourceLimits[key]) {
      failures.push(
        `${label} ${key} reached ${String(route.resources[key])} (limit ${String(expected.resourceLimits[key])}, owner ${
          route.owner
        }, remediation ${route.remediationTicket}).`
      );
    }
    if (route.activatedResources[key] > expected.activatedResourceLimits[key]) {
      failures.push(
        `${label} activated ${key} reached ${String(route.activatedResources[key])} (limit ${String(
          expected.activatedResourceLimits[key]
        )}, owner ${route.owner}, remediation ${route.remediationTicket}).`
      );
    }
  }

  const sourceDiff = diffSortedValues(expectedScriptSourceOwners, route.scriptSourceOwners);
  if (sourceDiff.added.length > 0 || sourceDiff.removed.length > 0) {
    failures.push(
      `${label} script source-owner set changed.${sourceDiff.added.length > 0 ? ` Added: ${sourceDiff.added.join(', ')}.` : ''}${
        sourceDiff.removed.length > 0 ? ` Removed: ${sourceDiff.removed.join(', ')}.` : ''
      }`
    );
  }

  if (!timingPolicy.enforce) {
    return failures;
  }

  const timingLimit = (value) => value * (1 + timingPolicy.tolerancePercent);
  const timingChecks = [
    ['domContentLoadedMedianMs', 'DOMContentLoaded median'],
    ['layoutAckMedianMs', 'layout-ack median'],
    ['layoutReturnSwitchMedianMs', 'layout-return-switch median'],
    ['layoutSwitchMedianMs', 'layout-switch median'],
    ['loadMedianMs', 'load median'],
    ['projectSwitchMedianMs', 'project-switch median'],
    ['routeReadyMedianMs', 'semantic route-ready median'],
  ];
  for (const [key, metricLabel] of timingChecks) {
    if (expected[key] > 0 && route[key] > timingLimit(expected[key])) {
      failures.push(
        `${label} ${metricLabel} exceeded its timing tolerance (owner ${route.owner}, remediation ${route.remediationTicket}).`
      );
    }
  }
  if (route.longestTaskMaxMs > Math.max(timingPolicy.longTaskTargetMs, timingLimit(expected.longestTaskMaxMs))) {
    failures.push(
      `${label} longest task exceeded its timing tolerance (owner ${route.owner}, remediation ${route.remediationTicket}).`
    );
  }

  return failures;
};
