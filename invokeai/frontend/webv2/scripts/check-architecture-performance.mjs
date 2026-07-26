import assert from 'node:assert/strict';
import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import ts from 'typescript-legacy';

import {
  BUILD_METRIC_KEYS,
  checkRouteBudget,
  measureRouteBuild,
  validateArchitectureBaseline,
  validateChunkSourceManifest,
} from './performance-budgets.mjs';
import { WIDGET_IMPLEMENTATION_PATTERN, WIDGET_SOURCES } from './widget-sources.mjs';

const root = resolve(import.meta.dirname, '..');
const baselinePath = resolve(root, 'performance/architecture-baseline.json');
const baselineInput = JSON.parse(await readFile(baselinePath, 'utf8'));
const manifest = JSON.parse(await readFile(resolve(root, 'dist/.vite/manifest.json'), 'utf8'));
const chunkSourceManifest = validateChunkSourceManifest(
  JSON.parse(await readFile(resolve(root, 'dist/.vite/chunk-sources.json'), 'utf8'))
);
const readAsset = async (file) => new Uint8Array(await readFile(resolve(root, 'dist', file)));
const updateBaseline = process.argv.includes('--update-baseline');

const assetCache = new Map();
for (const chunk of Object.values(manifest)) {
  for (const file of [chunk.file, ...(chunk.css ?? []), ...(chunk.assets ?? [])]) {
    if (file && !assetCache.has(file)) {
      assetCache.set(file, await readAsset(file));
    }
  }
}

const getLegacyOwnedLimit = (budget, measurement) => {
  if (
    typeof budget.baselineOwnedRawBytes !== 'number' ||
    typeof budget.maxGrowthPercent !== 'number' ||
    typeof budget.maxGrowthRawBytes !== 'number'
  ) {
    return measurement.ownedRawBytes;
  }

  return Math.max(
    measurement.ownedRawBytes,
    budget.baselineOwnedRawBytes +
      Math.min(budget.maxGrowthRawBytes, Math.floor(budget.baselineOwnedRawBytes * budget.maxGrowthPercent))
  );
};

const createLimits = (measurement, legacyBudget) =>
  Object.fromEntries(
    BUILD_METRIC_KEYS.map((key) => {
      if (key === 'ownedRawBytes') {
        return [key, getLegacyOwnedLimit(legacyBudget, measurement)];
      }
      if (key === 'requestCount' || key === 'scriptRequestCount') {
        return [key, measurement[key]];
      }
      return [key, Math.ceil(measurement[key] * 1.01)];
    })
  );

const routeEntries = Object.entries(baselineInput.build);
const measurements = routeEntries.map(([routeId, budget]) =>
  measureRouteBuild(manifest, chunkSourceManifest, routeId, budget.source, (file) => assetCache.get(file))
);

let baseline;
if (updateBaseline) {
  const build = Object.fromEntries(
    routeEntries.map(([routeId, previousBudget]) => {
      const measurement = measurements.find((candidate) => candidate.routeId === routeId);
      return [
        routeId,
        {
          baseline: {
            ...Object.fromEntries(BUILD_METRIC_KEYS.map((key) => [key, measurement[key]])),
            sourceOwners: measurement.sourceOwners,
          },
          limits: createLimits(measurement, previousBudget),
          owner: previousBudget.owner,
          remediationTicket: previousBudget.remediationTicket,
          source: previousBudget.source,
        },
      ];
    })
  );
  baseline = {
    build,
    capturedAt: new Date().toISOString().slice(0, 10),
    developmentInvalidation: baselineInput.developmentInvalidation,
    schemaVersion: 2,
    structural: {
      ...baselineInput.structural,
      editorForbiddenInitialChunkNames: [...baselineInput.structural.editorForbiddenInitialChunkNames].sort(),
      launchpadForbiddenInitialSources: [...baselineInput.structural.launchpadForbiddenInitialSources].sort(),
    },
  };
  validateArchitectureBaseline(baseline);
  await writeFile(baselinePath, `${JSON.stringify(baseline, null, 2)}\n`);
} else {
  baseline = validateArchitectureBaseline(baselineInput);
}

const failures = measurements.flatMap((measurement) =>
  checkRouteBudget(measurement, baseline.build[measurement.routeId])
);
const launchpad = measurements.find((measurement) => measurement.routeId === 'launchpad');
const editor = measurements.find((measurement) => measurement.routeId === 'editor');
const widgetImplementationSources = [...WIDGET_SOURCES.keys()];
for (const source of widgetImplementationSources) {
  assert.ok(manifest[source], `Registered widget implementation ${source} is missing from the build manifest.`);
}
for (const source of Object.keys(manifest)) {
  if (WIDGET_IMPLEMENTATION_PATTERN.test(source)) {
    assert.ok(
      WIDGET_SOURCES.has(source),
      `Widget implementation ${source} must be registered in scripts/widget-sources.mjs.`
    );
  }
}
const getInactiveWidgetFailures = (sources) =>
  widgetImplementationSources
    .filter((source) => sources.includes(source))
    .map((source) => ({
      message: `editor eagerly includes inactive widget implementation ${source}.`,
      owner: 'workbench',
      remediationTicket: 'deepen-widget-registry-loading',
      routeId: 'editor',
    }));

assert.equal(
  new Set(widgetImplementationSources.map((source) => manifest[source].file)).size,
  widgetImplementationSources.length,
  'Every first-party widget implementation must have an independently identifiable chunk.'
);
assert.equal(
  getInactiveWidgetFailures(['src/features/gallery/widget.ts']).length,
  1,
  'A synthetic eager widget implementation must fail the structural gate.'
);
failures.push(...getInactiveWidgetFailures(editor.sources));
for (const forbidden of baseline.structural.launchpadForbiddenInitialSources) {
  if (launchpad.sources.some((source) => source === forbidden || source.startsWith(forbidden))) {
    failures.push({
      message: `launchpad eagerly includes ${forbidden}.`,
      owner: 'app',
      remediationTicket: 'set-performance-budgets',
      routeId: 'launchpad',
    });
  }
}
for (const forbidden of baseline.structural.editorForbiddenInitialChunkNames) {
  if (editor.chunkNames.includes(forbidden)) {
    failures.push({
      message: `editor eagerly includes ${forbidden}.`,
      owner: 'workbench',
      remediationTicket: 'deepen-widget-registry-loading',
      routeId: 'editor',
    });
  }
}

const collectFiles = async (directory) => {
  const collected = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = resolve(directory, entry.name);
    if (entry.isDirectory()) {
      collected.push(...(await collectFiles(path)));
    } else if (/\.(?:ts|tsx)$/.test(entry.name) && !/\.test\.|\.type-test\.|\.testing\./.test(entry.name)) {
      collected.push(path);
    }
  }
  return collected;
};
const productionFiles = await collectFiles(resolve(root, 'src'));
const importerCounts = new Map(Object.values(baseline.developmentInvalidation).map((budget) => [budget.specifier, 0]));
for (const path of productionFiles) {
  const source = await readFile(path, 'utf8');
  const sourceFile = ts.createSourceFile(path, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  const seen = new Set();
  const visit = (node) => {
    if (
      (ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) &&
      node.moduleSpecifier &&
      ts.isStringLiteralLike(node.moduleSpecifier)
    ) {
      seen.add(node.moduleSpecifier.text);
    }
    ts.forEachChild(node, visit);
  };
  visit(sourceFile);
  for (const specifier of seen) {
    if (importerCounts.has(specifier)) {
      importerCounts.set(specifier, importerCounts.get(specifier) + 1);
    }
  }
}
for (const [metricId, budget] of Object.entries(baseline.developmentInvalidation)) {
  const actual = importerCounts.get(budget.specifier) ?? 0;
  if (actual > budget.maxDirectImporters) {
    failures.push({
      message: `${metricId} has ${actual} direct importers (budget ${budget.maxDirectImporters}).`,
      owner: budget.owner,
      remediationTicket: budget.remediationTicket,
      routeId: 'development-invalidation',
    });
  }
}

const reportPath = resolve(root, 'artifacts/architecture-performance/build-report.json');
await mkdir(dirname(reportPath), { recursive: true });
await writeFile(
  reportPath,
  `${JSON.stringify({ baselineCapturedAt: baseline.capturedAt, failures, importerCounts: Object.fromEntries(importerCounts), measurements }, null, 2)}\n`
);

if (failures.length > 0) {
  throw new Error(
    failures
      .map(
        (failure) =>
          `${failure.message} Owner: ${failure.owner}. Remediation: ${failure.remediationTicket}. Route: ${failure.routeId}.`
      )
      .join('\n')
  );
}

process.stdout.write(
  `${JSON.stringify({ importerCounts: Object.fromEntries(importerCounts), measurements }, null, 2)}\n`
);
