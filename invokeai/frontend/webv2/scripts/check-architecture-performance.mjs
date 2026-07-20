import assert from 'node:assert/strict';
import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import ts from 'typescript-legacy';

import { checkRouteBudget, measureRouteBuild } from './performance-budgets.mjs';
import { WIDGET_IMPLEMENTATION_PATTERN, WIDGET_SOURCES } from './widget-sources.mjs';

const root = resolve(import.meta.dirname, '..');
const baseline = JSON.parse(await readFile(resolve(root, 'performance/architecture-baseline.json'), 'utf8'));
const manifest = JSON.parse(await readFile(resolve(root, 'dist/.vite/manifest.json'), 'utf8'));
const readAsset = async (file) => new Uint8Array(await readFile(resolve(root, 'dist', file)));

const assetCache = new Map();
for (const chunk of Object.values(manifest)) {
  if (chunk.file && !assetCache.has(chunk.file)) {
    assetCache.set(chunk.file, await readAsset(chunk.file));
  }
}

const syntheticFailures = checkRouteBudget(
  {
    brotliBytes: 70,
    chunkNames: ['eager-gallery', 'entry'],
    files: ['entry.js', 'gallery.js'],
    gzipBytes: 80,
    initialRawBytes: 150,
    ownedRawBytes: 103,
    routeId: 'launchpad',
    source: 'entry.ts',
    sources: ['entry.ts', 'gallery.ts'],
  },
  {
    baselineOwnedRawBytes: 100,
    initialChunkNames: ['entry'],
    maxGrowthPercent: 0.02,
    maxGrowthRawBytes: 20,
    owner: 'gallery',
    remediationTicket: 'define-gallery-feature',
    source: 'entry.ts',
  }
);
assert.equal(syntheticFailures.length, 2, 'Synthetic byte and request-set regressions must both fail.');
assert.match(syntheticFailures.map((failure) => failure.message).join('\n'), /103 bytes/);
assert.match(syntheticFailures.map((failure) => failure.message).join('\n'), /eager-gallery/);

const measurements = Object.entries(baseline.build).map(([routeId, budget]) =>
  measureRouteBuild(manifest, routeId, budget.source, (file) => assetCache.get(file))
);
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
