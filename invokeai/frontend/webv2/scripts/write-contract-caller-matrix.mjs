import { mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, extname, join, relative, resolve } from 'node:path';

import { analyzeSource, closeSourceAnalysis, primeSourceAnalysis } from '../src/architecture/tsSourceAnalysis.ts';

const packageRoot = process.cwd();
const sourceRoot = resolve(packageRoot, 'src');
const normalize = (path) => path.replaceAll('\\', '/');
const isTypeScript = (path) => /\.[cm]?[jt]sx?$/.test(path);
const isTest = (path) => /(?:\.test|\.browser\.test|\.type-test)\.[^.]+$/.test(path);
const stripExtension = (path) => path.slice(0, -extname(path).length);
const paths = readdirSync(sourceRoot, { recursive: true, withFileTypes: true })
  .filter((entry) => entry.isFile())
  .map((entry) => normalize(relative(sourceRoot, join(entry.parentPath, entry.name))))
  .filter(isTypeScript)
  .sort();
const sources = new Map(paths.map((path) => [path, readFileSync(resolve(sourceRoot, path), 'utf8')]));
const pathByStem = new Map(paths.map((path) => [stripExtension(path), path]));
const aliases = [
  ['@app', 'app'],
  ['@features', 'features'],
  ['@platform', 'platform'],
  ['@theme', 'platform/ui/theme'],
  ['@workbench', 'workbench'],
  ['@', ''],
];
const hubs = [
  'features/queue/index.ts',
  'features/gallery/contracts.ts',
  'features/gallery/index.ts',
  'features/gallery/queries.ts',
  'features/gallery/react.ts',
  'features/gallery/utility.ts',
  'features/gallery/widget.ts',
  'features/identity/index.ts',
  'features/models/index.ts',
  'features/models/react.ts',
  'features/nodes/index.ts',
  'features/generation/components.ts',
  'features/generation/contracts.ts',
  'features/generation/graph.ts',
  'features/generation/index.ts',
  'features/generation/react.ts',
  'features/generation/settings.ts',
  'features/generation/widget.ts',
  'features/workflow/contracts.ts',
  'features/workflow/graph.ts',
  'features/workflow/index.ts',
  'features/workflow/preview.ts',
  'features/workflow/queries.ts',
  'features/workflow/react.ts',
  'features/workflow/utility.ts',
  'features/workflow/widget.ts',
  'features/upscale/index.ts',
  'features/upscale/widget.ts',
  'workbench/canvas-engine/contracts.ts',
  'workbench/canvas-engine/types.ts',
  'workbench/diagnostics/contracts.ts',
  'workbench/graphContracts.ts',
  'workbench/invocationContracts.ts',
  'workbench/layoutContracts.ts',
  'workbench/persistenceContracts.ts',
  'workbench/projectContracts.ts',
  'workbench/settings/contracts.ts',
  'workbench/widgetContracts.ts',
];

const normalizeSegments = (path) => {
  const parts = [];
  for (const part of path.split('/')) {
    if (!part || part === '.') {
      continue;
    }
    if (part === '..') {
      parts.pop();
    } else {
      parts.push(part);
    }
  }
  return parts.join('/');
};

const resolveImport = (sourcePath, specifier) => {
  let stem = null;
  if (specifier.startsWith('.')) {
    stem = normalizeSegments(`${sourcePath.slice(0, sourcePath.lastIndexOf('/'))}/${specifier}`);
  } else {
    for (const [alias, target] of aliases) {
      if (specifier === alias || specifier.startsWith(`${alias}/`)) {
        stem = normalizeSegments(`${target}${specifier.slice(alias.length)}`);
        break;
      }
    }
  }
  return stem ? (pathByStem.get(stem) ?? pathByStem.get(`${stem}/index`) ?? null) : null;
};

const callerRecords = new Map(hubs.map((hub) => [hub, new Map()]));
const allCallerRecords = new Map(hubs.map((hub) => [hub, new Set()]));
const addCaller = (hub, symbol, caller) => {
  const symbols = callerRecords.get(hub);
  const callers = symbols.get(symbol) ?? new Set();
  callers.add(caller);
  symbols.set(symbol, callers);
};

try {
  // Load the whole tree in one snapshot; parsing file by file costs a round trip each.
  primeSourceAnalysis(sources);

  for (const path of paths) {
    for (const reference of analyzeSource(path, sources.get(path)).moduleReferences) {
      if (reference.form !== 'import-declaration') {
        continue;
      }
      const target = resolveImport(path, reference.specifier);
      if (!target || !callerRecords.has(target)) {
        continue;
      }
      if (reference.symbols.length > 0) {
        allCallerRecords.get(target).add(path);
      }
      for (const symbol of reference.symbols) {
        if (symbol !== 'default' && symbol !== '*') {
          addCaller(target, symbol, path);
        }
      }
    }
  }

  const matrix = hubs.map((hub) => {
    const callersBySymbol = callerRecords.get(hub);
    const exportedSymbols = new Set(analyzeSource(hub, sources.get(hub)).publicExports);
    exportedSymbols.delete('default');
    exportedSymbols.delete('*');
    const symbols = [...exportedSymbols].sort().map((symbol) => {
      const callers = [...(callersBySymbol.get(symbol) ?? [])].sort();
      return {
        callers,
        productionCallers: callers.filter((path) => !isTest(path)),
        symbol,
      };
    });
    const allCallers = [...allCallerRecords.get(hub)].sort();
    return {
      allCallers,
      hub,
      productionCallerCount: allCallers.filter((path) => !isTest(path)).length,
      symbols,
    };
  });

  const artifactPath = resolve(packageRoot, 'artifacts/architecture/contract-caller-matrix.json');
  mkdirSync(dirname(artifactPath), { recursive: true });
  writeFileSync(artifactPath, `${JSON.stringify({ hubs: matrix }, null, 2)}\n`);
} finally {
  closeSourceAnalysis();
}
