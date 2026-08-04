import { mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, extname, join, relative, resolve } from 'node:path';
import {
  isExportDeclaration,
  isIdentifier,
  isImportDeclaration,
  isNamedExports,
  isNamedImports,
  isStringLiteralLikeNode,
  isVariableStatement,
  ModifierFlags,
} from 'typescript/unstable/ast';

import { parseSource, primeSources } from './parse-source.mjs';

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
// Load the whole tree in one snapshot; parsing file by file costs a round trip each.
primeSources(sources);
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

const parse = (path) => parseSource(path, sources.get(path));

const exportedSymbols = (path) => {
  const symbols = new Set();
  for (const statement of parse(path).statements) {
    if ((statement.modifierFlags & ModifierFlags.Export) !== 0) {
      if (statement.name && isIdentifier(statement.name)) {
        symbols.add(statement.name.text);
      }
      if (isVariableStatement(statement)) {
        for (const declaration of statement.declarationList.declarations) {
          if (isIdentifier(declaration.name)) {
            symbols.add(declaration.name.text);
          }
        }
      }
    }
    if (isExportDeclaration(statement) && statement.exportClause && isNamedExports(statement.exportClause)) {
      for (const element of statement.exportClause.elements) {
        symbols.add(element.name.text);
      }
    }
  }
  return [...symbols].sort();
};

const callerRecords = new Map(hubs.map((hub) => [hub, new Map()]));
const addCaller = (hub, symbol, caller) => {
  const symbols = callerRecords.get(hub);
  const callers = symbols.get(symbol) ?? new Set();
  callers.add(caller);
  symbols.set(symbol, callers);
};

for (const path of paths) {
  for (const statement of parse(path).statements) {
    if (
      !isImportDeclaration(statement) ||
      !statement.importClause ||
      !isStringLiteralLikeNode(statement.moduleSpecifier)
    ) {
      continue;
    }
    const target = resolveImport(path, statement.moduleSpecifier.text);
    if (!target || !callerRecords.has(target)) {
      continue;
    }
    const bindings = statement.importClause.namedBindings;
    if (bindings && isNamedImports(bindings)) {
      for (const element of bindings.elements) {
        addCaller(target, (element.propertyName ?? element.name).text, path);
      }
    } else {
      addCaller(target, '*', path);
    }
  }
}

const matrix = hubs.map((hub) => {
  const callersBySymbol = callerRecords.get(hub);
  const symbols = exportedSymbols(hub).map((symbol) => {
    const callers = [...(callersBySymbol.get(symbol) ?? [])].sort();
    return {
      callers,
      productionCallers: callers.filter((path) => !isTest(path)),
      symbol,
    };
  });
  const allCallers = [...new Set([...callersBySymbol.values()].flatMap((callers) => [...callers]))].sort();
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
