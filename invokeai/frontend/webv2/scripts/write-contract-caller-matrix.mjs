import { mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, extname, join, relative, resolve } from 'node:path';
import ts from 'typescript';

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
  'features/generation/batch.ts',
  'features/generation/canvas.ts',
  'features/generation/contracts.ts',
  'features/generation/draft-values.ts',
  'features/generation/drafts.ts',
  'features/generation/form.ts',
  'features/generation/graph.ts',
  'features/generation/hotkeys.ts',
  'features/generation/index.ts',
  'features/generation/prompt.ts',
  'features/generation/prompt-ui.ts',
  'features/generation/react.ts',
  'features/generation/reference-ui.ts',
  'features/generation/settings.ts',
  'features/generation/settings-ui.ts',
  'features/generation/utility.ts',
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

const parse = (path) =>
  ts.createSourceFile(
    path,
    sources.get(path),
    ts.ScriptTarget.Latest,
    true,
    path.endsWith('x') ? ts.ScriptKind.TSX : ts.ScriptKind.TS
  );

const exportedSymbols = (path) => {
  const symbols = new Set();
  for (const statement of parse(path).statements) {
    const modifiers = ts.canHaveModifiers(statement) ? ts.getModifiers(statement) : undefined;
    if (modifiers?.some((modifier) => modifier.kind === ts.SyntaxKind.ExportKeyword)) {
      if (statement.name && ts.isIdentifier(statement.name)) {
        symbols.add(statement.name.text);
      }
      if (ts.isVariableStatement(statement)) {
        for (const declaration of statement.declarationList.declarations) {
          if (ts.isIdentifier(declaration.name)) {
            symbols.add(declaration.name.text);
          }
        }
      }
    }
    if (ts.isExportDeclaration(statement) && statement.exportClause && ts.isNamedExports(statement.exportClause)) {
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
      !ts.isImportDeclaration(statement) ||
      !statement.importClause ||
      !ts.isStringLiteralLike(statement.moduleSpecifier)
    ) {
      continue;
    }
    const target = resolveImport(path, statement.moduleSpecifier.text);
    if (!target || !callerRecords.has(target)) {
      continue;
    }
    const bindings = statement.importClause.namedBindings;
    if (bindings && ts.isNamedImports(bindings)) {
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
