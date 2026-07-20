import { mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, extname, join, relative, resolve } from 'node:path';
import ts from 'typescript-legacy';

const packageRoot = process.cwd();
const sourceRoot = resolve(packageRoot, 'src');
const normalize = (path) => path.replaceAll('\\', '/');
const isTypeScript = (path) => /\.[cm]?[jt]sx?$/.test(path);
const isProduction = (path) => !/(?:\.test|\.browser\.test|\.type-test|\.stories)\.[^.]+$/.test(path);
const stripExtension = (path) => path.slice(0, -extname(path).length);
const paths = readdirSync(sourceRoot, { recursive: true, withFileTypes: true })
  .filter((entry) => entry.isFile())
  .map((entry) => normalize(relative(sourceRoot, join(entry.parentPath, entry.name))))
  .filter(isTypeScript)
  .sort();
const pathByStem = new Map(paths.map((path) => [stripExtension(path), path]));
const aliases = [
  ['@app', 'app'],
  ['@features', 'features'],
  ['@platform', 'platform'],
  ['@theme', 'platform/ui/theme'],
  ['@workbench', 'workbench'],
  ['@', ''],
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
    stem = normalizeSegments(`${dirname(sourcePath)}/${specifier}`);
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

const isCanvasOwned = (path) =>
  path.startsWith('workbench/canvas-engine/') || path.startsWith('workbench/canvas-operations/');
const isCanvasTarget = (path) => isCanvasOwned(path);
const isPublicTarget = (path) =>
  path === 'workbench/canvas-engine/api.ts' ||
  path === 'workbench/canvas-operations/api.ts' ||
  path === 'workbench/canvas-operations/react.ts';
const callerIntent = (path) => {
  if (path.startsWith('workbench/widgets/canvas/')) {
    return 'canvas-widget';
  }
  if (path.startsWith('workbench/widgets/layers/')) {
    return 'layers-widget';
  }
  if (path.startsWith('features/generation/')) {
    return 'generation';
  }
  if (path.startsWith('workbench/image-actions/')) {
    return 'image-actions';
  }
  if (path.startsWith('workbench/widgets/')) {
    return 'other-widget';
  }
  return path.split('/').slice(0, 2).join('/');
};
const importedSymbols = (clause) => {
  if (!clause) {
    return ['side-effect'];
  }
  const symbols = [];
  if (clause.name) {
    symbols.push('default');
  }
  if (clause.namedBindings && ts.isNamespaceImport(clause.namedBindings)) {
    symbols.push('*');
  }
  if (clause.namedBindings && ts.isNamedImports(clause.namedBindings)) {
    for (const element of clause.namedBindings.elements) {
      symbols.push((element.propertyName ?? element.name).text);
    }
  }
  return symbols.sort();
};

const imports = [];
for (const source of paths.filter(isProduction)) {
  if (isCanvasOwned(source)) {
    continue;
  }
  const text = readFileSync(resolve(sourceRoot, source), 'utf8');
  const file = ts.createSourceFile(
    source,
    text,
    ts.ScriptTarget.Latest,
    true,
    source.endsWith('x') ? ts.ScriptKind.TSX : ts.ScriptKind.TS
  );
  for (const statement of file.statements) {
    if (!ts.isImportDeclaration(statement) || !ts.isStringLiteralLike(statement.moduleSpecifier)) {
      continue;
    }
    const target = resolveImport(source, statement.moduleSpecifier.text);
    if (!target || !isCanvasTarget(target)) {
      continue;
    }
    imports.push({
      intent: callerIntent(source),
      interface: target.startsWith('workbench/canvas-engine/') ? 'canvas-engine' : 'canvas-operations',
      private: !isPublicTarget(target),
      source,
      symbols: importedSymbols(statement.importClause),
      target,
    });
  }
}

const grouped = Object.entries(Object.groupBy(imports, ({ intent }) => intent))
  .sort(([left], [right]) => left.localeCompare(right))
  .map(([intent, records]) => ({ intent, imports: records }));
const privateImports = imports.filter((record) => record.private);
const artifact = {
  generatedBy: 'scripts/write-canvas-import-matrix.mjs',
  groups: grouped,
  summary: {
    externalProductionCallerCount: new Set(imports.map(({ source }) => source)).size,
    importDeclarationCount: imports.length,
    privateCallerCount: new Set(privateImports.map(({ source }) => source)).size,
    privateImportDeclarationCount: privateImports.length,
  },
};

const artifactPath = resolve(packageRoot, 'artifacts/architecture/canvas-import-matrix.json');
mkdirSync(dirname(artifactPath), { recursive: true });
writeFileSync(artifactPath, `${JSON.stringify(artifact, null, 2)}\n`);
