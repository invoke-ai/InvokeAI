import { mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, extname, join, relative, resolve } from 'node:path';
import {
  isCallExpression,
  isEmptyStatement,
  isExportAssignment,
  isExportDeclaration,
  isIdentifier,
  isImportDeclaration,
  isImportTypeNode,
  isInterfaceDeclaration,
  isLiteralTypeNode,
  isNamedExports,
  isStringLiteralLikeNode,
  isTypeAliasDeclaration,
  isVariableStatement,
  ModifierFlags,
  SyntaxKind,
} from 'typescript/unstable/ast';

import { parseSource, primeSources } from './parse-source.mjs';

const packageRoot = process.cwd();
const sourceRoot = resolve(packageRoot, 'src');
const manifest = JSON.parse(readFileSync(resolve(sourceRoot, 'architecture/workbenchOwnershipManifest.json'), 'utf8'));
const isSource = (path) => /\.[cm]?[jt]sx?$/.test(path);
const isTest = (path) => /(?:\.test|\.browser\.test|\.type-test)\.[^.]+$/.test(path);
const stripExtension = (path) => path.slice(0, -extname(path).length);
const normalize = (path) => path.replaceAll('\\', '/').replace(/^src\//, '');
const paths = readdirSync(sourceRoot, { recursive: true, withFileTypes: true })
  .filter((entry) => entry.isFile())
  .map((entry) => normalize(relative(sourceRoot, join(entry.parentPath, entry.name))))
  .filter(isSource)
  .sort();
const productionPaths = paths.filter((path) => !isTest(path));
const sources = new Map(productionPaths.map((path) => [path, readFileSync(resolve(sourceRoot, path), 'utf8')]));
// Load the whole tree in one snapshot; parsing file by file costs a round trip each.
primeSources(sources, { jsx: true });
const pathByStem = new Map(productionPaths.map((path) => [stripExtension(path), path]));

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
  let resolved = null;
  if (specifier.startsWith('.')) {
    resolved = normalizeSegments(`${sourcePath.slice(0, sourcePath.lastIndexOf('/'))}/${specifier}`);
  } else {
    for (const [alias, target] of aliases) {
      if (specifier === alias || specifier.startsWith(`${alias}/`)) {
        resolved = normalizeSegments(`${target}${specifier.slice(alias.length)}`);
        break;
      }
    }
  }
  return resolved ? (pathByStem.get(resolved) ?? pathByStem.get(`${resolved}/index`) ?? null) : null;
};

const currentOwner = (path) => {
  const feature = /^features\/([^/]+)/.exec(path)?.[1];
  if (feature) {
    return `feature:${feature}`;
  }
  if (path.startsWith('app/')) {
    return 'app';
  }
  if (path.startsWith('platform/')) {
    return 'platform';
  }
  return 'workbench';
};

const matchesPrefix = (path, prefix) =>
  path === prefix || path.startsWith(`${prefix}.`) || path.startsWith(`${prefix}/`);
const targetRule = (path) => {
  const relativePath = path.replace(/^workbench\//, '');
  const rootOwner = manifest.rootFiles[relativePath];
  if (rootOwner) {
    const stem = stripExtension(relativePath);
    const targetPath =
      rootOwner === 'workbench'
        ? `workbench/${stem}`
        : rootOwner === 'platform'
          ? `platform/state/react/${stem}`
          : `features/${rootOwner.slice('feature:'.length)}/core/${stem}`;
    return { targetOwner: rootOwner, targetPath };
  }
  const override = manifest.overrides
    .filter((rule) => matchesPrefix(relativePath, rule.pathPrefix))
    .sort((a, b) => b.pathPrefix.length - a.pathPrefix.length)[0];
  if (override) {
    return override;
  }
  return manifest.directoryRules[relativePath.split('/')[0]] ?? null;
};
const targetOwner = (path) => (path.startsWith('workbench/') ? targetRule(path)?.targetOwner : currentOwner(path));

const parse = (path, source) => parseSource(path, source, { jsx: true });
const imports = (path, source) => {
  const sourceFile = parse(path, source);
  const specifiers = [];
  const visit = (node) => {
    if (
      (isImportDeclaration(node) || isExportDeclaration(node)) &&
      node.moduleSpecifier &&
      isStringLiteralLikeNode(node.moduleSpecifier)
    ) {
      specifiers.push(node.moduleSpecifier.text);
    } else if (
      isImportTypeNode(node) &&
      isLiteralTypeNode(node.argument) &&
      isStringLiteralLikeNode(node.argument.literal)
    ) {
      specifiers.push(node.argument.literal.text);
    } else if (isCallExpression(node) && node.expression.kind === SyntaxKind.ImportKeyword) {
      const argument = node.arguments[0];
      if (argument && isStringLiteralLikeNode(argument)) {
        specifiers.push(argument.text);
      }
    }
    node.forEachChild(visit);
  };
  visit(sourceFile);
  return specifiers;
};

const publicExports = (path, source) => {
  const names = new Set();
  for (const statement of parse(path, source).statements) {
    if ((statement.modifierFlags & ModifierFlags.Export) !== 0) {
      if (statement.name && isIdentifier(statement.name)) {
        names.add(statement.name.text);
      }
      if (isVariableStatement(statement)) {
        for (const declaration of statement.declarationList.declarations) {
          if (isIdentifier(declaration.name)) {
            names.add(declaration.name.text);
          }
        }
      }
    }
    if (isExportAssignment(statement)) {
      names.add('default');
    }
    if (isExportDeclaration(statement)) {
      if (!statement.exportClause) {
        names.add('*');
      } else if (isNamedExports(statement.exportClause)) {
        for (const element of statement.exportClause.elements) {
          names.add(element.name.text);
        }
      }
    }
  }
  return [...names].sort();
};

const isTypeOnly = (path, source) =>
  parse(path, source).statements.every((statement) => {
    if (isInterfaceDeclaration(statement) || isTypeAliasDeclaration(statement)) {
      return true;
    }
    if (isImportDeclaration(statement)) {
      return statement.importClause?.phaseModifier === SyntaxKind.TypeKeyword;
    }
    if (isExportDeclaration(statement)) {
      return statement.isTypeOnly;
    }
    return isEmptyStatement(statement);
  });

const inbound = new Map();
const graph = new Map();
const fileGraph = new Map();
for (const [sourcePath, source] of sources) {
  const sourceOwner = targetOwner(sourcePath);
  if (!sourceOwner) {
    throw new Error(`Unclassified source: ${sourcePath}`);
  }
  graph.set(sourceOwner, graph.get(sourceOwner) ?? new Set());
  fileGraph.set(sourcePath, fileGraph.get(sourcePath) ?? new Set());
  for (const specifier of imports(sourcePath, source)) {
    const targetPath = resolveImport(sourcePath, specifier);
    if (!targetPath) {
      continue;
    }
    const owner = targetOwner(targetPath);
    if (!owner) {
      throw new Error(`Unclassified target: ${targetPath}`);
    }
    fileGraph.get(sourcePath).add(targetPath);
    const inboundOwners = inbound.get(targetPath) ?? new Set();
    inboundOwners.add(sourceOwner);
    inbound.set(targetPath, inboundOwners);
    graph.set(owner, graph.get(owner) ?? new Set());
    if (owner !== sourceOwner) {
      graph.get(sourceOwner).add(owner);
    }
  }
}

const components = (dependencyGraph) => {
  let index = 0;
  const indexes = new Map();
  const low = new Map();
  const stack = [];
  const onStack = new Set();
  const result = [];
  const visit = (node) => {
    indexes.set(node, index);
    low.set(node, index++);
    stack.push(node);
    onStack.add(node);
    for (const target of dependencyGraph.get(node) ?? []) {
      if (!indexes.has(target)) {
        visit(target);
        low.set(node, Math.min(low.get(node), low.get(target)));
      } else if (onStack.has(target)) {
        low.set(node, Math.min(low.get(node), indexes.get(target)));
      }
    }
    if (low.get(node) === indexes.get(node)) {
      const component = [];
      let member;
      do {
        member = stack.pop();
        onStack.delete(member);
        component.push(member);
      } while (member !== node);
      if (component.length > 1) {
        result.push(component.sort());
      }
    }
  };
  for (const node of dependencyGraph.keys()) {
    if (!indexes.has(node)) {
      visit(node);
    }
  }
  return result.sort((a, b) => a.join().localeCompare(b.join()));
};

const workbenchPaths = productionPaths.filter((path) => path.startsWith('workbench/'));
const modules = workbenchPaths.map((path) => {
  const source = sources.get(path);
  const rule = targetRule(path);
  if (!rule) {
    throw new Error(`Unclassified Workbench module: ${path}`);
  }
  const outboundOwners = new Set(
    imports(path, source)
      .map((specifier) => resolveImport(path, specifier))
      .filter(Boolean)
      .map(targetOwner)
      .filter(Boolean)
  );
  const stem = stripExtension(path);
  return {
    currentOwner: 'workbench',
    inboundOwners: [...(inbound.get(path) ?? [])].sort(),
    moduleKind: isTypeOnly(path, source) ? 'type-only' : 'runtime',
    outboundOwners: [...outboundOwners].sort(),
    path,
    publicExports: publicExports(path, source),
    targetOwner: rule.targetOwner,
    targetPath: rule.targetPath,
    testCompanions: paths.filter((candidate) => candidate.startsWith(`${stem}.`) && isTest(candidate)).sort(),
  };
});

const artifact = {
  counts: {
    productionWorkbenchModules: modules.length,
    runtimeModules: modules.filter((module) => module.moduleKind === 'runtime').length,
    typeOnlyModules: modules.filter((module) => module.moduleKind === 'type-only').length,
  },
  generatedFromManifestVersion: manifest.version,
  fileCycles: components(fileGraph),
  modules,
  targetDependencyGraph: Object.fromEntries(
    [...graph].sort(([a], [b]) => a.localeCompare(b)).map(([owner, targets]) => [owner, [...targets].sort()])
  ),
  transitionalCycles: components(graph),
};
const artifactPath = resolve(packageRoot, 'artifacts/architecture/workbench-ownership-inventory.json');
mkdirSync(dirname(artifactPath), { recursive: true });
writeFileSync(artifactPath, `${JSON.stringify(artifact, null, 2)}\n`);
