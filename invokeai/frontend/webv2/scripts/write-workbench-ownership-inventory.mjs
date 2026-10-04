import { mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, extname, join, relative, resolve } from 'node:path';

import { analyzeSource, closeSourceAnalysis, primeSourceAnalysis } from '#architecture/source-analysis';

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

try {
  // Load the whole tree in one snapshot; parsing file by file costs a round trip each.
  primeSourceAnalysis(sources, { jsx: true });
  const analyses = new Map([...sources].map(([path, source]) => [path, analyzeSource(path, source, { jsx: true })]));

  const inbound = new Map();
  const graph = new Map();
  const fileGraph = new Map();
  for (const [sourcePath, analysis] of analyses) {
    const sourceOwner = targetOwner(sourcePath);
    if (!sourceOwner) {
      throw new Error(`Unclassified source: ${sourcePath}`);
    }
    graph.set(sourceOwner, graph.get(sourceOwner) ?? new Set());
    fileGraph.set(sourcePath, fileGraph.get(sourcePath) ?? new Set());
    for (const specifier of analysis.moduleReferences.map(({ specifier }) => specifier)) {
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

  const workbenchPaths = productionPaths.filter((path) => path.startsWith('workbench/'));
  const modules = workbenchPaths.map((path) => {
    const analysis = analyses.get(path);
    const rule = targetRule(path);
    if (!rule) {
      throw new Error(`Unclassified Workbench module: ${path}`);
    }
    const outboundOwners = new Set(
      analysis.moduleReferences
        .map(({ specifier }) => resolveImport(path, specifier))
        .filter(Boolean)
        .map(targetOwner)
        .filter(Boolean)
    );
    const stem = stripExtension(path);
    return {
      currentOwner: 'workbench',
      inboundOwners: [...(inbound.get(path) ?? [])].sort(),
      moduleKind: analysis.typeOnly ? 'type-only' : 'runtime',
      outboundOwners: [...outboundOwners].sort(),
      path,
      publicExports: analysis.publicExports,
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
} finally {
  closeSourceAnalysis();
}
