import { afterAll, describe, expect, it } from 'vitest';

import { collectImportReferences, getModuleOwner, primeImportSources, resolveImportPath } from './dependencyPolicy';
import { analyzeSource, closeSourceAnalysis } from './tsSourceAnalysis';
import { getWorkbenchTargetOwner, getWorkbenchTargetPath, workbenchOwnershipManifest } from './workbenchOwnership';

const sources = import.meta.glob('../**/*.{ts,tsx}', {
  eager: true,
  import: 'default',
  query: '?raw',
}) as Record<string, string>;

const toSourcePath = (path: string): string => path.replace(/^\.\.\//, '');
const isTest = (path: string): boolean => /(?:\.test|\.browser\.test|\.type-test)\.[^.]+$/.test(path);
const isProduction = (path: string): boolean => !isTest(path);
const stripExtension = (path: string): string => path.replace(/\.[cm]?[jt]sx?$/, '');

const productionSources: Record<string, string> = Object.fromEntries(
  Object.entries(sources)
    .map(([path, source]) => [toSourcePath(path), source])
    .filter(([path]) => isProduction(path))
);
const sourcePaths = Object.keys(sources).map(toSourcePath);

const pathByExtensionlessPath = new Map(Object.keys(productionSources).map((path) => [stripExtension(path), path]));

const resolveSourceFile = (sourcePath: string, specifier: string): string | null => {
  const resolved = resolveImportPath(sourcePath, specifier);
  if (!resolved) {
    return null;
  }
  return pathByExtensionlessPath.get(resolved) ?? pathByExtensionlessPath.get(`${resolved}/index`) ?? null;
};

const getTargetOwner = (path: string) =>
  path.startsWith('workbench/') ? getWorkbenchTargetOwner(path) : getModuleOwner(path);

afterAll(closeSourceAnalysis);

const stronglyConnectedComponents = (graph: Map<string, Set<string>>): string[][] => {
  let nextIndex = 0;
  const stack: string[] = [];
  const onStack = new Set<string>();
  const indexes = new Map<string, number>();
  const lowLinks = new Map<string, number>();
  const components: string[][] = [];

  const visit = (node: string): void => {
    indexes.set(node, nextIndex);
    lowLinks.set(node, nextIndex);
    nextIndex += 1;
    stack.push(node);
    onStack.add(node);

    for (const target of graph.get(node) ?? []) {
      if (!indexes.has(target)) {
        visit(target);
        lowLinks.set(node, Math.min(lowLinks.get(node)!, lowLinks.get(target)!));
      } else if (onStack.has(target)) {
        lowLinks.set(node, Math.min(lowLinks.get(node)!, indexes.get(target)!));
      }
    }

    if (lowLinks.get(node) === indexes.get(node)) {
      const component: string[] = [];
      let member: string;
      do {
        member = stack.pop()!;
        onStack.delete(member);
        component.push(member);
      } while (member !== node);
      components.push(component.sort());
    }
  };

  for (const node of graph.keys()) {
    if (!indexes.has(node)) {
      visit(node);
    }
  }
  return components.filter((component) => component.length > 1).sort((a, b) => a.join().localeCompare(b.join()));
};

describe('Workbench ownership manifest', () => {
  it('classifies every production Workbench module exactly once and emits an inspectable inventory', () => {
    const workbenchSources = Object.entries(productionSources).filter(([path]) => path.startsWith('workbench/'));
    primeImportSources(Object.entries(productionSources));
    const importReferencesByPath = new Map(
      Object.entries(productionSources).map(([path, source]) => [path, collectImportReferences(source, path)] as const)
    );
    const actualRootFiles = new Set<string>();

    for (const [path] of workbenchSources) {
      const relativePath = path.slice('workbench/'.length);
      if (!relativePath.includes('/')) {
        actualRootFiles.add(relativePath);
      }
    }

    const actualDirectories = [
      ...new Set([
        ...workbenchSources
          .map(([path]) => path.slice('workbench/'.length))
          .filter((path) => path.includes('/'))
          .map((path) => path.split('/')[0]!),
        ...workbenchOwnershipManifest.emptyDirectories,
      ]),
    ].sort();
    expect(actualDirectories).toEqual(Object.keys(workbenchOwnershipManifest.directoryRules).sort());
    expect([...actualRootFiles].sort()).toEqual(Object.keys(workbenchOwnershipManifest.rootFiles).sort());
    expect(new Set(workbenchOwnershipManifest.overrides.map((rule) => rule.pathPrefix)).size).toBe(
      workbenchOwnershipManifest.overrides.length
    );

    const inbound = new Map<string, Set<string>>();
    const graph = new Map<string, Set<string>>();

    for (const [sourcePath] of Object.entries(productionSources)) {
      const sourceOwner = getTargetOwner(sourcePath);
      expect(sourceOwner, `Unclassified source: ${sourcePath}`).not.toBeNull();
      graph.set(sourceOwner!, graph.get(sourceOwner!) ?? new Set<string>());

      for (const reference of importReferencesByPath.get(sourcePath) ?? []) {
        const targetFile = resolveSourceFile(sourcePath, reference.specifier);
        if (!targetFile) {
          continue;
        }
        const targetOwner = getTargetOwner(targetFile);
        expect(targetOwner, `Unclassified target: ${targetFile}`).not.toBeNull();

        const inboundOwners = inbound.get(targetFile) ?? new Set<string>();
        inboundOwners.add(sourceOwner!);
        inbound.set(targetFile, inboundOwners);
        graph.set(targetOwner!, graph.get(targetOwner!) ?? new Set<string>());
        if (targetOwner !== sourceOwner) {
          graph.get(sourceOwner!)!.add(targetOwner!);
        }
      }
    }

    const records = workbenchSources.map(([path, source]) => {
      const targetOwner = getWorkbenchTargetOwner(path);
      expect(targetOwner, `Unclassified: ${path}`).not.toBeNull();
      const analysis = analyzeSource(path, source, { jsx: true });

      const outboundOwners = new Set<string>();
      for (const reference of importReferencesByPath.get(path) ?? []) {
        const targetFile = resolveSourceFile(path, reference.specifier);
        if (!targetFile) {
          continue;
        }
        const outboundOwner = getTargetOwner(targetFile);
        if (outboundOwner) {
          outboundOwners.add(outboundOwner);
        }
      }

      const stem = stripExtension(path);
      const testCompanions = sourcePaths
        .filter((candidate) => candidate.startsWith(`${stem}.`) && isTest(candidate))
        .sort();

      return {
        currentOwner: 'workbench',
        inboundOwners: [] as string[],
        moduleKind: analysis.typeOnly ? 'type-only' : 'runtime',
        outboundOwners: [...outboundOwners].sort(),
        path,
        publicExports: [...analysis.publicExports],
        targetOwner,
        targetPath: getWorkbenchTargetPath(path),
        testCompanions,
      };
    });

    for (const record of records) {
      record.inboundOwners = [...(inbound.get(record.path) ?? [])].sort();
    }

    const artifact = {
      counts: {
        productionWorkbenchModules: records.length,
        runtimeModules: records.filter((record) => record.moduleKind === 'runtime').length,
        typeOnlyModules: records.filter((record) => record.moduleKind === 'type-only').length,
      },
      generatedFromManifestVersion: workbenchOwnershipManifest.version,
      modules: records.sort((a, b) => a.path.localeCompare(b.path)),
      targetDependencyGraph: Object.fromEntries(
        [...graph.entries()]
          .sort(([a], [b]) => a.localeCompare(b))
          .map(([owner, targets]) => [owner, [...targets].sort()])
      ),
      transitionalCycles: stronglyConnectedComponents(graph),
    };
    expect(records).toHaveLength(workbenchSources.length);
    expect(records.every((record) => record.targetPath)).toBe(true);
    expect(artifact.counts.productionWorkbenchModules).toBe(workbenchSources.length);
    expect(artifact.transitionalCycles).toHaveLength(0);
  }, 10_000);

  it('rejects responsibility-free target names', () => {
    const paths = [
      ...Object.values(workbenchOwnershipManifest.directoryRules).map((rule) => rule.targetPath),
      ...workbenchOwnershipManifest.overrides.map((rule) => rule.targetPath),
    ];
    expect(paths.filter((path) => /(?:^|\/)(?:common|shared|utils)(?:\/|$)/.test(path))).toEqual([]);
  });

  it('applies mixed-folder exact-file overrides before directory defaults', () => {
    expect(getWorkbenchTargetOwner('workbench/components/QueueProgressIndicator.tsx')).toBe('feature:queue');
  });
});
