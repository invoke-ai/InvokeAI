import ts from 'typescript';
import { describe, expect, it } from 'vitest';

import { collectImportReferences, getModuleOwner, resolveImportPath } from './dependencyPolicy';
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

const collectPublicExports = (source: string, path: string): string[] => {
  const sourceFile = ts.createSourceFile(path, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  const exports = new Set<string>();

  for (const statement of sourceFile.statements) {
    const modifiers = ts.canHaveModifiers(statement) ? ts.getModifiers(statement) : undefined;
    if (modifiers?.some((modifier) => modifier.kind === ts.SyntaxKind.ExportKeyword)) {
      if (
        (ts.isClassDeclaration(statement) ||
          ts.isFunctionDeclaration(statement) ||
          ts.isInterfaceDeclaration(statement) ||
          ts.isTypeAliasDeclaration(statement) ||
          ts.isEnumDeclaration(statement)) &&
        statement.name
      ) {
        exports.add(statement.name.text);
      } else if (ts.isVariableStatement(statement)) {
        for (const declaration of statement.declarationList.declarations) {
          if (ts.isIdentifier(declaration.name)) {
            exports.add(declaration.name.text);
          }
        }
      }
    }
    if (ts.isExportAssignment(statement)) {
      exports.add('default');
    }
    if (ts.isExportDeclaration(statement)) {
      if (!statement.exportClause) {
        exports.add('*');
      } else if (ts.isNamedExports(statement.exportClause)) {
        for (const element of statement.exportClause.elements) {
          exports.add(element.name.text);
        }
      }
    }
  }

  return [...exports].sort();
};

const isTypeOnlyModule = (source: string, path: string): boolean => {
  const sourceFile = ts.createSourceFile(path, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  return sourceFile.statements.every((statement) => {
    if (ts.isInterfaceDeclaration(statement) || ts.isTypeAliasDeclaration(statement)) {
      return true;
    }
    if (ts.isImportDeclaration(statement)) {
      return Boolean(statement.importClause?.isTypeOnly);
    }
    if (ts.isExportDeclaration(statement)) {
      return statement.isTypeOnly;
    }
    return ts.isEmptyStatement(statement);
  });
};

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

    for (const [sourcePath, source] of Object.entries(productionSources)) {
      const sourceOwner = getTargetOwner(sourcePath);
      expect(sourceOwner, `Unclassified source: ${sourcePath}`).not.toBeNull();
      graph.set(sourceOwner!, graph.get(sourceOwner!) ?? new Set<string>());

      for (const reference of collectImportReferences(source, sourcePath)) {
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

      const outboundOwners = new Set<string>();
      for (const reference of collectImportReferences(source, path)) {
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
      const testCompanions = Object.keys(sources)
        .map(toSourcePath)
        .filter((candidate) => candidate.startsWith(`${stem}.`) && isTest(candidate))
        .sort();

      return {
        currentOwner: 'workbench',
        inboundOwners: [] as string[],
        moduleKind: isTypeOnlyModule(source, path) ? 'type-only' : 'runtime',
        outboundOwners: [...outboundOwners].sort(),
        path,
        publicExports: collectPublicExports(source, path),
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
  });

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
