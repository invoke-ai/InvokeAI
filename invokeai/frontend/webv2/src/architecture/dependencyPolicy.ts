import ts from 'typescript-legacy';

import { FEATURE_PUBLIC_INTERFACES } from './featureInterfaces';

export type ModuleOwner = 'app' | 'platform' | 'workbench' | `feature:${string}`;

export interface ImportReference {
  exposesCanvasEngine: boolean;
  kind: 'dynamic-import' | 'export' | 'export-star' | 'import' | 'import-type';
  specifier: string;
}

export interface DependencyViolation {
  rule: string;
  source: string;
  target: string;
}

export interface MigrationException {
  /** Optional ISO date after which the exception is considered expired debt. */
  expires?: string;
  /** ISO date the exception was added. */
  introduced: string;
  owner: string;
  reason: string;
  removalTicket: string;
  rule: string;
  sourcePrefix: string;
  targetPrefix: string;
}

const ALIASES: ReadonlyArray<readonly [string, string]> = [
  ['@app', 'app'],
  ['@features', 'features'],
  ['@platform', 'platform'],
  ['@theme', 'platform/ui/theme'],
  ['@workbench', 'workbench'],
  ['@', ''],
];

const normalizePath = (path: string): string => {
  const parts: string[] = [];

  for (const part of path.replaceAll('\\', '/').split('/')) {
    if (!part || part === '.') {
      continue;
    }
    if (part === '..') {
      parts.pop();
      continue;
    }
    parts.push(part);
  }

  return parts.join('/');
};

export const collectImportReferences = (source: string, fileName = 'source.ts'): ImportReference[] => {
  const sourceFile = ts.createSourceFile(fileName, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  const references: ImportReference[] = [];

  const pushLiteral = (kind: ImportReference['kind'], node: ts.Node | undefined, exposesCanvasEngine = false): void => {
    if (node && ts.isStringLiteralLike(node)) {
      references.push({ exposesCanvasEngine, kind, specifier: node.text });
    }
  };

  const visit = (node: ts.Node): void => {
    if (ts.isImportDeclaration(node)) {
      const bindings = node.importClause?.namedBindings;
      const exposesCanvasEngine =
        Boolean(bindings && ts.isNamespaceImport(bindings)) ||
        Boolean(
          bindings &&
          ts.isNamedImports(bindings) &&
          bindings.elements.some((element) => (element.propertyName ?? element.name).text === 'CanvasEngine')
        );
      pushLiteral('import', node.moduleSpecifier, exposesCanvasEngine);
    } else if (ts.isExportDeclaration(node)) {
      const exposesCanvasEngine =
        !node.exportClause ||
        ts.isNamespaceExport(node.exportClause) ||
        node.exportClause.elements.some((element) => (element.propertyName ?? element.name).text === 'CanvasEngine');
      pushLiteral(
        !node.exportClause || ts.isNamespaceExport(node.exportClause) ? 'export-star' : 'export',
        node.moduleSpecifier,
        exposesCanvasEngine
      );
    } else if (ts.isImportTypeNode(node) && ts.isLiteralTypeNode(node.argument)) {
      pushLiteral('import-type', node.argument.literal, node.qualifier?.getText(sourceFile).includes('CanvasEngine'));
    } else if (ts.isCallExpression(node) && node.expression.kind === ts.SyntaxKind.ImportKeyword) {
      pushLiteral('dynamic-import', node.arguments[0]);
    }

    ts.forEachChild(node, visit);
  };

  visit(sourceFile);
  return references;
};

export const resolveImportPath = (sourcePath: string, specifier: string): string | null => {
  if (specifier.startsWith('.')) {
    const sourceDirectory = sourcePath.slice(0, Math.max(0, sourcePath.lastIndexOf('/')));
    return normalizePath(`${sourceDirectory}/${specifier}`);
  }

  for (const [alias, target] of ALIASES) {
    if (specifier === alias || specifier.startsWith(`${alias}/`)) {
      return normalizePath(`${target}${specifier.slice(alias.length)}`);
    }
  }

  return null;
};

export const getModuleOwner = (path: string): ModuleOwner => {
  const normalized = normalizePath(path).replace(/^src\//, '');
  const feature = /^features\/([^/]+)/.exec(normalized)?.[1];

  if (feature) {
    return `feature:${feature}`;
  }
  if (normalized.startsWith('app/')) {
    return 'app';
  }
  if (normalized.startsWith('platform/')) {
    return 'platform';
  }

  return 'workbench';
};

const FEATURE_ENTRY = /^features\/([^/]+)(?:\/([^/]+?)(?:\.[cm]?[jt]sx?)?)?$/;

const isFeaturePublicInterface = (path: string): boolean => {
  const match = FEATURE_ENTRY.exec(path);
  if (!match) {
    return false;
  }
  const surface = FEATURE_PUBLIC_INTERFACES[match[1]];
  if (surface === undefined) {
    return false;
  }
  const module = match[2];
  return module === undefined || module === 'index' || surface.includes(module);
};

const isCanvasPrivatePath = (path: string): boolean => {
  if (!path.startsWith('workbench/canvas-engine/') && !path.startsWith('workbench/canvas-operations/')) {
    return false;
  }

  return !/^workbench\/(?:canvas-engine\/api|canvas-operations\/(?:api|react))(?:\.[cm]?[jt]sx?)?$/.test(path);
};

const isCanvasOwnedPath = (path: string): boolean =>
  path.startsWith('workbench/canvas-engine/') || path.startsWith('workbench/canvas-operations/');

export const checkDependency = (source: string, specifier: string): DependencyViolation[] => {
  const target = resolveImportPath(source, specifier);

  if (!target) {
    return [];
  }

  const sourcePath = normalizePath(source).replace(/^src\//, '');
  const sourceOwner = getModuleOwner(sourcePath);
  const targetOwner = getModuleOwner(target);
  const violations: DependencyViolation[] = [];
  const add = (rule: string): void => {
    violations.push({ rule, source: sourcePath, target });
  };

  if (sourceOwner === 'platform' && (targetOwner === 'workbench' || targetOwner.startsWith('feature:'))) {
    add('platform-independence');
  }

  if (sourceOwner === 'workbench' && targetOwner === 'app') {
    add('app-composition-root');
  }

  if (sourceOwner.startsWith('feature:')) {
    if (targetOwner === 'app' || targetOwner === 'workbench') {
      add('feature-dependency-direction');
    }

    if (targetOwner.startsWith('feature:') && targetOwner !== sourceOwner && !isFeaturePublicInterface(target)) {
      add('feature-public-interface');
    }

    if (
      sourcePath.includes('/core/') &&
      (specifier === 'react' || target.includes('/ui/') || target.includes('/data/'))
    ) {
      add('feature-core-purity');
    }
  }

  if (!sourceOwner.startsWith('feature:') && targetOwner.startsWith('feature:') && !isFeaturePublicInterface(target)) {
    add('feature-private-interface');
  }

  if (!isCanvasOwnedPath(sourcePath) && isCanvasPrivatePath(target)) {
    add('canvas-private-interface');
  }

  if (target === 'workbench/types') {
    add('retired-contract-hub');
  }

  if (
    sourcePath.startsWith('workbench/canvas-engine/') &&
    [
      'features/generation',
      'workbench/backend',
      'workbench/canvas-operations',
      'workbench/generation',
      'workbench/widgets',
    ].some((prefix) => target.startsWith(prefix))
  ) {
    add('canvas-engine-independence');
  }

  return violations;
};

export const checkSource = (sourcePath: string, source: string): DependencyViolation[] =>
  collectImportReferences(source, sourcePath).flatMap(({ exposesCanvasEngine, kind, specifier }) => {
    const violations = checkDependency(sourcePath, specifier);
    const target = resolveImportPath(sourcePath, specifier);
    const normalizedSource = normalizePath(sourcePath).replace(/^src\//, '');
    const isFullEngineModule =
      target === 'workbench/canvas-engine/engine' || target === 'workbench/canvas-operations/createCanvasEngine';
    const mayConstructCanvas =
      normalizedSource === 'workbench/canvas-operations/createCanvasEngine.ts' ||
      normalizedSource === 'workbench/canvas-operations/engineRegistry.ts';

    if (kind === 'export-star' && target && getModuleOwner(normalizedSource) !== getModuleOwner(target)) {
      violations.push({ rule: 'cross-owner-star-export', source: normalizedSource, target });
    }

    if (exposesCanvasEngine && isFullEngineModule && !mayConstructCanvas) {
      violations.push({ rule: 'canvas-construction-private', source: normalizedSource, target });
    }

    return violations;
  });

export const formatViolation = ({ rule, source, target }: DependencyViolation): string =>
  `${rule}: ${source} -> ${target}`;

export const isExcepted = (violation: DependencyViolation, exception: MigrationException): boolean =>
  violation.rule === exception.rule &&
  violation.source.startsWith(exception.sourcePrefix) &&
  violation.target.startsWith(exception.targetPrefix);
