import { FEATURE_PUBLIC_INTERFACES } from './featureInterfaces';
import { analyzeSource, primeSourceAnalysis } from './tsSourceAnalysis';

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

/** Preload the exact [fileName, source] pairs before a sweep so subsequent checks reuse one parsed program. */
export const primeImportSources = (entries: Iterable<readonly [string, string]>): void => {
  primeSourceAnalysis(entries, { jsx: true });
};

export const collectImportReferences = (source: string, fileName = 'source.ts'): ImportReference[] =>
  analyzeSource(fileName, source, { jsx: true }).moduleReferences.map((reference) => {
    const exposesCanvasEngine =
      (reference.form === 'import-declaration' &&
        (reference.namespace || reference.symbols.includes('CanvasEngine'))) ||
      (reference.form === 'export-declaration' &&
        (reference.namespace || reference.symbols.includes('CanvasEngine'))) ||
      (reference.form === 'import-type' && reference.qualifier?.includes('CanvasEngine') === true);

    return { exposesCanvasEngine, kind: reference.kind, specifier: reference.specifier };
  });

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

const FEATURE_CORE_FORBIDDEN_PACKAGE_ROOTS = [
  '@chakra-ui/react',
  '@xyflow/react',
  'lucide-react',
  'react',
  'react-dom',
] as const;

const isPackageOrSubpath = (specifier: string, packageRoot: string): boolean =>
  specifier === packageRoot || specifier.startsWith(`${packageRoot}/`);

const isFeatureCoreForbiddenDependency = (specifier: string, target: string | null): boolean =>
  FEATURE_CORE_FORBIDDEN_PACKAGE_ROOTS.some((packageRoot) => isPackageOrSubpath(specifier, packageRoot)) ||
  Boolean(
    target &&
    (isPackageOrSubpath(target, 'platform/transport') ||
      isPackageOrSubpath(target, 'platform/ui') ||
      /\/(?:data|ui)(?:\/|$)/.test(target))
  );

/** The reducer shares the document seam's stack, anchor and repair helpers instead of the public API. */
const CANVAS_REDUCER = 'workbench/canvasProjectMutations.ts';
const REDUCER_DOCUMENT_MODULES = [
  'workbench/canvas-engine/document/documentIndex',
  'workbench/canvas-engine/document/documentTree',
  'workbench/canvas-engine/document/insertionAnchors',
  'workbench/canvas-engine/document/layerStacks',
  'workbench/canvas-engine/document/selectionRepair',
];
/** The loader repairs a persisted selection at ingress with the seam's own repair. */
const CANVAS_LOADER = 'workbench/canvasMigration.ts';
const LOADER_DOCUMENT_MODULES = ['workbench/canvas-engine/document/selectionRepair'];

const isSeamAllowance = (sourcePath: string, target: string): boolean =>
  (sourcePath === CANVAS_REDUCER && REDUCER_DOCUMENT_MODULES.includes(target)) ||
  (sourcePath === CANVAS_LOADER && LOADER_DOCUMENT_MODULES.includes(target));

const DOCUMENT_MODEL_ROOT = 'workbench/canvas-engine/document-model/';
const DOCUMENT_MODEL_DEPENDENCY_ROOTS = [DOCUMENT_MODEL_ROOT, 'workbench/canvas-engine/math/'];
const DOCUMENT_MODEL_DEPENDENCY_MODULES = [
  'workbench/canvas-engine/contracts',
  'workbench/canvas-engine/types',
  'workbench/canvas-engine/mutationContracts',
  'workbench/canvas-engine/document/commandRefusal',
  'workbench/canvas-engine/document/documentIndex',
  'workbench/canvas-engine/document/documentTree',
  'workbench/canvas-engine/document/insertionAnchors',
  'workbench/canvas-engine/document/layerEligibility',
  'workbench/canvas-engine/document/layerStacks',
  'workbench/canvas-engine/document/selectionRepair',
];

/** Accept extensionless targets as well as source filenames. */
const TEST_SUPPORT_SUFFIX = /\.(?:test|type-test|testing|stories)(?:\.[^.]+)?$/;

/** Exclude test support from production; importing it would pull runners and fixtures into application bundles. */
export const isProductionSourcePath = (path: string): boolean => !TEST_SUPPORT_SUFFIX.test(path);

/** The pure document model may only reach pure document facts, math, and contracts. */
const isDocumentModelDependency = (target: string | null): boolean =>
  target !== null &&
  (DOCUMENT_MODEL_DEPENDENCY_ROOTS.some((root) => target.startsWith(root)) ||
    DOCUMENT_MODEL_DEPENDENCY_MODULES.includes(target));

export const checkDependency = (source: string, specifier: string): DependencyViolation[] => {
  const sourcePath = normalizePath(source).replace(/^src\//, '');
  const target = resolveImportPath(source, specifier);
  const violations: DependencyViolation[] = [];
  const add = (rule: string, resolvedTarget = target ?? specifier): void => {
    violations.push({ rule, source: sourcePath, target: resolvedTarget });
  };

  if (
    sourcePath.startsWith('features/') &&
    sourcePath.includes('/core/') &&
    isFeatureCoreForbiddenDependency(specifier, target)
  ) {
    add('feature-core-purity');
  }

  if (
    sourcePath.startsWith(DOCUMENT_MODEL_ROOT) &&
    isProductionSourcePath(sourcePath) &&
    !isDocumentModelDependency(target)
  ) {
    add('document-model-purity');
  }

  if (!target) {
    return violations;
  }

  if (isProductionSourcePath(sourcePath) && !isProductionSourcePath(target)) {
    add('test-support-isolation');
  }

  const sourceOwner = getModuleOwner(sourcePath);
  const targetOwner = getModuleOwner(target);

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
  }

  if (!sourceOwner.startsWith('feature:') && targetOwner.startsWith('feature:') && !isFeaturePublicInterface(target)) {
    add('feature-private-interface');
  }

  if (!isCanvasOwnedPath(sourcePath) && isCanvasPrivatePath(target) && !isSeamAllowance(sourcePath, target)) {
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
