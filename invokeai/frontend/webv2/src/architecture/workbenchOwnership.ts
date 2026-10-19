import type { ModuleOwner } from './dependencyPolicy';

import manifestJson from './workbenchOwnershipManifest.json';

interface OwnershipRule {
  targetOwner: ModuleOwner;
  targetPath: string;
}

interface OwnershipOverride extends OwnershipRule {
  pathPrefix: string;
}

interface WorkbenchOwnershipManifest {
  currentOwner: 'workbench';
  directoryRules: Record<string, OwnershipRule>;
  emptyDirectories: string[];
  overrides: OwnershipOverride[];
  rootFiles: Record<string, ModuleOwner>;
  version: number;
}

export const workbenchOwnershipManifest = manifestJson as WorkbenchOwnershipManifest;

const stripWorkbenchPrefix = (path: string): string =>
  path
    .replaceAll('\\', '/')
    .replace(/^src\//, '')
    .replace(/^workbench\//, '');

const matchesPrefix = (path: string, prefix: string): boolean =>
  path === prefix || path.startsWith(`${prefix}.`) || path.startsWith(`${prefix}/`);

export const getWorkbenchTargetOwner = (path: string): ModuleOwner | null => {
  const relativePath = stripWorkbenchPrefix(path);
  const rootOwner = workbenchOwnershipManifest.rootFiles[relativePath];

  if (rootOwner) {
    return rootOwner;
  }

  const matchingOverrides = workbenchOwnershipManifest.overrides
    .filter((rule) => matchesPrefix(relativePath, rule.pathPrefix))
    .sort((a, b) => b.pathPrefix.length - a.pathPrefix.length);

  if (
    matchingOverrides.length > 1 &&
    matchingOverrides[0]?.pathPrefix.length === matchingOverrides[1]?.pathPrefix.length
  ) {
    return null;
  }

  if (matchingOverrides[0]) {
    return matchingOverrides[0].targetOwner;
  }

  const directory = relativePath.split('/')[0];
  return directory ? (workbenchOwnershipManifest.directoryRules[directory]?.targetOwner ?? null) : null;
};

export const getWorkbenchTargetPath = (path: string): string | null => {
  const relativePath = stripWorkbenchPrefix(path);
  const rootOwner = workbenchOwnershipManifest.rootFiles[relativePath];

  if (rootOwner) {
    const stem = relativePath.replace(/\.[cm]?[jt]sx?$/, '');
    if (rootOwner === 'workbench') {
      return `workbench/${stem}`;
    }
    if (rootOwner === 'platform') {
      return `platform/state/react/${stem}`;
    }
    if (rootOwner.startsWith('feature:')) {
      return `features/${rootOwner.slice('feature:'.length)}/core/${stem}`;
    }
  }

  const matchingOverride = workbenchOwnershipManifest.overrides
    .filter((rule) => matchesPrefix(relativePath, rule.pathPrefix))
    .sort((a, b) => b.pathPrefix.length - a.pathPrefix.length)[0];

  if (matchingOverride) {
    return matchingOverride.targetPath;
  }

  const directory = relativePath.split('/')[0];
  return directory ? (workbenchOwnershipManifest.directoryRules[directory]?.targetPath ?? null) : null;
};
