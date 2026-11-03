import type { ModelConfig, StarterModel } from '@features/models/core/types';
import type { ModelsSnapshot } from '@features/models/data/modelsStore';

import { canonicalLocalPath, isAbsoluteModelPath, resolveModelAbsolutePath } from '@features/models/core/schemas';
import { useModelsSelector } from '@features/models/data/modelsStore';

const areMapsEqual = (left: ReadonlyMap<string, string>, right: ReadonlyMap<string, string>): boolean =>
  left.size === right.size && [...left].every(([source, key]) => right.get(source) === key);

// Cache by input identity so repeated store notifications reuse the same map across subscribers.
let lastModels: ModelsSnapshot['models'] | null = null;
let lastModelsDir: string | null = null;
let lastSources: ReadonlyMap<string, string> = new Map<string, string>();

export const selectInstalledSourceKeys = (snapshot: ModelsSnapshot): ReadonlyMap<string, string> => {
  if (snapshot.models !== lastModels || snapshot.modelsDir !== lastModelsDir) {
    const sources = new Map<string, string>();

    for (const model of snapshot.models) {
      sources.set(model.source, model.key);
      // Local paths are indexed in canonical form too, so callers can match any spelling the server reports.
      if (isAbsoluteModelPath(model.source)) {
        sources.set(canonicalLocalPath(model.source), model.key);
      }
      const resolvedPath = resolveModelAbsolutePath(model.path, snapshot.modelsDir);
      sources.set(resolvedPath, model.key);
      sources.set(canonicalLocalPath(resolvedPath), model.key);
    }

    lastModels = snapshot.models;
    lastModelsDir = snapshot.modelsDir;
    lastSources = sources;
  }

  return lastSources;
};

/**
 * Map both recorded install sources and resolved paths to live model keys; scan-time snapshots would keep offering
 * Install after completion.
 */
export const useInstalledSourceKeys = (): ReadonlyMap<string, string> =>
  useModelsSelector(selectInstalledSourceKeys, areMapsEqual);

/** The installed model at a local path in any spelling: a Windows scan reports `C:\…` for a model recorded as `C:/…`. */
export const findInstalledLocalPathKey = (
  path: string,
  installedSourceKeys: ReadonlyMap<string, string>
): string | null => installedSourceKeys.get(canonicalLocalPath(path)) ?? null;

const starterIdentity = (base: string, type: string, name: string): string => `${base}\u0000${type}\u0000${name}`;

/**
 * The installed model a starter entry stands for, matched the way the backend
 * marks starters installed: by install source, else by name (or a previous
 * name) together with base and type.
 */
export const findInstalledStarterModelKey = (
  starter: Pick<StarterModel, 'base' | 'name' | 'previous_names' | 'source' | 'type'>,
  installedSourceKeys: ReadonlyMap<string, string>,
  models: readonly ModelConfig[]
): string | null => {
  const bySource = installedSourceKeys.get(starter.source);
  if (bySource !== undefined) {
    return bySource;
  }
  const names = new Set(
    [starter.name, ...(starter.previous_names ?? [])].map((name) => starterIdentity(starter.base, starter.type, name))
  );
  return models.find((model) => names.has(starterIdentity(model.base, model.type, model.name)))?.key ?? null;
};
