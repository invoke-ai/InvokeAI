import type { ModelsSnapshot } from '@features/models/data/modelsStore';

import { resolveModelAbsolutePath } from '@features/models/core/schemas';
import { useModelsSelector } from '@features/models/data/modelsStore';

const areSetsEqual = (left: ReadonlySet<string>, right: ReadonlySet<string>): boolean =>
  left.size === right.size && [...left].every((value) => right.has(value));

// Selectors run on every store notification for every subscriber; cache by the
// inputs' identities so repeat notifications reuse one set (and pass equality
// by reference instead of a member-by-member scan).
let lastModels: ModelsSnapshot['models'] | null = null;
let lastModelsDir: string | null = null;
let lastSources: ReadonlySet<string> = new Set<string>();

const selectInstalledSources = (snapshot: ModelsSnapshot): ReadonlySet<string> => {
  if (snapshot.models !== lastModels || snapshot.modelsDir !== lastModelsDir) {
    const sources = new Set<string>();

    for (const model of snapshot.models) {
      sources.add(model.source);
      sources.add(resolveModelAbsolutePath(model.path, snapshot.modelsDir));
    }

    lastModels = snapshot.models;
    lastModelsDir = snapshot.modelsDir;
    lastSources = sources;
  }

  return lastSources;
};

/**
 * Every string under which a library model is reachable as an install source:
 * its recorded install source plus its resolved absolute file path. Source
 * rows (folder scan, HuggingFace files) derive "installed" from this live set
 * — a scan-time snapshot would keep offering Install after the job finishes.
 */
export const useInstalledSources = (): ReadonlySet<string> => useModelsSelector(selectInstalledSources, areSetsEqual);
