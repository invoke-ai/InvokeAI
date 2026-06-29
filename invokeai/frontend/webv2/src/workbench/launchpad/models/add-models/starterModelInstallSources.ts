import type { ModelRecordChanges, StarterModel, StarterModelBundle } from '@workbench/models/types';

/** A source to install plus the curated config metadata to register it with. */
export interface StarterInstallSource {
  source: string;
  config: ModelRecordChanges;
}

export interface StarterModelInstallSourceOptions {
  /** Dependencies visible as their own rows in the active bundle should not be auto-installed by a row action. */
  dependencySourcesToSkip?: ReadonlySet<string>;
}

/**
 * The curated metadata a starter carries: passed as the install `config` so the
 * model is registered with its known name/base/type instead of relying purely on
 * server-side probing (matches the legacy install path). Built per-model, so a
 * dependency gets its own metadata rather than the parent's.
 */
const buildStarterConfig = (model: StarterModel | Omit<StarterModel, 'dependencies'>): ModelRecordChanges => ({
  base: model.base,
  description: model.description,
  name: model.name,
  type: model.type,
  ...(model.format ? { format: model.format } : {}),
  ...(model.variant ? { variant: model.variant } : {}),
});

const appendSource = (sources: StarterInstallSource[], seen: Set<string>, entry: StarterInstallSource): void => {
  if (seen.has(entry.source)) {
    return;
  }

  seen.add(entry.source);
  sources.push(entry);
};

export const getStarterModelInstallSources = (
  model: StarterModel,
  options: StarterModelInstallSourceOptions = {}
): StarterInstallSource[] => {
  const sources: StarterInstallSource[] = [];
  const seen = new Set<string>();

  for (const dependency of model.dependencies ?? []) {
    if (dependency.is_installed || options.dependencySourcesToSkip?.has(dependency.source)) {
      continue;
    }

    appendSource(sources, seen, { config: buildStarterConfig(dependency), source: dependency.source });
  }

  if (!model.is_installed) {
    appendSource(sources, seen, { config: buildStarterConfig(model), source: model.source });
  }

  return sources;
};

export const getStarterBundleInstallSources = (bundle: StarterModelBundle): StarterInstallSource[] => {
  const sources: StarterInstallSource[] = [];
  const seen = new Set<string>();

  for (const model of bundle.models) {
    if (model.is_installed) {
      continue;
    }

    for (const entry of getStarterModelInstallSources(model)) {
      appendSource(sources, seen, entry);
    }
  }

  return sources;
};
