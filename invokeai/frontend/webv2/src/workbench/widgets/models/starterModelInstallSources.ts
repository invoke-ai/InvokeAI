import type { StarterModel, StarterModelBundle } from '@workbench/models/types';

export interface StarterModelInstallSourceOptions {
  /** Dependencies visible as their own rows in the active bundle should not be auto-installed by a row action. */
  dependencySourcesToSkip?: ReadonlySet<string>;
}

const appendSource = (sources: string[], seen: Set<string>, source: string): void => {
  if (seen.has(source)) {
    return;
  }

  seen.add(source);
  sources.push(source);
};

export const getStarterModelInstallSources = (
  model: StarterModel,
  options: StarterModelInstallSourceOptions = {}
): string[] => {
  const sources: string[] = [];
  const seen = new Set<string>();

  for (const dependency of model.dependencies ?? []) {
    if (dependency.is_installed || options.dependencySourcesToSkip?.has(dependency.source)) {
      continue;
    }

    appendSource(sources, seen, dependency.source);
  }

  if (!model.is_installed) {
    appendSource(sources, seen, model.source);
  }

  return sources;
};

export const getStarterBundleInstallSources = (bundle: StarterModelBundle): string[] => {
  const sources: string[] = [];
  const seen = new Set<string>();

  for (const model of bundle.models) {
    if (model.is_installed) {
      continue;
    }

    for (const source of getStarterModelInstallSources(model)) {
      appendSource(sources, seen, source);
    }
  }

  return sources;
};
