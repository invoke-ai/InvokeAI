import type { StarterModel, StarterModelBundle } from '@features/models/core/types';

import { describe, expect, it } from 'vitest';

import type { StarterInstallSource } from './starterModelInstallSources';

import { getStarterBundleInstallSources, getStarterModelInstallSources } from './starterModelInstallSources';

const model = (source: string, overrides: Partial<StarterModel> = {}): StarterModel => ({
  base: 'sdxl',
  description: source,
  is_installed: false,
  name: source,
  source,
  type: 'main',
  ...overrides,
});

const sourcesOf = (entries: StarterInstallSource[]): string[] => entries.map((entry) => entry.source);

describe('starter model install sources', () => {
  it('queues uninstalled dependencies before the model', () => {
    const sources = getStarterModelInstallSources(
      model('main', { dependencies: [model('vae'), model('encoder', { is_installed: true })] })
    );

    expect(sourcesOf(sources)).toEqual(['vae', 'main']);
  });

  it('registers each entry with its own curated config', () => {
    const sources = getStarterModelInstallSources(
      model('main', { base: 'flux', dependencies: [model('vae', { base: 'sdxl', type: 'vae' })] })
    );

    expect(sources).toEqual([
      { config: { base: 'sdxl', description: 'vae', name: 'vae', type: 'vae' }, source: 'vae' },
      { config: { base: 'flux', description: 'main', name: 'main', type: 'main' }, source: 'main' },
    ]);
  });

  it('skips dependency sources already visible in the opened bundle', () => {
    const sources = getStarterModelInstallSources(model('main', { dependencies: [model('vae'), model('encoder')] }), {
      dependencySourcesToSkip: new Set(['vae']),
    });

    expect(sourcesOf(sources)).toEqual(['encoder', 'main']);
  });

  it('deduplicates sources across bundle models', () => {
    const bundle: StarterModelBundle = {
      models: [
        model('main-a', { dependencies: [model('shared-vae')] }),
        model('main-b', { dependencies: [model('shared-vae')] }),
      ],
      name: 'Bundle',
    };

    expect(sourcesOf(getStarterBundleInstallSources(bundle))).toEqual(['shared-vae', 'main-a', 'main-b']);
  });
});
