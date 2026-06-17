import type { StarterModel, StarterModelBundle } from '@workbench/models/types';

import { describe, expect, it } from 'vitest';

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

describe('starter model install sources', () => {
  it('queues uninstalled dependencies before the model', () => {
    const sources = getStarterModelInstallSources(
      model('main', { dependencies: [model('vae'), model('encoder', { is_installed: true })] })
    );

    expect(sources).toEqual(['vae', 'main']);
  });

  it('skips dependency sources already visible in the opened bundle', () => {
    const sources = getStarterModelInstallSources(model('main', { dependencies: [model('vae'), model('encoder')] }), {
      dependencySourcesToSkip: new Set(['vae']),
    });

    expect(sources).toEqual(['encoder', 'main']);
  });

  it('deduplicates sources across bundle models', () => {
    const bundle: StarterModelBundle = {
      models: [
        model('main-a', { dependencies: [model('shared-vae')] }),
        model('main-b', { dependencies: [model('shared-vae')] }),
      ],
      name: 'Bundle',
    };

    expect(getStarterBundleInstallSources(bundle)).toEqual(['shared-vae', 'main-a', 'main-b']);
  });
});
