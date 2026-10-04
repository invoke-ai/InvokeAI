import type { ModelConfig, StarterModel } from '@features/models/core/types';

import { describe, expect, it } from 'vitest';

import { findInstalledStarterModelKey } from './useInstalledSources';

const model = (overrides: Partial<ModelConfig>): ModelConfig =>
  ({
    base: 'sdxl',
    key: 'k',
    name: 'Juggernaut',
    source: 'https://example/j',
    type: 'main',
    ...overrides,
  }) as ModelConfig;

const starter: Pick<StarterModel, 'base' | 'name' | 'previous_names' | 'source' | 'type'> = {
  base: 'sdxl',
  name: 'Juggernaut XL',
  previous_names: ['Juggernaut'],
  source: 'https://example/jxl',
  type: 'main',
};

describe('findInstalledStarterModelKey', () => {
  it('prefers the model recorded under the starter source', () => {
    const bySource = new Map([['https://example/jxl', 'from-source']]);
    expect(findInstalledStarterModelKey(starter, bySource, [model({ key: 'by-name', name: 'Juggernaut XL' })])).toBe(
      'from-source'
    );
  });

  it('falls back to a name or previous name with the same base and type', () => {
    const models = [
      model({ base: 'sd-1', key: 'wrong-base', name: 'Juggernaut XL' }),
      model({ key: 'wrong-type', name: 'Juggernaut XL', type: 'lora' }),
      model({ key: 'renamed', name: 'Juggernaut' }),
    ];
    expect(findInstalledStarterModelKey(starter, new Map(), models)).toBe('renamed');
    expect(findInstalledStarterModelKey(starter, new Map(), models.slice(0, 2))).toBeNull();
  });
});
