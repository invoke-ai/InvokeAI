import type { ModelConfig, StarterModel } from '@features/models/core/types';
import type { ModelsSnapshot } from '@features/models/data/modelsStore';

import { describe, expect, it } from 'vitest';

import {
  findInstalledLocalPathKey,
  findInstalledStarterModelKey,
  selectInstalledSourceKeys,
} from './useInstalledSources';

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

describe('findInstalledLocalPathKey', () => {
  const installedKeys = (models: ModelConfig[], modelsDir: string | null = null) =>
    selectInstalledSourceKeys({ models, modelsDir } as ModelsSnapshot);

  it('finds a model recorded with forward slashes from a Windows scan path', () => {
    const keys = installedKeys([
      model({ key: 'lora', path: 'C:/Users/me/loras/x.safetensors', source: 'C:/Users/me/loras/x.safetensors' }),
    ]);

    expect(findInstalledLocalPathKey('C:\\Users\\me\\loras\\x.safetensors', keys)).toBe('lora');
    expect(findInstalledLocalPathKey('c:\\users\\ME\\loras\\x.safetensors', keys)).toBe('lora');
  });

  it('finds a managed model by its path under the models folder', () => {
    const keys = installedKeys(
      [model({ key: 'managed', path: 'loras/x.safetensors', source: 'https://example/x' })],
      'C:/invokeai/models'
    );

    expect(findInstalledLocalPathKey('C:\\invokeai\\models\\loras\\x.safetensors', keys)).toBe('managed');
  });

  it('keeps POSIX paths case-sensitive', () => {
    const keys = installedKeys([model({ key: 'posix', path: '/a/X.safetensors', source: '/a/X.safetensors' })]);

    expect(findInstalledLocalPathKey('/a/X.safetensors', keys)).toBe('posix');
    expect(findInstalledLocalPathKey('/a/x.safetensors', keys)).toBeNull();
  });
});
