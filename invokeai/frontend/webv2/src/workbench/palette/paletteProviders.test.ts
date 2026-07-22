import type { GenerationModelCatalogItem } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';
import type { TFunction } from 'i18next';

import { describe, expect, it, vi } from 'vitest';

const { ensureModelsLoaded, getModelsSnapshot } = vi.hoisted(() => ({
  ensureModelsLoaded: vi.fn(() => Promise.resolve()),
  getModelsSnapshot: vi.fn(),
}));

vi.mock('@features/models', () => ({
  ensureModelsLoaded,
  getModelBaseLabel: (base: string) => base,
  getModelsSnapshot,
}));

import { createModelsProvider } from './paletteProviders';

const model = (key: string, base: string, type = 'main'): ModelConfig =>
  ({ base, key, name: key, type }) as unknown as ModelConfig;
const t = ((key: string) => key) as TFunction;
const searchContext = () => ({ signal: new AbortController().signal });

describe('createModelsProvider', () => {
  it('lists only supported Generate models and applies one before opening the widget', async () => {
    const supported = model('supported-sdxl', 'sdxl');
    const external = model('external-provider', 'external', 'external_image_generator');
    const models = [
      supported,
      external,
      model('refiner', 'sdxl-refiner'),
      model('unknown', 'unknown'),
      model('any', 'any'),
      model('vae', 'sdxl', 'vae'),
    ];
    const order: string[] = [];
    const applyModel = vi.fn((_selected: GenerationModelCatalogItem, _models: readonly ModelConfig[]) => {
      order.push('apply');
    });
    const openGenerateWidget = vi.fn(() => {
      order.push('open');
    });
    const openModelManager = vi.fn();
    getModelsSnapshot.mockReturnValue({ models, status: 'loaded' });
    const provider = createModelsProvider({ applyModel, openGenerateWidget, openModelManager, t });

    const entries = await provider.search({ text: '' }, searchContext());

    expect(entries.map((entry) => entry.title)).toEqual(['supported-sdxl', 'external-provider']);

    entries[0]?.run();
    expect(applyModel).toHaveBeenCalledWith(supported, models);
    expect(order).toEqual(['apply', 'open']);

    entries[0]?.secondary?.run();
    expect(openModelManager).toHaveBeenCalledOnce();
  });

  it('retains model name and base text matching', async () => {
    const models = [model('cinematic-xl', 'sdxl'), model('fast-flux', 'flux')];
    getModelsSnapshot.mockReturnValue({ models, status: 'loaded' });
    const provider = createModelsProvider({
      applyModel: vi.fn(),
      openGenerateWidget: vi.fn(),
      openModelManager: vi.fn(),
      t,
    });

    expect((await provider.search({ text: 'cinematic' }, searchContext())).map((entry) => entry.title)).toEqual([
      'cinematic-xl',
    ]);
    expect((await provider.search({ text: 'flux' }, searchContext())).map((entry) => entry.title)).toEqual([
      'fast-flux',
    ]);
  });
});
