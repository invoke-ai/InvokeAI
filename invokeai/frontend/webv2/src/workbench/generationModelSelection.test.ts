import type {
  GenerateModelConfig,
  GenerateSettings,
  LoraModelConfig,
  VaeModelConfig,
} from '@features/generation/contracts';

import { getDefaultGenerateSettings } from '@features/generation/settings';
import { describe, expect, it, vi } from 'vitest';

import { selectProjectGenerateModel } from './generationModelSelection';

const createModel = (base: string, key = `${base}-model`): GenerateModelConfig => ({
  base,
  key,
  name: key,
  type: 'main',
});

const createSettings = (model: GenerateModelConfig, overrides: Partial<GenerateSettings> = {}): GenerateSettings => ({
  ...getDefaultGenerateSettings(model),
  seed: 7,
  shouldRandomizeSeed: false,
  ...overrides,
});

const sdxlLora: LoraModelConfig = { base: 'sdxl', key: 'sdxl-lora', name: 'SDXL LoRA', type: 'lora' };
const sdxlVae: VaeModelConfig = { base: 'sdxl', key: 'sdxl-vae', name: 'SDXL VAE', type: 'vae' };

describe('selectProjectGenerateModel', () => {
  it('atomically stores reconciled model settings and returns cleared labels', () => {
    const currentModel = createModel('sdxl');
    const model = createModel('flux');
    const setSettings = vi.fn();
    const currentValues = createSettings(currentModel, {
      batchCount: 3,
      loras: [{ isEnabled: true, model: sdxlLora, weight: 1 }],
      positivePrompt: 'a lighthouse',
      vae: sdxlVae,
    });

    const result = selectProjectGenerateModel({
      currentValues,
      generation: { setSettings },
      model,
      models: [currentModel, model, sdxlLora, sdxlVae],
      projectId: 'project-1',
    });

    expect(setSettings).toHaveBeenCalledOnce();
    expect(setSettings).toHaveBeenCalledWith(
      expect.objectContaining({
        batchCount: 3,
        loras: [],
        model,
        modelKey: model.key,
        positivePrompt: 'a lighthouse',
        seed: 7,
        vae: null,
      }),
      'project-1'
    );
    expect(result.clearedLabels).toEqual(expect.arrayContaining(['LoRAs', 'VAE']));
  });

  it('initializes complete settings when the project has no Generate values', () => {
    const model = createModel('sdxl');
    const setSettings = vi.fn();

    const result = selectProjectGenerateModel({
      currentValues: {},
      generation: { setSettings },
      model,
      models: [model],
      projectId: 'project-empty',
    });

    expect(result.settings).toMatchObject({ height: 1024, modelKey: model.key, width: 1024 });
    expect(setSettings).toHaveBeenCalledWith(
      expect.objectContaining({ height: 1024, model, modelKey: model.key, width: 1024 }),
      'project-empty'
    );
  });
});
