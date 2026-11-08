import type {
  GenerateModelConfig,
  GenerateSettings,
  LoraModelConfig,
  VaeModelConfig,
} from '@features/generation/contracts';

import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { getDefaultGenerateSettings } from '@features/generation/settings';
import { describe, expect, it, vi } from 'vitest';

import { recallProjectPromptHistoryItem, selectProjectGenerateModel } from './generationSettingsOrchestration';

const createModel = (base: string, key = `${base}-model`): GenerateModelConfig => ({
  base,
  key,
  name: key,
  type: 'main',
});

const createSettings = (model: GenerateModelConfig, overrides: Partial<GenerateSettings> = {}): GenerateSettings => ({
  ...getDefaultGenerateSettings(model),
  seed: 7,
  seedMode: 'fixed',
  ...overrides,
});

const sdxlLora: LoraModelConfig = { base: 'sdxl', key: 'sdxl-lora', name: 'SDXL LoRA', type: 'lora' };
const sdxlVae: VaeModelConfig = { base: 'sdxl', key: 'sdxl-vae', name: 'SDXL VAE', type: 'vae' };

seedArchitectureCapabilities();

describe('selectProjectGenerateModel', () => {
  it('stores a switch that loses nothing at once, with a revert patch for its Undo', () => {
    const currentModel = createModel('sdxl');
    const model = createModel('flux');
    const setSettings = vi.fn();
    const loras = [{ isEnabled: true, model: sdxlLora, weight: 1 }];
    const currentValues = {
      ...createSettings(currentModel, { batchCount: 3, loras, positivePrompt: 'a lighthouse' }),
      model: currentModel,
    };

    const selection = selectProjectGenerateModel({
      applyModelDefaults: false,
      isCanvasFrame: false,
      currentValues,
      generation: { setSettings },
      model,
      models: [currentModel, model, sdxlLora],
      projectId: 'project-1',
    });

    expect(setSettings).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({ batchCount: 3, loras, model, modelKey: model.key, positivePrompt: 'a lighthouse' }),
      'project-1'
    );
    expect(selection).toMatchObject({ result: { changes: [{ count: 1, kind: 'turnedOff', subject: 'concepts' }] } });
    expect(selection.status === 'switched' && selection.revertPatch).toMatchObject({
      model: currentModel,
      modelKey: currentModel.key,
    });
    expect(selection.status === 'switched' && selection.revertPatch).not.toHaveProperty('positivePrompt');
  });

  it("applies the new model's defaults when asked, and its revert patch restores the values they replaced", () => {
    const currentModel = createModel('sdxl');
    const model: GenerateModelConfig = {
      base: 'flux',
      default_settings: { guidance: 3.5, scheduler: 'heun', steps: 20 },
      key: 'flux-model',
      name: 'flux-model',
      type: 'main',
    };
    const setSettings = vi.fn();

    const selection = selectProjectGenerateModel({
      applyModelDefaults: true,
      isCanvasFrame: false,
      currentValues: { ...createSettings(currentModel, { scheduler: 'euler', steps: 42 }), model: currentModel },
      generation: { setSettings },
      model,
      models: [currentModel, model],
      projectId: 'project-1',
    });

    expect(setSettings).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({ cfgScale: 3.5, modelKey: model.key, scheduler: 'heun', steps: 20 }),
      'project-1'
    );
    expect(selection.status === 'switched' && selection.revertPatch).toMatchObject({ scheduler: 'euler', steps: 42 });
  });

  it('fills the only fitting encoder and VAE on a switch, and its revert patch empties them again', () => {
    const currentModel = createModel('sdxl');
    const model: GenerateModelConfig = {
      base: 'flux2',
      format: 'gguf_quantized',
      key: 'klein-9b',
      name: 'Klein 9B',
      type: 'main',
      variant: 'klein_9b',
    };
    const encoder = { base: 'any', key: 'qwen3-8b', name: 'Qwen3 8B', type: 'qwen3_encoder', variant: 'qwen3_8b' };
    const vae: VaeModelConfig = { base: 'flux2', key: 'flux2-vae', name: 'FLUX.2 VAE', type: 'vae' };
    const setSettings = vi.fn();

    const selection = selectProjectGenerateModel({
      applyModelDefaults: false,
      isCanvasFrame: false,
      currentValues: { ...createSettings(currentModel), model: currentModel },
      generation: { setSettings },
      model,
      models: [currentModel, model, encoder, vae],
      projectId: 'project-1',
    });

    expect(setSettings).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({ qwen3EncoderModel: encoder, vae }),
      'project-1'
    );
    expect(selection.status === 'switched' && selection.revertPatch).toMatchObject({
      qwen3EncoderModel: null,
      vae: null,
    });
  });

  it("fills required components on a project's first pick too", () => {
    const model: GenerateModelConfig = {
      base: 'flux2',
      format: 'gguf_quantized',
      key: 'klein-9b',
      name: 'Klein 9B',
      type: 'main',
      variant: 'klein_9b',
    };
    const encoder = { base: 'any', key: 'qwen3-8b', name: 'Qwen3 8B', type: 'qwen3_encoder', variant: 'qwen3_8b' };

    const selection = selectProjectGenerateModel({
      applyModelDefaults: false,
      isCanvasFrame: false,
      currentValues: {},
      generation: { setSettings: vi.fn() },
      model,
      models: [model, encoder],
      projectId: 'project-empty',
    });

    expect(selection).toMatchObject({ result: { settings: { qwen3EncoderModel: encoder } }, status: 'switched' });
  });

  it("keeps a Canvas frame's size when it applies the new model's defaults", () => {
    const currentModel = createModel('sdxl');
    const model: GenerateModelConfig = {
      base: 'sd-1',
      default_settings: { height: 512, steps: 25, width: 512 },
      key: 'sd1-model',
      name: 'sd1-model',
      type: 'main',
    };
    const setSettings = vi.fn();

    selectProjectGenerateModel({
      applyModelDefaults: true,
      isCanvasFrame: true,
      currentValues: { ...createSettings(currentModel, { height: 768, width: 1344 }), model: currentModel },
      generation: { setSettings },
      model,
      models: [currentModel, model],
      projectId: 'project-1',
    });

    expect(setSettings).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({ height: 768, steps: 25, width: 1344 }),
      'project-1'
    );
  });

  it('leaves a switch that removes something to the model card to confirm', () => {
    const currentModel = createModel('sdxl');
    const setSettings = vi.fn();

    const selection = selectProjectGenerateModel({
      applyModelDefaults: false,
      isCanvasFrame: false,
      currentValues: { ...createSettings(currentModel, { vae: sdxlVae }), model: currentModel },
      generation: { setSettings },
      model: createModel('flux'),
      models: [currentModel, sdxlVae],
      projectId: 'project-1',
    });

    expect(selection).toMatchObject({ status: 'needsConfirmation' });
    expect(selection.status === 'needsConfirmation' && selection.result.changes).toContainEqual({
      kind: 'removed',
      subject: 'vae',
    });
    expect(setSettings).not.toHaveBeenCalled();
  });

  it('writes nothing when the project already uses the model, so no pointless Undo is offered', () => {
    const model = createModel('sdxl');
    const setSettings = vi.fn();

    expect(
      selectProjectGenerateModel({
        applyModelDefaults: false,
        isCanvasFrame: false,
        currentValues: { ...createSettings(model), model },
        generation: { setSettings },
        model,
        models: [model],
        projectId: 'project-1',
      })
    ).toEqual({ status: 'unchanged' });
    expect(setSettings).not.toHaveBeenCalled();
  });

  it('initializes complete settings when the project has no Generate values', () => {
    const model = createModel('sdxl');
    const setSettings = vi.fn();

    const selection = selectProjectGenerateModel({
      applyModelDefaults: false,
      isCanvasFrame: false,
      currentValues: {},
      generation: { setSettings },
      model,
      models: [model],
      projectId: 'project-empty',
    });

    expect(selection).toMatchObject({
      result: { settings: { height: 1024, modelKey: model.key, width: 1024 } },
      revertPatch: null,
      status: 'switched',
    });
    expect(setSettings).toHaveBeenCalledWith(
      expect.objectContaining({ height: 1024, model, modelKey: model.key, width: 1024 }),
      'project-empty'
    );
  });
});

describe('recallProjectPromptHistoryItem', () => {
  it('patches the selected project once when current settings are valid', () => {
    const model = createModel('sdxl');
    const patchSettings = vi.fn();

    const didRecall = recallProjectPromptHistoryItem({
      currentValues: createSettings(model),
      generation: { patchSettings },
      item: { negativePrompt: 'low quality', positivePrompt: 'a lighthouse' },
      models: [model],
      projectId: 'project-recall',
    });

    expect(didRecall).toBe(true);
    expect(patchSettings).toHaveBeenCalledOnce();
    expect(patchSettings).toHaveBeenCalledWith(
      { negativePrompt: 'low quality', negativePromptEnabled: true, positivePrompt: 'a lighthouse' },
      'project-recall'
    );
  });

  it('preserves model-aware negative-prompt policy', () => {
    const model = createModel('flux');
    const patchSettings = vi.fn();

    recallProjectPromptHistoryItem({
      currentValues: createSettings(model, { negativePrompt: 'keep hidden value' }),
      generation: { patchSettings },
      item: { negativePrompt: 'history negative', positivePrompt: 'recalled prompt' },
      models: [model],
      projectId: 'project-flux',
    });

    expect(patchSettings).toHaveBeenCalledWith({ positivePrompt: 'recalled prompt' }, 'project-flux');
  });

  it('does not mutate invalid Generate state', () => {
    const patchSettings = vi.fn();

    const didRecall = recallProjectPromptHistoryItem({
      currentValues: { invalid: true },
      generation: { patchSettings },
      item: { negativePrompt: null, positivePrompt: 'recalled prompt' },
      models: undefined,
      projectId: 'project-invalid',
    });

    expect(didRecall).toBe(false);
    expect(patchSettings).not.toHaveBeenCalled();
  });
});
