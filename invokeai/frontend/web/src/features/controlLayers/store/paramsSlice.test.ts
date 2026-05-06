import type {
  ExternalApiModelConfig,
  ExternalApiModelDefaultSettings,
  ExternalImageSize,
  ExternalModelCapabilities,
  ExternalModelPanelSchema,
} from 'services/api/types';
import { describe, expect, it } from 'vitest';

import {
  modelChanged,
  paramsSliceConfig,
  selectHrfFinalDimensions,
  selectModelSupportsDimensions,
  selectModelSupportsGuidance,
  selectModelSupportsHrf,
  selectModelSupportsHrfUpscaleModel,
  selectModelSupportsNegativePrompt,
  selectModelSupportsRefImages,
  selectModelSupportsSeed,
  selectModelSupportsSteps,
  setHrfMethod,
} from './paramsSlice';
import { getInitialParamsState, type ParamsState } from './types';

const buildExternalModelIdentifier = (config: ExternalApiModelConfig) =>
  ({
    key: config.key,
    hash: config.hash,
    name: config.name,
    base: config.base,
    type: config.type,
  }) as const;

const createExternalConfig = (
  capabilities: ExternalModelCapabilities,
  panelSchema?: ExternalModelPanelSchema
): ExternalApiModelConfig => {
  const maxImageSize: ExternalImageSize = { width: 1024, height: 1024 };
  const defaultSettings: ExternalApiModelDefaultSettings = { width: 1024, height: 1024 };

  return {
    key: 'external-test',
    hash: 'external:openai:gpt-image-1',
    path: 'external://openai/gpt-image-1',
    file_size: 0,
    name: 'External Test',
    description: null,
    source: 'external://openai/gpt-image-1',
    source_type: 'url',
    source_api_response: null,
    cover_image: null,
    base: 'external',
    type: 'external_image_generator',
    format: 'external_api',
    provider_id: 'openai',
    provider_model_id: 'gpt-image-1',
    capabilities: { ...capabilities, max_image_size: maxImageSize },
    default_settings: defaultSettings,
    panel_schema: panelSchema,
    tags: ['external'],
    is_default: false,
  };
};

describe('paramsSlice selectors for external models', () => {
  it('returns false for negative prompt support on external models', () => {
    const config = createExternalConfig({
      modes: ['txt2img'],
      supports_reference_images: false,
    });
    const model = buildExternalModelIdentifier(config);

    expect(selectModelSupportsNegativePrompt.resultFunc(model)).toBe(false);
  });

  it('uses external capabilities for ref image support', () => {
    const config = createExternalConfig({
      modes: ['txt2img'],
      supports_reference_images: false,
    });
    const model = buildExternalModelIdentifier(config);

    expect(selectModelSupportsRefImages.resultFunc(model, config)).toBe(false);
  });

  it('returns false for guidance support on external models', () => {
    const config = createExternalConfig({
      modes: ['txt2img'],
      supports_reference_images: false,
    });
    const model = buildExternalModelIdentifier(config);

    expect(selectModelSupportsGuidance.resultFunc(model)).toBe(false);
  });

  it('uses external capabilities for seed support', () => {
    const config = createExternalConfig({
      modes: ['txt2img'],
      supports_reference_images: false,
      supports_seed: false,
    });
    const model = buildExternalModelIdentifier(config);

    expect(selectModelSupportsSeed.resultFunc(model, config)).toBe(false);
  });

  it('returns false for steps support on external models', () => {
    const config = createExternalConfig({
      modes: ['txt2img'],
      supports_reference_images: false,
    });
    const model = buildExternalModelIdentifier(config);

    expect(selectModelSupportsSteps.resultFunc(model)).toBe(false);
  });

  it('prefers panel schema over capabilities for control visibility', () => {
    const config = createExternalConfig(
      {
        modes: ['txt2img'],
        supports_reference_images: true,
        supports_seed: true,
      },
      {
        prompts: [{ name: 'reference_images' }],
        image: [{ name: 'dimensions' }],
        generation: [],
      }
    );
    const model = buildExternalModelIdentifier(config);

    expect(selectModelSupportsNegativePrompt.resultFunc(model)).toBe(false);
    expect(selectModelSupportsRefImages.resultFunc(model, config)).toBe(true);
    expect(selectModelSupportsGuidance.resultFunc(model)).toBe(false);
    expect(selectModelSupportsSeed.resultFunc(model, config)).toBe(false);
    expect(selectModelSupportsSteps.resultFunc(model)).toBe(false);
    expect(selectModelSupportsDimensions.resultFunc(model, config)).toBe(true);
  });

  it('supports HRF only for SD1.5 and SDXL models', () => {
    expect(
      selectModelSupportsHrf.resultFunc({
        key: 'sd1',
        hash: 'h',
        name: 'SD1',
        base: 'sd-1',
        type: 'main',
      })
    ).toBe(true);
    expect(
      selectModelSupportsHrf.resultFunc({
        key: 'sdxl',
        hash: 'h',
        name: 'SDXL',
        base: 'sdxl',
        type: 'main',
      })
    ).toBe(true);

    const unsupportedBases = ['sd-2', 'sd-3', 'flux', 'flux2', 'anima', 'cogview4', 'qwen-image', 'z-image'] as const;
    unsupportedBases.forEach((base) => {
      expect(
        selectModelSupportsHrf.resultFunc({
          key: base,
          hash: 'h',
          name: base,
          base,
          type: 'main',
        })
      ).toBe(false);
    });

    const config = createExternalConfig({ modes: ['txt2img'], supports_reference_images: false });
    expect(selectModelSupportsHrf.resultFunc(buildExternalModelIdentifier(config))).toBe(false);
  });
});

describe('paramsSlice HRF selectors', () => {
  it('rounds final dimensions down to the model grid', () => {
    expect(selectHrfFinalDimensions.resultFunc(513, 512, 1.5, 'flux')).toEqual({ width: 768, height: 768 });
  });

  it('supports upscale-model HRF only for SD1.5 and SDXL models', () => {
    expect(
      selectModelSupportsHrfUpscaleModel.resultFunc({
        key: 'sd1',
        hash: 'h',
        name: 'SD1',
        base: 'sd-1',
        type: 'main',
      })
    ).toBe(true);
    expect(
      selectModelSupportsHrfUpscaleModel.resultFunc({
        key: 'sdxl',
        hash: 'h',
        name: 'SDXL',
        base: 'sdxl',
        type: 'main',
      })
    ).toBe(true);
    expect(
      selectModelSupportsHrfUpscaleModel.resultFunc({
        key: 'flux',
        hash: 'h',
        name: 'FLUX',
        base: 'flux',
        type: 'main',
      })
    ).toBe(false);
  });
});

describe('paramsSlice HRF reducers', () => {
  it('clears upscale-model-only refinement overrides when switching to latent HRF', () => {
    const state = getInitialParamsState();
    state.hrfMethod = 'upscale_model';
    state.hrfSteps = 12;
    state.hrfModel = { key: 'hrf-model', hash: 'h', name: 'HRF Model', base: 'sdxl', type: 'main' };
    state.hrfLoraMode = 'dedicated';
    state.hrfLoras = [
      {
        id: 'lora',
        isEnabled: true,
        model: { key: 'lora', hash: 'h', name: 'LoRA', base: 'sdxl', type: 'lora' },
        weight: 0.6,
      },
    ] as ParamsState['hrfLoras'];

    const nextState = paramsSliceConfig.slice.reducer(state, setHrfMethod('latent'));

    expect(nextState.hrfMethod).toBe('latent');
    expect(nextState.hrfSteps).toBeNull();
    expect(nextState.hrfModel).toBeNull();
    expect(nextState.hrfLoraMode).toBe('reuse_generate');
    expect(nextState.hrfLoras).toEqual([]);
  });

  it('filters dedicated HRF LoRAs when the Generate model base changes', () => {
    const state = getInitialParamsState();
    const previousModel = { key: 'sdxl', hash: 'h', name: 'SDXL', base: 'sdxl', type: 'main' } as const;
    const nextModel = { key: 'sd1', hash: 'h', name: 'SD1', base: 'sd-1', type: 'main' } as const;
    state.model = previousModel;
    state.hrfMethod = 'upscale_model';
    state.hrfLoraMode = 'dedicated';
    state.hrfLoras = [
      {
        id: 'sdxl-lora',
        isEnabled: true,
        model: { key: 'sdxl-lora', hash: 'h', name: 'SDXL LoRA', base: 'sdxl', type: 'lora' },
        weight: 0.6,
      },
      {
        id: 'sd1-lora',
        isEnabled: true,
        model: { key: 'sd1-lora', hash: 'h', name: 'SD1 LoRA', base: 'sd-1', type: 'lora' },
        weight: 0.7,
      },
    ] as ParamsState['hrfLoras'];

    const nextState = paramsSliceConfig.slice.reducer(state, modelChanged({ model: nextModel, previousModel }));

    expect(nextState.hrfLoraMode).toBe('dedicated');
    expect(nextState.hrfLoras).toHaveLength(1);
    expect(nextState.hrfLoras[0]?.model.key).toBe('sd1-lora');
  });

  it('preserves HRF settings when switching through an unsupported base and back', () => {
    const state = getInitialParamsState();
    const sdxlModel = { key: 'sdxl', hash: 'h', name: 'SDXL', base: 'sdxl', type: 'main' } as const;
    const externalModel = buildExternalModelIdentifier(
      createExternalConfig({ modes: ['txt2img'], supports_reference_images: false })
    );
    state.model = sdxlModel;
    state.hrfEnabled = true;
    state.hrfMethod = 'upscale_model';
    state.hrfModel = { key: 'hrf-sdxl', hash: 'h', name: 'HRF SDXL', base: 'sdxl', type: 'main' };
    state.hrfTileControlNetModel = {
      key: 'tile-sdxl',
      hash: 'h',
      name: 'Tile SDXL',
      base: 'sdxl',
      type: 'controlnet',
    };
    state.hrfLoraMode = 'dedicated';
    state.hrfLoras = [
      {
        id: 'sdxl-lora',
        isEnabled: true,
        model: { key: 'sdxl-lora', hash: 'h', name: 'SDXL LoRA', base: 'sdxl', type: 'lora' },
        weight: 0.6,
      },
    ] as ParamsState['hrfLoras'];

    const unsupportedState = paramsSliceConfig.slice.reducer(
      state,
      modelChanged({ model: externalModel, previousModel: sdxlModel })
    );
    const returnedState = paramsSliceConfig.slice.reducer(
      unsupportedState,
      modelChanged({ model: sdxlModel, previousModel: externalModel })
    );

    expect(unsupportedState.hrfEnabled).toBe(true);
    expect(unsupportedState.hrfModel?.key).toBe('hrf-sdxl');
    expect(unsupportedState.hrfTileControlNetModel?.key).toBe('tile-sdxl');
    expect(unsupportedState.hrfLoraMode).toBe('dedicated');
    expect(unsupportedState.hrfLoras).toHaveLength(1);
    expect(returnedState.hrfEnabled).toBe(true);
    expect(returnedState.hrfModel?.key).toBe('hrf-sdxl');
    expect(returnedState.hrfTileControlNetModel?.key).toBe('tile-sdxl');
    expect(returnedState.hrfLoraMode).toBe('dedicated');
    expect(returnedState.hrfLoras[0]?.model.key).toBe('sdxl-lora');
  });

  it('cleans preserved HRF settings when returning to a different supported base', () => {
    const state = getInitialParamsState();
    const sdxlModel = { key: 'sdxl', hash: 'h', name: 'SDXL', base: 'sdxl', type: 'main' } as const;
    const sd1Model = { key: 'sd1', hash: 'h', name: 'SD1', base: 'sd-1', type: 'main' } as const;
    const fluxModel = { key: 'flux', hash: 'h', name: 'FLUX', base: 'flux', type: 'main' } as const;
    state.model = sdxlModel;
    state.hrfEnabled = true;
    state.hrfMethod = 'upscale_model';
    state.hrfModel = { key: 'hrf-sdxl', hash: 'h', name: 'HRF SDXL', base: 'sdxl', type: 'main' };
    state.hrfTileControlNetModel = {
      key: 'tile-sdxl',
      hash: 'h',
      name: 'Tile SDXL',
      base: 'sdxl',
      type: 'controlnet',
    };
    state.hrfLoraMode = 'dedicated';
    state.hrfLoras = [
      {
        id: 'sdxl-lora',
        isEnabled: true,
        model: { key: 'sdxl-lora', hash: 'h', name: 'SDXL LoRA', base: 'sdxl', type: 'lora' },
        weight: 0.6,
      },
    ] as ParamsState['hrfLoras'];

    const unsupportedState = paramsSliceConfig.slice.reducer(
      state,
      modelChanged({ model: fluxModel, previousModel: sdxlModel })
    );
    const sd1State = paramsSliceConfig.slice.reducer(
      unsupportedState,
      modelChanged({ model: sd1Model, previousModel: fluxModel })
    );

    expect(unsupportedState.hrfModel?.key).toBe('hrf-sdxl');
    expect(unsupportedState.hrfTileControlNetModel?.key).toBe('tile-sdxl');
    expect(unsupportedState.hrfLoras).toHaveLength(1);
    expect(sd1State.hrfEnabled).toBe(true);
    expect(sd1State.hrfModel).toBeNull();
    expect(sd1State.hrfTileControlNetModel).toBeNull();
    expect(sd1State.hrfLoraMode).toBe('dedicated');
    expect(sd1State.hrfLoras).toEqual([]);
  });
});

describe('paramsSlice HRF migrations', () => {
  it('migrates legacy HRF Structure to explicit Tile Control Weight', () => {
    const v5State = {
      ...getInitialParamsState(),
      _version: 5,
      hrfStructure: 0,
    } as unknown as Record<string, unknown>;
    delete v5State.hrfTileControlWeight;
    delete v5State.hrfSteps;
    delete v5State.hrfModel;
    delete v5State.hrfLoraMode;
    delete v5State.hrfLoras;

    const migrated = paramsSliceConfig.persistConfig!.migrate(v5State);

    expect(migrated).toMatchObject({
      _version: 6,
      hrfTileControlWeight: 0.625,
      hrfSteps: null,
      hrfModel: null,
      hrfLoraMode: 'reuse_generate',
      hrfLoras: [],
    });
    expect('hrfStructure' in migrated).toBe(false);
  });
});
