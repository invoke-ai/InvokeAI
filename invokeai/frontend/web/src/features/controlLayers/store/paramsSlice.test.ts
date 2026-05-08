import type {
  ExternalApiModelConfig,
  ExternalApiModelDefaultSettings,
  ExternalImageSize,
  ExternalModelCapabilities,
  ExternalModelPanelSchema,
} from 'services/api/types';
import { describe, expect, it } from 'vitest';

import { DETAILER_QUALITY_PRESETS, isDetailerQualityPresetAdjusted } from './detailerQualityPresets';
import {
  paramsSliceConfig,
  selectModelSupportsDimensions,
  selectModelSupportsFaceDetailer,
  selectModelSupportsGuidance,
  selectModelSupportsNegativePrompt,
  selectModelSupportsRefImages,
  selectModelSupportsSeed,
  selectModelSupportsSteps,
  setDetailerQuality,
} from './paramsSlice';
import { DEFAULT_FACE_DETAILER_PARAMS, getInitialParamsState } from './types';

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
});

describe('paramsSliceConfig persisted state migration', () => {
  const migrate = paramsSliceConfig.persistConfig?.migrate;

  it('backfills new Qwen Image fields when migrating from v2 and preserves existing params', () => {
    expect(migrate).toBeDefined();

    // Build a valid pre-PR v2 persisted state by removing the fields that were added in v3
    const initial = getInitialParamsState();
    const v2State: Record<string, unknown> = {
      ...initial,
      _version: 2,
      positivePrompt: 'a fluffy cat',
      seed: 42,
      shouldRandomizeSeed: false,
      dimensions: { ...initial.dimensions, width: 768, height: 768 },
    };
    delete v2State.qwenImageVaeModel;
    delete v2State.qwenImageQwenVLEncoderModel;

    const result = migrate?.(v2State) as ReturnType<typeof getInitialParamsState>;

    expect(result._version).toBe(11);
    expect(result.qwenImageVaeModel).toBeNull();
    expect(result.qwenImageQwenVLEncoderModel).toBeNull();
    expect(result).toMatchObject(DEFAULT_FACE_DETAILER_PARAMS);
    // Existing params should be preserved
    expect(result.positivePrompt).toBe('a fluffy cat');
    expect(result.seed).toBe(42);
    expect(result.shouldRandomizeSeed).toBe(false);
    expect(result.dimensions.width).toBe(768);
    expect(result.dimensions.height).toBe(768);
  });

  it('migrates current v3 params with current detailer defaults', () => {
    const v3Params = getInitialParamsState() as unknown as Record<string, unknown>;
    v3Params._version = 3;
    for (const key of Object.keys(DEFAULT_FACE_DETAILER_PARAMS)) {
      delete v3Params[key];
    }

    const migrated = paramsSliceConfig.persistConfig?.migrate(v3Params);

    expect(migrated).toMatchObject({
      _version: 11,
      ...DEFAULT_FACE_DETAILER_PARAMS,
    });
  });

  it('migrates legacy v3 detailer params to current defaults while preserving enabled state', () => {
    const v3Params = getInitialParamsState() as unknown as Record<string, unknown>;
    v3Params._version = 3;
    v3Params.detailerEnabled = true;
    v3Params.detailerStrength = 0.45;
    for (const key of [
      'detailerDetector',
      'detailerQuality',
      'detailerFaceSelection',
      'detailerDinoModel',
      'detailerSamModel',
      'detailerDetectionThreshold',
      'detailerTargetSize',
      'detailerMaxUpscale',
      'detailerMaxProcessSize',
      'detailerCropPadding',
      'detailerMaskExpand',
      'detailerMaskFeather',
    ]) {
      delete v3Params[key];
    }

    const migrated = paramsSliceConfig.persistConfig?.migrate(v3Params);

    expect(migrated).toMatchObject({
      _version: 11,
      ...DEFAULT_FACE_DETAILER_PARAMS,
      detailerEnabled: true,
    });
  });

  it('migrates v5 params to split mask settings and configurable target prompt', () => {
    const v5Params = getInitialParamsState() as unknown as Record<string, unknown>;
    v5Params._version = 5;
    v5Params.detailerEnabled = true;
    v5Params.detailerMaskExpand = 14;
    for (const key of [
      'detailerTargetPrompt',
      'detailerDenoiseMaskExpand',
      'detailerDenoiseMaskFeather',
      'detailerPasteMaskExpand',
      'detailerPasteMaskFeather',
      'detailerUpscaleMethod',
    ]) {
      delete v5Params[key];
    }

    const migrated = paramsSliceConfig.persistConfig?.migrate(v5Params);

    expect(migrated).toMatchObject({
      _version: 11,
      detailerEnabled: true,
      detailerTargetPrompt: 'face',
      detailerDenoiseMaskExpand: 14,
      detailerDenoiseMaskFeather: 6,
      detailerPasteMaskExpand: 0,
      detailerPasteMaskFeather: 6,
      detailerUpscaleMethod: 'pixel_crop_resize',
      detailerColorCorrectMode: 'off',
    });
  });

  it('migrates v6 params to stabilized quality defaults', () => {
    const v6Params = getInitialParamsState() as unknown as Record<string, unknown>;
    v6Params._version = 6;
    v6Params.detailerEnabled = true;
    v6Params.detailerTargetPrompt = 'hands';
    v6Params.detailerTargetSize = 512;
    v6Params.detailerMaxUpscale = 4;
    v6Params.detailerPasteMaskExpand = 2;
    v6Params.detailerPasteMaskFeather = 12;
    delete v6Params.detailerColorCorrectMode;

    const migrated = paramsSliceConfig.persistConfig?.migrate(v6Params);

    expect(migrated).toMatchObject({
      _version: 11,
      detailerEnabled: true,
      detailerTargetPrompt: 'hands',
      detailerTargetSize: 768,
      detailerMaxUpscale: 8,
      detailerMaxProcessSize: 1024,
      detailerStrength: 0.26,
      detailerPasteMaskExpand: 0,
      detailerPasteMaskFeather: 6,
      detailerColorCorrectMode: 'off',
    });
  });

  it('migrates v7 params to opt-in color matching', () => {
    const v7Params = getInitialParamsState() as unknown as Record<string, unknown>;
    v7Params._version = 7;
    v7Params.detailerColorCorrectMode = 'YCbCr-Luma';

    const migrated = paramsSliceConfig.persistConfig?.migrate(v7Params);

    expect(migrated).toMatchObject({
      _version: 11,
      detailerColorCorrectMode: 'off',
    });
  });

  it('preserves explicit v7 non-default color matching during migration', () => {
    const v7Params = getInitialParamsState() as unknown as Record<string, unknown>;
    v7Params._version = 7;
    v7Params.detailerColorCorrectMode = 'RGB';

    const migrated = paramsSliceConfig.persistConfig?.migrate(v7Params);

    expect(migrated).toMatchObject({
      _version: 11,
      detailerColorCorrectMode: 'RGB',
    });
  });

  it('migrates v8 params to add disabled debug output', () => {
    const v8Params = getInitialParamsState() as unknown as Record<string, unknown>;
    v8Params._version = 8;
    delete v8Params.detailerDebugEnabled;

    const migrated = paramsSliceConfig.persistConfig?.migrate(v8Params);

    expect(migrated).toMatchObject({
      _version: 11,
      detailerDebugEnabled: false,
    });
  });

  it('migrates untouched v9 quality preset values to crop-fidelity defaults', () => {
    const v9Params = getInitialParamsState() as unknown as Record<string, unknown>;
    Object.assign(v9Params, {
      _version: 9,
      detailerQuality: 'balanced',
      detailerTargetSize: 768,
      detailerMaxUpscale: 8,
      detailerMaxProcessSize: 768,
      detailerCropPadding: 64,
      detailerStrength: 0.28,
      detailerSteps: 14,
      detailerCfgScale: 7.5,
      detailerDenoiseMaskFeather: 8,
      detailerPasteMaskExpand: 0,
      detailerPasteMaskFeather: 8,
      detailerSamModel: 'segment-anything-2-small',
    });

    const migrated = paramsSliceConfig.persistConfig?.migrate(v9Params);

    expect(migrated).toMatchObject({
      _version: 11,
      detailerQuality: 'balanced',
      detailerTargetSize: 768,
      detailerMaxUpscale: 8,
      detailerMaxProcessSize: 1024,
      detailerCropPadding: 48,
      detailerStrength: 0.26,
      detailerSteps: 14,
      detailerCfgScale: 4.5,
      detailerDenoiseMaskFeather: 6,
      detailerPasteMaskExpand: 0,
      detailerPasteMaskFeather: 6,
      detailerSamModel: 'segment-anything-2-base',
    });
  });

  it('preserves custom v9 quality values during migration', () => {
    const v9Params = getInitialParamsState() as unknown as Record<string, unknown>;
    Object.assign(v9Params, {
      _version: 9,
      detailerQuality: 'balanced',
      detailerTargetSize: 768,
      detailerMaxUpscale: 8,
      detailerMaxProcessSize: 768,
      detailerCropPadding: 64,
      detailerStrength: 0.27,
      detailerSteps: 14,
      detailerCfgScale: 7.5,
      detailerDenoiseMaskFeather: 8,
      detailerPasteMaskExpand: 0,
      detailerPasteMaskFeather: 8,
      detailerSamModel: 'segment-anything-2-large',
    });

    const migrated = paramsSliceConfig.persistConfig?.migrate(v9Params);

    expect(migrated).toMatchObject({
      _version: 11,
      detailerMaxProcessSize: 768,
      detailerCropPadding: 64,
      detailerStrength: 0.27,
      detailerCfgScale: 7.5,
      detailerDenoiseMaskFeather: 8,
      detailerPasteMaskFeather: 8,
      detailerSamModel: 'segment-anything-2-large',
    });
  });

  it('migrates untouched v10 detailer CFG and crop padding defaults', () => {
    const v10Params = getInitialParamsState() as unknown as Record<string, unknown>;
    Object.assign(v10Params, {
      _version: 10,
      detailerQuality: 'high',
      detailerTargetSize: 1024,
      detailerMaxUpscale: 12,
      detailerMaxProcessSize: 1024,
      detailerCropPadding: 64,
      detailerStrength: 0.28,
      detailerSteps: 18,
      detailerCfgScale: 7.5,
      detailerDenoiseMaskFeather: 6,
      detailerPasteMaskExpand: 0,
      detailerPasteMaskFeather: 6,
      detailerSamModel: 'segment-anything-2-large',
    });

    const migrated = paramsSliceConfig.persistConfig?.migrate(v10Params);

    expect(migrated).toMatchObject({
      _version: 11,
      detailerQuality: 'high',
      detailerCropPadding: 32,
      detailerCfgScale: 5,
    });
  });

  it('applies quality preset values consistently', () => {
    const state = paramsSliceConfig.slice.reducer(
      { ...getInitialParamsState(), detailerColorCorrectMode: 'RGB' },
      setDetailerQuality('high')
    );

    expect(state).toMatchObject({
      detailerQuality: 'high',
      detailerTargetSize: 1024,
      detailerMaxUpscale: 12,
      detailerMaxProcessSize: 1024,
      detailerCropPadding: 32,
      detailerStrength: 0.28,
      detailerSteps: 18,
      detailerCfgScale: 5,
      detailerDenoiseMaskFeather: 6,
      detailerPasteMaskExpand: 0,
      detailerPasteMaskFeather: 6,
      detailerSamModel: 'segment-anything-2-large',
      detailerColorCorrectMode: 'RGB',
    });
  });

  it('preserves manual detailer tuning when clicking the selected quality again', () => {
    const state = paramsSliceConfig.slice.reducer(
      {
        ...getInitialParamsState(),
        detailerQuality: 'balanced',
        detailerCfgScale: 3.5,
        detailerCropPadding: 24,
        detailerSamModel: 'segment-anything-2-large',
        detailerStrength: 0.18,
        detailerSteps: 16,
      },
      setDetailerQuality('balanced')
    );

    expect(state).toMatchObject({
      detailerQuality: 'balanced',
      detailerCfgScale: 3.5,
      detailerCropPadding: 24,
      detailerSamModel: 'segment-anything-2-large',
      detailerStrength: 0.18,
      detailerSteps: 16,
    });
  });

  it('detects detailer quality values adjusted away from the selected preset', () => {
    const state = getInitialParamsState();
    const fastState = { ...state, detailerQuality: 'fast' as const, ...DETAILER_QUALITY_PRESETS.fast };
    const highState = { ...state, detailerQuality: 'high' as const, ...DETAILER_QUALITY_PRESETS.high };

    expect(isDetailerQualityPresetAdjusted(state)).toBe(false);
    expect(isDetailerQualityPresetAdjusted(fastState)).toBe(false);
    expect(isDetailerQualityPresetAdjusted(highState)).toBe(false);
    expect(isDetailerQualityPresetAdjusted({ ...state, detailerCfgScale: 3.5 })).toBe(true);
    expect(isDetailerQualityPresetAdjusted({ ...state, detailerCropPadding: 24 })).toBe(true);
    expect(isDetailerQualityPresetAdjusted({ ...state, detailerSamModel: 'segment-anything-2-large' })).toBe(true);
    expect(isDetailerQualityPresetAdjusted({ ...state, detailerStrength: 0.18 })).toBe(true);
    expect(isDetailerQualityPresetAdjusted({ ...state, detailerSteps: 16 })).toBe(true);
  });

  it('supports the face detailer only for SD1, SD2, and SDXL', () => {
    const makeModel = (base: 'sd-1' | 'sd-2' | 'sdxl' | 'flux' | 'external') =>
      ({
        key: `${base}-model`,
        hash: `${base}-hash`,
        name: `${base} model`,
        base,
        type: base === 'external' ? 'external_image_generator' : 'main',
      }) as Parameters<typeof selectModelSupportsFaceDetailer.resultFunc>[0];

    expect(selectModelSupportsFaceDetailer.resultFunc(makeModel('sd-1'))).toBe(true);
    expect(selectModelSupportsFaceDetailer.resultFunc(makeModel('sd-2'))).toBe(true);
    expect(selectModelSupportsFaceDetailer.resultFunc(makeModel('sdxl'))).toBe(true);
    expect(selectModelSupportsFaceDetailer.resultFunc(makeModel('flux'))).toBe(false);
    expect(selectModelSupportsFaceDetailer.resultFunc(makeModel('external'))).toBe(false);
    expect(selectModelSupportsFaceDetailer.resultFunc(null)).toBe(false);
  });
});
