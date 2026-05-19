import { describe, expect, it, vi } from 'vitest';

import { getDetailerSettingsValueParts, ImageMetadataHandlers, MetadataUtils } from './parsing';

const createMockStore = () => ({
  dispatch: vi.fn(),
  getState: vi.fn(() => ({
    params: { model: null },
  })),
});

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const createStore = () => createMockStore() as any;

describe('Qwen metadata parsing', () => {
  it('does not report missing Qwen metadata keys as available', async () => {
    const store = createStore();

    const hasMetadata = await MetadataUtils.hasMetadataByHandlers({
      metadata: {},
      handlers: [
        ImageMetadataHandlers.QwenImageComponentSource,
        ImageMetadataHandlers.QwenImageQuantization,
        ImageMetadataHandlers.QwenImageShift,
      ],
      store,
      require: 'all',
    });

    // Handlers reject when keys are absent, so hasMetadata should be false
    expect(hasMetadata).toBe(false);
  });

  it('does not recall Qwen values when metadata keys are absent', async () => {
    const store = createStore();

    const recalled = await MetadataUtils.recallByHandlers({
      metadata: {},
      handlers: [
        ImageMetadataHandlers.QwenImageComponentSource,
        ImageMetadataHandlers.QwenImageQuantization,
        ImageMetadataHandlers.QwenImageShift,
      ],
      store,
      silent: true,
    });

    // No keys present → handlers reject → 0 recalls, no dispatches
    expect(recalled.size).toBe(0);
    const mockStore = store as ReturnType<typeof createMockStore>;
    expect(mockStore.dispatch).not.toHaveBeenCalled();
  });

  it('recalls Qwen handlers with actual values when metadata keys are present', async () => {
    const store = createStore();

    const recalled = await MetadataUtils.recallByHandlers({
      metadata: {
        qwen_image_component_source: { key: 'test-key', hash: 'test', name: 'Test', base: 'qwen-image', type: 'main' },
        qwen_image_quantization: 'nf4',
        qwen_image_shift: 3.0,
      },
      handlers: [
        ImageMetadataHandlers.QwenImageComponentSource,
        ImageMetadataHandlers.QwenImageQuantization,
        ImageMetadataHandlers.QwenImageShift,
      ],
      store,
      silent: true,
    });

    expect(recalled.size).toBe(3);
    const mockStore = store as ReturnType<typeof createMockStore>;
    expect(mockStore.dispatch).toHaveBeenCalledTimes(3);
  });

  it('recalls standalone Qwen Image VAE and Qwen VL encoder when metadata keys are present', async () => {
    const store = createStore();

    const recalled = await MetadataUtils.recallByHandlers({
      metadata: {
        qwen_image_vae: { key: 'vae-key', hash: 'vae-hash', name: 'Qwen VAE', base: 'qwen-image', type: 'vae' },
        qwen_image_qwen_vl_encoder: {
          key: 'enc-key',
          hash: 'enc-hash',
          name: 'Qwen VL Encoder',
          base: 'qwen-image',
          type: 'qwen_vl_encoder',
        },
      },
      handlers: [ImageMetadataHandlers.QwenImageVaeModel, ImageMetadataHandlers.QwenImageQwenVLEncoderModel],
      store,
      silent: true,
    });

    expect(recalled.size).toBe(2);
    const mockStore = store as ReturnType<typeof createMockStore>;
    expect(mockStore.dispatch).toHaveBeenCalledTimes(2);
  });

  it('does not recall standalone Qwen Image VAE/encoder when keys are absent', async () => {
    const store = createStore();

    const recalled = await MetadataUtils.recallByHandlers({
      metadata: {},
      handlers: [ImageMetadataHandlers.QwenImageVaeModel, ImageMetadataHandlers.QwenImageQwenVLEncoderModel],
      store,
      silent: true,
    });

    expect(recalled.size).toBe(0);
    const mockStore = store as ReturnType<typeof createMockStore>;
    expect(mockStore.dispatch).not.toHaveBeenCalled();
  });

  it('recalls Qwen component source as null when key is present but value is null', async () => {
    const store = createStore();

    const recalled = await MetadataUtils.recallByHandlers({
      metadata: {
        qwen_image_component_source: null,
      },
      handlers: [ImageMetadataHandlers.QwenImageComponentSource],
      store,
      silent: true,
    });

    // Key is present with null value → handler resolves with null → 1 recall
    expect(recalled.size).toBe(1);
    const mockStore = store as ReturnType<typeof createMockStore>;
    expect(mockStore.dispatch).toHaveBeenCalledTimes(1);
  });
});

describe('Detailer metadata parsing', () => {
  it('builds localized Detailer metadata display parts', () => {
    expect(
      getDetailerSettingsValueParts({
        enabled: true,
        targetPrompt: 'face',
        quality: 'high',
        detector: 'grounding-dino-sam',
      })
    ).toEqual([
      { type: 'i18n', key: 'common.on' },
      { type: 'text', value: 'face' },
      { type: 'i18n', key: 'parameters.faceDetailer.qualities.high' },
      { type: 'i18n', key: 'parameters.faceDetailer.detectors.groundedSam' },
    ]);

    expect(
      getDetailerSettingsValueParts({
        enabled: false,
        detector: 'mediapipe',
      })
    ).toEqual([
      { type: 'i18n', key: 'common.off' },
      { type: 'i18n', key: 'parameters.faceDetailer.detectors.mediapipe' },
    ]);
  });

  it('does not recall Detailer settings when detailer_enabled metadata is absent', async () => {
    const store = createStore();

    const recalled = await MetadataUtils.recallByHandlers({
      metadata: {
        detailer_quality: 'high',
        detailer_target_prompt: 'face',
      },
      handlers: [ImageMetadataHandlers.DetailerSettings],
      store,
      silent: true,
    });

    expect(recalled.size).toBe(0);
    const mockStore = store as ReturnType<typeof createMockStore>;
    expect(mockStore.dispatch).not.toHaveBeenCalled();
  });

  it('recalls stored detailer params instead of Body effective runtime values', async () => {
    const store = createStore();

    const recalled = await MetadataUtils.recallByHandlers({
      metadata: {
        detailer_enabled: true,
        detailer_detector: 'grounding-dino-sam',
        detailer_quality: 'balanced',
        detailer_target_profile: 'person',
        detailer_target_prompt: 'person',
        detailer_dino_model: 'grounding-dino-base',
        detailer_sam_model: 'segment-anything-2-large',
        detailer_detection_threshold: 0.3,
        detailer_target_size: 1024,
        detailer_param_target_size: 640,
        detailer_max_upscale: 8,
        detailer_param_max_upscale: 3,
        detailer_max_process_size: 1024,
        detailer_param_max_process_size: 896,
        detailer_denoise_mask_expand: 0,
        detailer_param_denoise_mask_expand: 10,
        detailer_denoise_mask_feather: 2,
        detailer_param_denoise_mask_feather: 8,
        detailer_paste_mask_expand: 0,
        detailer_param_paste_mask_expand: 2,
        detailer_paste_mask_feather: 4,
        detailer_param_paste_mask_feather: 12,
        detailer_strength: 0.14,
        detailer_param_strength: 0.4,
        detailer_steps: 14,
        detailer_param_steps: 22,
        detailer_cfg_scale: 4.5,
        detailer_param_cfg_scale: 9,
        detailer_debug_enabled: true,
      },
      handlers: [ImageMetadataHandlers.DetailerSettings],
      store,
      silent: true,
    });

    expect(recalled.size).toBe(1);
    const mockStore = store as ReturnType<typeof createMockStore>;
    expect(mockStore.dispatch).toHaveBeenCalledWith(expect.objectContaining({ type: 'params/setDetailerEnabled' }));
    expect(mockStore.dispatch).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'params/setDetailerTargetSize', payload: 640 })
    );
    expect(mockStore.dispatch).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'params/setDetailerMaxUpscale', payload: 3 })
    );
    expect(mockStore.dispatch).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'params/setDetailerMaxProcessSize', payload: 896 })
    );
    expect(mockStore.dispatch).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'params/setDetailerStrength', payload: 0.4 })
    );
    expect(mockStore.dispatch).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'params/setDetailerSteps', payload: 22 })
    );
    expect(mockStore.dispatch).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'params/setDetailerCfgScale', payload: 9 })
    );
    expect(mockStore.dispatch).not.toHaveBeenCalledWith(
      expect.objectContaining({ type: 'params/setDetailerDebugEnabled' })
    );
  });

  it('does not recall legacy Body effective values into visible sliders', async () => {
    const store = createStore();

    const recalled = await MetadataUtils.recallByHandlers({
      metadata: {
        detailer_enabled: true,
        detailer_quality: 'balanced',
        detailer_target_profile: 'person',
        detailer_target_prompt: 'person',
        detailer_target_size: 1024,
        detailer_max_process_size: 1024,
        detailer_strength: 0.14,
        detailer_steps: 14,
        detailer_cfg_scale: 4.5,
      },
      handlers: [ImageMetadataHandlers.DetailerSettings],
      store,
      silent: true,
    });

    expect(recalled.size).toBe(1);
    const mockStore = store as ReturnType<typeof createMockStore>;
    expect(mockStore.dispatch).not.toHaveBeenCalledWith(
      expect.objectContaining({ type: 'params/setDetailerTargetSize' })
    );
    expect(mockStore.dispatch).not.toHaveBeenCalledWith(
      expect.objectContaining({ type: 'params/setDetailerMaxProcessSize' })
    );
    expect(mockStore.dispatch).not.toHaveBeenCalledWith(
      expect.objectContaining({ type: 'params/setDetailerStrength' })
    );
    expect(mockStore.dispatch).not.toHaveBeenCalledWith(expect.objectContaining({ type: 'params/setDetailerSteps' }));
    expect(mockStore.dispatch).not.toHaveBeenCalledWith(
      expect.objectContaining({ type: 'params/setDetailerCfgScale' })
    );
  });
});
