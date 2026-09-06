import type { AppStore } from 'app/store/store';
import {
  setHiDiffusionAutoRatios,
  setHiDiffusionEnabled,
  setHiDiffusionT1Ratio,
  setHiDiffusionT2Ratio,
} from 'features/controlLayers/store/paramsSlice';
import { describe, expect, it, vi } from 'vitest';

import { ImageMetadataHandlers, MetadataUtils, parseMetadataHandler } from './parsing';

const createMockStore = () => ({
  dispatch: vi.fn(),
  getState: vi.fn(() => ({
    params: { model: null },
  })),
});

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const createStore = () => createMockStore() as any;

describe('Qwen metadata parsing', () => {
  it('normalizes synchronous parser throws into rejected promises', async () => {
    const store = createStore();
    let result: Promise<unknown> | undefined;

    expect(() => {
      result = parseMetadataHandler({}, ImageMetadataHandlers.RefinerSteps, store);
    }).not.toThrow();

    await expect(result).rejects.toThrow();
  });

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

  it('does not report metadata as available when require some and all handlers reject', async () => {
    const store = createStore();

    const hasMetadata = await MetadataUtils.hasMetadataByHandlers({
      metadata: {},
      handlers: [
        ImageMetadataHandlers.QwenImageComponentSource,
        ImageMetadataHandlers.QwenImageQuantization,
        ImageMetadataHandlers.QwenImageShift,
      ],
      store,
      require: 'some',
    });

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

describe('HiDiffusion metadata parsing', () => {
  it('recalls null ratios as automatic thresholds', async () => {
    const store = createStore();
    const metadata = { hidiffusion_t1_ratio: null, hidiffusion_t2_ratio: null };

    const t1 = await parseMetadataHandler(metadata, ImageMetadataHandlers.HiDiffusionT1Ratio, store);
    const t2 = await parseMetadataHandler(metadata, ImageMetadataHandlers.HiDiffusionT2Ratio, store);
    ImageMetadataHandlers.HiDiffusionT1Ratio.recall(t1, store);
    ImageMetadataHandlers.HiDiffusionT2Ratio.recall(t2, store);

    expect(store.dispatch).toHaveBeenCalledWith(setHiDiffusionAutoRatios(true));
    expect(store.dispatch).not.toHaveBeenCalledWith(setHiDiffusionT1Ratio(expect.anything()));
    expect(store.dispatch).not.toHaveBeenCalledWith(setHiDiffusionT2Ratio(expect.anything()));
  });

  it('recalls numeric ratios as manual thresholds', async () => {
    const store = createStore();
    const metadata = { hidiffusion_t1_ratio: 0.65, hidiffusion_t2_ratio: 0.2 };

    const t1 = await parseMetadataHandler(metadata, ImageMetadataHandlers.HiDiffusionT1Ratio, store);
    const t2 = await parseMetadataHandler(metadata, ImageMetadataHandlers.HiDiffusionT2Ratio, store);
    ImageMetadataHandlers.HiDiffusionT1Ratio.recall(t1, store);
    ImageMetadataHandlers.HiDiffusionT2Ratio.recall(t2, store);

    expect(store.dispatch).toHaveBeenCalledWith(setHiDiffusionAutoRatios(false));
    expect(store.dispatch).toHaveBeenCalledWith(setHiDiffusionT1Ratio(0.65));
    expect(store.dispatch).toHaveBeenCalledWith(setHiDiffusionT2Ratio(0.2));
  });

  it('disables HiDiffusion when recalling all metadata from an older image', async () => {
    let hiDiffusionEnabled = true;
    let hiDiffusionAutoRatios = false;
    let hiDiffusionT1Ratio = 0.8;
    let hiDiffusionT2Ratio = 0.6;
    const store = {
      dispatch: vi.fn((action) => {
        if (action.type === setHiDiffusionEnabled.type) {
          hiDiffusionEnabled = action.payload;
        } else if (action.type === setHiDiffusionAutoRatios.type) {
          hiDiffusionAutoRatios = action.payload;
        } else if (action.type === setHiDiffusionT1Ratio.type) {
          hiDiffusionT1Ratio = action.payload;
        } else if (action.type === setHiDiffusionT2Ratio.type) {
          hiDiffusionT2Ratio = action.payload;
        }
        return action;
      }),
      getState: vi.fn(() => ({
        params: { model: null },
      })),
    } as unknown as AppStore;

    await MetadataUtils.recallAllImageMetadata(
      {
        generation_mode: 'txt2img',
        width: 512,
        height: 512,
        steps: 20,
        cfg_scale: 7.5,
        scheduler: 'euler',
        positive_prompt: 'an older image',
        negative_prompt: '',
      },
      store
    );

    expect(store.dispatch).toHaveBeenCalledWith(setHiDiffusionEnabled(false));
    expect(hiDiffusionEnabled).toBe(false);
    expect(hiDiffusionAutoRatios).toBe(true);
    expect(hiDiffusionT1Ratio).toBe(0.8);
    expect(hiDiffusionT2Ratio).toBe(0.6);
  });
});
