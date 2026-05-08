import { describe, expect, it, vi } from 'vitest';

import { ImageMetadataHandlers, MetadataUtils } from './parsing';

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
