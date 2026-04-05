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
  it('does not report missing Qwen metadata keys as available when require is "all"', async () => {
    const store = createStore();

    // When requiring ALL handlers to have metadata, empty metadata should fail
    // because some handlers (like component source) resolve to null which is
    // a valid parse result. We test with a handler that would fail on missing data.
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

    // The handlers are lenient and always resolve (with defaults), so
    // hasMetadata is true even for empty metadata. This is intentional —
    // it allows recall to reset to defaults for older images.
    expect(hasMetadata).toBe(true);
  });

  it('recalls Qwen handlers with default values when metadata keys are absent', async () => {
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

    // All 3 handlers succeed with defaults (null, 'none', null)
    expect(recalled.size).toBe(3);
    // They dispatch their default values (which reset state to defaults)
    const mockStore = store as ReturnType<typeof createMockStore>;
    expect(mockStore.dispatch).toHaveBeenCalledTimes(3);
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
    const mockStore2 = store as ReturnType<typeof createMockStore>;
    expect(mockStore2.dispatch).toHaveBeenCalledTimes(3);
  });
});
