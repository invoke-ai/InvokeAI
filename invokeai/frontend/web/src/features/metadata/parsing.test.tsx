import type { AppStore } from 'app/store/store';
import type * as paramsSliceModule from 'features/controlLayers/store/paramsSlice';
import { ImageMetadataHandlers } from 'features/metadata/parsing';
import type * as modelsApiModule from 'services/api/endpoints/models';
import { beforeEach, describe, expect, it, vi } from 'vitest';

// ---------------------------------------------------------------------------
// Module mocks
//
// We are testing only the *gating* logic of the model-related metadata
// handlers (`VAEModel`, `KleinVAEModel`, `KleinQwen3EncoderModel`). The actual
// model lookup goes through `parseModelIdentifier`, which dispatches RTK
// Query thunks. We stub the models endpoint so that any lookup resolves to a
// canned model identifier — the parse step then succeeds and the assertions
// inside each handler become observable.
// ---------------------------------------------------------------------------

let currentBase: string | null = 'flux2';

vi.mock('features/controlLayers/store/paramsSlice', async (importOriginal) => {
  const mod = await importOriginal<typeof paramsSliceModule>();
  return { ...mod, selectBase: () => currentBase };
});

const fakeModel = (type: 'vae' | 'qwen3_encoder', base: string) => ({
  key: `${type}-key`,
  hash: 'hash',
  name: `Some ${type}`,
  base,
  type,
});

let nextResolved: ReturnType<typeof fakeModel> = fakeModel('vae', 'flux2');

vi.mock('services/api/endpoints/models', async (importOriginal) => {
  const mod = await importOriginal<typeof modelsApiModule>();
  return {
    ...mod,
    modelsApi: {
      ...mod.modelsApi,
      endpoints: {
        ...mod.modelsApi.endpoints,
        getModelConfig: { initiate: (key: string) => ({ type: 'rtkq/initiate', key }) },
      },
    },
  };
});

const makeStore = (): AppStore =>
  ({
    dispatch: vi.fn(() => ({
      unwrap: () => Promise.resolve(nextResolved),
    })),
    getState: () => ({}),
  }) as unknown as AppStore;

beforeEach(() => {
  currentBase = 'flux2';
  nextResolved = fakeModel('vae', 'flux2');
});

describe('ImageMetadataHandlers — Klein recall gating', () => {
  describe('KleinVAEModel', () => {
    it('parses metadata.vae when the current main model is FLUX.2 Klein', async () => {
      currentBase = 'flux2';
      nextResolved = fakeModel('vae', 'flux2');
      const store = makeStore();

      const parsed = await ImageMetadataHandlers.KleinVAEModel.parse({ vae: nextResolved }, store);

      expect(parsed.key).toBe('vae-key');
      expect(parsed.type).toBe('vae');
    });

    it('rejects parsing when the current main model is not FLUX.2 Klein', async () => {
      currentBase = 'sdxl';
      nextResolved = fakeModel('vae', 'flux2');
      const store = makeStore();

      await expect(ImageMetadataHandlers.KleinVAEModel.parse({ vae: nextResolved }, store)).rejects.toThrow();
    });
  });

  describe('KleinQwen3EncoderModel', () => {
    it('parses metadata.qwen3_encoder when the current main model is FLUX.2 Klein', async () => {
      currentBase = 'flux2';
      nextResolved = fakeModel('qwen3_encoder', 'flux2');
      const store = makeStore();

      const parsed = await ImageMetadataHandlers.KleinQwen3EncoderModel.parse(
        { qwen3_encoder: nextResolved },
        store
      );

      expect(parsed.key).toBe('qwen3_encoder-key');
      expect(parsed.type).toBe('qwen3_encoder');
    });

    it('rejects parsing when the current main model is not FLUX.2 Klein', async () => {
      currentBase = 'sdxl';
      nextResolved = fakeModel('qwen3_encoder', 'flux2');
      const store = makeStore();

      await expect(
        ImageMetadataHandlers.KleinQwen3EncoderModel.parse({ qwen3_encoder: nextResolved }, store)
      ).rejects.toThrow();
    });
  });

  describe('VAEModel (generic)', () => {
    // The generic VAEModel handler must NOT also fire for FLUX.2 / Z-Image
    // images, otherwise the metadata viewer renders duplicate VAE rows next
    // to the dedicated KleinVAEModel / ZImageVAEModel handlers.
    it.each(['flux2', 'z-image'])('rejects parsing when current base is %s', async (base) => {
      currentBase = base;
      nextResolved = fakeModel('vae', base);
      const store = makeStore();

      await expect(ImageMetadataHandlers.VAEModel.parse({ vae: nextResolved }, store)).rejects.toThrow();
    });

    it('parses successfully for non-Klein, non-Z-Image bases', async () => {
      currentBase = 'sdxl';
      nextResolved = fakeModel('vae', 'sdxl');
      const store = makeStore();

      const parsed = await ImageMetadataHandlers.VAEModel.parse({ vae: nextResolved }, store);
      expect(parsed.key).toBe('vae-key');
    });
  });
});
