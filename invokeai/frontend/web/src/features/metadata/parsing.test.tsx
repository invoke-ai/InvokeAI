import type { AppStore } from 'app/store/store';
import type * as paramsSliceModule from 'features/controlLayers/store/paramsSlice';
import { ImageMetadataHandlers, MetadataUtils } from 'features/metadata/parsing';
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

const fakeModel = (type: 'main' | 'vae' | 'qwen3_encoder' | 'lora', base: string) => ({
  key: `${type}-key`,
  hash: 'hash',
  name: `Some ${type}`,
  base,
  type,
});

let nextResolved: ReturnType<typeof fakeModel> = fakeModel('vae', 'flux2');
let resolvedModels: Record<string, ReturnType<typeof fakeModel>> = {};

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
    dispatch: vi.fn((action) => {
      if (action?.type === 'generation/modelSelected') {
        currentBase = action.payload.base;
        return action;
      }
      return {
        unwrap: () => Promise.resolve(resolvedModels[action?.key] ?? nextResolved),
      };
    }),
    getState: () => ({}),
  }) as unknown as AppStore;

beforeEach(() => {
  currentBase = 'flux2';
  nextResolved = fakeModel('vae', 'flux2');
  resolvedModels = {};
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

      const parsed = await ImageMetadataHandlers.KleinQwen3EncoderModel.parse({ qwen3_encoder: nextResolved }, store);

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

  describe('Guidance (legacy FLUX.2 gating)', () => {
    // Prior to the Klein guidance cleanup, FLUX.2 images wrote a `guidance`
    // field into metadata. The guidance scalar is inert for all current Klein
    // variants, so legacy values must not be recalled into the shared guidance
    // state — otherwise they leak back into FLUX.1 when the user switches
    // models.
    it('rejects parsing when the image was generated with a FLUX.2 model', async () => {
      const store = makeStore();

      await expect(
        Promise.resolve().then(() =>
          ImageMetadataHandlers.Guidance.parse(
            {
              model: { key: 'k', hash: 'h', name: 'Klein 9B Base', base: 'flux2', type: 'main' },
              guidance: 3.5,
            },
            store
          )
        )
      ).rejects.toThrow();
    });

    it('parses successfully for FLUX.1 metadata', async () => {
      const store = makeStore();

      const parsed = await ImageMetadataHandlers.Guidance.parse(
        {
          model: { key: 'k', hash: 'h', name: 'FLUX Dev', base: 'flux', type: 'main' },
          guidance: 3.5,
        },
        store
      );

      expect(parsed).toBe(3.5);
    });

    it('parses successfully when no model metadata is present', async () => {
      // Metadata without a model field should still parse (back-compat for
      // images where only scalar params were saved).
      const store = makeStore();

      const parsed = await ImageMetadataHandlers.Guidance.parse({ guidance: 3.5 }, store);
      expect(parsed).toBe(3.5);
    });
  });

  describe('HRF LoRAs', () => {
    it('parses dedicated HRF LoRAs against the image metadata model base', async () => {
      currentBase = 'sd-1';
      const mainModel = fakeModel('main', 'sdxl');
      const hrfLora = fakeModel('lora', 'sdxl');
      resolvedModels = {
        [hrfLora.key]: hrfLora,
      };
      const store = makeStore();

      const parsed = await ImageMetadataHandlers.HrfLoRAs.parse(
        {
          model: mainModel,
          hrf_loras: [{ model: hrfLora, weight: 0.6 }],
        },
        store
      );

      expect(parsed).toEqual([
        expect.objectContaining({ model: expect.objectContaining({ key: hrfLora.key }), weight: 0.6 }),
      ]);
    });

    it('filters HRF LoRAs that do not match the metadata HRF model base', async () => {
      currentBase = 'sdxl';
      const mainModel = fakeModel('main', 'sdxl');
      const hrfModel = fakeModel('main', 'sd-1');
      const hrfLora = fakeModel('lora', 'sdxl');
      resolvedModels = {
        [hrfLora.key]: hrfLora,
      };
      const store = makeStore();

      const parsed = await ImageMetadataHandlers.HrfLoRAs.parse(
        {
          model: mainModel,
          hrf_model: hrfModel,
          hrf_loras: [{ model: hrfLora, weight: 0.6 }],
        },
        store
      );

      expect(parsed).toEqual([]);
    });

    it('recalls dedicated HRF LoRAs after the recalled main model changes base', async () => {
      currentBase = 'sd-1';
      const mainModel = fakeModel('main', 'sdxl');
      const hrfLora = fakeModel('lora', 'sdxl');
      resolvedModels = {
        [mainModel.key]: mainModel,
        [hrfLora.key]: hrfLora,
      };
      const store = makeStore();

      const recalled = await MetadataUtils.recallByHandlers({
        metadata: {
          model: mainModel,
          hrf_loras: [{ model: hrfLora, weight: 0.6 }],
        },
        handlers: [ImageMetadataHandlers.HrfLoRAs, ImageMetadataHandlers.MainModel],
        store,
        silent: true,
      });

      expect(Object.keys(ImageMetadataHandlers).indexOf('HrfLoRAs')).toBeGreaterThan(
        Object.keys(ImageMetadataHandlers).indexOf('MainModel')
      );
      expect(recalled.has(ImageMetadataHandlers.MainModel)).toBe(true);
      expect(recalled.has(ImageMetadataHandlers.HrfLoRAs)).toBe(true);
      expect(store.dispatch).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'generation/modelSelected',
          payload: expect.objectContaining({ base: 'sdxl' }),
        })
      );
      expect(store.dispatch).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'params/setHrfLoras',
          payload: [expect.objectContaining({ model: expect.objectContaining({ key: hrfLora.key }), weight: 0.6 })],
        })
      );
    });
  });
});
