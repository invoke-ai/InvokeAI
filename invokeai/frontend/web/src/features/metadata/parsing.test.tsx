import type { AppStore } from 'app/store/store';
import type * as paramsSliceModule from 'features/controlLayers/store/paramsSlice';
import { ImageMetadataHandlers } from 'features/metadata/parsing';
import type * as modelsApiModule from 'services/api/endpoints/models';
import { beforeEach, describe, expect, it, vi } from 'vitest';

// ---------------------------------------------------------------------------
// Module mocks
//
// We are testing only the *gating* logic of the model-related metadata
// handlers (`VAEModel`, `Flux2VAEModel`, `KleinQwen3EncoderModel`,
// `Flux2DevMistralEncoderModel`). The model lookup goes
// through `parseModelIdentifier`, which dispatches an RTK Query thunk. We stub
// the models endpoint so any lookup resolves to a canned model identifier —
// the parse step then succeeds and the assertions inside each handler become
// observable.
// ---------------------------------------------------------------------------

let currentBase: string | null = 'flux2';

vi.mock('features/controlLayers/store/paramsSlice', async (importOriginal) => {
  const mod = await importOriginal<typeof paramsSliceModule>();
  return { ...mod, selectBase: () => currentBase };
});

const fakeModel = (type: 'vae' | 'qwen3_encoder' | 'mistral_encoder', base: string) => ({
  key: `${type}-key`,
  hash: 'hash',
  name: `Some ${type}`,
  base,
  type,
});

// FLUX.2 main-model config. The `variant` is what the dev-vs-Klein VAE
// disambiguation resolves against (`dev` => [dev], `klein_*` => Klein), mirroring
// the graph builder's `isFlux2Dev = model.variant === 'dev'`.
const fakeMainModel = (variant: 'dev' | 'klein_9b') => ({
  key: 'main-key',
  hash: 'main-hash',
  name: `FLUX.2 ${variant}`,
  base: 'flux2',
  type: 'main',
  variant,
});

let nextResolved: ReturnType<typeof fakeModel> = fakeModel('vae', 'flux2');

// Registry consulted by the store's `dispatch` mock, keyed by the model key that
// `getModelConfig.initiate` was called with. Lets a single test resolve both a
// VAE lookup (`vae-key`) and the image's main model (`main-key`) to distinct
// configs. Unregistered keys fall back to `nextResolved`.
let modelRegistry: Record<string, unknown> = {};

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
    dispatch: vi.fn((action: { key?: string }) => ({
      unwrap: () => Promise.resolve((action?.key && modelRegistry[action.key]) || nextResolved),
    })),
    getState: () => ({}),
  }) as unknown as AppStore;

beforeEach(() => {
  currentBase = 'flux2';
  nextResolved = fakeModel('vae', 'flux2');
  modelRegistry = {};
});

describe('ImageMetadataHandlers — Klein recall gating', () => {
  describe('Flux2VAEModel', () => {
    // Klein and [dev] share a single flux2VaeModel slot, so one handler recalls both
    // variants' VAE from metadata.vae — no dev/Klein disambiguation.
    it.each(['klein_9b', 'dev'] as const)(
      'parses metadata.vae for FLUX.2 %s images when base is flux2',
      async (variant) => {
        currentBase = 'flux2';
        nextResolved = fakeModel('vae', 'flux2');
        modelRegistry['main-key'] = fakeMainModel(variant);
        const store = makeStore();

        const parsed = await ImageMetadataHandlers.Flux2VAEModel.parse(
          { vae: nextResolved, model: fakeMainModel(variant) },
          store
        );

        expect(parsed.key).toBe('vae-key');
        expect(parsed.type).toBe('vae');
      }
    );

    it('rejects when base is not flux2', async () => {
      currentBase = 'sdxl';
      nextResolved = fakeModel('vae', 'flux2');
      const store = makeStore();

      await expect(
        ImageMetadataHandlers.Flux2VAEModel.parse({ vae: nextResolved, model: fakeMainModel('klein_9b') }, store)
      ).rejects.toThrow();
    });
  });

  describe('KleinQwen3EncoderModel', () => {
    it('parses metadata.qwen3_encoder when base is flux2', async () => {
      currentBase = 'flux2';
      nextResolved = fakeModel('qwen3_encoder', 'flux2');
      const store = makeStore();

      const parsed = await ImageMetadataHandlers.KleinQwen3EncoderModel.parse({ qwen3_encoder: nextResolved }, store);

      expect(parsed.key).toBe('qwen3_encoder-key');
      expect(parsed.type).toBe('qwen3_encoder');
    });

    it('rejects when base is not flux2', async () => {
      currentBase = 'sdxl';
      nextResolved = fakeModel('qwen3_encoder', 'flux2');
      const store = makeStore();

      await expect(
        ImageMetadataHandlers.KleinQwen3EncoderModel.parse({ qwen3_encoder: nextResolved }, store)
      ).rejects.toThrow();
    });
  });

  describe('Flux2DevMistralEncoderModel', () => {
    it('parses metadata.mistral_encoder when base is flux2', async () => {
      currentBase = 'flux2';
      nextResolved = fakeModel('mistral_encoder', 'flux2');
      const store = makeStore();

      const parsed = await ImageMetadataHandlers.Flux2DevMistralEncoderModel.parse(
        { mistral_encoder: nextResolved },
        store
      );

      expect(parsed.key).toBe('mistral_encoder-key');
      expect(parsed.type).toBe('mistral_encoder');
    });

    it('rejects when base is not flux2', async () => {
      currentBase = 'sdxl';
      nextResolved = fakeModel('mistral_encoder', 'flux2');
      const store = makeStore();

      await expect(
        ImageMetadataHandlers.Flux2DevMistralEncoderModel.parse({ mistral_encoder: nextResolved }, store)
      ).rejects.toThrow();
    });
  });

  describe('VAEModel (generic)', () => {
    // The generic VAEModel handler must NOT also fire for FLUX.2 / Z-Image
    // images, otherwise the metadata viewer renders duplicate VAE rows next
    // to the dedicated Flux2VAEModel / ZImageVAEModel handlers.
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
});
