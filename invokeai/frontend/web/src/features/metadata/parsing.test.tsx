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

const fakeModel = (type: 'vae' | 'qwen3_encoder' | 'qwen3_vl_encoder', base: string) => ({
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
    // The generic VAEModel handler must NOT also fire for FLUX.2 / Z-Image /
    // Krea-2 images, otherwise the metadata viewer renders duplicate VAE rows
    // next to the dedicated KleinVAEModel / ZImageVAEModel / Krea2VAEModel
    // handlers (and recalls into the wrong, shared VAE slot).
    it.each(['flux2', 'z-image', 'krea-2'])('rejects parsing when current base is %s', async (base) => {
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

describe('ImageMetadataHandlers — Krea-2 recall gating', () => {
  // Krea-2 borrows the Qwen-Image VAE and a standalone Qwen3-VL encoder for single-file / GGUF
  // transformers, recalled into dedicated (krea2VaeModel / krea2Qwen3VlEncoderModel) slots — but only when
  // the current main model is actually Krea-2.
  describe('Krea2VAEModel', () => {
    it.each(['qwen-image', 'anima'] as const)(
      'parses a supported %s VAE when the current and metadata main models are Krea-2',
      async (vaeBase) => {
        currentBase = 'krea-2';
        nextResolved = fakeModel('vae', vaeBase);
        const store = makeStore();

        const parsed = await ImageMetadataHandlers.Krea2VAEModel.parse(
          { model: fakeModel('main', 'krea-2'), vae: nextResolved },
          store
        );

        expect(parsed.key).toBe('vae-key');
        expect(parsed.type).toBe('vae');
        expect(parsed.base).toBe(vaeBase);
      }
    );

    it('rejects parsing when the current main model is not Krea-2', async () => {
      currentBase = 'sdxl';
      nextResolved = fakeModel('vae', 'krea-2');
      const store = makeStore();

      await expect(ImageMetadataHandlers.Krea2VAEModel.parse({ vae: nextResolved }, store)).rejects.toThrow();
    });

    it('rejects VAE metadata from a non-Krea-2 image even when Krea-2 is currently selected', async () => {
      currentBase = 'krea-2';
      nextResolved = fakeModel('vae', 'sdxl');
      const store = makeStore();

      await expect(
        ImageMetadataHandlers.Krea2VAEModel.parse(
          { model: fakeModel('qwen3_vl_encoder', 'sdxl'), vae: nextResolved },
          store
        )
      ).rejects.toThrow();
    });
  });

  describe('Krea2Qwen3VlEncoderModel', () => {
    it('parses metadata.qwen3_vl_encoder when the current main model is Krea-2', async () => {
      currentBase = 'krea-2';
      nextResolved = fakeModel('qwen3_vl_encoder', 'krea-2');
      const store = makeStore();

      const parsed = await ImageMetadataHandlers.Krea2Qwen3VlEncoderModel.parse(
        { model: fakeModel('main', 'krea-2'), qwen3_vl_encoder: nextResolved },
        store
      );

      expect(parsed.key).toBe('qwen3_vl_encoder-key');
      expect(parsed.type).toBe('qwen3_vl_encoder');
    });

    it('rejects parsing when the current main model is not Krea-2', async () => {
      currentBase = 'flux';
      nextResolved = fakeModel('qwen3_vl_encoder', 'krea-2');
      const store = makeStore();

      await expect(
        ImageMetadataHandlers.Krea2Qwen3VlEncoderModel.parse({ qwen3_vl_encoder: nextResolved }, store)
      ).rejects.toThrow();
    });

    it('rejects encoder metadata from a non-Krea-2 image even when Krea-2 is currently selected', async () => {
      currentBase = 'krea-2';
      nextResolved = fakeModel('qwen3_vl_encoder', 'any');
      const store = makeStore();

      await expect(
        ImageMetadataHandlers.Krea2Qwen3VlEncoderModel.parse(
          { model: fakeModel('qwen3_vl_encoder', 'flux'), qwen3_vl_encoder: nextResolved },
          store
        )
      ).rejects.toThrow();
    });
  });

  // The conditioning-enhancer settings are Krea-2-only scalars. Their parse is gated on the current base so
  // recalling an unrelated (older / non-Krea-2) image does NOT clobber the user's hidden enhancer state.
  // The base check throws synchronously, which the parse runner turns into a rejected promise.
  describe('conditioning-enhancer gating', () => {
    const enhancerCases = [
      { handler: 'Krea2SeedVarianceEnabled', field: 'krea2_seed_variance_enabled', value: true },
      { handler: 'Krea2SeedVarianceStrength', field: 'krea2_seed_variance_strength', value: 20 },
      { handler: 'Krea2SeedVarianceRandomizePercent', field: 'krea2_seed_variance_randomize_percent', value: 50 },
      { handler: 'Krea2RebalanceEnabled', field: 'krea2_rebalance_enabled', value: true },
      { handler: 'Krea2RebalanceMultiplier', field: 'krea2_rebalance_multiplier', value: 4 },
      { handler: 'Krea2RebalanceWeights', field: 'krea2_rebalance_weights', value: '1,1,1,1,1,1,1,2.5,5,1.1,4,1' },
    ] as const;

    // The six handlers have different value types (boolean/number/string), so index into a loosely-typed
    // view to keep the union of parse signatures callable.
    const getHandler = (name: (typeof enhancerCases)[number]['handler']) =>
      ImageMetadataHandlers[name] as unknown as {
        parse: (metadata: Record<string, unknown>, store: AppStore) => Promise<unknown>;
      };

    it.each(enhancerCases)('$handler parses when the current base is Krea-2', async ({ handler, field, value }) => {
      currentBase = 'krea-2';
      const store = makeStore();

      const parsed = await getHandler(handler).parse({ model: { base: 'krea-2' }, [field]: value }, store);

      expect(parsed).toBe(value);
    });

    it.each(enhancerCases)(
      '$handler rejects (does not clobber) when the current base is not Krea-2',
      async ({ handler, field, value }) => {
        currentBase = 'sdxl';
        const store = makeStore();

        await expect(
          Promise.resolve().then(() => getHandler(handler).parse({ model: { base: 'krea-2' }, [field]: value }, store))
        ).rejects.toThrow();
      }
    );

    it.each(enhancerCases)(
      '$handler rejects metadata from another model base even when Krea-2 is selected',
      async ({ handler, field, value }) => {
        currentBase = 'krea-2';
        const store = makeStore();

        await expect(
          Promise.resolve().then(() => getHandler(handler).parse({ model: { base: 'sdxl' }, [field]: value }, store))
        ).rejects.toThrow();
      }
    );
  });
});
