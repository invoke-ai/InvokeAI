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

const fakeModel = (
  type: 'vae' | 'qwen3_encoder' | 'mistral_encoder' | 'qwen3_vl_encoder' | 't5_encoder',
  base: string
) => ({
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
    // The generic VAEModel handler must NOT also fire for FLUX.2 / Z-Image /
    // Krea-2 images, otherwise the metadata viewer renders duplicate VAE rows
    // next to the dedicated Flux2VAEModel / ZImageVAEModel / Krea2VAEModel
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

  describe('T5EncoderModel', () => {
    it('parses metadata.t5_encoder into a t5_encoder model identifier', async () => {
      nextResolved = fakeModel('t5_encoder', 'any');
      const store = makeStore();

      const parsed = await ImageMetadataHandlers.T5EncoderModel.parse({ t5_encoder: nextResolved }, store);

      expect(parsed.key).toBe('t5_encoder-key');
      expect(parsed.type).toBe('t5_encoder');
    });

    it('rejects parsing when the resolved model is not a t5_encoder', async () => {
      nextResolved = fakeModel('vae', 'flux');
      const store = makeStore();

      await expect(ImageMetadataHandlers.T5EncoderModel.parse({ t5_encoder: nextResolved }, store)).rejects.toThrow();
    });
  });

  describe('Guidance (FLUX.2 variant gating)', () => {
    // guidance_embeds is inert for FLUX.2 Klein, so a legacy Klein `guidance`
    // value must not be recalled into the shared guidance state — otherwise it
    // leaks back into FLUX.1 when the user switches models. FLUX.2 [dev] genuinely
    // consumes guidance, so it must parse and recall. The handler resolves the
    // image's own model to read its variant.
    it('rejects parsing when the image was generated with a FLUX.2 Klein model', async () => {
      modelRegistry['k'] = fakeMainModel('klein_9b');
      const store = makeStore();

      await expect(
        ImageMetadataHandlers.Guidance.parse(
          {
            model: { key: 'k', hash: 'h', name: 'Klein 9B Base', base: 'flux2', type: 'main' },
            guidance: 3.5,
          },
          store
        )
      ).rejects.toThrow();
    });

    it('parses successfully when the image was generated with a FLUX.2 [dev] model', async () => {
      modelRegistry['k'] = fakeMainModel('dev');
      const store = makeStore();

      const parsed = await ImageMetadataHandlers.Guidance.parse(
        {
          model: { key: 'k', hash: 'h', name: 'FLUX.2 dev', base: 'flux2', type: 'main' },
          guidance: 3.5,
        },
        store
      );

      expect(parsed).toBe(3.5);
    });

    it('rejects when the FLUX.2 model can no longer be resolved', async () => {
      // Uninstalled/unresolvable model: we cannot confirm it was [dev], so fall
      // back to the safe Klein behavior and skip rather than leak a stale value.
      modelRegistry = {};
      nextResolved = fakeModel('vae', 'flux2'); // no variant field
      const store = makeStore();

      await expect(
        ImageMetadataHandlers.Guidance.parse(
          {
            model: { key: 'gone', hash: 'h', name: 'Uninstalled', base: 'flux2', type: 'main' },
            guidance: 3.5,
          },
          store
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

    it.each(['sdxl', 'flux'] as const)('rejects an incompatible %s VAE from Krea-2 image metadata', async (vaeBase) => {
      currentBase = 'krea-2';
      nextResolved = fakeModel('vae', vaeBase);
      const store = makeStore();

      await expect(
        ImageMetadataHandlers.Krea2VAEModel.parse({ model: fakeModel('main', 'krea-2'), vae: nextResolved }, store)
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
      { handler: 'Krea2SeedVarianceStrength', field: 'krea2_seed_variance_strength', value: 0.5 },
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

  // The numeric enhancer scalars must parse exactly the ranges the slider / param-state / invocation accept,
  // so recalling an image never dispatches a value the backend rejects (regression for review 4791964047:
  // strength is a 0..2 std-multiplier, randomize-percent allows 0 as the disabled value).
  describe('conditioning-enhancer numeric ranges', () => {
    const parseStrength = (value: unknown, store: AppStore) =>
      (
        ImageMetadataHandlers.Krea2SeedVarianceStrength as unknown as {
          parse: (m: Record<string, unknown>, s: AppStore) => Promise<unknown>;
        }
      ).parse({ model: { base: 'krea-2' }, krea2_seed_variance_strength: value }, store);

    const parsePercent = (value: unknown, store: AppStore) =>
      (
        ImageMetadataHandlers.Krea2SeedVarianceRandomizePercent as unknown as {
          parse: (m: Record<string, unknown>, s: AppStore) => Promise<unknown>;
        }
      ).parse({ model: { base: 'krea-2' }, krea2_seed_variance_randomize_percent: value }, store);

    it.each([0, 0.1, 2])('strength accepts %s (0..2 std-multiplier)', async (value) => {
      currentBase = 'krea-2';
      expect(await parseStrength(value, makeStore())).toBe(value);
    });

    it.each([-0.1, 2.1, 20])('strength rejects out-of-range %s (e.g. the stale absolute value 20)', async (value) => {
      currentBase = 'krea-2';
      const store = makeStore();
      // `.parse()` throws synchronously, so wrap it to observe the rejection (as the gating tests do).
      await expect(Promise.resolve().then(() => parseStrength(value, store))).rejects.toThrow();
    });

    it.each([0, 1, 50, 100])('randomize percent accepts %s (0 = disabled)', async (value) => {
      currentBase = 'krea-2';
      expect(await parsePercent(value, makeStore())).toBe(value);
    });

    it.each([-1, 101])('randomize percent rejects out-of-range %s', async (value) => {
      currentBase = 'krea-2';
      const store = makeStore();
      await expect(Promise.resolve().then(() => parsePercent(value, store))).rejects.toThrow();
    });
  });

  // The rebalance weights are free text forwarded to the backend, which requires exactly 12 finite numbers.
  // Recall must reject anything else so it can't dispatch state that fails at generation (review 4800904928).
  describe('Krea2RebalanceWeights recall validation', () => {
    const parseWeights = (value: unknown, store: AppStore) =>
      (
        ImageMetadataHandlers.Krea2RebalanceWeights as unknown as {
          parse: (m: Record<string, unknown>, s: AppStore) => Promise<unknown>;
        }
      ).parse({ model: { base: 'krea-2' }, krea2_rebalance_weights: value }, store);

    it('recalls exactly 12 finite comma-separated numbers', async () => {
      currentBase = 'krea-2';
      const value = '1,1,1,1,1,1,1,2.5,5,1.1,4,1';
      expect(await parseWeights(value, makeStore())).toBe(value);
    });

    it.each([
      ['too few', '1,2,3'],
      ['too many', '1,2,3,4,5,6,7,8,9,10,11,12,13'],
      ['nonnumeric', '1,2,3,4,5,6,7,8,9,10,11,x'],
      ['nan', '1,2,3,4,5,6,7,8,9,10,11,nan'],
      ['inf', '1,2,3,4,5,6,7,8,9,10,11,inf'],
    ])('rejects %s', async (_label, value) => {
      currentBase = 'krea-2';
      const store = makeStore();
      await expect(Promise.resolve().then(() => parseWeights(value, store))).rejects.toThrow();
    });
  });
});
