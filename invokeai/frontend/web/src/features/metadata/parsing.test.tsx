import type { AppStore } from 'app/store/store';
import type * as paramsSliceModule from 'features/controlLayers/store/paramsSlice';
import { ImageMetadataHandlers, recallIfStillValid } from 'features/metadata/parsing';
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
  type: 'vae' | 'qwen3_encoder' | 'mistral_encoder' | 'qwen3_vl_encoder' | 't5_encoder' | 'main',
  base: string,
  // Extra config fields, for handlers that gate on more than base/type (e.g. a Wan VAE's
  // `latent_channels`). The store mock resolves these objects as the model's full config.
  over: Record<string, unknown> = {}
) => ({
  key: `${type}-key`,
  hash: 'hash',
  name: `Some ${type}`,
  base,
  type,
  ...over,
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

      const parsed = await ImageMetadataHandlers.KleinQwen3EncoderModel.parse(
        { qwen3_encoder: nextResolved, model: fakeMainModel('klein_9b') },
        store
      );

      expect(parsed.key).toBe('qwen3_encoder-key');
      expect(parsed.type).toBe('qwen3_encoder');
    });

    it('rejects when base is not flux2', async () => {
      currentBase = 'sdxl';
      nextResolved = fakeModel('qwen3_encoder', 'flux2');
      const store = makeStore();

      await expect(
        ImageMetadataHandlers.KleinQwen3EncoderModel.parse(
          { qwen3_encoder: nextResolved, model: fakeMainModel('klein_9b') },
          store
        )
      ).rejects.toThrow();
    });

    // Z-Image and Anima write the same field into their own slots. Without the provenance check their
    // encoder would be recalled into `params.kleinQwen3EncoderModel` (review 4966712044).
    it.each(['z-image', 'anima'])('rejects %s image metadata while FLUX.2 is selected', async (base) => {
      currentBase = 'flux2';
      nextResolved = fakeModel('qwen3_encoder', base);
      const store = makeStore();

      await expect(
        ImageMetadataHandlers.KleinQwen3EncoderModel.parse(
          { qwen3_encoder: nextResolved, model: { key: 'main-key', hash: 'h', name: base, base, type: 'main' } },
          store
        )
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
    // The generic VAEModel handler must NOT also fire for bases in
    // BASES_WITH_DEDICATED_VAE_HANDLER (FLUX.1 / FLUX.2 / Z-Image / Krea-2 /
    // Anima), otherwise the metadata viewer renders duplicate VAE rows next to
    // the dedicated handlers and recalls into the wrong, shared VAE slot.
    it.each(['flux', 'flux2', 'z-image', 'krea-2', 'anima'])(
      'rejects parsing when current base is %s',
      async (base) => {
        currentBase = base;
        nextResolved = fakeModel('vae', base);
        const store = makeStore();

        await expect(ImageMetadataHandlers.VAEModel.parse({ vae: nextResolved }, store)).rejects.toThrow();
      }
    );

    it.each(['sd-1', 'sdxl'])('parses successfully for %s, which uses the shared VAE slot', async (base) => {
      currentBase = base;
      nextResolved = fakeModel('vae', base);
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

// `vae` and `qwen3_encoder` are shared metadata fields written by several bases, each recalling into its
// own params slot. Two axes must line up for a handler to fire: the *provenance* of the image
// (metadata.model.base) and the *currently selected* base (which decides which slot is live). These tests
// pin both, so a handler can never write a foreign base's model into its slot.
describe('ImageMetadataHandlers — Anima / Z-Image / FLUX.1 recall gating', () => {
  const fakeMain = (base: string) => ({
    key: 'main-key',
    hash: 'main-hash',
    name: `Some ${base} model`,
    base,
    type: 'main',
  });

  describe('AnimaVAEModel', () => {
    it('parses an Anima VAE from an Anima image while Anima is selected', async () => {
      currentBase = 'anima';
      nextResolved = fakeModel('vae', 'anima');
      const store = makeStore();

      const parsed = await ImageMetadataHandlers.AnimaVAEModel.parse(
        { model: fakeMain('anima'), vae: nextResolved },
        store
      );

      expect(parsed.key).toBe('vae-key');
      expect(parsed.base).toBe('anima');
    });

    // Krea-2 may legitimately use an Anima VAE, so without the provenance check a Krea-2 image recalled
    // while Anima is selected would push its VAE into the Anima slot.
    it('rejects a Krea-2 image even while Anima is selected', async () => {
      currentBase = 'anima';
      nextResolved = fakeModel('vae', 'anima');
      const store = makeStore();

      await expect(
        ImageMetadataHandlers.AnimaVAEModel.parse({ model: fakeMain('krea-2'), vae: nextResolved }, store)
      ).rejects.toThrow();
    });

    it('rejects metadata that carries no model at all', async () => {
      currentBase = 'anima';
      nextResolved = fakeModel('vae', 'anima');
      const store = makeStore();

      await expect(ImageMetadataHandlers.AnimaVAEModel.parse({ vae: nextResolved }, store)).rejects.toThrow();
    });

    it('rejects when the current base is not Anima', async () => {
      currentBase = 'sdxl';
      nextResolved = fakeModel('vae', 'anima');
      const store = makeStore();

      await expect(
        ImageMetadataHandlers.AnimaVAEModel.parse({ model: fakeMain('anima'), vae: nextResolved }, store)
      ).rejects.toThrow();
    });

    // The Anima model loader's VAE input has no `ui_model_base`, and anima_l2i / anima_i2l branch
    // explicitly on FluxAutoEncoder - so a workflow-built Anima image may legitimately carry a FLUX VAE,
    // and the Anima VAE picker offers those too (isAnimaCompatibleVAEModelConfig). Review 4966712044.
    it('parses a FLUX VAE from an Anima image', async () => {
      currentBase = 'anima';
      nextResolved = fakeModel('vae', 'flux');
      const store = makeStore();

      const parsed = await ImageMetadataHandlers.AnimaVAEModel.parse(
        { model: fakeMain('anima'), vae: nextResolved },
        store
      );

      expect(parsed.base).toBe('flux');
    });

    it.each(['qwen-image', 'flux2', 'sdxl'])('rejects a %s VAE from Anima metadata', async (vaeBase) => {
      currentBase = 'anima';
      nextResolved = fakeModel('vae', vaeBase);
      const store = makeStore();

      await expect(
        ImageMetadataHandlers.AnimaVAEModel.parse({ model: fakeMain('anima'), vae: nextResolved }, store)
      ).rejects.toThrow();
    });

    // Anima's transformer works in a 16-channel latent space, so the A14B Wan VAE is as valid an input as
    // an Anima-base one - `anima_l2i` / `anima_i2l` treat both as AutoencoderKLWan. Recall must gate on
    // that geometry (`latent_channels`), not on the base alone (review 4972570279).
    it('parses a 16-channel Wan VAE from an Anima image', async () => {
      currentBase = 'anima';
      nextResolved = fakeModel('vae', 'wan', { latent_channels: 16 });
      const store = makeStore();

      const parsed = await ImageMetadataHandlers.AnimaVAEModel.parse(
        { model: fakeMain('anima'), vae: nextResolved },
        store
      );

      expect(parsed.base).toBe('wan');
    });

    // The 48-channel Wan2.2-VAE (TI2V-5B) is the same class but a different latent space - recalling it
    // would swap in a VAE that cannot decode Anima latents.
    it('rejects a 48-channel Wan VAE from Anima metadata', async () => {
      currentBase = 'anima';
      nextResolved = fakeModel('vae', 'wan', { latent_channels: 48 });
      const store = makeStore();

      await expect(
        ImageMetadataHandlers.AnimaVAEModel.parse({ model: fakeMain('anima'), vae: nextResolved }, store)
      ).rejects.toThrow();
    });

    it('rejects a Wan VAE whose latent channel count is unknown', async () => {
      currentBase = 'anima';
      nextResolved = fakeModel('vae', 'wan');
      const store = makeStore();

      await expect(
        ImageMetadataHandlers.AnimaVAEModel.parse({ model: fakeMain('anima'), vae: nextResolved }, store)
      ).rejects.toThrow();
    });
  });

  describe('AnimaQwen3EncoderModel', () => {
    // Anima encoders are identified by variant (qwen3_06b), not by base — a base assert here would be
    // wrong. Provenance plus the current base carry the whole gate.
    it('parses regardless of the encoder base', async () => {
      currentBase = 'anima';
      nextResolved = fakeModel('qwen3_encoder', 'any');
      const store = makeStore();

      const parsed = await ImageMetadataHandlers.AnimaQwen3EncoderModel.parse(
        { model: fakeMain('anima'), qwen3_encoder: nextResolved },
        store
      );

      expect(parsed.key).toBe('qwen3_encoder-key');
      expect(parsed.type).toBe('qwen3_encoder');
    });

    it.each(['z-image', 'flux2'])('rejects %s image metadata, which writes the same field', async (base) => {
      currentBase = 'anima';
      nextResolved = fakeModel('qwen3_encoder', 'any');
      const store = makeStore();

      await expect(
        ImageMetadataHandlers.AnimaQwen3EncoderModel.parse(
          { model: fakeMain(base), qwen3_encoder: nextResolved },
          store
        )
      ).rejects.toThrow();
    });

    it('rejects when the current base is not Anima', async () => {
      currentBase = 'z-image';
      nextResolved = fakeModel('qwen3_encoder', 'any');
      const store = makeStore();

      await expect(
        ImageMetadataHandlers.AnimaQwen3EncoderModel.parse(
          { model: fakeMain('anima'), qwen3_encoder: nextResolved },
          store
        )
      ).rejects.toThrow();
    });
  });

  describe('ZImageQwen3EncoderModel', () => {
    it('parses a Z-Image image while Z-Image is selected', async () => {
      currentBase = 'z-image';
      nextResolved = fakeModel('qwen3_encoder', 'any');
      const store = makeStore();

      const parsed = await ImageMetadataHandlers.ZImageQwen3EncoderModel.parse(
        { model: fakeMain('z-image'), qwen3_encoder: nextResolved },
        store
      );

      expect(parsed.key).toBe('qwen3_encoder-key');
    });

    // This handler recalls into the Z-Image slots (and nulls zImageQwen3SourceModel). Anima and FLUX.2
    // Klein write the same metadata field, so without the base gate they would clobber those slots.
    it.each(['anima', 'flux2'])('rejects when the current base is %s', async (base) => {
      currentBase = base;
      nextResolved = fakeModel('qwen3_encoder', 'any');
      const store = makeStore();

      await expect(
        ImageMetadataHandlers.ZImageQwen3EncoderModel.parse(
          { model: fakeMain(base), qwen3_encoder: nextResolved },
          store
        )
      ).rejects.toThrow();
    });

    // The selected base alone is not enough: viewing an Anima or Klein image while Z-Image is selected
    // must not populate the Z-Image encoder slot nor clear the source slot (review 4966712044).
    it.each(['anima', 'flux2'])('rejects %s image metadata while Z-Image is selected', async (base) => {
      currentBase = 'z-image';
      nextResolved = fakeModel('qwen3_encoder', 'any');
      const store = makeStore();

      await expect(
        ImageMetadataHandlers.ZImageQwen3EncoderModel.parse(
          { model: fakeMain(base), qwen3_encoder: nextResolved },
          store
        )
      ).rejects.toThrow();
    });
  });

  describe('ZImageVAEModel', () => {
    it('parses a Z-Image image while Z-Image is selected', async () => {
      currentBase = 'z-image';
      nextResolved = fakeModel('vae', 'z-image');
      const store = makeStore();

      const parsed = await ImageMetadataHandlers.ZImageVAEModel.parse(
        { model: fakeMain('z-image'), vae: nextResolved },
        store
      );

      expect(parsed.key).toBe('vae-key');
    });

    it.each(['flux', 'anima'])('rejects %s image metadata while Z-Image is selected', async (base) => {
      currentBase = 'z-image';
      nextResolved = fakeModel('vae', base);
      const store = makeStore();

      await expect(
        ImageMetadataHandlers.ZImageVAEModel.parse({ model: fakeMain(base), vae: nextResolved }, store)
      ).rejects.toThrow();
    });

    it('rejects when the current base is not Z-Image', async () => {
      currentBase = 'flux';
      nextResolved = fakeModel('vae', 'z-image');
      const store = makeStore();

      await expect(
        ImageMetadataHandlers.ZImageVAEModel.parse({ model: fakeMain('z-image'), vae: nextResolved }, store)
      ).rejects.toThrow();
    });
  });

  describe('ZImageQwen3SourceModel', () => {
    it('parses a Z-Image image while Z-Image is selected', async () => {
      currentBase = 'z-image';
      nextResolved = fakeModel('main', 'z-image');
      const store = makeStore();

      const parsed = await ImageMetadataHandlers.ZImageQwen3SourceModel.parse(
        { model: fakeMain('z-image'), qwen3_source: nextResolved },
        store
      );

      expect(parsed.key).toBe('main-key');
    });

    it('rejects foreign image metadata while Z-Image is selected', async () => {
      currentBase = 'z-image';
      nextResolved = fakeModel('main', 'z-image');
      const store = makeStore();

      await expect(
        ImageMetadataHandlers.ZImageQwen3SourceModel.parse(
          { model: fakeMain('anima'), qwen3_source: nextResolved },
          store
        )
      ).rejects.toThrow();
    });
  });

  describe('Flux1VAEModel', () => {
    it('parses a FLUX VAE from a FLUX.1 image while FLUX.1 is selected', async () => {
      currentBase = 'flux';
      nextResolved = fakeModel('vae', 'flux');
      const store = makeStore();

      const parsed = await ImageMetadataHandlers.Flux1VAEModel.parse(
        { model: fakeMain('flux'), vae: nextResolved },
        store
      );

      expect(parsed.key).toBe('vae-key');
    });

    it('rejects when the current base is flux2, which has its own VAE slot', async () => {
      currentBase = 'flux2';
      nextResolved = fakeModel('vae', 'flux');
      const store = makeStore();

      await expect(
        ImageMetadataHandlers.Flux1VAEModel.parse({ model: fakeMain('flux'), vae: nextResolved }, store)
      ).rejects.toThrow();
    });

    it('rejects a non-FLUX VAE', async () => {
      currentBase = 'flux';
      nextResolved = fakeModel('vae', 'sdxl');
      const store = makeStore();

      await expect(
        ImageMetadataHandlers.Flux1VAEModel.parse({ model: fakeMain('flux'), vae: nextResolved }, store)
      ).rejects.toThrow();
    });

    // A non-FLUX.1 image may legitimately carry a FLUX VAE (Z-Image accepts one). Without the provenance
    // check that VAE would be recalled into `params.fluxVAE` (review 4966712044).
    it('rejects a FLUX VAE that came from a Z-Image image', async () => {
      currentBase = 'flux';
      nextResolved = fakeModel('vae', 'flux');
      const store = makeStore();

      await expect(
        ImageMetadataHandlers.Flux1VAEModel.parse({ model: fakeMain('z-image'), vae: nextResolved }, store)
      ).rejects.toThrow();
    });
  });

  describe('Flux2VAEModel', () => {
    it('parses a FLUX.2 image while FLUX.2 is selected', async () => {
      currentBase = 'flux2';
      nextResolved = fakeModel('vae', 'flux2');
      const store = makeStore();

      const parsed = await ImageMetadataHandlers.Flux2VAEModel.parse(
        { model: fakeMain('flux2'), vae: nextResolved },
        store
      );

      expect(parsed.key).toBe('vae-key');
    });

    it('rejects a Krea-2 image while FLUX.2 is selected', async () => {
      currentBase = 'flux2';
      nextResolved = fakeModel('vae', 'qwen-image');
      const store = makeStore();

      await expect(
        ImageMetadataHandlers.Flux2VAEModel.parse({ model: fakeMain('krea-2'), vae: nextResolved }, store)
      ).rejects.toThrow();
    });
  });
});

// A metadata row is parsed once, and the metadata viewer stays open while the user switches models. The
// row's recall therefore re-runs the handler's gate at click time: `parse` reads the *currently* selected
// base, so a row left over from the previous base must not write into a slot that is no longer in play.
// Review 4987208275.
describe('recallIfStillValid', () => {
  const animaVae = () => fakeModel('vae', 'anima');
  const animaMetadata = () => ({ model: { key: 'main-key', hash: 'h', name: 'anima', base: 'anima', type: 'main' } });

  const run = (handler: (typeof ImageMetadataHandlers)['AnimaVAEModel'], metadata: Record<string, unknown>) => {
    const recall = vi.fn();
    const store = makeStore();
    return recallIfStillValid({ metadata, handler, recall, value: nextResolved, store }).then((didRecall) => ({
      didRecall,
      recall,
      store,
    }));
  };

  it('recalls while the handler still admits the metadata', async () => {
    currentBase = 'anima';
    nextResolved = animaVae();

    const { didRecall, recall } = await run(ImageMetadataHandlers.AnimaVAEModel, {
      ...animaMetadata(),
      vae: nextResolved,
    });

    expect(didRecall).toBe(true);
    expect(recall).toHaveBeenCalledTimes(1);
  });

  it('does nothing once the selected base has moved on', async () => {
    // The row was parsed under Anima; the user switched to SDXL before the click landed.
    currentBase = 'sdxl';
    nextResolved = animaVae();

    const { didRecall, recall } = await run(ImageMetadataHandlers.AnimaVAEModel, {
      ...animaMetadata(),
      vae: nextResolved,
    });

    expect(didRecall).toBe(false);
    expect(recall).not.toHaveBeenCalled();
  });

  it('does nothing when the metadata itself no longer passes the gate', async () => {
    // Provenance failure rather than a base switch - same outcome, same single gate.
    currentBase = 'anima';
    nextResolved = animaVae();

    const { didRecall, recall } = await run(ImageMetadataHandlers.AnimaVAEModel, {
      model: { key: 'main-key', hash: 'h', name: 'krea-2', base: 'krea-2', type: 'main' },
      vae: nextResolved,
    });

    expect(didRecall).toBe(false);
    expect(recall).not.toHaveBeenCalled();
  });

  it('passes through the value the row owns, not the reparsed one', async () => {
    // A collection row owns a single item; recalling whatever `parse` returned would recall the whole set.
    currentBase = 'anima';
    nextResolved = animaVae();
    const store = makeStore();
    const recall = vi.fn();
    const sentinel = { mine: true };

    await recallIfStillValid({
      metadata: { ...animaMetadata(), vae: nextResolved },
      handler: ImageMetadataHandlers.AnimaVAEModel,
      recall,
      value: sentinel,
      store,
    });

    expect(recall).toHaveBeenCalledWith(sentinel, store);
  });
});
