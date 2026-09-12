import { describe, expect, it } from 'vitest';

import type { MainModelConfig } from './types';
import {
  isAnimaCompatibleVAEModelConfig,
  isAnimaVAEModelConfig,
  isFlux2DiffusersMainModelConfig,
  isWanLowNoisePartnerOption,
  isZImageDiffusersMainModelConfig,
  selectPrimaryMainModelOptions,
} from './types';

const partialConfig = (base: 'flux2' | 'z-image', submodels: Record<string, unknown>) => ({
  type: 'main',
  base,
  format: 'sdnq_quantized',
  variant: 'klein_4b',
  name: 'partial-sdnq-pipeline',
  submodels,
});

describe('SDNQ pipeline model predicates', () => {
  it.each([
    ['flux2', isFlux2DiffusersMainModelConfig],
    ['z-image', isZImageDiffusersMainModelConfig],
  ] as const)('rejects a pipeline with only a transformer submodel', (base, predicate) => {
    expect(predicate(partialConfig(base, { transformer: {} }) as never)).toBe(false);
  });

  it.each([
    ['flux2', isFlux2DiffusersMainModelConfig],
    ['z-image', isZImageDiffusersMainModelConfig],
  ] as const)('rejects a pipeline with no transformer submodel', (base, predicate) => {
    expect(predicate(partialConfig(base, { vae: {}, text_encoder: {}, tokenizer: {} }) as never)).toBe(false);
  });
});

// The Anima loader accepts a FLUX VAE and a 16-channel Wan VAE beside the Wan/QwenImage one, but Krea-2
// draws its own VAE pool from the base-driven `isAnimaVAEModelConfig` and must not be offered either.
// The two predicates must therefore stay distinct - collapsing them would widen Krea-2's picker as a
// side effect.
describe('Anima VAE predicates', () => {
  const vae = (base: string, over: Record<string, unknown> = {}) =>
    ({ key: `${base}-vae`, type: 'vae', base, name: `${base} vae`, ...over }) as never;

  it('accepts an Anima VAE in both predicates', () => {
    expect(isAnimaVAEModelConfig(vae('anima'))).toBe(true);
    expect(isAnimaCompatibleVAEModelConfig(vae('anima'))).toBe(true);
  });

  it('accepts a FLUX VAE only as Anima-compatible, not as an Anima-base VAE', () => {
    expect(isAnimaVAEModelConfig(vae('flux'))).toBe(false);
    expect(isAnimaCompatibleVAEModelConfig(vae('flux'))).toBe(true);
  });

  // Anima's transformer works in a 16-channel latent space, which the A14B Wan VAE provides. The
  // 48-channel Wan2.2-VAE (TI2V-5B) is the same AutoencoderKLWan class but a different latent space.
  it('accepts a 16-channel Wan VAE as Anima-compatible', () => {
    expect(isAnimaCompatibleVAEModelConfig(vae('wan', { latent_channels: 16 }))).toBe(true);
    expect(isAnimaVAEModelConfig(vae('wan', { latent_channels: 16 }))).toBe(false);
  });

  it('rejects a 48-channel Wan VAE', () => {
    expect(isAnimaCompatibleVAEModelConfig(vae('wan', { latent_channels: 48 }))).toBe(false);
  });

  // A main model's bundled `vae` submodel carries no `latent_channels`, so its geometry is unverifiable.
  it('rejects a Wan VAE whose latent channel count is unknown', () => {
    expect(isAnimaCompatibleVAEModelConfig(vae('wan'))).toBe(false);
  });

  it.each(['flux2', 'qwen-image', 'sdxl'])('rejects a %s VAE in both predicates', (base) => {
    expect(isAnimaVAEModelConfig(vae(base))).toBe(false);
    expect(isAnimaCompatibleVAEModelConfig(vae(base))).toBe(false);
  });

  // Every VAE picker is built from one of these guards called with a *single* argument, so main models
  // with a bundled `vae` submodel are part of each slot's domain. The `excludeSubmodels` parameter is
  // what distinguishes that from a geometry check the submodel cannot answer - see the Wan arm below.
  describe('main-model VAE submodels', () => {
    const mainWithVae = (base: string) =>
      ({ key: `${base}-main`, type: 'main', base, name: `${base} main`, submodels: { vae: {} } }) as never;

    it('accepts a FLUX main-model VAE submodel unless submodels are excluded', () => {
      expect(isAnimaCompatibleVAEModelConfig(mainWithVae('flux'))).toBe(true);
      expect(isAnimaCompatibleVAEModelConfig(mainWithVae('flux'), true)).toBe(false);
    });

    it('accepts an Anima main-model VAE submodel unless submodels are excluded', () => {
      expect(isAnimaVAEModelConfig(mainWithVae('anima'))).toBe(true);
      expect(isAnimaVAEModelConfig(mainWithVae('anima'), true)).toBe(false);
      expect(isAnimaCompatibleVAEModelConfig(mainWithVae('anima'))).toBe(true);
    });

    // A bundled submodel carries no `latent_channels`, so a Wan main model can never clear the geometry
    // check - regardless of what is passed for `excludeSubmodels`.
    it('never accepts a Wan main-model VAE submodel', () => {
      expect(isAnimaCompatibleVAEModelConfig(mainWithVae('wan'))).toBe(false);
      expect(isAnimaCompatibleVAEModelConfig(mainWithVae('wan'), true)).toBe(false);
    });

    // Point-free `.filter(guard)` would hand these guards the array *index* as `excludeSubmodels`.
    // TypeScript rejects that at this call site, but not inside `buildModelsSelector`, whose generic
    // signature declares the guard as single-parameter - see modelsByType.test.ts for that coverage.
    it('changes its verdict based on the second argument', () => {
      const config = mainWithVae('flux');

      expect(isAnimaCompatibleVAEModelConfig(config, false)).toBe(true);
      expect(isAnimaCompatibleVAEModelConfig(config, true)).toBe(false);
    });
  });
});

const wanMain = (over: Record<string, unknown>) =>
  ({
    key: 'k',
    type: 'main',
    base: 'wan',
    format: 'checkpoint',
    variant: 't2v_a14b',
    name: 'wan',
    ...over,
  }) as unknown as MainModelConfig;

describe('Wan low-noise partner picker', () => {
  it('offers an untagged single-file A14B', () => {
    // The case the whole branch exists to support. `expert` comes from a filename
    // heuristic, is absent from `ModelRecordChanges`, and installed records are never
    // re-probed — so requiring `expert === 'low'` here strands untagged pairs
    // permanently, with no correction short of delete-and-reinstall.
    expect(isWanLowNoisePartnerOption(wanMain({ key: 'untagged', expert: 'none' }))).toBe(true);
  });

  it('offers a tagged low expert, and a GGUF one — the pair need not share a format', () => {
    expect(isWanLowNoisePartnerOption(wanMain({ key: 'low', expert: 'low' }))).toBe(true);
    expect(isWanLowNoisePartnerOption(wanMain({ key: 'low-gguf', expert: 'low', format: 'gguf_quantized' }))).toBe(
      true
    );
  });

  it('does not offer a tagged high expert or a Diffusers main', () => {
    expect(isWanLowNoisePartnerOption(wanMain({ key: 'high', expert: 'high' }))).toBe(false);
    expect(isWanLowNoisePartnerOption(wanMain({ key: 'diff', format: 'diffusers' }))).toBe(false);
  });

  it('does not offer a TI2V-5B — it is single-transformer and has no partner', () => {
    expect(isWanLowNoisePartnerOption(wanMain({ key: '5b', variant: 'ti2v_5b', expert: 'none' }))).toBe(false);
  });

  it('lets an untagged pair be assembled: both halves stay in the primary picker too', () => {
    // The two pickers have to agree. Widening this one must not start hiding untagged
    // models from the primary list — `selectPrimaryMainModelOptions` keys on the narrow
    // tag test for exactly that reason.
    const a = wanMain({ key: 'a', expert: 'none', name: 'pair-part-1' });
    const b = wanMain({ key: 'b', expert: 'none', name: 'pair-part-2' });

    expect([a, b].filter(isWanLowNoisePartnerOption)).toHaveLength(2);
    expect(selectPrimaryMainModelOptions([a, b])).toHaveLength(2);
  });

  it('never leaves a TI2V-5B invisible in both pickers', () => {
    // The partner picker excludes every TI2V-5B, so the primary picker must not hide one
    // either — a single-transformer model has no partner slot to be steered toward. The
    // two exclusions have to agree or the model is reachable from nowhere in the linear UI.
    //
    // Reachable with a real record: the pre-branch GGUF probe applied the expert tag
    // without consulting the variant, so a 5B whose stem contained `low_noise` was stored
    // as `expert='low'` and still is. `hasPartner` then matches it against any second
    // TI2V-5B, which is all it takes to hide it.
    const taggedLow5b = wanMain({ key: 'ti2v-low', variant: 'ti2v_5b', expert: 'low' });
    const plain5b = wanMain({ key: 'ti2v-plain', variant: 'ti2v_5b', expert: 'none' });
    const library = [taggedLow5b, plain5b];

    expect(selectPrimaryMainModelOptions(library).map((c) => c.key)).toEqual(['ti2v-low', 'ti2v-plain']);
    expect(library.filter(isWanLowNoisePartnerOption)).toHaveLength(0);
  });

  it('keeps an untagged model in the primary picker even next to a tagged high expert', () => {
    // The case that actually catches `selectPrimaryMainModelOptions` being switched to the
    // wide predicate. With two untagged models the wide test classes both as low experts,
    // so neither has a partner and neither is hidden — the mistake hides behind itself.
    // Add a same-variant `high` and the untagged model suddenly has a partner, so keying
    // the primary filter on the wide test would drop it from the main picker entirely.
    const high = wanMain({ key: 'high', expert: 'high', name: 'high' });
    const untagged = wanMain({ key: 'untagged', expert: 'none', name: 'untagged' });

    expect(selectPrimaryMainModelOptions([high, untagged]).map((c) => c.key)).toEqual(['high', 'untagged']);
  });

  it('still hides a tagged low expert from the primary picker when it has a partner', () => {
    const high = wanMain({ key: 'high', expert: 'high', name: 'high' });
    const low = wanMain({ key: 'low', expert: 'low', name: 'low' });

    expect(selectPrimaryMainModelOptions([high, low]).map((c) => c.key)).toEqual(['high']);
  });
});
