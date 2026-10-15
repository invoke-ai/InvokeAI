import { resetArchitectureCapabilities } from '@features/generation/core/architectureCapabilities';
import {
  seedArchitectureCapabilities,
  architectureCapabilitiesFixture,
} from '@features/generation/core/architectureCapabilities.testing';
import { describe, expect, it } from 'vitest';

import type { ComponentModelConfig, GenerateModelConfig, VaeModelConfig } from './types';

import {
  getCompatibleSelectedComponentKey,
  isAnimaQwen3Encoder,
  isErnieImageMistralEncoder,
  isFlux2DiffusersSourceForModel,
  isFlux2MistralEncoder,
  isFlux2Qwen3EncoderForModel,
  isNonAnimaQwen3Encoder,
  isSelfContainedSDNQFlux1Pipeline,
  isSelfContainedSDNQPipeline,
  isVaeAcceptedByBase,
  isVaeCompatibleWithGenerateModel,
  type GenerateComponentCandidate,
} from './componentCompatibility';

const flux2Model = (variant: string): GenerateModelConfig => ({
  base: 'flux2',
  key: `flux2-${variant}`,
  name: `FLUX.2 ${variant}`,
  type: 'main',
  variant,
});

const candidate = (overrides: Partial<GenerateComponentCandidate>): GenerateComponentCandidate => ({
  base: 'any',
  type: 'qwen3_encoder',
  ...overrides,
});

const STANDARD_SDNQ_COMPONENTS = ['transformer', 'vae', 'text_encoder', 'tokenizer'] as const;
const FLUX1_SDNQ_COMPONENTS = [...STANDARD_SDNQ_COMPONENTS, 'text_encoder_2', 'tokenizer_2'] as const;

const submodelsWithout = (components: readonly string[], missing: string): Record<string, object> =>
  Object.fromEntries(components.filter((component) => component !== missing).map((component) => [component, {}]));

seedArchitectureCapabilities();

describe('Generate component compatibility', () => {
  it('only treats SDNQ folders with every required pipeline component as self-contained', () => {
    const complete = candidate({
      format: 'sdnq_quantized',
      submodels: { text_encoder: {}, tokenizer: {}, transformer: {}, vae: {} },
    });
    const partial = candidate({
      format: 'sdnq_quantized',
      submodels: { transformer: {} },
    });

    expect(isSelfContainedSDNQPipeline(complete)).toBe(true);
    expect(isSelfContainedSDNQPipeline(partial)).toBe(false);
  });

  it.each(STANDARD_SDNQ_COMPONENTS)('rejects a standard SDNQ pipeline missing %s', (missing) => {
    expect(
      isSelfContainedSDNQPipeline(
        candidate({ format: 'sdnq_quantized', submodels: submodelsWithout(STANDARD_SDNQ_COMPONENTS, missing) })
      )
    ).toBe(false);
  });

  it('requires both text-encoder pairs before a FLUX.1 SDNQ folder is self-contained', () => {
    const complete = candidate({
      format: 'sdnq_quantized',
      submodels: {
        text_encoder: {},
        text_encoder_2: {},
        tokenizer: {},
        tokenizer_2: {},
        transformer: {},
        vae: {},
      },
    });
    const missingT5Pair = candidate({
      format: 'sdnq_quantized',
      submodels: { text_encoder: {}, tokenizer: {}, transformer: {}, vae: {} },
    });

    expect(isSelfContainedSDNQFlux1Pipeline(complete)).toBe(true);
    expect(isSelfContainedSDNQFlux1Pipeline(missingT5Pair)).toBe(false);
  });

  it.each(FLUX1_SDNQ_COMPONENTS)('rejects a FLUX.1 SDNQ pipeline missing %s', (missing) => {
    expect(
      isSelfContainedSDNQFlux1Pipeline(
        candidate({ format: 'sdnq_quantized', submodels: submodelsWithout(FLUX1_SDNQ_COMPONENTS, missing) })
      )
    ).toBe(false);
  });

  it('separates Anima Qwen3 0.6B encoders from other Qwen3 encoders', () => {
    const qwen06b = candidate({ variant: 'qwen3_06b' });
    const qwen4b = candidate({ variant: 'qwen3_4b' });

    expect(isAnimaQwen3Encoder(qwen06b)).toBe(true);
    expect(isAnimaQwen3Encoder(qwen4b)).toBe(false);
    expect(isNonAnimaQwen3Encoder(qwen06b)).toBe(false);
    expect(isNonAnimaQwen3Encoder(qwen4b)).toBe(true);
  });

  it('matches FLUX.2 Klein models to compatible Qwen3 encoder variants', () => {
    const filter = isFlux2Qwen3EncoderForModel(flux2Model('klein_9b'));

    expect(filter(candidate({ variant: 'qwen3_8b' }))).toBe(true);
    expect(filter(candidate({ variant: 'qwen3_4b' }))).toBe(false);
    expect(filter(candidate({ variant: 'qwen3_06b' }))).toBe(false);
  });

  it('matches FLUX.2 component sources by shared Qwen3 encoder variant', () => {
    const filter = isFlux2DiffusersSourceForModel(flux2Model('klein_9b'));

    expect(filter(candidate({ base: 'flux2', format: 'diffusers', type: 'main', variant: 'klein_9b_base' }))).toBe(
      true
    );
    expect(filter(candidate({ base: 'flux2', format: 'diffusers', type: 'main', variant: 'klein_4b' }))).toBe(false);
    expect(filter(candidate({ base: 'flux2', format: 'checkpoint', type: 'main', variant: 'klein_9b' }))).toBe(false);
    expect(filter(candidate({ base: 'flux2', format: 'diffusers', type: 'main' }))).toBe(false);
  });

  it('uses Mistral components and dev Diffusers sources for FLUX.2 [dev]', () => {
    const model = flux2Model('dev');

    expect(isFlux2MistralEncoder(candidate({ type: 'mistral_encoder', variant: 'cow_mistral3_small' }))).toBe(true);
    expect(isFlux2MistralEncoder(candidate({ type: 'qwen3_encoder' }))).toBe(false);
    expect(isFlux2Qwen3EncoderForModel(model)(candidate({ variant: 'qwen3_8b' }))).toBe(false);
    expect(
      isFlux2DiffusersSourceForModel(model)(
        candidate({ base: 'flux2', format: 'diffusers', type: 'main', variant: 'dev' })
      )
    ).toBe(true);
    expect(
      isFlux2DiffusersSourceForModel(model)(
        candidate({ base: 'flux2', format: 'diffusers', type: 'main', variant: 'klein_9b' })
      )
    ).toBe(false);
  });

  it('keeps each Mistral-family encoder out of the other family slot', () => {
    // Filter variants even when they share a model type.
    const ministral = candidate({ type: 'mistral_encoder', variant: 'ministral3_3b' });
    const mistralSmall3 = candidate({ type: 'mistral_encoder', variant: 'cow_mistral3_small' });
    const mistral24b = candidate({ type: 'mistral_encoder', variant: 'mistral3_24b' });

    expect(isFlux2MistralEncoder(ministral)).toBe(false);
    expect([mistralSmall3, mistral24b].every(isFlux2MistralEncoder)).toBe(true);

    expect(isErnieImageMistralEncoder(ministral)).toBe(true);
    expect([mistralSmall3, mistral24b].some(isErnieImageMistralEncoder)).toBe(false);
    expect(isErnieImageMistralEncoder(candidate({ type: 'qwen3_encoder', variant: 'ministral3_3b' }))).toBe(false);
  });

  it('allows only backend-supported Anima VAE families', () => {
    const isAnimaVae = isVaeAcceptedByBase('anima');

    expect(isAnimaVae(candidate({ base: 'anima', type: 'vae' }))).toBe(true);
    expect(isAnimaVae(candidate({ base: 'qwen-image', type: 'vae' }))).toBe(true);
    expect(isAnimaVae(candidate({ base: 'wan', latent_channels: 16, type: 'vae' }))).toBe(true);
    expect(isAnimaVae(candidate({ base: 'wan', latent_channels: 48, type: 'vae' }))).toBe(false);
    // A FLUX VAE decodes an Anima latent without raising and returns a magenta smear.
    expect(isAnimaVae(candidate({ base: 'flux', type: 'vae' }))).toBe(false);
    expect(isAnimaVae(candidate({ base: 'sdxl', type: 'vae' }))).toBe(false);
  });

  it('offers no VAE at all before the capability table has loaded', () => {
    resetArchitectureCapabilities();

    expect(isVaeAcceptedByBase('anima')(candidate({ base: 'anima', type: 'vae' }))).toBe(false);
    expect(
      isVaeCompatibleWithGenerateModel({ base: 'sdxl', key: 'sdxl', name: 'SDXL', type: 'main' }, {
        base: 'sdxl',
        key: 'sdxl-vae',
        name: 'SDXL VAE',
        type: 'vae',
      } as VaeModelConfig)
    ).toBe(false);
  });

  it('centralizes generate VAE compatibility for cross-base families', () => {
    const vae = (base: string) => ({ base, key: `${base}-vae`, name: `${base} VAE`, type: 'vae' as const });

    expect(isVaeCompatibleWithGenerateModel(flux2Model('klein_9b'), vae('flux2'))).toBe(true);
    expect(isVaeCompatibleWithGenerateModel(flux2Model('klein_9b'), vae('flux'))).toBe(false);
    expect(
      isVaeCompatibleWithGenerateModel({ base: 'z-image', key: 'z-image', name: 'Z-Image', type: 'main' }, vae('flux'))
    ).toBe(true);
    expect(
      isVaeCompatibleWithGenerateModel({ base: 'anima', key: 'anima', name: 'Anima', type: 'main' }, vae('qwen-image'))
    ).toBe(true);
  });

  it('hides a stale selected component when it no longer passes the picker filter', () => {
    const staleVae: ComponentModelConfig = { base: 'sdxl', key: 'sdxl-vae', name: 'SDXL VAE', type: 'vae' };

    expect(getCompatibleSelectedComponentKey(staleVae, isVaeAcceptedByBase('anima'))).toBeNull();
  });

  it('offers each Wan variant only the VAE width its decode accepts', () => {
    // Wan A14B and 5B require compatible VAE channel widths.
    const wanMain = (variant: string) =>
      ({ base: 'wan', key: `wan-${variant}`, name: `Wan ${variant}`, type: 'main', variant }) as GenerateModelConfig;
    const wanVae = (latentChannels: number) =>
      ({
        base: 'wan',
        key: `wan-vae-${latentChannels}`,
        latent_channels: latentChannels,
        name: 'Wan VAE',
        type: 'vae',
      }) as VaeModelConfig;

    expect(isVaeCompatibleWithGenerateModel(wanMain('i2v_a14b'), wanVae(16))).toBe(true);
    expect(isVaeCompatibleWithGenerateModel(wanMain('i2v_a14b'), wanVae(48))).toBe(false);
    expect(isVaeCompatibleWithGenerateModel(wanMain('ti2v_5b'), wanVae(48))).toBe(true);
    expect(isVaeCompatibleWithGenerateModel(wanMain('ti2v_5b'), wanVae(16))).toBe(false);
  });
});

describe('the VAE filters and the backend declarations', () => {
  // Exercise each variant row with its own variant.
  const mainOf = (base: string, variant: string | null = null) =>
    ({
      base,
      key: `${base}-main`,
      name: base,
      type: 'main' as const,
      ...(variant === null ? {} : { variant }),
    }) as unknown as GenerateModelConfig;

  const vaeOf = (base: string, latentChannels?: number) =>
    ({
      base,
      key: `${base}-vae`,
      name: `${base} vae`,
      type: 'vae' as const,
      ...(latentChannels === undefined ? {} : { latent_channels: latentChannels }),
    }) as unknown as VaeModelConfig;

  const ALL_BASES = [...new Set(architectureCapabilitiesFixture.map((row) => row.base))];

  /** Check parity with served VAE policy instead of duplicating frontend rules. */
  it('accepts every VAE base the backend declares for that architecture', () => {
    const declared = architectureCapabilitiesFixture.filter((row) => row.vae);
    expect(declared.length).toBeGreaterThan(0);

    for (const row of declared) {
      for (const accepted of row.vae!.accepted) {
        expect(
          // Only wan distinguishes VAEs by latent width; the others leave it null.
          isVaeCompatibleWithGenerateModel(
            mainOf(row.base, row.variant),
            vaeOf(accepted.base, accepted.latent_channels ?? undefined)
          ),
          `${row.base} should accept a VAE registered under ${accepted.base}`
        ).toBe(true);
      }
    }
  });

  /** Negative parity cases must reject an always-true filter. */
  it('rejects every VAE base the backend does not declare for that architecture', () => {
    for (const row of architectureCapabilitiesFixture) {
      const accepted = row.vae?.accepted ?? [{ base: row.base, latent_channels: null }];
      const acceptedBases = new Set(accepted.map((entry) => entry.base));

      for (const base of ALL_BASES) {
        if (acceptedBases.has(base)) {
          continue;
        }
        expect(
          isVaeCompatibleWithGenerateModel(mainOf(row.base, row.variant), vaeOf(base)),
          `${row.base} should refuse a VAE registered under ${base}`
        ).toBe(false);
      }

      // Same-base VAEs are still constrained by channel width.
      for (const entry of accepted) {
        if (entry.latent_channels === null) {
          continue;
        }
        const otherWidth = entry.latent_channels === 16 ? 48 : 16;
        if (accepted.some((other) => other.base === entry.base && other.latent_channels === otherWidth)) {
          continue;
        }
        expect(
          isVaeCompatibleWithGenerateModel(mainOf(row.base, row.variant), vaeOf(entry.base, otherWidth)),
          `${row.base} should refuse a ${otherWidth}-channel ${entry.base} VAE`
        ).toBe(false);
      }
    }
  });

  it('refuses an architecture the table says nothing about', () => {
    expect(isVaeCompatibleWithGenerateModel(mainOf('not-an-architecture'), vaeOf('not-an-architecture'))).toBe(false);
  });
});
