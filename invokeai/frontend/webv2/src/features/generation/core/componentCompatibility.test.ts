import { resetArchitectureCapabilities } from '@features/models/core/architectureCapabilities';
import { seedArchitectureCapabilities } from '@features/models/core/architectureCapabilities.testing';
import { describe, expect, it } from 'vitest';

import type { ComponentModelConfig, GenerateModelConfig, VaeModelConfig } from './types';

import {
  getCompatibleSelectedComponentKey,
  isAnimaQwen3Encoder,
  isErnieImageMistralEncoder,
  isBundledFlux2Main,
  isFlux2EncoderSourceForModel,
  isFlux2MistralEncoder,
  isFlux2PipelineForModel,
  isFlux2Qwen3EncoderForModel,
  isSelfContainedSDNQFlux1Pipeline,
  isSelfContainedSDNQPipeline,
  isVaeAcceptedByBase,
  isVaeCompatibleWithGenerateModel,
  isZImageQwen3Encoder,
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

  it('gives Anima the Qwen3 0.6B encoder and Z-Image only the 4B one', () => {
    const qwen06b = candidate({ variant: 'qwen3_06b' });
    const qwen4b = candidate({ variant: 'qwen3_4b' });
    const qwen8b = candidate({ variant: 'qwen3_8b' });

    expect([qwen06b, qwen4b, qwen8b].map(isAnimaQwen3Encoder)).toEqual([true, false, false]);
    // The 8B encoder is 4096 wide; Z-Image's caption embedder takes 2560.
    expect([qwen06b, qwen4b, qwen8b].map(isZImageQwen3Encoder)).toEqual([false, true, false]);
    expect(isZImageQwen3Encoder(candidate({ type: 'qwen3_vl_encoder', variant: 'qwen3_4b' }))).toBe(false);
  });

  it('matches FLUX.2 Klein models to compatible Qwen3 encoder variants', () => {
    const filter = isFlux2Qwen3EncoderForModel(flux2Model('klein_9b'));

    expect(filter(candidate({ variant: 'qwen3_8b' }))).toBe(true);
    expect(filter(candidate({ variant: 'qwen3_4b' }))).toBe(false);
    expect(filter(candidate({ variant: 'qwen3_06b' }))).toBe(false);
  });

  it('lets a FLUX.2 Klein model of unknown size take either Klein encoder but never the Anima one', () => {
    const filter = isFlux2Qwen3EncoderForModel(flux2Model('klein_unknown'));

    expect(filter(candidate({ variant: 'qwen3_4b' }))).toBe(true);
    expect(filter(candidate({ variant: 'qwen3_8b' }))).toBe(true);
    expect(filter(candidate({ variant: 'qwen3_06b' }))).toBe(false);
    expect(filter(candidate({ type: 'mistral_encoder', variant: 'qwen3_8b' }))).toBe(false);
  });

  it('matches FLUX.2 component sources by shared Qwen3 encoder variant', () => {
    const filter = isFlux2EncoderSourceForModel(flux2Model('klein_9b'));

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
      isFlux2EncoderSourceForModel(model)(
        candidate({ base: 'flux2', format: 'diffusers', type: 'main', variant: 'dev' })
      )
    ).toBe(true);
    expect(
      isFlux2EncoderSourceForModel(model)(
        candidate({ base: 'flux2', format: 'diffusers', type: 'main', variant: 'klein_9b' })
      )
    ).toBe(false);
  });

  it('takes FLUX.2 [dev] components only from Diffusers pipelines and Klein ones from SDNQ pipelines too', () => {
    const pipeline = (format: string, variant: string) =>
      candidate({
        base: 'flux2',
        format,
        type: 'main',
        variant,
        ...(format === 'sdnq_quantized'
          ? { submodels: { text_encoder: {}, tokenizer: {}, transformer: {}, vae: {} } }
          : {}),
      });
    const sdnqKlein = pipeline('sdnq_quantized', 'klein_9b');
    const sdnqDev = pipeline('sdnq_quantized', 'dev');
    const diffusersKlein = pipeline('diffusers', 'klein_9b');
    const diffusersDev = pipeline('diffusers', 'dev');
    const pipelines = [sdnqKlein, sdnqDev, diffusersKlein, diffusersDev];

    // Any Diffusers pipeline gives [dev] its VAE; only a [dev] one its Mistral encoder.
    expect(pipelines.map(isFlux2PipelineForModel(flux2Model('dev')))).toEqual([false, false, true, true]);
    expect(pipelines.map(isFlux2EncoderSourceForModel(flux2Model('dev')))).toEqual([false, false, false, true]);
    expect(pipelines.map(isFlux2PipelineForModel(flux2Model('klein_9b')))).toEqual([true, true, true, true]);
    expect(pipelines.map(isFlux2EncoderSourceForModel(flux2Model('klein_9b')))).toEqual([true, false, true, false]);
    // The selected install counts as bundled by the same rule.
    expect(pipelines.map(isBundledFlux2Main)).toEqual([true, false, true, true]);
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

  it('asks the served VAE rule only about VAEs, and never for an external model', () => {
    // The rule itself is Models' (`architectureAcceptsVae`); these check what Generation hands it.
    expect(isVaeAcceptedByBase('anima')(candidate({ base: 'anima', type: 'vae' }))).toBe(true);
    expect(isVaeAcceptedByBase('anima')(candidate({ base: 'anima', type: 'lora' }))).toBe(false);
    expect(
      isVaeCompatibleWithGenerateModel(
        {
          base: 'external',
          key: 'gpt-image',
          name: 'GPT Image',
          type: 'external_image_generator',
        } as GenerateModelConfig,
        { base: 'external', key: 'vae', name: 'VAE', type: 'vae' } as VaeModelConfig
      )
    ).toBe(false);
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
