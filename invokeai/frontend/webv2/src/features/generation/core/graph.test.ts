import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type {
  ComponentModelConfig,
  ExternalImageGeneratorModelConfig,
  GenerateModelConfig,
  GenerateSettings,
  LoraModelConfig,
  MainModelConfig,
  VaeModelConfig,
} from './types';

import { getDefaultGenerateSettings, isSupportedGenerateModel } from './baseGenerationPolicies';
import { compileGenerateGraph, resolveGenerateSeed } from './graph';
import { generateGraphCases, satisfiedSettingsFor } from './graphCoverage.testing';

const sd1Model: MainModelConfig = { base: 'sd-1', key: 'sd1-model', name: 'SD 1.5', type: 'main' };
const sd2Model: MainModelConfig = { base: 'sd-2', key: 'sd2-model', name: 'SD 2', type: 'main' };
const sdxlModel: MainModelConfig = { base: 'sdxl', key: 'sdxl-model', name: 'SDXL', type: 'main' };
const sd3Model: MainModelConfig = { base: 'sd-3', key: 'sd3-model', name: 'SD3', type: 'main' };
const fluxModel: MainModelConfig = { base: 'flux', key: 'flux-model', name: 'FLUX dev', type: 'main' };
const flux2Model: MainModelConfig = {
  base: 'flux2',
  format: 'diffusers',
  key: 'flux2-model',
  name: 'FLUX.2',
  type: 'main',
};
const flux2Klein9bModel: MainModelConfig = {
  base: 'flux2',
  format: 'gguf_quantized',
  key: 'flux2-klein-9b',
  name: 'FLUX.2 Klein 9B',
  type: 'main',
  variant: 'klein_9b',
};
const completeSdnqPipeline = { text_encoder: {}, tokenizer: {}, transformer: {}, vae: {} };
const completeFlux1SdnqPipeline = { ...completeSdnqPipeline, text_encoder_2: {}, tokenizer_2: {} };
const flux2DevModel: MainModelConfig = {
  base: 'flux2',
  format: 'diffusers',
  key: 'flux2-dev',
  name: 'FLUX.2 [dev]',
  type: 'main',
  variant: 'dev',
};
const flux2DevGgufModel: MainModelConfig = {
  ...flux2DevModel,
  format: 'gguf_quantized',
  key: 'flux2-dev-gguf',
};
const flux2DevSource: MainModelConfig = {
  ...flux2DevModel,
  key: 'flux2-dev-source',
};
const fluxKontextModel: MainModelConfig = {
  base: 'flux',
  format: 'diffusers',
  key: 'flux-kontext',
  name: 'FLUX Kontext dev',
  type: 'main',
};
const incompatibleFlux2Source: MainModelConfig = {
  base: 'flux2',
  format: 'diffusers',
  key: 'flux2-klein-4b-source',
  name: 'FLUX.2 Klein 4B Source',
  type: 'main',
  variant: 'klein_4b',
};
const cogView4Model: MainModelConfig = { base: 'cogview4', key: 'cog-model', name: 'CogView4', type: 'main' };
const qwenImageModel: MainModelConfig = {
  base: 'qwen-image',
  format: 'diffusers',
  key: 'qwen-model',
  name: 'Qwen Image',
  type: 'main',
};
const zImageModel: MainModelConfig = {
  base: 'z-image',
  format: 'diffusers',
  key: 'z-model',
  name: 'Z-Image',
  type: 'main',
};
const animaModel: MainModelConfig = { base: 'anima', key: 'anima-model', name: 'Anima', type: 'main' };
const externalModel: GenerateModelConfig = {
  base: 'external',
  capabilities: { modes: ['txt2img'], supports_seed: true },
  format: 'external_api',
  key: 'external-model',
  name: 'OpenAI Image',
  provider_id: 'openai',
  type: 'external_image_generator',
};
const sd1Lora: LoraModelConfig = { base: 'sd-1', key: 'sd1-lora', name: 'SD 1.5 LoRA', type: 'lora' };
const sd2Lora: LoraModelConfig = { base: 'sd-2', key: 'sd2-lora', name: 'SD 2 LoRA', type: 'lora' };
const sdxlLora: LoraModelConfig = { base: 'sdxl', key: 'sdxl-lora', name: 'SDXL LoRA', type: 'lora' };
const fluxLora: LoraModelConfig = { base: 'flux', key: 'flux-lora', name: 'FLUX LoRA', type: 'lora' };
const flux2DevLora: LoraModelConfig = {
  base: 'flux2',
  key: 'flux2-dev-lora',
  name: 'FLUX.2 [dev] LoRA',
  type: 'lora',
  variant: 'dev',
};
const sd1IpAdapter: ComponentModelConfig = {
  base: 'sd-1',
  key: 'sd1-ip-adapter',
  name: 'SD 1 IP Adapter',
  type: 'ip_adapter',
};
const fluxRedux: ComponentModelConfig = {
  base: 'flux',
  key: 'flux-redux',
  name: 'FLUX Redux',
  type: 'flux_redux',
};
const vae: VaeModelConfig = { base: 'sd-1', key: 'vae-model', name: 'Custom VAE', type: 'vae' };
const fluxVae: VaeModelConfig = { base: 'flux', key: 'flux-vae', name: 'FLUX VAE', type: 'vae' };
const flux2Vae: VaeModelConfig = { base: 'flux2', key: 'flux2-vae', name: 'FLUX.2 VAE', type: 'vae' };
const qwenImageVae: VaeModelConfig = { base: 'qwen-image', key: 'qwen-vae', name: 'Qwen VAE', type: 'vae' };
const t5Encoder: ComponentModelConfig = { base: 'any', key: 't5', name: 'T5 Encoder', type: 't5_encoder' };
const clipEmbed: ComponentModelConfig = {
  base: 'any',
  key: 'clip',
  name: 'CLIP Embed',
  type: 'clip_embed',
  variant: 'large',
};
const mistralEncoder: ComponentModelConfig = {
  base: 'any',
  key: 'mistral',
  name: 'Mistral Encoder',
  type: 'mistral_encoder',
};
const ministralEncoder: ComponentModelConfig = {
  base: 'any',
  key: 'ministral',
  name: 'Ministral 3B Encoder',
  type: 'mistral_encoder',
  variant: 'ministral3_3b',
};
const qwen3Encoder: ComponentModelConfig = {
  base: 'any',
  key: 'qwen3',
  name: 'Qwen3 Encoder',
  type: 'qwen3_encoder',
  variant: 'qwen3_06b',
};

const createSettings = (model: GenerateModelConfig, overrides: Partial<GenerateSettings> = {}): GenerateSettings => ({
  ...getDefaultGenerateSettings(model),
  seed: 1,
  seedMode: 'fixed',
  ...overrides,
});

const refImage = {
  original: { image: { height: 768, image_name: 'reference.png', width: 512 } },
};

const compile = (model: GenerateModelConfig, overrides: Partial<GenerateSettings> = {}) =>
  compileGenerateGraph(createSettings(model, overrides), model, 'gallery', {
    useCpuNoise: true,
    useNsfwChecker: false,
    useWatermark: false,
  }).backendGraph;

const getEdge = (graph: ReturnType<typeof compile>, targetNodeId: string, targetField: string) =>
  graph.edges.find((edge) => edge.destination.node_id === targetNodeId && edge.destination.field === targetField);

const getNodeByType = (graph: ReturnType<typeof compile>, type: string) =>
  Object.values(graph.nodes).find((node) => node.type === type);

afterEach(() => {
  vi.restoreAllMocks();
});

seedArchitectureCapabilities();

describe('compileGenerateGraph', () => {
  it('recognizes the legacy-supported generate model families', () => {
    expect(
      [
        sd1Model,
        sdxlModel,
        sd3Model,
        fluxModel,
        flux2Model,
        cogView4Model,
        qwenImageModel,
        zImageModel,
        animaModel,
        externalModel,
      ].every(isSupportedGenerateModel)
    ).toBe(true);
  });

  it('builds a plain txt2img graph without optional nodes', () => {
    const graph = compile(sd1Model);

    expect(graph.nodes.seamless).toBeUndefined();
    expect(graph.nodes.vae_loader).toBeUndefined();
    expect(getEdge(graph, 'denoise_latents', 'unet')?.source.node_id).toBe('model_loader');
    expect(getEdge(graph, 'canvas_output', 'vae')?.source.node_id).toBe('model_loader');
    expect(graph.nodes.canvas_output?.fp32).toBe(true);
    expect(graph.nodes.canvas_output?.color_compensation).toBe('None');
  });

  it('records the accelerator supplied by the orchestration boundary in metadata', () => {
    const graph = compileGenerateGraph(
      createSettings(sd1Model),
      sd1Model,
      'gallery',
      { useCpuNoise: false, useNsfwChecker: false, useWatermark: false },
      'xpu'
    ).backendGraph;

    expect(getNodeByType(graph, 'core_metadata')?.rand_device).toBe('xpu');
  });

  it('passes SDXL color compensation to the VAE decode node when enabled', () => {
    const graph = compile(sdxlModel, { colorCompensation: true });

    expect(graph.nodes.canvas_output?.color_compensation).toBe('SDXL');
  });

  it('coerces invalid SD schedulers before graph submission', () => {
    const graph = compile(sd1Model, { scheduler: 'not-a-scheduler' });
    const metadata = getNodeByType(graph, 'core_metadata');

    expect(graph.nodes.denoise_latents?.scheduler).toBe('euler_a');
    expect(metadata?.scheduler).toBe('euler_a');
  });

  it('wires CLIP skip into the SD1 conditioning chain', () => {
    const graph = compile(sd1Model, { clipSkip: 2 });

    expect(graph.nodes.clip_skip?.skipped_layers).toBe(2);
  });

  it.each([sd1Model, sdxlModel])('passes enabled HiDiffusion settings to $base denoise and metadata', (model) => {
    const graph = compile(model, {
      hiDiffusionEnabled: true,
      hiDiffusionRauNetEnabled: false,
      hiDiffusionT1Ratio: 0.35,
      hiDiffusionT2Ratio: 0.15,
      hiDiffusionWindowAttentionEnabled: true,
    });
    const denoise = graph.nodes.denoise_latents;
    const metadata = getNodeByType(graph, 'core_metadata');

    expect(denoise).toMatchObject({
      hidiffusion: true,
      hidiffusion_raunet: false,
      hidiffusion_t1_ratio: 0.35,
      hidiffusion_t2_ratio: 0.15,
      hidiffusion_window_attn: true,
    });
    expect(metadata).toMatchObject({
      hidiffusion: true,
      hidiffusion_raunet: false,
      hidiffusion_t1_ratio: 0.35,
      hidiffusion_t2_ratio: 0.15,
      hidiffusion_window_attn: true,
    });
  });

  it('omits HiDiffusion ratios from denoise while disabled but records all metadata settings', () => {
    const graph = compile(sd1Model, {
      hiDiffusionEnabled: false,
      hiDiffusionRauNetEnabled: false,
      hiDiffusionT1Ratio: 0.6,
      hiDiffusionT2Ratio: 0.2,
      hiDiffusionWindowAttentionEnabled: true,
    });
    const denoise = graph.nodes.denoise_latents;
    const metadata = getNodeByType(graph, 'core_metadata');

    expect(denoise).toMatchObject({
      hidiffusion: false,
      hidiffusion_raunet: false,
      hidiffusion_window_attn: true,
    });
    expect(denoise).not.toHaveProperty('hidiffusion_t1_ratio');
    expect(denoise).not.toHaveProperty('hidiffusion_t2_ratio');
    expect(metadata).toMatchObject({
      hidiffusion: false,
      hidiffusion_raunet: false,
      hidiffusion_t1_ratio: 0.6,
      hidiffusion_t2_ratio: 0.2,
      hidiffusion_window_attn: true,
    });
  });

  it('omits HiDiffusion inputs and metadata for unsupported SD2 models', () => {
    const graph = compile(sd2Model, {
      hiDiffusionEnabled: true,
      hiDiffusionRauNetEnabled: true,
      hiDiffusionT1Ratio: 0.4,
      hiDiffusionT2Ratio: 0,
      hiDiffusionWindowAttentionEnabled: true,
    });
    const metadata = getNodeByType(graph, 'core_metadata');

    for (const key of [
      'hidiffusion',
      'hidiffusion_raunet',
      'hidiffusion_t1_ratio',
      'hidiffusion_t2_ratio',
      'hidiffusion_window_attn',
    ]) {
      expect(graph.nodes.denoise_latents).not.toHaveProperty(key);
      expect(metadata).not.toHaveProperty(key);
    }
  });

  it('routes the UNet and VAE through the seamless node when tiling is enabled', () => {
    const graph = compile(sd1Model, { seamlessXAxis: true, vae });

    expect(graph.nodes.seamless?.seamless_x).toBe(true);
    expect(graph.nodes.seamless?.seamless_y).toBe(false);
    expect(graph.nodes.vae_loader?.vae_model).toEqual(vae);
    expect(getEdge(graph, 'seamless', 'unet')?.source.node_id).toBe('model_loader');
    expect(getEdge(graph, 'seamless', 'vae')?.source.node_id).toBe('vae_loader');
    expect(getEdge(graph, 'denoise_latents', 'unet')?.source.node_id).toBe('seamless');
    expect(getEdge(graph, 'canvas_output', 'vae')?.source.node_id).toBe('seamless');
  });

  it('uses a VAE override only when it matches the model base', () => {
    const matching = compile(sd1Model, { vae, vaePrecision: 'fp16' });

    expect(getEdge(matching, 'canvas_output', 'vae')?.source.node_id).toBe('vae_loader');
    expect(matching.nodes.canvas_output?.fp32).toBe(false);

    const mismatched = compile(sdxlModel, { vae });

    expect(mismatched.nodes.vae_loader).toBeUndefined();
    expect(getEdge(mismatched, 'canvas_output', 'vae')?.source.node_id).toBe('model_loader');
  });

  it('keeps an SD3 VAE override its picker would hide out of the graph', () => {
    // Builders must filter stale optional overrides that required-slot validation never sees.
    const graph = compile(sd3Model, { vae: fluxVae });

    expect(graph.nodes.model_loader?.vae_model).toBeUndefined();
    expect(compile(sd3Model, { vae: { ...fluxVae, base: 'sd-3', key: 'sd3-vae' } }).nodes.model_loader).toMatchObject({
      vae_model: { key: 'sd3-vae' },
    });
  });

  it('sends the Qwen Image loader every VAE its picker offers', () => {
    // Picker and compiler must agree for a Qwen VAE registered under anima.
    const animaRegisteredVae: VaeModelConfig = {
      base: 'anima',
      key: 'anima-qwen-vae',
      name: 'Qwen VAE (installed for Anima)',
      type: 'vae',
    };
    const qwenVlEncoder: ComponentModelConfig = {
      base: 'any',
      key: 'qwen-vl',
      name: 'Qwen VL',
      type: 'qwen_vl_encoder',
    };
    const qwenImageCheckpoint: MainModelConfig = { ...qwenImageModel, format: 'checkpoint', key: 'qwen-checkpoint' };

    expect(
      compile(qwenImageCheckpoint, { qwenVLEncoderModel: qwenVlEncoder, vae: animaRegisteredVae }).nodes.model_loader
    ).toMatchObject({ vae_model: animaRegisteredVae });
    expect(compile(qwenImageModel, { vae: animaRegisteredVae }).nodes.model_loader).toMatchObject({
      vae_model: animaRegisteredVae,
    });
  });

  it('routes SD LoRAs through the UNet and CLIP conditioning chain', () => {
    const graph = compile(sd1Model, {
      loras: [{ isEnabled: true, model: sd1Lora, weight: 0.5 }],
    });
    const loraLoader = getNodeByType(graph, 'lora_collection_loader');
    const metadata = getNodeByType(graph, 'core_metadata');

    expect(loraLoader).toBeDefined();
    expect(getEdge(graph, 'denoise_latents', 'unet')?.source.node_id).toBe(loraLoader?.id);
    expect(getEdge(graph, 'pos_cond', 'clip')?.source.node_id).toBe(loraLoader?.id);
    expect(getEdge(graph, 'neg_cond', 'clip')?.source.node_id).toBe(loraLoader?.id);
    expect(metadata?.loras).toEqual([
      { model: { base: 'sd-1', key: 'sd1-lora', name: 'SD 1.5 LoRA', type: 'lora' }, weight: 0.5 },
    ]);
  });

  it('routes SDXL LoRAs through both CLIP encoders', () => {
    const graph = compile(sdxlModel, {
      loras: [{ isEnabled: true, model: sdxlLora, weight: 0.65 }],
    });
    const loraLoader = getNodeByType(graph, 'sdxl_lora_collection_loader');

    expect(loraLoader).toBeDefined();
    expect(getEdge(graph, 'denoise_latents', 'unet')?.source.node_id).toBe(loraLoader?.id);
    expect(getEdge(graph, 'pos_cond', 'clip')?.source.node_id).toBe(loraLoader?.id);
    expect(getEdge(graph, 'pos_cond', 'clip2')?.source.node_id).toBe(loraLoader?.id);
    expect(getEdge(graph, 'neg_cond', 'clip2')?.source.node_id).toBe(loraLoader?.id);
  });

  it('ignores disabled and incompatible LoRAs', () => {
    const graph = compile(sd1Model, {
      loras: [
        { isEnabled: true, model: sd2Lora, weight: 1 },
        { isEnabled: false, model: sd1Lora, weight: 1 },
      ],
    });
    const metadata = getNodeByType(graph, 'core_metadata');

    expect(getNodeByType(graph, 'lora_collection_loader')).toBeUndefined();
    expect(getEdge(graph, 'denoise_latents', 'unet')?.source.node_id).toBe('model_loader');
    expect(metadata?.loras).toBeUndefined();
  });

  it('builds a FLUX graph with required components and FLUX LoRAs', () => {
    const graph = compile(fluxModel, {
      clipEmbedModel: clipEmbed,
      loras: [{ isEnabled: true, model: fluxLora, weight: 0.8 }],
      t5EncoderModel: t5Encoder,
      vae: fluxVae,
    });
    const loraLoader = getNodeByType(graph, 'flux_lora_collection_loader');

    expect(graph.nodes.model_loader?.type).toBe('flux_model_loader');
    expect(graph.nodes.model_loader?.t5_encoder_model).toEqual(t5Encoder);
    expect(graph.nodes.model_loader?.clip_embed_model).toEqual(clipEmbed);
    expect(graph.nodes.denoise_latents?.type).toBe('flux_denoise');
    expect(graph.nodes.canvas_output?.type).toBe('flux_vae_decode');
    expect(loraLoader).toBeDefined();
    expect(getEdge(graph, 'denoise_latents', 'transformer')?.source.node_id).toBe(loraLoader?.id);
  });

  it('fails early when required FLUX components are missing', () => {
    expect(() => compile(fluxModel)).toThrow('Generate needs a T5 Encoder for FLUX models.');
  });

  it('builds modern Diffusers-family graphs without separate component overrides', () => {
    expect(compile(sd3Model).nodes.model_loader?.type).toBe('sd3_model_loader');
    expect(compile(flux2Model).nodes.model_loader?.type).toBe('flux2_klein_model_loader');
    expect(compile(flux2DevModel).nodes.model_loader?.type).toBe('flux2_dev_model_loader');
    expect(compile(cogView4Model).nodes.model_loader?.type).toBe('cogview4_model_loader');
    expect(compile(qwenImageModel).nodes.model_loader?.type).toBe('qwen_image_model_loader');
    expect(compile(zImageModel).nodes.model_loader?.type).toBe('z_image_model_loader');
  });

  it('builds complete SDNQ pipeline graphs without standalone component overrides', () => {
    const flux = compile({
      ...fluxModel,
      format: 'sdnq_quantized',
      submodels: completeFlux1SdnqPipeline,
    });
    const flux2 = compile({
      ...flux2Klein9bModel,
      format: 'sdnq_quantized',
      submodels: completeSdnqPipeline,
    });
    const zImage = compile({
      ...zImageModel,
      format: 'sdnq_quantized',
      submodels: completeSdnqPipeline,
    });

    expect(flux.nodes.model_loader).toMatchObject({
      clip_embed_model: undefined,
      t5_encoder_model: undefined,
      vae_model: undefined,
    });
    expect(flux2.nodes.model_loader).toMatchObject({ qwen3_encoder_model: undefined, vae_model: undefined });
    expect(zImage.nodes.model_loader).toMatchObject({ qwen3_encoder_model: undefined, vae_model: undefined });
  });

  it('sends a standalone Z-Image transformer the Qwen3 4B encoder and refuses the 8B one', () => {
    const zImageGguf: MainModelConfig = { ...zImageModel, format: 'gguf_quantized' };
    const qwen3_4b: ComponentModelConfig = { ...qwen3Encoder, key: 'qwen3-4b', variant: 'qwen3_4b' };
    const qwen3_8b: ComponentModelConfig = { ...qwen3Encoder, key: 'qwen3-8b', variant: 'qwen3_8b' };
    const graph = compile(zImageGguf, { qwen3EncoderModel: qwen3_4b, vae: fluxVae });

    expect(graph.nodes.model_loader).toMatchObject({
      qwen3_encoder_model: qwen3_4b,
      type: 'z_image_model_loader',
      vae_model: fluxVae,
    });
    expect(() => compile(zImageGguf, { qwen3EncoderModel: qwen3_8b, vae: fluxVae })).toThrow(
      'The selected Qwen3 Encoder is not the 4B one Z-Image conditions on.'
    );
  });

  it('builds FLUX.2 [dev] graphs with standalone components and dev LoRAs', () => {
    const graph = compile(flux2DevGgufModel, {
      loras: [{ isEnabled: true, model: flux2DevLora, weight: 0.75 }],
      mistralEncoderModel: mistralEncoder,
      vae: flux2Vae,
    });
    const loraLoader = getNodeByType(graph, 'flux2_dev_lora_collection_loader');
    const metadata = getNodeByType(graph, 'core_metadata');

    expect(graph.nodes.model_loader).toMatchObject({
      mistral_encoder_model: mistralEncoder,
      type: 'flux2_dev_model_loader',
      vae_model: flux2Vae,
    });
    expect(graph.nodes.pos_cond?.type).toBe('flux2_dev_text_encoder');
    expect(loraLoader).toBeDefined();
    expect(getEdge(graph, 'pos_cond', 'mistral_encoder')?.source.node_id).toBe(loraLoader?.id);
    expect(getEdge(graph, 'denoise_latents', 'transformer')?.source.node_id).toBe(loraLoader?.id);
    expect(metadata?.mistral_encoder).toEqual(mistralEncoder);
  });

  it('uses a compatible Diffusers model as the FLUX.2 [dev] component source', () => {
    const graph = compile(flux2DevGgufModel, { componentSourceModel: flux2DevSource });

    expect(graph.nodes.model_loader).toMatchObject({
      mistral_source_model: flux2DevSource,
      type: 'flux2_dev_model_loader',
    });
    expect(graph.nodes.model_loader?.mistral_encoder_model).toBeUndefined();
    expect(graph.nodes.model_loader?.vae_model).toBeUndefined();
  });

  it('uses a complete FLUX.2 SDNQ component source without standalone overrides', () => {
    const source: MainModelConfig = {
      ...flux2Klein9bModel,
      format: 'sdnq_quantized',
      key: 'flux2-klein-9b-sdnq-source',
      submodels: completeSdnqPipeline,
    };
    const graph = compile(flux2Klein9bModel, { componentSourceModel: source });

    expect(graph.nodes.model_loader).toMatchObject({
      qwen3_encoder_model: undefined,
      qwen3_source_model: source,
      vae_model: undefined,
    });
  });

  it('rejects a partial FLUX.2 SDNQ component source during graph compilation', () => {
    const source: MainModelConfig = {
      ...flux2Klein9bModel,
      format: 'sdnq_quantized',
      key: 'flux2-klein-9b-partial-sdnq-source',
      submodels: { text_encoder: {}, transformer: {}, vae: {} },
    };

    expect(() => compile(flux2Klein9bModel, { componentSourceModel: source })).toThrow(
      'Generate needs a Qwen3 Encoder for non-Diffusers FLUX.2 models.'
    );
  });

  it('does not enable FLUX.2 CFG without negative conditioning', () => {
    const graph = compile(flux2Model, { cfgScale: 4 });

    expect(graph.nodes.denoise_latents?.cfg_scale).toBe(1);
    expect(getEdge(graph, 'denoise_latents', 'negative_text_conditioning')).toBeUndefined();
  });

  it('routes FLUX.2 positive conditioning through a stable collector', () => {
    const graph = compile(flux2Model);

    expect(graph.nodes.pos_cond_collect?.type).toBe('collect');
    expect(getEdge(graph, 'pos_cond_collect', 'item')?.source.node_id).toBe('pos_cond');
    expect(getEdge(graph, 'denoise_latents', 'positive_text_conditioning')?.source.node_id).toBe('pos_cond_collect');
  });

  it('rejects a stale incompatible FLUX.2 component source when Qwen3 is missing', () => {
    expect(() => compile(flux2Klein9bModel, { componentSourceModel: incompatibleFlux2Source, vae: flux2Vae })).toThrow(
      'Generate needs a Qwen3 Encoder for non-Diffusers FLUX.2 models.'
    );
  });

  it('rejects dimensions that are not on the selected model family grid', () => {
    expect(() => compile(flux2Model, { height: 888 })).toThrow('Generate height must be a multiple of 16.');
  });

  it('builds an Anima graph with required components', () => {
    expect(() => compile(animaModel)).toThrow('Generate needs a Qwen3 Encoder for Anima models.');

    const graph = compile(animaModel, { qwen3EncoderModel: qwen3Encoder, vae: qwenImageVae });

    expect(graph.nodes.model_loader?.type).toBe('anima_model_loader');
    expect(graph.nodes.model_loader?.qwen3_encoder_model).toEqual(qwen3Encoder);
    expect(graph.nodes.model_loader?.vae_model).toEqual(qwenImageVae);
    expect(graph.nodes.canvas_output?.type).toBe('anima_l2i');
  });

  it('builds an external provider graph', () => {
    const graph = compile(externalModel);

    expect(graph.nodes.canvas_output?.type).toBe('openai_image_generation');
    expect(graph.nodes.canvas_output?.model).toEqual(externalModel);
    expect(getEdge(graph, 'canvas_output', 'prompt')?.source.node_id).toBe('positive_prompt');
    expect(getEdge(graph, 'canvas_output', 'seed')?.source.node_id).toBe('seed');
  });

  it('does not wire unsupported external provider seed or negative prompt inputs', () => {
    const graph = compile({ ...externalModel, capabilities: { modes: ['txt2img'] } });

    expect(getEdge(graph, 'canvas_output', 'prompt')?.source.node_id).toBe('positive_prompt');
    expect(getEdge(graph, 'canvas_output', 'seed')).toBeUndefined();
    expect(getEdge(graph, 'canvas_output', 'negative_prompt')).toBeUndefined();
  });

  it('wires external negative prompt only when the provider advertises support', () => {
    const graph = compile({
      ...externalModel,
      capabilities: { modes: ['txt2img'], supports_negative_prompt: true, supports_seed: false },
    });

    expect(getEdge(graph, 'canvas_output', 'negative_prompt')?.source.node_id).toBe('negative_prompt');
    expect(getEdge(graph, 'canvas_output', 'seed')).toBeUndefined();
  });

  it('wires external provider reference images when supported', () => {
    const graph = compile(
      { ...externalModel, capabilities: { modes: ['txt2img'], supports_reference_images: true } },
      {
        referenceImages: [
          { id: 'ref-1', isEnabled: true, config: { type: 'external_reference_image', image: refImage } },
        ],
      }
    );

    expect(graph.nodes.canvas_output?.reference_images).toEqual([{ image_name: 'reference.png' }]);
  });

  describe('external provider options', () => {
    const providerSettings: Partial<GenerateSettings> = {
      externalImageSize: '2K',
      geminiTemperature: 0.4,
      geminiThinkingLevel: 'high',
      openaiBackground: 'transparent',
      openaiInputFidelity: 'high',
      openaiQuality: 'low',
      seedreamOptimizePrompt: true,
      seedreamWatermark: true,
    };
    const providerModel = (
      providerId: string,
      capabilities: ExternalImageGeneratorModelConfig['capabilities']
    ): GenerateModelConfig =>
      ({
        ...externalModel,
        capabilities,
        provider_id: providerId,
        provider_model_id: providerId === 'openai' ? 'gpt-image-1' : `${providerId}-model`,
      }) as GenerateModelConfig;

    it('compiles OpenAI quality, background and input fidelity, recording them in metadata', () => {
      const model = providerModel('openai', { modes: ['txt2img', 'img2img'] });
      const graph = compile(model, providerSettings);

      expect(graph.nodes.canvas_output).toMatchObject({
        background: 'transparent',
        input_fidelity: 'high',
        quality: 'low',
      });
      // Fields of other providers' nodes never reach this one.
      expect(graph.nodes.canvas_output).not.toHaveProperty('temperature');
      expect(graph.nodes.canvas_output).not.toHaveProperty('watermark');
      expect(getNodeByType(graph, 'core_metadata')).toMatchObject({
        openai_background: 'transparent',
        openai_input_fidelity: 'high',
        openai_quality: 'low',
      });
    });

    it('omits OpenAI input fidelity for a model with no image inputs, and when unset', () => {
      const noInputs = compile(providerModel('openai', { modes: ['txt2img'] }), providerSettings);

      expect(noInputs.nodes.canvas_output).toMatchObject({ background: 'transparent', quality: 'low' });
      expect(noInputs.nodes.canvas_output).not.toHaveProperty('input_fidelity');

      const unset = compile(providerModel('openai', { modes: ['txt2img', 'img2img'] }), {
        ...providerSettings,
        openaiInputFidelity: null,
      });

      expect(unset.nodes.canvas_output).not.toHaveProperty('input_fidelity');
      expect(getNodeByType(unset, 'core_metadata')).not.toHaveProperty('openai_input_fidelity');
    });

    it('compiles Gemini temperature and thinking level only when set', () => {
      const model = providerModel('gemini', { modes: ['txt2img'] });
      const graph = compile(model, providerSettings);

      expect(graph.nodes.canvas_output).toMatchObject({ temperature: 0.4, thinking_level: 'high' });
      expect(graph.nodes.canvas_output).not.toHaveProperty('quality');
      expect(getNodeByType(graph, 'core_metadata')).toMatchObject({
        gemini_temperature: 0.4,
        gemini_thinking_level: 'high',
      });

      const defaults = compile(model, { ...providerSettings, geminiTemperature: null, geminiThinkingLevel: null });

      expect(defaults.nodes.canvas_output).not.toHaveProperty('temperature');
      expect(defaults.nodes.canvas_output).not.toHaveProperty('thinking_level');
    });

    it('compiles Seedream watermark and prompt optimization as explicit booleans', () => {
      const model = providerModel('seedream', { modes: ['txt2img'] });

      expect(compile(model, providerSettings).nodes.canvas_output).toMatchObject({
        optimize_prompt: true,
        watermark: true,
      });
      expect(
        compile(model, { ...providerSettings, seedreamOptimizePrompt: false, seedreamWatermark: false }).nodes
          .canvas_output
      ).toMatchObject({ optimize_prompt: false, watermark: false });
      expect(getNodeByType(compile(model, providerSettings), 'core_metadata')).toMatchObject({
        seedream_optimize_prompt: true,
        seedream_watermark: true,
      });
    });

    it('sends the named size only when the model lists it in a resolution preset', () => {
      const presets = [
        { aspect_ratio: '1:1', height: 2048, image_size: '2K', label: '1:1 (2K)', width: 2048 },
        { aspect_ratio: '1:1', height: 1024, image_size: '1K', label: '1:1 (1K)', width: 1024 },
      ];
      const listed = compile(providerModel('gemini', { modes: ['txt2img'], resolution_presets: presets }), {
        ...providerSettings,
        height: 2048,
        width: 2048,
      });

      expect(listed.nodes.canvas_output).toMatchObject({ height: 2048, image_size: '2K', width: 2048 });
      expect(getNodeByType(listed, 'core_metadata')).toMatchObject({ image_size: '2K' });

      // A size picked for another model, or a provider without named sizes, falls back to the provider default.
      const unlisted = compile(providerModel('gemini', { modes: ['txt2img'] }), providerSettings);

      expect(unlisted.nodes.canvas_output?.image_size).toBeNull();
      expect(getNodeByType(unlisted, 'core_metadata')).not.toHaveProperty('image_size');

      // Dimensions typed after the pick are custom: the named size would otherwise override them at the provider.
      const typed = compile(providerModel('gemini', { modes: ['txt2img'], resolution_presets: presets }), {
        ...providerSettings,
        height: 2048,
        width: 1536,
      });

      expect(typed.nodes.canvas_output).toMatchObject({ height: 2048, image_size: null, width: 1536 });
      expect(getNodeByType(typed, 'core_metadata')).not.toHaveProperty('image_size');
    });

    it('accepts provider sizes that are not multiples of a local model grid', () => {
      const model = providerModel('gemini', { modes: ['txt2img'] });
      const graph = compile(model, { height: 2048, width: 1365 });

      expect(graph.nodes.canvas_output).toMatchObject({ height: 2048, width: 1365 });
    });
  });

  it('wires SD IP Adapter reference images into denoise', () => {
    const graph = compile(sd1Model, {
      referenceImages: [
        {
          id: 'ref-1',
          isEnabled: true,
          config: {
            beginEndStepPct: [0.1, 0.8],
            clipVisionModel: 'ViT-H',
            image: refImage,
            method: 'style',
            model: sd1IpAdapter,
            type: 'ip_adapter',
            weight: 0.6,
          },
        },
      ],
    });
    const adapter = getNodeByType(graph, 'ip_adapter');

    expect(adapter).toMatchObject({
      begin_step_percent: 0.1,
      clip_vision_model: 'ViT-H',
      end_step_percent: 0.8,
      image: { image_name: 'reference.png' },
      ip_adapter_model: sd1IpAdapter,
      method: 'style',
      weight: 0.6,
    });
    expect(getEdge(graph, 'denoise_latents', 'ip_adapter')?.source.node_id).toContain('ip_adapter_collector');
  });

  it('generates from the crop while serializing complete original and crop metadata', () => {
    const croppedReferenceImage = {
      crop: {
        box: { height: 256, width: 320, x: 40, y: 50 },
        image: { height: 256, image_name: 'reference-crop.png', width: 320 },
        ratio: null,
      },
      original: { image: { height: 768, image_name: 'reference-original.png', width: 512 } },
    };
    const graph = compile(sd1Model, {
      referenceImages: [
        {
          config: {
            beginEndStepPct: [0, 1],
            clipVisionModel: 'ViT-H',
            image: croppedReferenceImage,
            method: 'full',
            model: sd1IpAdapter,
            type: 'ip_adapter',
            weight: 1,
          },
          id: 'cropped-ref',
          isEnabled: true,
        },
      ],
    });
    const metadata = getNodeByType(graph, 'core_metadata');

    expect(getNodeByType(graph, 'ip_adapter')?.image).toEqual({ image_name: 'reference-crop.png' });
    expect(metadata?.ref_images).toMatchObject([
      {
        config: { image: croppedReferenceImage },
        id: 'cropped-ref',
        isEnabled: true,
      },
    ]);
  });

  it('wires FLUX Redux reference images into denoise', () => {
    const graph = compile(fluxModel, {
      clipEmbedModel: clipEmbed,
      referenceImages: [
        {
          id: 'ref-redux',
          isEnabled: true,
          config: { image: refImage, imageInfluence: 'medium', model: fluxRedux, type: 'flux_redux' },
        },
      ],
      t5EncoderModel: t5Encoder,
      vae: fluxVae,
    });
    const redux = getNodeByType(graph, 'flux_redux');

    expect(redux).toMatchObject({ image: { image_name: 'reference.png' }, redux_model: fluxRedux });
    expect(getEdge(graph, 'denoise_latents', 'redux_conditioning')?.source.node_id).toContain('flux_redux_collector');
  });

  it('wires FLUX Kontext reference images into denoise', () => {
    const graph = compile(fluxKontextModel, {
      clipEmbedModel: clipEmbed,
      referenceImages: [
        {
          id: 'ref-kontext',
          isEnabled: true,
          config: { image: refImage, model: fluxKontextModel, type: 'flux_kontext_reference_image' },
        },
      ],
      t5EncoderModel: t5Encoder,
      vae: fluxVae,
    });

    expect(getNodeByType(graph, 'flux_kontext')).toBeDefined();
    expect(getEdge(graph, 'denoise_latents', 'kontext_conditioning')?.source.node_id).toContain('flux_kontext_collect');
  });

  it('wires FLUX.2 built-in reference images into denoise', () => {
    const graph = compile(flux2Model, {
      referenceImages: [
        { id: 'ref-flux2', isEnabled: true, config: { image: refImage, type: 'flux2_reference_image' } },
      ],
    });

    expect(getNodeByType(graph, 'flux_kontext')).toBeDefined();
    expect(getEdge(graph, 'denoise_latents', 'kontext_conditioning')?.source.node_id).toContain(
      'flux2_kontext_collect'
    );
  });

  it('wires Qwen Image Edit reference images into text encoder and denoise latents', () => {
    const graph = compile(
      { ...qwenImageModel, variant: 'edit' },
      {
        referenceImages: [
          { id: 'ref-qwen', isEnabled: true, config: { image: refImage, type: 'qwen_image_reference_image' } },
        ],
      }
    );

    expect(getEdge(graph, 'pos_cond', 'reference_images')).toBeDefined();
    expect(getEdge(graph, 'denoise_latents', 'reference_latents')).toBeDefined();
  });
});

describe('generate seeds', () => {
  it('keeps the entered seed in every mode but random', () => {
    expect(resolveGenerateSeed(createSettings(sdxlModel, { seed: 123, seedMode: 'fixed' }))).toBe(123);
    expect(resolveGenerateSeed(createSettings(sdxlModel, { seed: 123, seedMode: 'increment' }))).toBe(123);
    expect(resolveGenerateSeed(createSettings(sdxlModel, { seed: 123, seedMode: 'decrement' }))).toBe(123);
  });

  it('draws a fresh seed in random mode', () => {
    vi.spyOn(Math, 'random').mockReturnValue(0.5);

    const settings = createSettings(sdxlModel, { seed: 123, seedMode: 'random' });

    expect(resolveGenerateSeed(settings)).toBe(2147483647);
  });
});

describe('Krea-2, Ideogram 4 and Wan graphs', () => {
  const krea2Diffusers: MainModelConfig = {
    base: 'krea-2',
    format: 'diffusers',
    key: 'krea2-diffusers',
    name: 'Krea-2 Turbo',
    type: 'main',
    variant: 'krea2_turbo',
  };
  const krea2Checkpoint: MainModelConfig = {
    base: 'krea-2',
    format: 'checkpoint',
    key: 'krea2-checkpoint',
    name: 'Krea-2 Raw',
    type: 'main',
    variant: 'krea2_base',
  };
  const ideogram4Model: MainModelConfig = {
    base: 'ideogram-4',
    format: 'diffusers',
    key: 'ideogram4',
    name: 'Ideogram 4',
    type: 'main',
  };
  const ideogram4Conditional: MainModelConfig = {
    base: 'ideogram-4',
    branch: 'conditional',
    format: 'checkpoint',
    key: 'ideogram4-cond',
    name: 'Ideogram 4 (single file, fp8)',
    type: 'main',
  };
  const ideogram4Unconditional: MainModelConfig = {
    base: 'ideogram-4',
    branch: 'unconditional',
    format: 'checkpoint',
    key: 'ideogram4-uncond',
    name: 'Ideogram 4 Unconditional (single file, fp8)',
    type: 'main',
  };
  const ideogram4Vae: VaeModelConfig = { base: 'flux2', key: 'flux2-vae', name: 'FLUX.2 VAE', type: 'vae' };
  const qwen3Vl8bEncoder: ComponentModelConfig = {
    base: 'any',
    key: 'qwen3-vl-8b',
    name: 'Qwen3-VL 8B',
    type: 'qwen3_vl_encoder',
    variant: 'qwen3_vl_8b',
  };
  const wanDiffusers: MainModelConfig = {
    base: 'wan',
    format: 'diffusers',
    key: 'wan-diffusers',
    name: 'Wan 2.2 TI2V 5B',
    type: 'main',
    variant: 'ti2v_5b',
  };
  const krea2Vae: VaeModelConfig = { base: 'qwen-image', key: 'krea2-vae', name: 'Qwen VAE', type: 'vae' };
  const qwen3VlEncoder: ComponentModelConfig = {
    base: 'any',
    key: 'qwen3-vl',
    name: 'Qwen3-VL',
    type: 'qwen3_vl_encoder',
    variant: 'qwen3_vl_4b',
  };
  const wanLowNoise: MainModelConfig = {
    base: 'wan',
    format: 'diffusers',
    key: 'wan-low',
    name: 'Wan low noise',
    type: 'main',
    variant: 'i2v_a14b',
  };

  it('decodes Krea-2 with the Qwen-Image VAE node it shares', () => {
    const graph = compile(krea2Diffusers);

    expect(graph.nodes.model_loader?.type).toBe('krea2_model_loader');
    expect(graph.nodes.canvas_output?.type).toBe('qwen_image_l2i');
    expect(graph.nodes.denoise_latents?.type).toBe('krea2_denoise');
  });

  it('omits Krea-2 negative conditioning while CFG is disabled, and adds it when enabled', () => {
    // Krea-2-Turbo runs at cfg_scale 1 by default, where the model takes no negative input.
    expect(compile(krea2Diffusers, { cfgScale: 1 }).nodes.neg_cond).toBeUndefined();

    const withCfg = compile(krea2Diffusers, { cfgScale: 4.5 });

    expect(withCfg.nodes.neg_cond?.type).toBe('krea2_text_encoder');
    expect(getEdge(withCfg, 'denoise_latents', 'negative_conditioning')).toBeDefined();
  });

  it('passes standalone Krea-2 submodels to the loader for a non-diffusers transformer', () => {
    const graph = compile(krea2Checkpoint, { qwen3VLEncoderModel: qwen3VlEncoder, vae: krea2Vae });

    expect(graph.nodes.model_loader).toMatchObject({
      qwen3_vl_encoder_model: qwen3VlEncoder,
      vae_model: krea2Vae,
    });
  });

  it('sends an anima-registered Qwen-Image VAE to the Krea-2 loader', () => {
    // Graph compilation accepts compatible cross-registration VAEs.
    const animaRegisteredVae: VaeModelConfig = {
      base: 'anima',
      key: 'anima-qwen-vae',
      name: 'Qwen VAE (installed for Anima)',
      type: 'vae',
    };
    const graph = compile(krea2Checkpoint, {
      qwen3VLEncoderModel: qwen3VlEncoder,
      vae: animaRegisteredVae,
    });

    expect(graph.nodes.model_loader).toMatchObject({ vae_model: animaRegisteredVae });
  });

  it('refuses to compile a non-diffusers Krea-2 with a missing submodel', () => {
    // User validation errors must precede internal builder errors.
    expect(() => compile(krea2Checkpoint, { qwen3VLEncoderModel: qwen3VlEncoder })).toThrow(
      'Generate needs a VAE for non-Diffusers Krea-2 models.'
    );
    expect(() => compile(krea2Checkpoint, { vae: krea2Vae })).toThrow(
      'Generate needs a Qwen3-VL Encoder for non-Diffusers Krea-2 models.'
    );
  });

  it('leaves the Krea-2 conditioning path untouched while both enhancers are off', () => {
    const graph = compile(krea2Diffusers);

    expect(getNodeByType(graph, 'krea2_conditioning_rebalance')).toBeUndefined();
    expect(getNodeByType(graph, 'krea2_seed_variance')).toBeUndefined();
    expect(getEdge(graph, 'pos_cond_collect', 'item')?.source.node_id).toBe('pos_cond');
    expect(getEdge(graph, 'denoise_latents', 'positive_conditioning')?.source.node_id).toBe('pos_cond_collect');
  });

  it('chains rebalance then seed variance between the Krea-2 encoder and denoise', () => {
    const graph = compile(krea2Diffusers, {
      krea2RebalanceEnabled: true,
      krea2SeedVarianceEnabled: true,
      krea2SeedVarianceStrength: 0.2,
    });

    expect(getEdge(graph, 'krea2_rebalance', 'conditioning')?.source.node_id).toBe('pos_cond');
    expect(getEdge(graph, 'krea2_seed_variance', 'conditioning')?.source.node_id).toBe('krea2_rebalance');
    expect(getEdge(graph, 'pos_cond_collect', 'item')?.source.node_id).toBe('krea2_seed_variance');
    expect(getEdge(graph, 'denoise_latents', 'positive_conditioning')?.source.node_id).toBe('pos_cond_collect');
    // Seed variance needs the seed to be reproducible run-to-run.
    expect(getEdge(graph, 'krea2_seed_variance', 'variance_seed')?.source.node_id).toBe('seed');
  });

  it('skips Krea-2 seed variance at zero strength, which would be a no-op node', () => {
    const graph = compile(krea2Diffusers, { krea2SeedVarianceEnabled: true, krea2SeedVarianceStrength: 0 });

    expect(getNodeByType(graph, 'krea2_seed_variance')).toBeUndefined();
  });

  it('routes the Ideogram 4 prompt through the caption builder and omits negative conditioning', () => {
    const graph = compile(ideogram4Model);

    expect(getEdge(graph, 'caption_builder', 'prompt')?.source.node_id).toBe('positive_prompt');
    expect(getEdge(graph, 'pos_cond', 'prompt')?.source.node_id).toBe('caption_builder');
    // Ideogram 4's denoise node has no negative conditioning input at all.
    expect(graph.nodes.neg_cond).toBeUndefined();
    expect(getEdge(graph, 'denoise_latents', 'negative_conditioning')).toBeUndefined();
  });

  it('omits unset Ideogram 4 overrides so the sampler preset keeps deciding', () => {
    const graph = compile(ideogram4Model);

    expect(graph.nodes.denoise_latents).toMatchObject({ sampler_preset: 'V4_QUALITY_48' });
    expect(graph.nodes.denoise_latents).not.toHaveProperty('steps');
    expect(graph.nodes.denoise_latents).not.toHaveProperty('guidance_scale');
    expect(graph.nodes.denoise_latents).not.toHaveProperty('mu');
  });

  it('sends Ideogram 4 overrides once set', () => {
    const graph = compile(ideogram4Model, {
      ideogram4GuidanceScale: 6,
      ideogram4Mu: 1.5,
      ideogram4SamplerPreset: 'V4_TURBO_12',
      ideogram4Steps: 12,
    });

    expect(graph.nodes.denoise_latents).toMatchObject({
      guidance_scale: 6,
      mu: 1.5,
      sampler_preset: 'V4_TURBO_12',
      steps: 12,
    });
  });

  it('wires both Ideogram 4 branches and its standalone components for a single-file transformer', () => {
    // Wire both Ideogram branches for every denoise step.
    const graph = compile(ideogram4Conditional, {
      ideogram4UnconditionalModel: ideogram4Unconditional,
      qwen3VLEncoderModel: qwen3Vl8bEncoder,
      vae: ideogram4Vae,
    });

    expect(graph.nodes.model_loader).toMatchObject({
      qwen3_vl_encoder_model: qwen3Vl8bEncoder,
      unconditional_model: ideogram4Unconditional,
      vae_model: ideogram4Vae,
    });
    expect(getEdge(graph, 'denoise_latents', 'unconditional_transformer')?.source).toMatchObject({
      field: 'unconditional_transformer',
      node_id: 'model_loader',
    });
    // Two files made the image; recording one of them would not identify what ran.
    expect(getNodeByType(graph, 'core_metadata')).toMatchObject({
      ideogram4_unconditional_model: ideogram4Unconditional,
      qwen3_vl_encoder: qwen3Vl8bEncoder,
    });
  });

  it('drops a component selection that does not suit the Ideogram 4 model about to run', () => {
    // Filter stale optional encoder overrides for bundled models.
    const graph = compile(ideogram4Model, { qwen3VLEncoderModel: qwen3VlEncoder });

    expect(graph.nodes.model_loader?.qwen3_vl_encoder_model).toBeUndefined();
  });

  it('leaves the Ideogram 4 unconditional input unconnected for a diffusers pipeline', () => {
    // The bundled transformer already contains both branches; another branch is invalid.
    const graph = compile(ideogram4Model);

    expect(getEdge(graph, 'denoise_latents', 'unconditional_transformer')).toBeUndefined();
    expect(graph.nodes.model_loader?.unconditional_model).toBeUndefined();
  });

  it('forwards the Ideogram 4 color palette to the caption builder', () => {
    const graph = compile(ideogram4Model, { ideogram4ColorPalette: ['teal', 'ochre'] });

    expect(graph.nodes.caption_builder).toMatchObject({ color_palette: ['teal', 'ochre'] });
  });

  it('emits a single-frame image output for Wan, not a video', () => {
    const graph = compile(wanDiffusers);

    expect(graph.nodes.model_loader?.type).toBe('wan_model_loader');
    expect(graph.nodes.canvas_output?.type).toBe('wan_l2i');
    expect(getNodeByType(graph, 'wan_l2v')).toBeUndefined();
  });

  it('always gives Wan a negative prompt, which it supports unconditionally', () => {
    const graph = compile(wanDiffusers, { cfgScale: 1 });

    expect(graph.nodes.neg_cond?.type).toBe('wan_text_encoder');
    expect(getEdge(graph, 'denoise_latents', 'negative_conditioning')?.source.node_id).toBe('neg_cond');
  });

  it('passes the optional Wan low-noise expert through and leaves it unset otherwise', () => {
    // Omit undefined optional fields instead of serializing null.
    expect(compile(wanDiffusers).nodes.model_loader?.transformer_low_noise_model).toBeUndefined();
    expect(compile(wanDiffusers, { wanLowNoiseModel: wanLowNoise }).nodes.model_loader).toMatchObject({
      transformer_low_noise_model: wanLowNoise,
    });
  });

  it('omits the Wan low-noise guidance unless overridden, letting the backend reuse guidance_scale', () => {
    expect(compile(wanDiffusers).nodes.denoise_latents).not.toHaveProperty('guidance_scale_low_noise');
    expect(compile(wanDiffusers, { wanGuidanceScaleLowNoise: 2.5 }).nodes.denoise_latents).toMatchObject({
      guidance_scale_low_noise: 2.5,
    });
  });

  it('routes Wan LoRAs through the transformer only, leaving the text encoder unpatched', () => {
    const a14bLora: LoraModelConfig = {
      base: 'wan',
      key: 'wan-lora',
      name: '5B LoRA',
      type: 'lora',
      variant: '5b',
    };
    const graph = compile(wanDiffusers, { loras: [{ isEnabled: true, model: a14bLora, weight: 1 }] });
    const loader = getNodeByType(graph, 'wan_lora_collection_loader');

    expect(loader).toBeDefined();
    expect(getEdge(graph, 'denoise_latents', 'transformer')?.source.node_id).toBe(loader?.id);
    // The Wan encoder is fed straight from the model loader — Wan LoRAs do not touch it.
    expect(getEdge(graph, 'pos_cond', 'wan_t5_encoder')?.source.node_id).toBe('model_loader');
  });
});

describe('Qwen-Image-2.1 graphs', () => {
  const qwen21Diffusers: MainModelConfig = {
    base: 'qwen-image-2-1',
    format: 'diffusers',
    key: 'qwen21-diffusers',
    name: 'Qwen-Image-2.1',
    type: 'main',
    variant: 'qwen_image_21_base',
  };
  const qwen21Gguf: MainModelConfig = {
    base: 'qwen-image-2-1',
    format: 'gguf_quantized',
    key: 'qwen21-gguf',
    name: 'Qwen-Image-2.1 Q4_K_M',
    type: 'main',
    variant: 'qwen_image_21_base',
  };
  const qwen21Vae: VaeModelConfig = {
    base: 'qwen-image-2-1',
    key: 'qwen21-vae',
    name: 'Qwen-Image-2.1 VAE',
    type: 'vae',
  };
  const qwen3Vl8bEncoder: ComponentModelConfig = {
    base: 'any',
    key: 'qwen3-vl-8b',
    name: 'Qwen3-VL 8B',
    type: 'qwen3_vl_encoder',
    variant: 'qwen3_vl_8b',
  };

  it('runs the negative prompt only with CFG above 1, wired through its own encoder', () => {
    // Qwen-Image-2.1 is sampled without guidance by default; there the model takes no negative input.
    const withoutCfg = compile(qwen21Diffusers, { cfgScale: 1 });

    expect(withoutCfg.nodes.neg_cond).toBeUndefined();
    expect(getEdge(withoutCfg, 'denoise_latents', 'negative_conditioning')).toBeUndefined();

    const withCfg = compile(qwen21Diffusers, { cfgScale: 4 });

    expect(withCfg.nodes.neg_cond?.type).toBe('qwen_image_2_1_text_encoder');
    expect(getEdge(withCfg, 'denoise_latents', 'negative_conditioning')?.source.node_id).toBe('neg_cond');
    expect(getEdge(withCfg, 'neg_cond', 'prompt')?.source.node_id).toBe('negative_prompt');
    expect(getEdge(withCfg, 'neg_cond', 'qwen3_vl_encoder')?.source).toEqual({
      field: 'qwen3_vl_encoder',
      node_id: 'model_loader',
    });
    expect(withCfg.nodes.denoise_latents).toMatchObject({ cfg_scale: 4 });
  });

  it('requires the VAE and the encoder for a GGUF transformer, and passes them to the loader', () => {
    expect(() => compile(qwen21Gguf, { qwen3VLEncoderModel: qwen3Vl8bEncoder })).toThrow(
      'Generate needs a VAE for non-Diffusers Qwen-Image-2.1 models.'
    );
    expect(() => compile(qwen21Gguf, { vae: qwen21Vae })).toThrow(
      'Generate needs a Qwen3-VL 8B Encoder for non-Diffusers Qwen-Image-2.1 models.'
    );

    const graph = compile(qwen21Gguf, { qwen3VLEncoderModel: qwen3Vl8bEncoder, vae: qwen21Vae });

    expect(graph.nodes.model_loader).toMatchObject({ qwen3_vl_encoder_model: qwen3Vl8bEncoder, vae_model: qwen21Vae });
  });
});

describe('ERNIE-Image graphs', () => {
  const ernieModel: MainModelConfig = {
    base: 'ernie-image',
    format: 'diffusers',
    key: 'ernie-image',
    name: 'ERNIE-Image',
    type: 'main',
  };

  it('encodes the negative prompt only while CFG is on', () => {
    // Exercise explicit CFG-off wiring: negative conditioning is required only above 1.
    const withCfg = compile(ernieModel, { cfgScale: 4 });

    expect(withCfg.nodes.neg_cond?.type).toBe('ernie_image_text_encoder');
    expect(getEdge(withCfg, 'neg_cond', 'text_encoder')?.source.node_id).toBe('model_loader');
    expect(getEdge(withCfg, 'neg_cond', 'prompt')?.source.node_id).toBe('negative_prompt');
    expect(getEdge(withCfg, 'denoise_latents', 'negative_conditioning')?.source.node_id).toBe('neg_cond');
    expect(withCfg.nodes.denoise_latents?.guidance_scale).toBe(4);

    const withoutCfg = compile(ernieModel, { cfgScale: 1 });

    expect(withoutCfg.nodes.neg_cond).toBeUndefined();
    expect(getEdge(withoutCfg, 'denoise_latents', 'negative_conditioning')).toBeUndefined();
    expect(withoutCfg.nodes.denoise_latents?.guidance_scale).toBe(1);
  });

  it('wires a single-file transformer to the encoder and VAE it was given', () => {
    // Assert encoder/VAE edges because their identifier types permit accidental swapping.
    const singleFile: MainModelConfig = { ...ernieModel, format: 'checkpoint', key: 'ernie-image-single' };

    const graph = compile(singleFile, { mistralEncoderModel: ministralEncoder, vae: flux2Vae });

    expect(graph.nodes.model_loader).toMatchObject({
      text_encoder_model: { key: 'ministral' },
      vae_model: { key: 'flux2-vae' },
    });
  });

  it('keeps a FLUX.2 Mistral encoder its picker would hide out of the graph', () => {
    // Filter optional encoder variants even when required-slot validation skips them.
    const graph = compile(ernieModel, { mistralEncoderModel: mistralEncoder });

    expect(graph.nodes.model_loader?.text_encoder_model).toBeUndefined();
  });

  it('builds the rest of the graph out of the one bundled pipeline', () => {
    const graph = compile(ernieModel, { scheduler: 'dpmpp_2m' });

    expect(graph.nodes.canvas_output?.type).toBe('ernie_image_vae_decode');
    expect(getEdge(graph, 'canvas_output', 'vae')?.source.node_id).toBe('model_loader');
    expect(getEdge(graph, 'denoise_latents', 'transformer')?.source.node_id).toBe('model_loader');
    // Unload the unused prompt enhancer because it cannot idle-offload.
    expect(graph.nodes.model_loader?.use_prompt_enhancer).toBe(false);
    // Coerce to ERNIE's supported scheduler set.
    expect(graph.nodes.denoise_latents?.scheduler).toBe('euler');
  });
});

describe('PiD decode', () => {
  const pidDecoder = (base: string): ComponentModelConfig => ({
    base,
    key: `pid-${base}`,
    name: `PiD ${base}`,
    type: 'pid_decoder',
  });
  const gemma2: ComponentModelConfig = { base: 'any', key: 'gemma2', name: 'Gemma 2 2B', type: 'gemma2_encoder' };

  const pidSettings = (base: string, overrides: Partial<GenerateSettings> = {}): Partial<GenerateSettings> => ({
    gemma2EncoderModel: gemma2,
    pidDecoderModel: pidDecoder(base),
    pidMode: 'fit',
    ...overrides,
  });

  it('keeps the ordinary VAE decode when PiD is off', () => {
    const graph = compile(fluxModel, { clipEmbedModel: clipEmbed, t5EncoderModel: t5Encoder, vae: fluxVae });

    expect(getNodeByType(graph, 'flux_vae_decode')).toBeDefined();
    expect(getNodeByType(graph, 'flux_pid_decode')).toBeUndefined();
  });

  it('replaces the VAE decode with a PiD decode and downscales in fit mode', () => {
    const graph = compile(fluxModel, {
      clipEmbedModel: clipEmbed,
      t5EncoderModel: t5Encoder,
      vae: fluxVae,
      ...pidSettings('flux'),
      height: 1024,
      width: 1024,
    });

    expect(getNodeByType(graph, 'flux_vae_decode')).toBeUndefined();
    const decode = getNodeByType(graph, 'flux_pid_decode');
    expect(decode).toBeDefined();
    // The resize is the output, because the 4x decode must come back to the asked-for size.
    const output = graph.nodes.canvas_output;
    expect(output?.type).toBe('img_resize');
    expect(output).toMatchObject({ height: 1024, width: 1024 });
    expect(getEdge(graph, 'canvas_output', 'image')?.source.node_id).toBe(decode?.id);
  });

  it('generates at the full size in fit mode', () => {
    const graph = compile(fluxModel, {
      clipEmbedModel: clipEmbed,
      t5EncoderModel: t5Encoder,
      vae: fluxVae,
      ...pidSettings('flux'),
      height: 1024,
      width: 1024,
    });

    expect(graph.nodes.denoise_latents).toMatchObject({ height: 1024, width: 1024 });
  });

  it('generates at a quarter size in native mode and makes the decode the output', () => {
    const graph = compile(fluxModel, {
      clipEmbedModel: clipEmbed,
      t5EncoderModel: t5Encoder,
      vae: fluxVae,
      ...pidSettings('flux', { pidMode: 'native' }),
      height: 2048,
      width: 2048,
    });

    expect(graph.nodes.denoise_latents).toMatchObject({ height: 512, width: 512 });
    // No downscale: PiD's 4x output IS the result.
    expect(graph.nodes.canvas_output?.type).toBe('flux_pid_decode');
    expect(getNodeByType(graph, 'img_resize')).toBeUndefined();
  });

  it('wires the prompt, seed and both loaders into the decode', () => {
    const graph = compile(fluxModel, {
      clipEmbedModel: clipEmbed,
      t5EncoderModel: t5Encoder,
      vae: fluxVae,
      ...pidSettings('flux'),
    });
    const decode = getNodeByType(graph, 'flux_pid_decode');

    expect(getEdge(graph, decode!.id, 'prompt')?.source.node_id).toBe('positive_prompt');
    expect(getEdge(graph, decode!.id, 'seed')?.source.node_id).toBe('seed');
    expect(getEdge(graph, decode!.id, 'latents')?.source.node_id).toBe('denoise_latents');
    expect(getNodeByType(graph, 'pid_decoder_loader')).toMatchObject({ pid_decoder_model: pidDecoder('flux') });
    expect(getNodeByType(graph, 'gemma2_encoder_loader')).toMatchObject({ gemma2_model: gemma2 });
  });

  it('does not wire a VAE into decoders that use fixed constants', () => {
    const graph = compile(fluxModel, {
      clipEmbedModel: clipEmbed,
      t5EncoderModel: t5Encoder,
      vae: fluxVae,
      ...pidSettings('flux'),
    });
    const decode = getNodeByType(graph, 'flux_pid_decode');

    // FLUX.1 and SD3 use fixed VAE constants and expose no `vae` input.
    expect(getEdge(graph, decode!.id, 'vae')).toBeUndefined();
  });

  it('wires a VAE into decoders that read its constants at runtime', () => {
    const graph = compile(qwenImageModel, { ...pidSettings('qwen-image') });
    const decode = getNodeByType(graph, 'qwen_image_pid_decode');

    expect(getEdge(graph, decode!.id, 'vae')?.source.node_id).toBe('model_loader');
  });

  it('uses a FLUX decoder for Z-Image, which shares its VAE', () => {
    const graph = compile(zImageModel, { ...pidSettings('flux') });

    expect(getNodeByType(graph, 'z_image_pid_decode')).toBeDefined();
    expect(getNodeByType(graph, 'pid_decoder_loader')).toMatchObject({ pid_decoder_model: pidDecoder('flux') });
  });

  it('decodes through PiD for SDXL, sizing the noise node instead of the denoise node', () => {
    const graph = compile(sdxlModel, { ...pidSettings('sdxl', { pidMode: 'native' }), height: 2048, width: 2048 });

    // SDXL is sized via its noise node; its grid is 8, so 2048 / 4 = 512.
    expect(graph.nodes.noise).toMatchObject({ height: 512, width: 512 });
    expect(graph.nodes.canvas_output?.type).toBe('sdxl_pid_decode');
  });

  it('decodes through PiD for SD3', () => {
    const graph = compile(sd3Model, { ...pidSettings('sd-3') });

    expect(getNodeByType(graph, 'sd3_pid_decode')).toBeDefined();
    expect(getNodeByType(graph, 'sd3_l2i')).toBeUndefined();
  });

  it('decodes through PiD for FLUX.2', () => {
    const graph = compile(flux2Model, { ...pidSettings('flux2') });

    expect(getNodeByType(graph, 'flux2_pid_decode')).toBeDefined();
    expect(getNodeByType(graph, 'flux2_vae_decode')).toBeUndefined();
  });

  it('records the PiD settings in metadata', () => {
    const graph = compile(fluxModel, {
      clipEmbedModel: clipEmbed,
      t5EncoderModel: t5Encoder,
      vae: fluxVae,
      ...pidSettings('flux'),
    });
    const metadata = getNodeByType(graph, 'core_metadata');

    expect(metadata).toMatchObject({ pid_mode: 'fit', pid_steps: 4 });
  });

  it('refuses to compile PiD on an unsupported base rather than silently ignoring it', () => {
    // Reject unsupported PiD rather than silently use ordinary decode.
    expect(() => compile(cogView4Model, { pidMode: 'fit' })).toThrow(/PiD is not supported/);
  });

  it('requires both PiD models before compiling', () => {
    expect(() =>
      compile(fluxModel, {
        clipEmbedModel: clipEmbed,
        t5EncoderModel: t5Encoder,
        vae: fluxVae,
        pidMode: 'fit',
        pidDecoderModel: pidDecoder('flux'),
      })
    ).toThrow(/Gemma-2 encoder/);
    expect(() =>
      compile(fluxModel, {
        clipEmbedModel: clipEmbed,
        t5EncoderModel: t5Encoder,
        vae: fluxVae,
        pidMode: 'fit',
        gemma2EncoderModel: gemma2,
      })
    ).toThrow(/PiD decoder/);
  });

  it('enforces the 4x grid on the requested size in native mode', () => {
    // FLUX's grid is 16, so native requires multiples of 64.
    expect(() =>
      compile(fluxModel, {
        clipEmbedModel: clipEmbed,
        t5EncoderModel: t5Encoder,
        vae: fluxVae,
        ...pidSettings('flux', { pidMode: 'native' }),
        height: 2048,
        width: 2032,
      })
    ).toThrow(/multiple of 64/);
  });
});

describe('SDXL Refiner', () => {
  const refinerModel: MainModelConfig = {
    base: 'sdxl-refiner',
    hash: 'blake3:refiner',
    key: 'sdxl-refiner-model',
    name: 'SDXL Refiner',
    type: 'main',
  };
  const refinerSettings: Partial<GenerateSettings> = {
    refinerCfgScale: 5,
    refinerModel,
    refinerNegativeAestheticScore: 2,
    refinerPositiveAestheticScore: 7,
    refinerScheduler: 'dpmpp_2m',
    refinerStart: 0.7,
    refinerSteps: 12,
  };

  it('leaves the SDXL graph unchanged without a refiner', () => {
    const graph = compile(sdxlModel);

    expect(graph.nodes.refiner_denoise_latents).toBeUndefined();
    expect(graph.nodes.refiner_model_loader).toBeUndefined();
    expect(graph.nodes.denoise_latents).toMatchObject({ denoising_end: 1, denoising_start: 0 });
    expect(getEdge(graph, 'canvas_output', 'latents')?.source.node_id).toBe('denoise_latents');
    expect(getNodeByType(graph, 'core_metadata')).not.toHaveProperty('refiner_model');
  });

  it('hands the base latents to a refiner stage that feeds the decode', () => {
    const graph = compile(sdxlModel, refinerSettings);
    const refinerDenoise = graph.nodes.refiner_denoise_latents;

    expect(graph.nodes.denoise_latents).toMatchObject({ denoising_end: 0.7, denoising_start: 0 });
    expect(graph.nodes.refiner_model_loader).toMatchObject({
      model: { base: 'sdxl-refiner', hash: 'blake3:refiner', key: 'sdxl-refiner-model', type: 'main' },
      type: 'sdxl_refiner_model_loader',
    });
    expect(refinerDenoise).toMatchObject({
      cfg_scale: 5,
      denoising_end: 1,
      denoising_start: 0.7,
      scheduler: 'dpmpp_2m',
      steps: 12,
      type: 'denoise_latents',
    });
    expect(graph.nodes.refiner_pos_cond).toMatchObject({ aesthetic_score: 7, type: 'sdxl_refiner_compel_prompt' });
    expect(graph.nodes.refiner_neg_cond).toMatchObject({ aesthetic_score: 2, type: 'sdxl_refiner_compel_prompt' });

    expect(getEdge(graph, 'refiner_denoise_latents', 'unet')?.source).toEqual({
      field: 'unet',
      node_id: 'refiner_model_loader',
    });
    expect(getEdge(graph, 'refiner_pos_cond', 'clip2')?.source.node_id).toBe('refiner_model_loader');
    expect(getEdge(graph, 'refiner_neg_cond', 'clip2')?.source.node_id).toBe('refiner_model_loader');
    expect(getEdge(graph, 'refiner_pos_cond', 'style')?.source.node_id).toBe('positive_prompt');
    expect(getEdge(graph, 'refiner_neg_cond', 'style')?.source.node_id).toBe('negative_prompt');
    expect(getEdge(graph, 'refiner_denoise_latents', 'positive_conditioning')?.source.node_id).toBe('refiner_pos_cond');
    expect(getEdge(graph, 'refiner_denoise_latents', 'negative_conditioning')?.source.node_id).toBe('refiner_neg_cond');
    expect(getEdge(graph, 'refiner_denoise_latents', 'latents')?.source.node_id).toBe('denoise_latents');
    expect(getEdge(graph, 'canvas_output', 'latents')?.source.node_id).toBe('refiner_denoise_latents');
    // The base VAE still decodes; the refiner has none of its own.
    expect(getEdge(graph, 'canvas_output', 'vae')?.source.node_id).toBe('model_loader');
  });

  it('records the refiner and its tuning in metadata', () => {
    const graph = compile(sdxlModel, refinerSettings);

    expect(getNodeByType(graph, 'core_metadata')).toMatchObject({
      refiner_cfg_scale: 5,
      refiner_model: { base: 'sdxl-refiner', hash: 'blake3:refiner', key: 'sdxl-refiner-model', type: 'main' },
      refiner_negative_aesthetic_score: 2,
      refiner_positive_aesthetic_score: 7,
      refiner_scheduler: 'dpmpp_2m',
      refiner_start: 0.7,
      refiner_steps: 12,
    });
  });

  it('wraps the refiner UNet in its own seamless node when tiling is on', () => {
    const graph = compile(sdxlModel, { ...refinerSettings, seamlessXAxis: true });

    expect(graph.nodes.refiner_seamless).toMatchObject({ seamless_x: true, seamless_y: false, type: 'seamless' });
    expect(getEdge(graph, 'refiner_seamless', 'unet')?.source.node_id).toBe('refiner_model_loader');
    expect(getEdge(graph, 'refiner_denoise_latents', 'unet')?.source.node_id).toBe('refiner_seamless');
  });

  it('runs the refiner pass with the base HiDiffusion settings and coerces its scheduler', () => {
    const graph = compile(sdxlModel, {
      ...refinerSettings,
      hiDiffusionEnabled: true,
      hiDiffusionT1Ratio: 0.3,
      refinerScheduler: 'not-a-scheduler',
    });

    expect(graph.nodes.refiner_denoise_latents).toMatchObject({
      hidiffusion: true,
      hidiffusion_t1_ratio: 0.3,
      scheduler: 'euler_a',
    });
    expect(getNodeByType(graph, 'core_metadata')?.refiner_scheduler).toBe('euler_a');
  });

  it('ignores a stored refiner on a non-SDXL main model', () => {
    const graph = compile(sd1Model, refinerSettings);

    expect(graph.nodes.refiner_denoise_latents).toBeUndefined();
    expect(getNodeByType(graph, 'core_metadata')).not.toHaveProperty('refiner_model');
  });

  it('rejects a refiner together with PiD decoding', () => {
    const gemma2: ComponentModelConfig = { base: 'any', key: 'gemma2', name: 'Gemma 2 2B', type: 'gemma2_encoder' };

    expect(() =>
      compile(sdxlModel, {
        ...refinerSettings,
        gemma2EncoderModel: gemma2,
        pidDecoderModel: { base: 'sdxl', key: 'pid-sdxl', name: 'PiD SDXL', type: 'pid_decoder' },
        pidMode: 'fit',
      })
    ).toThrow('PiD decode is not compatible with the SDXL Refiner.');
  });

  it('refuses a non-refiner model stored in the refiner slot', () => {
    expect(() => compile(sdxlModel, { ...refinerSettings, refinerModel: sdxlModel })).toThrow(
      'SDXL is not an SDXL Refiner model.'
    );
  });

  it('hands the whole schedule to the refiner at a start of 0', () => {
    const graph = compile(sdxlModel, { ...refinerSettings, refinerStart: 0 });

    // The base pass only seeds the latents from noise; the refiner denoises every step.
    expect(graph.nodes.denoise_latents).toMatchObject({ denoising_end: 0, denoising_start: 0 });
    expect(graph.nodes.refiner_denoise_latents).toMatchObject({ denoising_end: 1, denoising_start: 0 });
    expect(getEdge(graph, 'canvas_output', 'latents')?.source.node_id).toBe('refiner_denoise_latents');
    expect(getNodeByType(graph, 'core_metadata')?.refiner_start).toBe(0);
  });

  it('runs without the refiner at a start of 1, where its pass would be empty', () => {
    const graph = compile(sdxlModel, {
      ...refinerSettings,
      gemma2EncoderModel: { base: 'any', key: 'gemma2', name: 'Gemma 2 2B', type: 'gemma2_encoder' },
      pidDecoderModel: { base: 'sdxl', key: 'pid-sdxl', name: 'PiD SDXL', type: 'pid_decoder' },
      pidMode: 'fit',
      refinerStart: 1,
    });

    // Neither the refiner model nor its PiD conflict applies: the graph is the plain SDXL graph.
    expect(graph.nodes.refiner_model_loader).toBeUndefined();
    expect(graph.nodes.refiner_denoise_latents).toBeUndefined();
    expect(graph.nodes.denoise_latents).toMatchObject({ denoising_end: 1, denoising_start: 0 });
    expect(getNodeByType(graph, 'core_metadata')).not.toHaveProperty('refiner_model');
  });

  it('keeps IP Adapter reference images on the base pass', () => {
    const sdxlIpAdapter: ComponentModelConfig = {
      base: 'sdxl',
      key: 'sdxl-ip-adapter',
      name: 'SDXL IP Adapter',
      type: 'ip_adapter',
    };
    const graph = compile(sdxlModel, {
      ...refinerSettings,
      referenceImages: [
        {
          id: 'ref-1',
          isEnabled: true,
          config: {
            beginEndStepPct: [0, 1],
            clipVisionModel: 'ViT-H',
            image: refImage,
            method: 'full',
            model: sdxlIpAdapter,
            type: 'ip_adapter',
            weight: 1,
          },
        },
      ],
    });

    expect(getEdge(graph, 'denoise_latents', 'ip_adapter')?.source.node_id).toContain('ip_adapter_collector');
    expect(getEdge(graph, 'refiner_denoise_latents', 'ip_adapter')).toBeUndefined();
    expect(
      graph.edges
        .filter((edge) => edge.destination.node_id === 'refiner_denoise_latents')
        .map((e) => e.destination.field)
    ).toEqual(['unet', 'positive_conditioning', 'negative_conditioning', 'latents']);
  });
});

describe('model-family tuning', () => {
  const fluxComponents: Partial<GenerateSettings> = {
    clipEmbedModel: clipEmbed,
    t5EncoderModel: t5Encoder,
    vae: fluxVae,
  };
  const krea2Model: MainModelConfig = {
    base: 'krea-2',
    format: 'diffusers',
    key: 'krea2-tuning',
    name: 'Krea-2 Turbo',
    type: 'main',
    variant: 'krea2_turbo',
  };
  const styleReference = (styleStrength: number, isEnabled = true, id = 'style-ref') => ({
    config: { image: refImage, styleStrength, type: 'krea2_reference_image' as const },
    id,
    isEnabled,
  });

  it('sends no DyPE fields while the preset is off, but records off so recall can restore it', () => {
    const graph = compile(fluxModel, fluxComponents);
    const metadata = getNodeByType(graph, 'core_metadata');

    expect(graph.nodes.denoise_latents).not.toHaveProperty('dype_preset');
    expect(graph.nodes.denoise_latents).not.toHaveProperty('dype_scale');
    expect(metadata).toMatchObject({ dype_preset: 'off' });
    expect(metadata).not.toHaveProperty('dype_scale');
  });

  it('sends only the DyPE preset for the derived presets, and scale and exponent under manual', () => {
    const auto = compile(fluxModel, { ...fluxComponents, fluxDypePreset: 'auto', fluxDypeScale: 3 });
    const manual = compile(fluxModel, {
      ...fluxComponents,
      fluxDypeExponent: 0.5,
      fluxDypePreset: 'manual',
      fluxDypeScale: 3,
    });

    expect(auto.nodes.denoise_latents).toMatchObject({ dype_preset: 'auto' });
    expect(auto.nodes.denoise_latents).not.toHaveProperty('dype_scale');
    expect(manual.nodes.denoise_latents).toMatchObject({ dype_exponent: 0.5, dype_preset: 'manual', dype_scale: 3 });
    expect(getNodeByType(manual, 'core_metadata')).toMatchObject({
      dype_exponent: 0.5,
      dype_preset: 'manual',
      dype_scale: 3,
    });
  });

  it('leaves the Z-Image shift to the node unless overridden, and records which', () => {
    const auto = compile(zImageModel);
    const fixed = compile(zImageModel, { zImageShift: 3 });

    expect(auto.nodes.denoise_latents).not.toHaveProperty('shift');
    expect(getNodeByType(auto, 'core_metadata')?.z_image_shift).toBe('auto');
    expect(fixed.nodes.denoise_latents?.shift).toBe(3);
    expect(getNodeByType(fixed, 'core_metadata')?.z_image_shift).toBe(3);
  });

  it('inserts the Z-Image seed variance enhancer between the encoder and the collector, fed by the seed', () => {
    const graph = compile(zImageModel, {
      zImageSeedVarianceEnabled: true,
      zImageSeedVarianceRandomizePercent: 25,
      zImageSeedVarianceStrength: 0.3,
    });
    const enhancer = getNodeByType(graph, 'z_image_seed_variance_enhancer');

    expect(enhancer).toMatchObject({ randomize_percent: 25, strength: 0.3 });
    expect(getEdge(graph, enhancer!.id, 'conditioning')?.source.node_id).toBe('pos_cond');
    expect(getEdge(graph, enhancer!.id, 'seed')?.source.node_id).toBe('seed');
    expect(getEdge(graph, 'pos_cond_collect', 'item')?.source.node_id).toBe(enhancer!.id);
    expect(getNodeByType(graph, 'core_metadata')).toMatchObject({
      z_image_seed_variance_enabled: true,
      z_image_seed_variance_randomize_percent: 25,
      z_image_seed_variance_strength: 0.3,
    });
  });

  it('skips the Z-Image enhancer while it is off or at zero strength, keeping the direct conditioning edge', () => {
    for (const graph of [
      compile(zImageModel),
      compile(zImageModel, { zImageSeedVarianceEnabled: true, zImageSeedVarianceStrength: 0 }),
    ]) {
      expect(getNodeByType(graph, 'z_image_seed_variance_enhancer')).toBeUndefined();
      expect(getEdge(graph, 'pos_cond_collect', 'item')?.source.node_id).toBe('pos_cond');
    }
  });

  it('sends the Qwen-Image shift only when overridden', () => {
    expect(compile(qwenImageModel).nodes.denoise_latents).not.toHaveProperty('shift');
    expect(getNodeByType(compile(qwenImageModel), 'core_metadata')?.qwen_image_shift).toBe('auto');

    const fixed = compile(qwenImageModel, { qwenImageShift: 3 });

    expect(fixed.nodes.denoise_latents?.shift).toBe(3);
    expect(getNodeByType(fixed, 'core_metadata')?.qwen_image_shift).toBe(3);
  });

  it('passes the Qwen-Image encoder quantization to both text encoders, and nothing for none', () => {
    const plain = compile(qwenImageModel, { cfgScale: 4 });
    const quantized = compile(qwenImageModel, { cfgScale: 4, qwenImageEncoderQuantization: 'nf4' });

    expect(plain.nodes.pos_cond).not.toHaveProperty('quantization');
    expect(plain.nodes.neg_cond).not.toHaveProperty('quantization');
    expect(quantized.nodes.pos_cond?.quantization).toBe('nf4');
    expect(quantized.nodes.neg_cond?.quantization).toBe('nf4');
    expect(getNodeByType(quantized, 'core_metadata')?.qwen_image_quantization).toBe('nf4');
  });

  it('encodes the enabled Krea-2 style reference at the denoise size and feeds it to the denoise node', () => {
    // The served limit is one enabled reference; a disabled card stays out of the graph.
    const graph = compile(krea2Model, {
      height: 768,
      referenceImages: [styleReference(1.5, false, 'disabled'), styleReference(0.8, true, 'used')],
      width: 1280,
    });
    const styleNodes = Object.values(graph.nodes).filter((node) => node.type === 'krea2_style_reference');

    expect(styleNodes).toHaveLength(1);
    expect(styleNodes[0]).toMatchObject({
      height: 768,
      image: { image_name: 'reference.png' },
      style_strength: 0.8,
      width: 1280,
    });
    expect(getEdge(graph, styleNodes[0]!.id, 'vae')?.source.node_id).toBe('model_loader');
    expect(getEdge(graph, 'denoise_latents', 'style_reference')?.source.node_id).toBe(styleNodes[0]!.id);
    expect(getNodeByType(graph, 'core_metadata')?.krea2_style_strength).toBe(0.8);
    expect((getNodeByType(graph, 'core_metadata')!.ref_images as { id: string }[]).map((entry) => entry.id)).toEqual([
      'used',
    ]);
  });

  it('encodes a cropped Krea-2 reference from its crop', () => {
    const cropped = {
      crop: {
        box: { height: 256, width: 256, x: 0, y: 0 },
        image: { height: 256, image_name: 'crop.png', width: 256 },
        ratio: 1,
      },
      original: refImage.original,
    };
    const graph = compile(krea2Model, {
      referenceImages: [
        { config: { image: cropped, styleStrength: 1, type: 'krea2_reference_image' }, id: 'r', isEnabled: true },
      ],
    });

    expect(getNodeByType(graph, 'krea2_style_reference')?.image).toEqual({ image_name: 'crop.png' });
  });

  it('leaves the Krea-2 graph without a style node when every reference is bypassed', () => {
    const graph = compile(krea2Model, { referenceImages: [styleReference(0)] });

    expect(getNodeByType(graph, 'krea2_style_reference')).toBeUndefined();
    expect(getEdge(graph, 'denoise_latents', 'style_reference')).toBeUndefined();
    expect(getNodeByType(graph, 'core_metadata')).not.toHaveProperty('krea2_style_strength');
  });
});

describe('output filters', () => {
  it.each(generateGraphCases)('$label chains the checker and watermark after its decode', ({ base, shape }) => {
    const { model, settings } = satisfiedSettingsFor(base, shape);
    const plain = compileGenerateGraph(settings, model, 'gallery', {
      useCpuNoise: true,
      useNsfwChecker: false,
      useWatermark: false,
    }).backendGraph;
    const graph = compileGenerateGraph(settings, model, 'canvas', {
      useCpuNoise: true,
      useNsfwChecker: true,
      useWatermark: true,
    }).backendGraph;

    expect(Object.values(plain.nodes).some((node) => node.type === 'img_nsfw' || node.type === 'img_watermark')).toBe(
      false
    );
    // The image that used to be the output now feeds the checker, and only the watermark is the result.
    expect(graph.nodes.canvas_unfiltered_output?.type).toBe(plain.nodes.canvas_output?.type);
    expect(graph.nodes.canvas_unfiltered_output?.is_intermediate).toBe(true);
    expect(graph.nodes.canvas_img_nsfw).toMatchObject({ is_intermediate: true, type: 'img_nsfw', use_cache: false });
    expect(graph.nodes.canvas_output).toMatchObject({ is_intermediate: true, type: 'img_watermark', use_cache: false });
    expect(getEdge(graph, 'canvas_img_nsfw', 'image')?.source).toEqual({
      field: 'image',
      node_id: 'canvas_unfiltered_output',
    });
    expect(getEdge(graph, 'canvas_output', 'image')?.source).toEqual({ field: 'image', node_id: 'canvas_img_nsfw' });
    expect(getEdge(graph, 'canvas_output', 'metadata')?.source.node_id).toBe(getNodeByType(graph, 'core_metadata')?.id);
    expect(getEdge(graph, 'canvas_unfiltered_output', 'metadata')).toBeUndefined();
  });

  it('runs a single filter as the durable output itself', () => {
    const graph = compileGenerateGraph(createSettings(sd1Model), sd1Model, 'gallery', {
      useCpuNoise: true,
      useNsfwChecker: false,
      useWatermark: true,
    }).backendGraph;

    expect(graph.nodes.canvas_img_nsfw).toBeUndefined();
    expect(graph.nodes.canvas_output).toMatchObject({ is_intermediate: false, type: 'img_watermark' });
    expect(getEdge(graph, 'canvas_output', 'image')?.source.node_id).toBe('canvas_unfiltered_output');
  });

  it('keeps the filters after a PiD decode, which already ends on the output node', () => {
    const gemma2: ComponentModelConfig = { base: 'any', key: 'gemma2', name: 'Gemma 2 2B', type: 'gemma2_encoder' };
    const graph = compileGenerateGraph(
      createSettings(sdxlModel, {
        gemma2EncoderModel: gemma2,
        pidDecoderModel: { base: 'sdxl', key: 'pid-sdxl', name: 'PiD SDXL', type: 'pid_decoder' },
        pidMode: 'native',
      }),
      sdxlModel,
      'gallery',
      { useCpuNoise: true, useNsfwChecker: true, useWatermark: false }
    ).backendGraph;

    expect(graph.nodes.canvas_unfiltered_output?.type).toBe('sdxl_pid_decode');
    expect(graph.nodes.canvas_output?.type).toBe('img_nsfw');
  });

  it('does not filter an external provider, which returns finished images', () => {
    const graph = compileGenerateGraph(createSettings(externalModel), externalModel, 'gallery', {
      useCpuNoise: true,
      useNsfwChecker: true,
      useWatermark: true,
    }).backendGraph;

    expect(getNodeByType(graph, 'img_nsfw')).toBeUndefined();
    expect(getNodeByType(graph, 'img_watermark')).toBeUndefined();
  });
});
