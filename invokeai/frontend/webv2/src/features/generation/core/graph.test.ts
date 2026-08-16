import { afterEach, describe, expect, it, vi } from 'vitest';

import type {
  ComponentModelConfig,
  GenerateModelConfig,
  GenerateSettings,
  LoraModelConfig,
  MainModelConfig,
  VaeModelConfig,
} from './types';

import { getDefaultGenerateSettings, isSupportedGenerateModel } from './baseGenerationPolicies';
import { compileGenerateGraph, generateSeedSequence, resolveGenerateSeed } from './graph';

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
const clipEmbed: ComponentModelConfig = { base: 'any', key: 'clip', name: 'CLIP Embed', type: 'clip_embed' };
const mistralEncoder: ComponentModelConfig = {
  base: 'any',
  key: 'mistral',
  name: 'Mistral Encoder',
  type: 'mistral_encoder',
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
  shouldRandomizeSeed: false,
  ...overrides,
});

const refImage = {
  original: { image: { height: 768, image_name: 'reference.png', width: 512 } },
};

const compile = (model: GenerateModelConfig, overrides: Partial<GenerateSettings> = {}) =>
  compileGenerateGraph(createSettings(model, overrides), model, 'gallery', { useCpuNoise: true }).backendGraph;

const getEdge = (graph: ReturnType<typeof compile>, targetNodeId: string, targetField: string) =>
  graph.edges.find((edge) => edge.destination.node_id === targetNodeId && edge.destination.field === targetField);

const getNodeByType = (graph: ReturnType<typeof compile>, type: string) =>
  Object.values(graph.nodes).find((node) => node.type === type);

afterEach(() => {
  vi.restoreAllMocks();
});

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
  it('keeps an explicit seed when randomization is disabled', () => {
    const settings = createSettings(sdxlModel, { seed: 123, shouldRandomizeSeed: false });

    expect(resolveGenerateSeed(settings)).toBe(123);
  });

  it('resolves a random seed when randomization is enabled', () => {
    vi.spyOn(Math, 'random').mockReturnValue(0.5);

    const settings = createSettings(sdxlModel, { seed: 123, shouldRandomizeSeed: true });

    expect(resolveGenerateSeed(settings)).toBe(2147483647);
  });

  it('builds the legacy-style sequential seed batch starting from the resolved seed', () => {
    expect(generateSeedSequence(10, 4)).toEqual([10, 11, 12, 13]);
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
    // Regression: the builder filtered VAEs to base `qwen-image`, so a Qwen-Image VAE
    // installed under the `anima` base was dropped and the loader got no VAE at all.
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
    // Validation runs ahead of the builder, so the user-facing reason surfaces rather than the
    // builder's internal guard. Both layers exist; this pins which one the user actually sees.
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
    // Unset optional model fields stay `undefined` here, matching every other builder; they
    // drop out when the graph is serialized rather than reaching the backend as null.
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
    // CogView4 has no PiD decode node. Falling back to an ordinary decode would leave the
    // user wondering why the output is not 4x.
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
