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
  height: 768,
  imageName: 'reference.png',
  imageUrl: '/api/v1/images/i/reference.png/full',
  queuedAt: '2026-01-01T00:00:00.000Z',
  sourceQueueItemId: 'backend-gallery',
  thumbnailUrl: '/api/v1/images/i/reference.png/thumbnail',
  width: 512,
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
    expect(compile(cogView4Model).nodes.model_loader?.type).toBe('cogview4_model_loader');
    expect(compile(qwenImageModel).nodes.model_loader?.type).toBe('qwen_image_model_loader');
    expect(compile(zImageModel).nodes.model_loader?.type).toBe('z_image_model_loader');
  });

  it('does not enable FLUX.2 CFG without negative conditioning', () => {
    const graph = compile(flux2Model, { cfgScale: 4 });

    expect(graph.nodes.denoise_latents?.cfg_scale).toBe(1);
    expect(getEdge(graph, 'denoise_latents', 'negative_text_conditioning')).toBeUndefined();
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
