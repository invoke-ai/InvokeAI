import { seedArchitectureCapabilities } from '@features/models/core/architectureCapabilities.testing';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type {
  ComponentModelConfig,
  ExternalImageGeneratorModelConfig,
  GenerateModelConfig,
  GenerateReferenceImage,
  GenerateSettings,
  LoraModelConfig,
  MainModelConfig,
  VaeModelConfig,
} from './types';

import { getDefaultGenerateSettings, type SupportedGenerateBase } from './baseGenerationPolicies';
import { compileGenerateGraph, getFamilyGraphBuilder, GRAPH_BUILDERS, resolveGenerateSeed } from './graph';
import {
  fullyFilledSettingsFor,
  generateGraphCases,
  satisfiedSettingsFor,
  tunedSettingsFor,
} from './graphCoverage.testing';
import { isPidDecodeNodeType } from './pidGraph';
import { getReferenceImagePromptTags } from './referenceImage';
import { REFINER_DENOISE_NODE_ID } from './sdxlRefinerGraph';
const sd1Model: MainModelConfig = { base: 'sd-1', key: 'sd1-model', name: 'SD 1.5', type: 'main' };
const sdxlModel: MainModelConfig = { base: 'sdxl', key: 'sdxl-model', name: 'SDXL', type: 'main' };
const sd3Model: MainModelConfig = { base: 'sd-3', format: 'diffusers', key: 'sd3-model', name: 'SD3', type: 'main' };
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
const fluxKontextModel: MainModelConfig = {
  base: 'flux',
  format: 'diffusers',
  key: 'flux-kontext',
  name: 'FLUX Kontext dev',
  type: 'main',
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
  it('builds a plain txt2img graph without optional nodes', () => {
    const graph = compile(sd1Model);

    expect(graph.nodes.seamless).toBeUndefined();
    expect(graph.nodes.vae_loader).toBeUndefined();
    expect(getEdge(graph, 'denoise_latents', 'unet')?.source.node_id).toBe('model_loader');
    expect(getEdge(graph, 'canvas_output', 'vae')?.source.node_id).toBe('model_loader');
    expect(graph.nodes.canvas_output?.fp32).toBe(true);
    expect(graph.nodes.canvas_output).not.toHaveProperty('color_compensation');
  });

  it('coerces invalid SD schedulers before graph submission', () => {
    const graph = compile(sd1Model, { scheduler: 'not-a-scheduler' });
    const metadata = getNodeByType(graph, 'core_metadata');

    expect(graph.nodes.denoise_latents?.scheduler).toBe('euler_a');
    expect(metadata?.scheduler).toBe('euler_a');
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

  it('routes the SD3 transformer and both CLIP encoders through SD3 LoRAs, and T5 around them', () => {
    const sd3Lora: LoraModelConfig = { base: 'sd-3', key: 'sd3-lora', name: 'SD3 LoRA', type: 'lora' };
    const qwenLora: LoraModelConfig = { base: 'qwen-image', key: 'qwen-lora', name: 'Qwen LoRA', type: 'lora' };

    const graph = compile(sd3Model, {
      loras: [
        { isEnabled: true, model: sd3Lora, weight: 0.6 },
        { isEnabled: true, model: qwenLora, weight: 1 },
      ],
    });
    const loraLoader = getNodeByType(graph, 'sd3_lora_collection_loader');

    expect(loraLoader).toBeDefined();
    for (const field of ['transformer', 'clip_l', 'clip_g']) {
      expect(getEdge(graph, loraLoader?.id ?? '', field)?.source.node_id).toBe('model_loader');
    }
    expect(getEdge(graph, 'denoise_latents', 'transformer')?.source.node_id).toBe(loraLoader?.id);
    for (const cond of ['pos_cond', 'neg_cond']) {
      expect(getEdge(graph, cond, 'clip_l')?.source.node_id).toBe(loraLoader?.id);
      expect(getEdge(graph, cond, 'clip_g')?.source.node_id).toBe(loraLoader?.id);
      expect(getEdge(graph, cond, 't5_encoder')?.source.node_id).toBe('model_loader');
    }
    expect(getNodeByType(graph, 'core_metadata')?.loras).toEqual([
      { model: { base: 'sd-3', key: 'sd3-lora', name: 'SD3 LoRA', type: 'lora' }, weight: 0.6 },
    ]);
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

  describe('FLUX.2 [dev] component sources', () => {
    const sdnqKleinSource: MainModelConfig = {
      ...flux2Klein9bModel,
      format: 'sdnq_quantized',
      key: 'flux2-klein-9b-sdnq-source',
      submodels: completeSdnqPipeline,
    };
    it('never sends an SDNQ pipeline as the Mistral source', () => {
      expect(() =>
        compile(flux2DevGgufModel, { componentSourceModel: sdnqKleinSource, mistralEncoderModel: mistralEncoder })
      ).toThrow('Generate needs a FLUX.2 VAE or a FLUX.2 Diffusers pipeline for this FLUX.2 [dev] model.');

      const graph = compile(flux2DevGgufModel, {
        componentSourceModel: sdnqKleinSource,
        mistralEncoderModel: mistralEncoder,
        vae: flux2Vae,
      });

      expect(graph.nodes.model_loader).toMatchObject({
        mistral_encoder_model: mistralEncoder,
        mistral_source_model: undefined,
        vae_model: flux2Vae,
      });
    });
  });

  it('rejects dimensions that are not on the selected model family grid', () => {
    expect(() => compile(flux2Model, { height: 888 })).toThrow('Height must be a multiple of 16.');
  });

  it('builds an Anima graph with required components', () => {
    expect(() => compile(animaModel)).toThrow('Generate needs a Qwen3 Encoder for Anima models.');

    const graph = compile(animaModel, { qwen3EncoderModel: qwen3Encoder, vae: qwenImageVae });

    expect(graph.nodes.model_loader?.type).toBe('anima_model_loader');
    expect(graph.nodes.model_loader?.qwen3_encoder_model).toEqual(qwen3Encoder);
    expect(graph.nodes.model_loader?.vae_model).toEqual(qwenImageVae);
    expect(graph.nodes.canvas_output?.type).toBe('anima_l2i');
  });

  describe('Anima IP-Adapter', () => {
    const animaQwen3Model: MainModelConfig = { ...animaModel, format: 'checkpoint', variant: 'anima_qwen3' };
    const animaIpAdapter: ComponentModelConfig = {
      base: 'anima',
      key: 'anima-ip-adapter',
      name: 'Anima IP-Adapter',
      type: 'ip_adapter',
    };
    const animaReference = (overrides: { isEnabled?: boolean; weight?: number } = {}) => ({
      config: {
        beginEndStepPct: [0.2, 0.9] as [number, number],
        clipVisionModel: 'ViT-H' as const,
        image: refImage,
        method: 'style' as const,
        model: animaIpAdapter,
        type: 'ip_adapter' as const,
        weight: overrides.weight ?? 1.5,
      },
      id: 'anima-ref',
      isEnabled: overrides.isEnabled ?? true,
    });
    const compileAnima = (model: MainModelConfig, referenceImages: GenerateSettings['referenceImages']) =>
      compile(model, { qwen3EncoderModel: qwen3Encoder, referenceImages, vae: qwenImageVae });

    it('sends the reference through anima_ip_adapter with a strength and no encoder or method', () => {
      const graph = compileAnima(animaQwen3Model, [animaReference()]);
      const adapter = getNodeByType(graph, 'anima_ip_adapter');

      expect(adapter).toEqual({
        begin_step_percent: 0.2,
        end_step_percent: 0.9,
        id: adapter?.id,
        image: { image_name: 'reference.png' },
        ip_adapter_model: animaIpAdapter,
        is_intermediate: true,
        strength: 1.5,
        type: 'anima_ip_adapter',
        use_cache: true,
      });
      expect(getNodeByType(graph, 'ip_adapter')).toBeUndefined();

      const collector = getEdge(graph, 'denoise_latents', 'ip_adapter')?.source.node_id;
      expect(graph.nodes[collector!]?.type).toBe('collect');
      expect(getEdge(graph, collector!, 'item')?.source).toEqual({ field: 'ip_adapter', node_id: adapter?.id });
      expect(getNodeByType(graph, 'core_metadata')?.ref_images).toMatchObject([
        { config: { model: animaIpAdapter, type: 'ip_adapter', weight: 1.5 }, id: 'anima-ref' },
      ]);
    });
  });

  it('builds an external provider graph', () => {
    const graph = compile(externalModel);

    expect(graph.nodes.canvas_output?.type).toBe('openai_image_generation');
    expect(graph.nodes.canvas_output?.model).toEqual(externalModel);
    expect(getEdge(graph, 'canvas_output', 'prompt')?.source.node_id).toBe('positive_prompt');
    expect(getEdge(graph, 'canvas_output', 'seed')?.source.node_id).toBe('seed');
  });

  it('wires external negative prompt only when the provider advertises support', () => {
    const graph = compile({
      ...externalModel,
      capabilities: { modes: ['txt2img'], supports_negative_prompt: true, supports_seed: false },
    });

    expect(getEdge(graph, 'canvas_output', 'negative_prompt')?.source.node_id).toBe('negative_prompt');
    expect(getEdge(graph, 'canvas_output', 'seed')).toBeUndefined();
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
      expect(graph.nodes.canvas_output).not.toHaveProperty('temperature');
      expect(graph.nodes.canvas_output).not.toHaveProperty('watermark');
      expect(getNodeByType(graph, 'core_metadata')).toMatchObject({
        openai_background: 'transparent',
        openai_input_fidelity: 'high',
        openai_quality: 'low',
      });
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

  it('generates from the crop while serializing complete original and crop metadata, with the name', () => {
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
          name: 'Hero pose',
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
        name: 'Hero pose',
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

  describe('Qwen-Image-2.1 reference images', () => {
    const qwenImage21Model: MainModelConfig = {
      base: 'qwen-image-2-1',
      format: 'diffusers',
      key: 'qwen21',
      name: 'Qwen-Image-2.1',
      type: 'main',
    };
    const references = (...names: string[]): GenerateReferenceImage[] =>
      names.map((name) => ({
        config: {
          image: { original: { image: { height: 768, image_name: name, width: 512 } } },
          type: 'qwen_image_2_1_reference_image',
        },
        id: name,
        isEnabled: true,
      }));

    it('reads each reference in both prompts and, in the same order, as latents in denoise', () => {
      const graph = compile(qwenImage21Model, { cfgScale: 4, referenceImages: references('a.png', 'b.png') });

      const nameOf = (imageNode: string) =>
        (graph.nodes[imageNode]?.image as { image_name: string } | undefined)?.image_name;
      const chain = (lastCollect: string) => [
        getEdge(graph, lastCollect, 'collection')?.source.node_id ?? '',
        lastCollect,
      ];

      const imagesInto = getEdge(graph, 'pos_cond', 'reference_images')?.source.node_id ?? '';
      expect(getEdge(graph, 'neg_cond', 'reference_images')?.source.node_id).toBe(imagesInto);
      expect(chain(imagesInto).map((collect) => nameOf(getEdge(graph, collect, 'item')?.source.node_id ?? ''))).toEqual(
        ['a.png', 'b.png']
      );

      const latentsInto = getEdge(graph, 'denoise_latents', 'reference_latents')?.source.node_id ?? '';
      const latentImages = chain(latentsInto).map((collect) => {
        const i2l = getEdge(graph, collect, 'item')?.source.node_id ?? '';
        expect(graph.nodes[i2l]).toMatchObject({ reference: true, type: 'qwen_image_2_1_i2l' });
        return nameOf(getEdge(graph, i2l, 'image')?.source.node_id ?? '');
      });
      expect(latentImages).toEqual(['a.png', 'b.png']);
    });
  });
});

describe('getFamilyGraphBuilder', () => {
  it('returns builders for supported bases only, never Object.prototype members', () => {
    expect(getFamilyGraphBuilder('sdxl')).toBeTypeOf('function');
    expect(getFamilyGraphBuilder('constructor')).toBeNull();
    expect(getFamilyGraphBuilder('toString')).toBeNull();
    expect(getFamilyGraphBuilder('__proto__')).toBeNull();
  });
});

describe('family graph facts', () => {
  const projectSettings = { useCpuNoise: true, useNsfwChecker: false, useWatermark: false };
  const build = (model: MainModelConfig, settings: GenerateSettings) =>
    GRAPH_BUILDERS[model.base as SupportedGenerateBase](settings, model, false, projectSettings);
  const recipes = { filled: fullyFilledSettingsFor, required: satisfiedSettingsFor, tuned: tunedSettingsFor };
  const familyCases = generateGraphCases.flatMap((testCase) =>
    (Object.keys(recipes) as (keyof typeof recipes)[]).map((recipe) => ({
      ...testCase,
      label: `${testCase.label} / ${recipe}`,
      recipe,
    }))
  );

  it.each(familyCases)('$label hands out the facts its emitted graph carries', ({ base, recipe, shape }) => {
    const { model, settings } = recipes[recipe](base, shape);
    const { conditioning, denoise, denoiseSize, graph, latents, layoutPrompt } = build(model, settings);
    const sourceOf = (nodeId: string, field: string) => {
      const edge = getEdge(graph, nodeId, field);
      return edge ? { field: edge.source.field, nodeId: edge.source.node_id } : undefined;
    };
    const named = (output: { node: { id: string }; field: string } | undefined) =>
      output ? { field: output.field, nodeId: output.node.id } : undefined;
    const reaches = (fromId: string, toId: string): boolean => {
      const seen = new Set([fromId]);
      const queue = [fromId];
      for (let id = queue.shift(); id !== undefined; id = queue.shift()) {
        if (id === toId) {
          return true;
        }
        for (const edge of graph.edges) {
          if (edge.source.node_id === id && !seen.has(edge.destination.node_id)) {
            seen.add(edge.destination.node_id);
            queue.push(edge.destination.node_id);
          }
        }
      }
      return false;
    };
    const decode =
      Object.values(graph.nodes).find((node) => isPidDecodeNodeType(node.type)) ?? graph.nodes.canvas_output!;

    const decodeLatents = sourceOf(decode.id, 'latents');
    expect(
      decodeLatents?.nodeId === REFINER_DENOISE_NODE_ID
        ? sourceOf(REFINER_DENOISE_NODE_ID, 'latents')?.nodeId
        : decodeLatents?.nodeId
    ).toBe(denoise.id);
    expect(denoise).toBe(graph.nodes[denoise.id]);

    const { height, width } = graph.nodes.noise ?? denoise;
    expect(denoiseSize).toEqual({ height, width });

    expect(latents === null).toBe(['ernie-image', 'ideogram-4'].includes(base));
    if (latents) {
      expect(named(latents.vae)).toEqual(sourceOf(decode.id, 'vae') ?? { field: 'vae', nodeId: 'model_loader' });
      expect(named(latents.unet)).toEqual(sourceOf(denoise.id, 'unet'));
    }

    expect(conditioning === null).toBe(
      !['sd-1', 'sd-2', 'sdxl', 'flux', 'flux2', 'krea-2', 'z-image', 'anima'].includes(base)
    );
    if (conditioning) {
      expect(reaches(conditioning.positive.id, conditioning.positiveCollector.id)).toBe(true);
      expect(
        graph.edges.some(
          (edge) => edge.source.node_id === conditioning.positiveCollector.id && edge.destination.node_id === denoise.id
        )
      ).toBe(true);
      if (conditioning.negative) {
        expect(reaches(conditioning.negative.node.id, conditioning.negative.collector.id)).toBe(true);
        expect(sourceOf(denoise.id, 'negative_conditioning')?.nodeId).toBe(conditioning.negative.collector.id);
      }
    }

    expect(layoutPrompt === null).toBe(!['flux3', 'ideogram-4', 'moss-image'].includes(base));
    if (layoutPrompt) {
      expect(sourceOf(layoutPrompt.id, 'prompt')?.nodeId).toBe('positive_prompt');
    }
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
  it('decodes Krea-2 with the Qwen-Image VAE node it shares', () => {
    const graph = compile(krea2Diffusers);

    expect(graph.nodes.model_loader?.type).toBe('krea2_model_loader');
    expect(graph.nodes.canvas_output?.type).toBe('qwen_image_l2i');
    expect(graph.nodes.denoise_latents?.type).toBe('krea2_denoise');
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
    expect(getEdge(graph, 'krea2_seed_variance', 'variance_seed')?.source.node_id).toBe('seed');
  });

  it('routes the Ideogram 4 prompt through the caption builder and omits negative conditioning', () => {
    const graph = compile(ideogram4Model);

    expect(getEdge(graph, 'caption_builder', 'prompt')?.source.node_id).toBe('positive_prompt');
    expect(getEdge(graph, 'pos_cond', 'prompt')?.source.node_id).toBe('caption_builder');
    const metadata = getNodeByType(graph, 'core_metadata')!;
    expect(getEdge(graph, metadata.id, 'ideogram4_caption')?.source).toEqual({
      field: 'value',
      node_id: 'caption_builder',
    });
    expect(getEdge(graph, metadata.id, 'positive_prompt')?.source.node_id).toBe('positive_prompt');
    expect(graph.nodes.neg_cond).toBeUndefined();
    expect(getEdge(graph, 'denoise_latents', 'negative_conditioning')).toBeUndefined();
  });

  it('emits a single-frame image output for Wan, not a video', () => {
    const graph = compile(wanDiffusers);

    expect(graph.nodes.model_loader?.type).toBe('wan_model_loader');
    expect(graph.nodes.canvas_output?.type).toBe('wan_l2i');
    expect(getNodeByType(graph, 'wan_l2v')).toBeUndefined();
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

  describe('Vision Tower', () => {
    const ggufEncoder: ComponentModelConfig = { ...qwen3Vl8bEncoder, format: 'gguf_quantized' };
    const tower = (variant: string): ComponentModelConfig => ({
      base: 'any',
      format: 'gguf_quantized',
      key: `mmproj-${variant}`,
      name: 'mmproj',
      type: 'qwen3_vl_vision_tower',
      variant,
    });
    const referenceImages: GenerateReferenceImage[] = [
      {
        config: {
          image: { original: { image: { height: 512, image_name: 'a.png', width: 512 } } },
          type: 'qwen_image_2_1_reference_image',
        },
        id: 'a.png',
        isEnabled: true,
      },
    ];

    it('sends the tower with a GGUF 8B encoder, and the references ride on the unchanged encoder edge', () => {
      const graph = compile(qwen21Gguf, {
        qwen3VLEncoderModel: ggufEncoder,
        qwen3VLVisionTowerModel: tower('qwen3_vl_8b'),
        referenceImages,
        vae: qwen21Vae,
      });

      expect(graph.nodes.model_loader).toMatchObject({ qwen3_vl_vision_tower_model: { key: 'mmproj-qwen3_vl_8b' } });
      expect(getEdge(graph, 'pos_cond', 'reference_images')).toBeDefined();
      expect(getEdge(graph, 'pos_cond', 'qwen3_vl_encoder')?.source.node_id).toBe('model_loader');
      expect(getNodeByType(graph, 'core_metadata')).toMatchObject({
        qwen3_vl_vision_tower: { key: 'mmproj-qwen3_vl_8b' },
      });
    });
  });
});

describe('FLUX.3 graphs', () => {
  const kleinModel: MainModelConfig = {
    base: 'flux3',
    format: 'diffusers',
    key: 'flux3-klein',
    name: 'FLUX.3 Klein',
    type: 'main',
    variant: 'flux3_klein',
  };
  const reference = (id: string, isEnabled = true): GenerateReferenceImage => ({
    config: {
      image: { original: { image: { height: 512, image_name: `${id}.png`, width: 512 } } },
      type: 'flux2_reference_image',
    },
    id,
    isEnabled,
  });

  const denoiserReferenceOrder = (graph: ReturnType<typeof compile>): string[] => {
    const names: string[] = [];
    let collect = getEdge(graph, 'denoise_latents', 'kontext_conditioning')?.source.node_id;
    while (collect) {
      const kontextId = getEdge(graph, collect, 'item')?.source.node_id;
      const kontext = kontextId ? graph.nodes[kontextId] : undefined;
      names.unshift((kontext as { image?: { image_name: string } } | undefined)?.image?.image_name ?? '?');
      collect = getEdge(graph, collect, 'collection')?.source.node_id;
    }
    return names;
  };

  it('hands the denoiser the references in the order their prompt tags number them', () => {
    const referenceImages = [reference('a'), reference('off', false), reference('b')];

    const graph = compile(kleinModel, { referenceImages });

    expect(denoiserReferenceOrder(graph)).toEqual(['a.png', 'b.png']);
    expect([...getReferenceImagePromptTags('flux3', referenceImages)]).toEqual([
      ['a', '<ref_image_0>'],
      ['b', '<ref_image_1>'],
    ]);
  });

  it('sends the prompt through the layout node, whose boxes become edit targets once references are wired', () => {
    const plain = compile(kleinModel);
    const withReference = compile(kleinModel, { referenceImages: [reference('a')] });
    const withDisabledReference = compile(kleinModel, { referenceImages: [reference('off', false)] });

    expect(getEdge(plain, 'layout_prompt', 'prompt')?.source.node_id).toBe('positive_prompt');
    expect(getEdge(plain, 'pos_cond', 'prompt')?.source).toEqual({ field: 'value', node_id: 'layout_prompt' });
    expect(plain.nodes.layout_prompt).toMatchObject({ type: 'flux3_layout_prompt' });
    expect(plain.nodes.layout_prompt).not.toHaveProperty('regions');
    expect(getEdge(plain, getNodeByType(plain, 'core_metadata')!.id, 'flux3_caption')).toBeUndefined();
    expect(plain.nodes.layout_prompt).not.toHaveProperty('edit_targets');
    expect(withReference.nodes.layout_prompt?.edit_targets).toBe(true);
    expect(getEdge(withDisabledReference, 'denoise_latents', 'kontext_conditioning')).toBeUndefined();
    expect(withDisabledReference.nodes.layout_prompt).not.toHaveProperty('edit_targets');
  });

  describe('with the prompt upsampler on', () => {
    const upsampler = {
      base: 'any',
      key: 'pup',
      name: 'FLUX.3 Prompt Upsampler',
      supports_images: true,
      type: 'text_llm',
    };
    const upsample = (model: MainModelConfig, overrides: Partial<GenerateSettings> = {}) =>
      compile(model, { flux3PromptUpsamplerEnabled: true, flux3PromptUpsamplerModel: upsampler, ...overrides });

    it('replaces the layout node and sends the typed prompt through the upsampler to the encoder', () => {
      const graph = upsample(kleinModel);
      const metadata = getNodeByType(graph, 'core_metadata')!;

      expect(graph.nodes.layout_prompt).toBeUndefined();
      expect(graph.nodes.prompt_upsampler).toMatchObject({
        type: 'flux3_prompt_upsampler',
        upsampler_model: { key: 'pup', name: 'FLUX.3 Prompt Upsampler', type: 'text_llm' },
      });
      expect(graph.nodes.prompt_upsampler).not.toHaveProperty('edit_targets');
      expect(graph.nodes.prompt_upsampler).not.toHaveProperty('max_new_tokens');
      expect(getEdge(graph, 'prompt_upsampler', 'prompt')?.source).toEqual({
        field: 'value',
        node_id: 'positive_prompt',
      });
      expect(getEdge(graph, 'pos_cond', 'prompt')?.source).toEqual({ field: 'value', node_id: 'prompt_upsampler' });
      expect(getEdge(graph, metadata.id, 'flux3_caption')?.source).toEqual({
        field: 'value',
        node_id: 'prompt_upsampler',
      });
      expect(getEdge(graph, metadata.id, 'positive_prompt')?.source.node_id).toBe('positive_prompt');
      expect(metadata.flux3_prompt_upsampler).toMatchObject({ key: 'pup' });
      expect(getEdge(graph, 'prompt_upsampler', 'reference_images')).toBeUndefined();
    });
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
});

describe('MOSS-Image graphs', () => {
  const mossDiffusers: MainModelConfig = {
    base: 'moss-image',
    format: 'diffusers',
    key: 'moss-diffusers',
    name: 'MOSS-Image Base',
    type: 'main',
  };
  it('builds a Diffusers pipeline graph that conditions without a collector and decodes as FLUX.2', () => {
    const graph = compile(mossDiffusers);

    expect(graph.nodes.model_loader).toMatchObject({ type: 'moss_image_model_loader' });
    expect(graph.nodes.model_loader?.qwen3_5_encoder_model).toBeUndefined();
    expect(graph.nodes.model_loader?.vae_model).toBeUndefined();
    expect(graph.nodes.denoise_latents).toMatchObject({
      cfg_scale: 3,
      height: 1024,
      steps: 50,
      type: 'moss_image_denoise',
      width: 1024,
    });
    expect(getNodeByType(graph, 'collect')).toBeUndefined();
    expect(getEdge(graph, 'denoise_latents', 'positive_conditioning')?.source).toEqual({
      field: 'conditioning',
      node_id: 'pos_cond',
    });
    expect(getEdge(graph, 'pos_cond', 'qwen3_5_encoder')?.source).toEqual({
      field: 'qwen3_5_encoder',
      node_id: 'model_loader',
    });
    expect(graph.nodes.canvas_output?.type).toBe('flux2_vae_decode');
    expect(getEdge(graph, 'canvas_output', 'vae')?.source).toEqual({ field: 'vae', node_id: 'model_loader' });
    expect(getEdge(graph, 'denoise_latents', 'vae')?.source).toEqual({ field: 'vae', node_id: 'model_loader' });
    expect(getNodeByType(graph, 'core_metadata')?.generation_mode).toBe('moss_image_txt2img');
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
    const output = graph.nodes.canvas_output;
    expect(output?.type).toBe('img_resize');
    expect(output).toMatchObject({ height: 1024, width: 1024 });
    expect(getEdge(graph, 'canvas_output', 'image')?.source.node_id).toBe(decode?.id);
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
    expect(graph.nodes.canvas_output?.type).toBe('flux_pid_decode');
    expect(getNodeByType(graph, 'img_resize')).toBeUndefined();
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
    expect(() => compile(cogView4Model, { pidMode: 'fit' })).toThrow(/does not support PiD/);
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

  describe('Qwen-Image LoRAs', () => {
    const qwenLora: LoraModelConfig = { base: 'qwen-image', key: 'qwen-lora', name: 'A LoRA', type: 'lora' };
    const loras = [{ isEnabled: true, model: qwenLora, weight: 0.7 }];

    it('routes the Qwen2.5-VL encoder to both prompts through the LoRA collection loader', () => {
      const graph = compile(qwenImageModel, { cfgScale: 4, loras });
      const loader = getNodeByType(graph, 'qwen_image_lora_collection_loader');

      expect(loader).toBeDefined();
      expect(getEdge(graph, loader?.id ?? '', 'qwen_vl_encoder')?.source.node_id).toBe('model_loader');
      expect(getEdge(graph, 'pos_cond', 'qwen_vl_encoder')?.source.node_id).toBe(loader?.id);
      expect(getEdge(graph, 'neg_cond', 'qwen_vl_encoder')?.source.node_id).toBe(loader?.id);
      expect(getEdge(graph, 'denoise_latents', 'transformer')?.source.node_id).toBe(loader?.id);
    });
  });

  it('encodes the enabled Krea-2 style reference at the denoise size and feeds it to the denoise node', () => {
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
  const editReference = (imageName: string, isEnabled = true) => ({
    config: {
      image: { original: { image: { height: 640, image_name: imageName, width: 480 } } },
      type: 'krea2_edit_reference_image' as const,
    },
    id: imageName,
    isEnabled,
  });

  const getReferenceLatentEncoders = (graph: ReturnType<typeof compile>) => {
    const encoders: Array<ReturnType<typeof compile>['nodes'][string] | undefined> = [];
    let collectId = getEdge(graph, 'denoise_latents', 'reference_latents')?.source.node_id;

    while (collectId) {
      const encoderId = getEdge(graph, collectId, 'item')?.source.node_id;

      encoders.unshift(encoderId ? graph.nodes[encoderId] : undefined);
      collectId = getEdge(graph, collectId, 'collection')?.source.node_id;
    }

    return encoders;
  };

  it('reads the enabled Krea-2 edit references, in list order, into the prompt and the denoise', () => {
    const cropped = {
      config: {
        image: {
          crop: {
            box: { height: 256, width: 256, x: 0, y: 0 },
            image: { height: 256, image_name: 'pose-crop.png', width: 256 },
            ratio: 1,
          },
          original: { image: { height: 1024, image_name: 'pose.png', width: 1024 } },
        },
        type: 'krea2_edit_reference_image' as const,
      },
      id: 'pose',
      isEnabled: true,
    };
    const graph = compile(krea2Model, {
      referenceImages: [
        editReference('depth.png'),
        editReference('off.png', false),
        cropped,
        editReference('room.png'),
      ],
    });
    const expectedImages = [{ image_name: 'depth.png' }, { image_name: 'pose-crop.png' }, { image_name: 'room.png' }];
    const encoders = getReferenceLatentEncoders(graph);

    expect(graph.nodes.pos_cond?.reference_images).toEqual(expectedImages);
    expect(graph.nodes.neg_cond).toBeUndefined();
    expect(encoders.map((encoder) => encoder?.image)).toEqual(expectedImages);
    for (const encoder of encoders) {
      expect(encoder).toMatchObject({ reference: true, type: 'qwen_image_i2l' });
      expect(encoder).not.toHaveProperty('width');
      expect(encoder).not.toHaveProperty('height');
      expect(getEdge(graph, encoder!.id, 'vae')?.source).toEqual({ field: 'vae', node_id: 'model_loader' });
    }
    expect(graph.nodes.denoise_latents).not.toHaveProperty('reference_kv_cache');
    expect(getNodeByType(graph, 'krea2_style_reference')).toBeUndefined();

    const metadata = getNodeByType(graph, 'core_metadata')!;

    expect(metadata).not.toHaveProperty('krea2_reference_kv_cache');
    expect((metadata.ref_images as { id: string }[]).map((entry) => entry.id)).toEqual([
      'depth.png',
      'pose',
      'room.png',
    ]);
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
    expect(graph.nodes.canvas_unfiltered_output?.type).toBe(plain.nodes.canvas_output?.type);
    expect(graph.nodes.canvas_unfiltered_output?.is_intermediate).toBe(true);
    expect(graph.nodes.canvas_img_nsfw).toMatchObject({ is_intermediate: true, type: 'img_nsfw', use_cache: false });
    expect(graph.nodes.canvas_output).toMatchObject({ is_intermediate: true, type: 'img_watermark', use_cache: false });
    expect(getEdge(graph, 'canvas_img_nsfw', 'image')?.source).toEqual({
      field: 'image',
      node_id: 'canvas_unfiltered_output',
    });
    expect(getEdge(graph, 'canvas_output', 'image')?.source).toEqual({ field: 'image', node_id: 'canvas_img_nsfw' });
    const metadataSource =
      base === 'qwen-image-2-1' ? 'denoise_latents' : (getNodeByType(graph, 'core_metadata')?.id ?? '');
    expect(getEdge(graph, 'canvas_output', 'metadata')?.source).toEqual({ field: 'metadata', node_id: metadataSource });
    expect(getEdge(graph, 'canvas_unfiltered_output', 'metadata')).toBeUndefined();
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
