import type { BackendGraphContract, GenerationProjectSettings } from '@features/generation/core/contracts';
import type {
  ComponentModelConfig,
  GenerateModelConfig,
  GenerateSettings,
  MainModelConfig,
  VaeModelConfig,
} from '@features/generation/core/types';

import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { getDefaultGenerateSettings } from '@features/generation/core/baseGenerationPolicies';
import { getPidDecoderBaseForMainBase } from '@features/generation/core/pid';
import { isPidDecodeNodeType } from '@features/generation/core/pidGraph';
import { describe, expect, it, vi } from 'vitest';

import type { ControlLayerGraphInput } from './addControlLayers';
import type { RegionalGuidanceInput } from './addRegionalGuidance';
import type { CanvasCompileMode, CanvasCompositingSettings, CanvasScalingSettings, Rect } from './types';

import { compileCanvasGraph } from './compileCanvasGraph';

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
const flux2DevModel: MainModelConfig = {
  ...flux2Model,
  key: 'flux2-dev-model',
  name: 'FLUX.2 [dev]',
  variant: 'dev',
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
const krea2Model: MainModelConfig = {
  base: 'krea-2',
  format: 'diffusers',
  key: 'krea2-model',
  name: 'Krea-2',
  type: 'main',
};
const qwenImage21Model: MainModelConfig = {
  base: 'qwen-image-2-1',
  format: 'diffusers',
  key: 'qwen-image-2-1-model',
  name: 'Qwen-Image-2.1',
  type: 'main',
};
const externalModel: GenerateModelConfig = {
  base: 'external',
  capabilities: { modes: ['txt2img'], supports_seed: true },
  format: 'external_api',
  key: 'external-model',
  name: 'OpenAI Image',
  provider_id: 'openai',
  type: 'external_image_generator',
};

const fluxVae: VaeModelConfig = { base: 'flux', key: 'flux-vae', name: 'FLUX VAE', type: 'vae' };
const qwenImageVae: VaeModelConfig = { base: 'qwen-image', key: 'qwen-vae', name: 'Qwen VAE', type: 'vae' };
const t5Encoder: ComponentModelConfig = { base: 'any', key: 't5', name: 'T5 Encoder', type: 't5_encoder' };
const clipEmbed: ComponentModelConfig = {
  base: 'any',
  key: 'clip',
  name: 'CLIP Embed',
  type: 'clip_embed',
  variant: 'large',
};
const qwen3Encoder: ComponentModelConfig = {
  base: 'any',
  key: 'qwen3',
  name: 'Qwen3 Encoder',
  type: 'qwen3_encoder',
  variant: 'qwen3_06b',
};

const PROJECT_SETTINGS: GenerationProjectSettings = { useCpuNoise: true, useNsfwChecker: false, useWatermark: false };

const settingsFor = (model: GenerateModelConfig, overrides: Partial<GenerateSettings> = {}): GenerateSettings => ({
  ...getDefaultGenerateSettings(model),
  seed: 1,
  seedMode: 'fixed',
  ...overrides,
});

/** Per-base overrides needed to pass component validation. */
const COMPONENT_OVERRIDES: Partial<Record<string, Partial<GenerateSettings>>> = {
  flux: { clipEmbedModel: clipEmbed, t5EncoderModel: t5Encoder, vae: fluxVae },
  anima: { qwen3EncoderModel: qwen3Encoder, vae: qwenImageVae },
};

const bbox: Rect = { height: 1024, width: 768, x: 128, y: 64 };

const COMPOSITING_FIXTURE: CanvasCompositingSettings = {
  coherenceEdgeSize: 16,
  coherenceMinDenoise: 0,
  coherenceMode: 'Gaussian Blur',
  infillColorValue: { a: 1, b: 0, g: 0, r: 0 },
  infillMethod: 'lama',
  infillPatchmatchDownscaleSize: 1,
  infillTileSize: 32,
  maskBlur: 16,
  outputOnlyMaskedRegions: false,
};

const compile = (
  model: GenerateModelConfig,
  mode: CanvasCompileMode,
  overrides: {
    settings?: Partial<GenerateSettings>;
    bbox?: Rect;
    compositeImageName?: string | null;
    maskImageName?: string | null;
    noiseMaskImageName?: string | null;
    compositing?: Partial<CanvasCompositingSettings>;
    scaling?: CanvasScalingSettings;
    strength?: number;
    destination?: 'canvas' | 'gallery';
    outputOnlyMaskedRegions?: boolean;
    controlLayers?: readonly ControlLayerGraphInput[];
    regionalGuidance?: readonly RegionalGuidanceInput[];
    projectSettings?: Partial<GenerationProjectSettings>;
  } = {}
) =>
  compileCanvasGraph({
    bbox: overrides.bbox ?? bbox,
    compositeImageName:
      'compositeImageName' in overrides ? (overrides.compositeImageName ?? null) : 'canvas-composite.png',
    compositing: {
      ...COMPOSITING_FIXTURE,
      outputOnlyMaskedRegions: overrides.outputOnlyMaskedRegions ?? false,
      ...overrides.compositing,
    },
    controlLayers: overrides.controlLayers,
    regionalGuidance: overrides.regionalGuidance,
    scaling: overrides.scaling,
    destination: overrides.destination ?? 'canvas',
    maskImageName:
      'maskImageName' in overrides ? overrides.maskImageName : mode === 'inpaint' ? 'canvas-mask.png' : null,
    mode,
    model,
    noiseMaskImageName: overrides.noiseMaskImageName ?? null,
    projectSettings: { ...PROJECT_SETTINGS, ...overrides.projectSettings },
    settings: settingsFor(model, {
      ...COMPONENT_OVERRIDES[model.base],
      ...overrides.settings,
    }),
    strength: overrides.strength ?? 0.6,
  });

const getEdge = (graph: BackendGraphContract, targetNodeId: string, targetField: string) =>
  graph.edges.find((edge) => edge.destination.node_id === targetNodeId && edge.destination.field === targetField);

const getNodeByType = (graph: BackendGraphContract, type: string) =>
  Object.values(graph.nodes).find((node) => node.type === type);

interface BaseCase {
  model: MainModelConfig;
  encodeType: string;
  outputType: string;
  denoiseType: string;
  txt2imgMode: string;
  img2imgMode: string;
  /** sd-3 / flux / flux2 rescale strength with an exponent of 0.2 (legacy parity). */
  optimizedDenoising: boolean;
}

/** Legacy-equivalent denoising_start for a strength on a given base. */
const expectedDenoisingStart = (strength: number, optimizedDenoising: boolean): number =>
  1 - strength ** (optimizedDenoising ? 0.2 : 1);

const BASE_CASES: BaseCase[] = [
  {
    model: sd1Model,
    encodeType: 'i2l',
    outputType: 'l2i',
    denoiseType: 'denoise_latents',
    txt2imgMode: 'txt2img',
    img2imgMode: 'img2img',
    optimizedDenoising: false,
  },
  {
    model: sd2Model,
    encodeType: 'i2l',
    outputType: 'l2i',
    denoiseType: 'denoise_latents',
    txt2imgMode: 'txt2img',
    img2imgMode: 'img2img',
    optimizedDenoising: false,
  },
  {
    model: sdxlModel,
    encodeType: 'i2l',
    outputType: 'l2i',
    denoiseType: 'denoise_latents',
    txt2imgMode: 'sdxl_txt2img',
    img2imgMode: 'sdxl_img2img',
    optimizedDenoising: false,
  },
  {
    model: sd3Model,
    encodeType: 'sd3_i2l',
    outputType: 'sd3_l2i',
    denoiseType: 'sd3_denoise',
    txt2imgMode: 'sd3_txt2img',
    img2imgMode: 'sd3_img2img',
    optimizedDenoising: true,
  },
  {
    model: fluxModel,
    encodeType: 'flux_vae_encode',
    outputType: 'flux_vae_decode',
    denoiseType: 'flux_denoise',
    txt2imgMode: 'flux_txt2img',
    img2imgMode: 'flux_img2img',
    optimizedDenoising: true,
  },
  {
    model: flux2Model,
    encodeType: 'flux2_vae_encode',
    outputType: 'flux2_vae_decode',
    denoiseType: 'flux2_denoise',
    txt2imgMode: 'flux2_txt2img',
    img2imgMode: 'flux2_img2img',
    optimizedDenoising: true,
  },
  {
    model: cogView4Model,
    encodeType: 'cogview4_i2l',
    outputType: 'cogview4_l2i',
    denoiseType: 'cogview4_denoise',
    txt2imgMode: 'cogview4_txt2img',
    img2imgMode: 'cogview4_img2img',
    optimizedDenoising: false,
  },
  {
    model: qwenImageModel,
    encodeType: 'qwen_image_i2l',
    outputType: 'qwen_image_l2i',
    denoiseType: 'qwen_image_denoise',
    txt2imgMode: 'qwen_image_txt2img',
    img2imgMode: 'qwen_image_img2img',
    optimizedDenoising: false,
  },
  {
    model: zImageModel,
    encodeType: 'z_image_i2l',
    outputType: 'z_image_l2i',
    denoiseType: 'z_image_denoise',
    txt2imgMode: 'z_image_txt2img',
    img2imgMode: 'z_image_img2img',
    optimizedDenoising: false,
  },
  {
    model: animaModel,
    encodeType: 'anima_i2l',
    outputType: 'anima_l2i',
    denoiseType: 'anima_denoise',
    txt2imgMode: 'anima_txt2img',
    img2imgMode: 'anima_img2img',
    optimizedDenoising: false,
  },
  {
    model: qwenImage21Model,
    encodeType: 'qwen_image_2_1_i2l',
    outputType: 'qwen_image_2_1_l2i',
    denoiseType: 'qwen_image_2_1_denoise',
    txt2imgMode: 'qwen_image_2_1_txt2img',
    img2imgMode: 'qwen_image_2_1_img2img',
    optimizedDenoising: false,
  },
];

seedArchitectureCapabilities();

describe('compileCanvasGraph', () => {
  describe('txt2img per base', () => {
    it.each(BASE_CASES)(
      'builds a $model.base txt2img graph sized to the bbox',
      ({ model, outputType, txt2imgMode }) => {
        const { backendGraph, graph, mode } = compile(model, 'txt2img');

        // No img2img encode node in a pure txt2img graph.
        expect(getNodeByType(backendGraph, 'i2l')).toBeUndefined();
        expect(getNodeByType(backendGraph, `${model.base}_i2l`)).toBeUndefined();
        expect(getNodeByType(backendGraph, `${model.base}_vae_encode`)).toBeUndefined();

        // Canvas destination: intermediate output of the base builder's decode type.
        expect(backendGraph.nodes.canvas_output?.type).toBe(outputType);
        expect(backendGraph.nodes.canvas_output?.is_intermediate).toBe(true);

        // Deterministic prompt/seed node ids are present.
        expect(backendGraph.nodes.positive_prompt).toBeDefined();
        expect(backendGraph.nodes.seed).toBeDefined();

        // Dimensions come from the bbox, not the settings.
        expect(getNodeByType(backendGraph, 'core_metadata')?.width).toBe(bbox.width);
        expect(getNodeByType(backendGraph, 'core_metadata')?.height).toBe(bbox.height);
        expect(getNodeByType(backendGraph, 'core_metadata')?.generation_mode).toBe(txt2imgMode);

        expect(mode).toBe('txt2img');
        expect(graph.label).toBe(`${model.name} txt2img`);
      }
    );

    it('applies bbox dimensions to the SD noise node', () => {
      const { backendGraph } = compile(sd1Model, 'txt2img');

      expect(backendGraph.nodes.noise?.width).toBe(bbox.width);
      expect(backendGraph.nodes.noise?.height).toBe(bbox.height);
    });

    it('produces a durable (non-intermediate) output for a Gallery destination', () => {
      // Canvas destination stages an intermediate; Gallery must be a durable image.
      expect(compile(sd1Model, 'txt2img').backendGraph.nodes.canvas_output?.is_intermediate).toBe(true);
      expect(
        compile(sd1Model, 'txt2img', { destination: 'gallery' }).backendGraph.nodes.canvas_output?.is_intermediate
      ).toBe(false);
    });

    it('applies bbox dimensions to a flow-model denoise node', () => {
      const { backendGraph } = compile(fluxModel, 'txt2img');

      expect(backendGraph.nodes.denoise_latents?.width).toBe(bbox.width);
      expect(backendGraph.nodes.denoise_latents?.height).toBe(bbox.height);
    });
  });

  describe('img2img per base', () => {
    it.each(BASE_CASES)(
      'grafts a $model.base image-to-latents encode node',
      ({ model, encodeType, denoiseType, img2imgMode, optimizedDenoising }) => {
        const { backendGraph, mode } = compile(model, 'img2img', { strength: 0.6 });

        const encode = getNodeByType(backendGraph, encodeType);
        expect(encode).toBeDefined();
        expect(encode?.image).toEqual({ image_name: 'canvas-composite.png' });

        // image → encode → denoise.latents
        expect(getEdge(backendGraph, encode!.id, 'vae')).toBeDefined();
        expect(getEdge(backendGraph, 'denoise_latents', 'latents')?.source.node_id).toBe(encode?.id);

        expect(backendGraph.nodes.denoise_latents?.type).toBe(denoiseType);
        expect(backendGraph.nodes.denoise_latents?.denoising_start).toBeCloseTo(
          expectedDenoisingStart(0.6, optimizedDenoising),
          10
        );
        expect(backendGraph.nodes.denoise_latents?.denoising_end).toBe(1);

        // metadata reflects the img2img mode + strength.
        const metadata = getNodeByType(backendGraph, 'core_metadata');
        expect(metadata?.generation_mode).toBe(img2imgMode);
        expect(metadata?.strength).toBe(0.6);

        expect(mode).toBe('img2img');
      }
    );

    it('feeds the encode node from the same VAE source as the decode node', () => {
      const { backendGraph } = compile(sd1Model, 'img2img');
      const encode = getNodeByType(backendGraph, 'i2l');
      const decodeVaeSource = getEdge(backendGraph, 'canvas_output', 'vae')?.source.node_id;
      const encodeVaeSource = getEdge(backendGraph, encode!.id, 'vae')?.source.node_id;

      expect(encodeVaeSource).toBe(decodeVaeSource);
      expect(encodeVaeSource).toBe('model_loader');
    });

    it('routes the encode VAE through the seamless node when tiling is enabled', () => {
      const { backendGraph } = compile(sd1Model, 'img2img', { settings: { seamlessXAxis: true } });
      const encode = getNodeByType(backendGraph, 'i2l');

      expect(getEdge(backendGraph, encode!.id, 'vae')?.source.node_id).toBe('seamless');
    });

    it('mirrors the decode precision onto the SD encode node', () => {
      const fp16 = compile(sd1Model, 'img2img', { settings: { vaePrecision: 'fp16' } });
      expect(getNodeByType(fp16.backendGraph, 'i2l')?.fp32).toBe(false);

      const fp32 = compile(sd1Model, 'img2img', { settings: { vaePrecision: 'fp32' } });
      expect(getNodeByType(fp32.backendGraph, 'i2l')?.fp32).toBe(true);
    });

    it('does not attach fp32 to non-SD encode nodes', () => {
      const { backendGraph } = compile(fluxModel, 'img2img');

      expect(getNodeByType(backendGraph, 'flux_vae_encode')?.fp32).toBeUndefined();
    });

    it('threads the composite image name into the encode node', () => {
      const { backendGraph } = compile(sdxlModel, 'img2img', { compositeImageName: 'my-upload.png' });

      expect(getNodeByType(backendGraph, 'i2l')?.image).toEqual({ image_name: 'my-upload.png' });
    });

    it('reflects strength in denoising_start', () => {
      const { backendGraph } = compile(sd1Model, 'img2img', { strength: 0.25 });

      expect(backendGraph.nodes.denoise_latents?.denoising_start).toBeCloseTo(0.75, 10);
    });

    it.each([sd3Model, fluxModel, flux2Model])(
      'rescales strength with the optimized (exponent-0.2) curve for $base',
      (model) => {
        const { backendGraph } = compile(model, 'img2img', { strength: 0.75 });

        // Legacy parity: 1 - 0.75 ** 0.2 ≈ 0.056 (not the linear 0.25).
        expect(backendGraph.nodes.denoise_latents?.denoising_start).toBeCloseTo(1 - 0.75 ** 0.2, 10);
      }
    );
  });

  describe('unsupported models', () => {
    it('rejects external image generators for txt2img', () => {
      expect(() => compile(externalModel, 'txt2img')).toThrow('does not support canvas generation');
    });

    it('rejects external image generators for img2img', () => {
      expect(() => compile(externalModel, 'img2img')).toThrow('does not support canvas generation');
    });
  });

  describe('validation', () => {
    it('rejects img2img without a composite image', () => {
      expect(() => compile(sd1Model, 'img2img', { compositeImageName: null })).toThrow(
        'Canvas generation requires a composited source image.'
      );
    });

    it('rejects a strength of zero', () => {
      expect(() => compile(sd1Model, 'img2img', { strength: 0 })).toThrow(
        'Canvas denoising strength must be greater than 0 and at most 1.'
      );
    });

    it('rejects a strength above one', () => {
      expect(() => compile(sd1Model, 'img2img', { strength: 1.5 })).toThrow(
        'Canvas denoising strength must be greater than 0 and at most 1.'
      );
    });

    it('accepts a strength of exactly one', () => {
      const { backendGraph } = compile(sd1Model, 'img2img', { strength: 1 });

      expect(backendGraph.nodes.denoise_latents?.denoising_start).toBe(0);
    });

    it('rejects a zero-area bbox', () => {
      expect(() => compile(sd1Model, 'txt2img', { bbox: { height: 0, width: 512, x: 0, y: 0 } })).toThrow(
        'Canvas bounding box must have a positive area.'
      );
    });

    it('rejects fractional bbox dimensions before emitting backend resize fields', () => {
      expect(() => compile(flux2Model, 'txt2img', { bbox: { height: 888, width: 1024.5, x: 0, y: 0 } })).toThrow(
        'Canvas bounding box dimensions must be whole pixels.'
      );
    });

    it('processes an off-grid bbox on the model grid and resizes the output back to the bbox', () => {
      const { backendGraph } = compile(flux2Model, 'txt2img', {
        bbox: { height: 888, width: 1025, x: 0, y: 0 },
      });

      expect(backendGraph.nodes.denoise_latents).toMatchObject({ height: 896, width: 1024 });
      expect(backendGraph.nodes.canvas_output).toMatchObject({
        height: 888,
        type: 'img_resize',
        width: 1025,
      });
      expect(getEdge(backendGraph, 'canvas_output', 'image')?.source.node_id).toBe('canvas_processing_output');
      expect(getNodeByType(backendGraph, 'core_metadata')).toMatchObject({ height: 888, width: 1025 });
      expect(getEdge(backendGraph, 'canvas_output', 'metadata')?.source.node_id).toBe(
        getNodeByType(backendGraph, 'core_metadata')?.id
      );
    });

    it('does not silently upscale a small grid-valid bbox to the model optimal area', () => {
      const { backendGraph } = compile(sd1Model, 'txt2img', {
        bbox: { height: 64, width: 64, x: 0, y: 0 },
      });

      expect(backendGraph.nodes.noise).toMatchObject({ height: 64, width: 64 });
      expect(backendGraph.nodes.canvas_processing_output).toBeUndefined();
      expect(backendGraph.nodes.canvas_output?.type).toBe('l2i');
    });

    it('denoises a small txt2img bbox at the optimal area when scaling is automatic', () => {
      // SD 1.x: grid 8, optimal 512. 2:1 grows to 728×368 (720×360 is just under 512²).
      const { backendGraph } = compile(sd1Model, 'txt2img', {
        bbox: { height: 128, width: 256, x: 0, y: 0 },
        scaling: { height: null, method: 'auto', width: null },
      });

      expect(backendGraph.nodes.noise).toMatchObject({ height: 368, width: 728 });
      expect(backendGraph.nodes.canvas_output).toMatchObject({ height: 128, type: 'img_resize', width: 256 });
    });

    it('scales an img2img source up to the optimal area for processing when scaling is automatic', () => {
      // FLUX.2: grid 16, optimal 1024. 2:1 grows to 1456×736 (1440×720 is just under 1024²).
      const { backendGraph } = compile(flux2Model, 'img2img', {
        bbox: { height: 256, width: 512, x: 0, y: 0 },
        scaling: { height: null, method: 'auto', width: null },
      });

      expect(backendGraph.nodes.canvas_resize_initial_to_processing).toMatchObject({
        height: 736,
        type: 'img_resize',
        width: 1456,
      });
      expect(backendGraph.nodes.canvas_output).toMatchObject({ height: 256, type: 'img_resize', width: 512 });
    });

    it('processes at the manual size and restores the bbox footprint', () => {
      const { backendGraph } = compile(flux2Model, 'img2img', {
        bbox: { height: 512, width: 512, x: 0, y: 0 },
        scaling: { height: 768, method: 'manual', width: 1024 },
      });

      expect(backendGraph.nodes.canvas_resize_initial_to_processing).toMatchObject({ height: 768, width: 1024 });
      expect(backendGraph.nodes.canvas_output).toMatchObject({ height: 512, type: 'img_resize', width: 512 });
    });

    it('leaves an on-grid bbox alone when scaling is off', () => {
      const { backendGraph } = compile(flux2Model, 'img2img', {
        bbox: { height: 512, width: 512, x: 0, y: 0 },
        scaling: { height: 768, method: 'none', width: 1024 },
      });

      expect(backendGraph.nodes.canvas_resize_initial_to_processing).toBeUndefined();
      expect(backendGraph.nodes.canvas_output?.type).not.toBe('img_resize');
    });

    it('resizes an off-grid img2img source before processing and restores the bbox footprint', () => {
      const { backendGraph } = compile(flux2Model, 'img2img', {
        bbox: { height: 888, width: 1025, x: 0, y: 0 },
      });

      expect(backendGraph.nodes.canvas_resize_initial_to_processing).toMatchObject({
        height: 896,
        type: 'img_resize',
        width: 1024,
      });
      expect(getEdge(backendGraph, 'canvas_i2l', 'image')?.source.node_id).toBe('canvas_resize_initial_to_processing');
      expect(backendGraph.nodes.canvas_output).toMatchObject({ height: 888, type: 'img_resize', width: 1025 });
    });

    it('resizes off-grid inpaint inputs for processing, then composites at the exact bbox size', () => {
      const { backendGraph } = compile(flux2Model, 'inpaint', {
        bbox: { height: 888, width: 1025, x: 0, y: 0 },
      });

      expect(backendGraph.nodes.canvas_resize_initial_to_processing).toMatchObject({ height: 896, width: 1024 });
      expect(backendGraph.nodes.canvas_resize_mask_to_processing).toMatchObject({ height: 896, width: 1024 });
      expect(backendGraph.nodes.canvas_resize_generated_to_bbox).toMatchObject({ height: 888, width: 1025 });
      expect(backendGraph.nodes.canvas_resize_output_mask_to_bbox).toMatchObject({ height: 888, width: 1025 });
      expect(getEdge(backendGraph, 'canvas_output', 'layer_upper')?.source.node_id).toBe(
        'canvas_resize_generated_to_bbox'
      );
      expect(getEdge(backendGraph, 'canvas_output', 'mask')?.source.node_id).toBe('canvas_resize_output_mask_to_bbox');
    });

    it('resizes the combined off-grid outpaint mask for processing, then composites at the bbox size', () => {
      const { backendGraph } = compile(flux2Model, 'outpaint', {
        bbox: { height: 888, width: 1025, x: 0, y: 0 },
      });

      expect(backendGraph.nodes.canvas_resize_initial_to_processing).toMatchObject({ height: 896, width: 1024 });
      expect(backendGraph.nodes.canvas_resize_outpaint_mask_to_processing).toMatchObject({
        height: 896,
        width: 1024,
      });
      expect(backendGraph.nodes.canvas_resize_generated_to_bbox).toMatchObject({ height: 888, width: 1025 });
      expect(backendGraph.nodes.canvas_resize_output_mask_to_bbox).toMatchObject({ height: 888, width: 1025 });
    });

    it.each(['inpaint', 'outpaint'] as const)(
      'resizes an off-grid %s noise mask before applying it to the processing image',
      (mode) => {
        const { backendGraph } = compile(flux2Model, mode, {
          bbox: { height: 888, width: 1025, x: 0, y: 0 },
          noiseMaskImageName: 'noise-mask.png',
        });

        expect(backendGraph.nodes.canvas_resize_noise_mask_to_processing).toMatchObject({
          height: 896,
          image: { image_name: 'noise-mask.png' },
          type: 'img_resize',
          width: 1024,
        });
        expect(getEdge(backendGraph, 'add_inpaint_noise', 'mask')?.source.node_id).toBe(
          'canvas_resize_noise_mask_to_processing'
        );
        expect(getEdge(backendGraph, 'add_inpaint_noise', 'image')?.source.node_id).toBe(
          mode === 'inpaint' ? 'canvas_resize_initial_to_processing' : 'infill'
        );
      }
    );

    it('keeps an off-grid masked-only output transparent at the exact bbox size', () => {
      const { backendGraph } = compile(flux2Model, 'inpaint', {
        bbox: { height: 888, width: 1025, x: 0, y: 0 },
        outputOnlyMaskedRegions: true,
      });

      expect(backendGraph.nodes.canvas_output).toMatchObject({
        invert_mask: true,
        type: 'apply_mask_to_image',
      });
      expect(getEdge(backendGraph, 'canvas_output', 'image')?.source.node_id).toBe('canvas_resize_generated_to_bbox');
      expect(getEdge(backendGraph, 'canvas_output', 'mask')?.source.node_id).toBe('canvas_resize_output_mask_to_bbox');
    });

    it('surfaces missing-component validation for the bbox-sized settings', () => {
      // FLUX with no component overrides fails the shared generate validation.
      expect(() =>
        compileCanvasGraph({
          bbox,
          compositeImageName: null,
          compositing: COMPOSITING_FIXTURE,
          destination: 'canvas',
          mode: 'txt2img',
          model: fluxModel,
          projectSettings: PROJECT_SETTINGS,
          settings: settingsFor(fluxModel),
          strength: 0.6,
        })
      ).toThrow('Generate needs a T5 Encoder for FLUX models.');
    });
  });
});

describe('compileCanvasGraph — inpaint per base', () => {
  it.each(BASE_CASES)(
    'grafts a $model.base inpaint pipeline (encode, gradient mask, blend composite-back)',
    ({ model, encodeType, outputType, denoiseType, txt2imgMode, optimizedDenoising }) => {
      const { backendGraph, mode } = compile(model, 'inpaint', { strength: 0.6 });
      expect(mode).toBe('inpaint');

      // Encode fed by the initial composite image → denoise.latents.
      const encode = getNodeByType(backendGraph, encodeType);
      expect(encode).toBeDefined();
      expect(encode?.image).toEqual({ image_name: 'canvas-composite.png' });
      expect(getEdge(backendGraph, 'denoise_latents', 'latents')?.source.node_id).toBe(encode?.id);

      // create_gradient_mask carries coherence params + the mask image, and feeds denoise_mask.
      const gradient = backendGraph.nodes.create_gradient_mask;
      expect(gradient?.type).toBe('create_gradient_mask');
      expect(gradient?.coherence_mode).toBe('Gaussian Blur');
      expect(gradient?.edge_radius).toBe(16);
      expect(gradient?.minimum_denoise).toBe(0);
      expect(gradient?.image).toEqual({ image_name: 'canvas-composite.png' });
      expect(gradient?.mask).toEqual({ image_name: 'canvas-mask.png' });
      expect(getEdge(backendGraph, 'denoise_latents', 'denoise_mask')?.source.node_id).toBe('create_gradient_mask');

      // fp32 only on the SD gradient mask (mirrors the SD i2l).
      expect(gradient?.fp32).toBe(encodeType === 'i2l');

      // The base decode is demoted to an intermediate canvas_l2i; the blend is canvas_output.
      expect(backendGraph.nodes.canvas_l2i?.type).toBe(outputType);
      expect(backendGraph.nodes.canvas_l2i?.is_intermediate).toBe(true);
      const blend = backendGraph.nodes.canvas_output;
      expect(blend?.type).toBe('invokeai_img_blend');
      expect(blend?.is_intermediate).toBe(true);
      expect(blend?.layer_base).toEqual({ image_name: 'canvas-composite.png' });
      expect(getEdge(backendGraph, 'canvas_output', 'layer_upper')?.source.node_id).toBe('canvas_l2i');

      // expand_mask_with_fade uses the mask blur and feeds the blend mask.
      expect(backendGraph.nodes.expand_mask?.type).toBe('expand_mask_with_fade');
      expect(backendGraph.nodes.expand_mask?.fade_size_px).toBe(16);
      expect(getEdge(backendGraph, 'expand_mask', 'mask')?.source.node_id).toBe('create_gradient_mask');
      expect(getEdge(backendGraph, 'canvas_output', 'mask')?.source.node_id).toBe('expand_mask');

      // Strength curve + metadata mode.
      expect(backendGraph.nodes.denoise_latents?.type).toBe(denoiseType);
      expect(backendGraph.nodes.denoise_latents?.denoising_start).toBeCloseTo(
        expectedDenoisingStart(0.6, optimizedDenoising)
      );
      const metadata = getNodeByType(backendGraph, 'core_metadata');
      expect(metadata?.generation_mode).toBe(txt2imgMode.replace('txt2img', 'inpaint'));
      expect(metadata?.strength).toBe(0.6);
    }
  );

  it('re-points core_metadata onto the final blend output (not the intermediate decode)', () => {
    const { backendGraph } = compile(sd1Model, 'inpaint');
    const metadata = getNodeByType(backendGraph, 'core_metadata')!;
    const metaEdge = backendGraph.edges.find(
      (edge) => edge.source.node_id === metadata.id && edge.destination.field === 'metadata'
    );
    expect(metaEdge?.destination.node_id).toBe('canvas_output');
  });

  it('outputs only the generated mask region with transparency when requested', () => {
    const { backendGraph } = compile(sd1Model, 'inpaint', { outputOnlyMaskedRegions: true });

    const output = backendGraph.nodes.canvas_output;
    expect(output?.type).toBe('apply_mask_to_image');
    expect(output?.invert_mask).toBe(true);
    expect(output?.is_intermediate).toBe(true);
    expect(backendGraph.nodes.canvas_l2i?.is_intermediate).toBe(true);
    expect(getEdge(backendGraph, 'canvas_output', 'image')?.source.node_id).toBe('canvas_l2i');
    expect(getEdge(backendGraph, 'canvas_output', 'mask')?.source.node_id).toBe('expand_mask');
    expect(output?.layer_base).toBeUndefined();

    const metadata = getNodeByType(backendGraph, 'core_metadata')!;
    const metaEdge = backendGraph.edges.find(
      (edge) => edge.source.node_id === metadata.id && edge.destination.field === 'metadata'
    );
    expect(metaEdge?.destination.node_id).toBe('canvas_output');
  });

  it('composites the generated mask region over the initial image when masked-only output is disabled', () => {
    const { backendGraph } = compile(sd1Model, 'inpaint', { outputOnlyMaskedRegions: false });

    expect(backendGraph.nodes.canvas_output).toMatchObject({
      layer_base: { image_name: 'canvas-composite.png' },
      type: 'invokeai_img_blend',
    });
  });

  it('wires a UNet edge into create_gradient_mask only for SD-family models', () => {
    const sd = compile(sd1Model, 'inpaint').backendGraph;
    expect(getEdge(sd, 'create_gradient_mask', 'unet')).toBeDefined();
    const flux = compile(fluxModel, 'inpaint').backendGraph;
    expect(getEdge(flux, 'create_gradient_mask', 'unet')).toBeUndefined();
  });

  it('inserts an img_noise node before encode when a noise mask is present', () => {
    const { backendGraph } = compile(sd1Model, 'inpaint', { noiseMaskImageName: 'noise.png' });
    const noise = backendGraph.nodes.add_inpaint_noise;
    expect(noise?.type).toBe('img_noise');
    expect(noise?.image).toEqual({ image_name: 'canvas-composite.png' });
    expect(noise?.mask).toEqual({ image_name: 'noise.png' });
    expect(getEdge(backendGraph, 'add_inpaint_noise', 'seed')?.source.node_id).toBe('seed');
    // noise → i2l.image
    expect(getEdge(backendGraph, 'canvas_i2l', 'image')?.source.node_id).toBe('add_inpaint_noise');
  });

  it('feeds the composite directly into i2l when there is no noise mask', () => {
    const { backendGraph } = compile(sd1Model, 'inpaint');
    expect(backendGraph.nodes.add_inpaint_noise).toBeUndefined();
    expect(backendGraph.nodes.canvas_i2l?.image).toEqual({ image_name: 'canvas-composite.png' });
  });

  it('rejects inpaint without a mask image', () => {
    expect(() => compile(sd1Model, 'inpaint', { maskImageName: null })).toThrow(
      'Canvas inpainting requires an inpaint mask.'
    );
  });

  it('produces a durable blend output for a Gallery destination', () => {
    expect(
      compile(sd1Model, 'inpaint', { destination: 'gallery' }).backendGraph.nodes.canvas_output?.is_intermediate
    ).toBe(false);
  });
});

describe('compileCanvasGraph — outpaint per base', () => {
  it.each(BASE_CASES)('grafts a $model.base outpaint pipeline (infill + alpha mask)', ({ model, txt2imgMode }) => {
    const { backendGraph, mode } = compile(model, 'outpaint', { maskImageName: null });
    expect(mode).toBe('outpaint');

    // Infill applied to the initial image (default method: lama).
    const infill = backendGraph.nodes.infill;
    expect(infill?.type).toBe('infill_lama');
    expect(infill?.image).toEqual({ image_name: 'canvas-composite.png' });

    // Mask derived from the raster alpha (tomask) → gradient mask (no inpaint mask present).
    expect(backendGraph.nodes.image_alpha_to_mask?.type).toBe('tomask');
    expect(getEdge(backendGraph, 'create_gradient_mask', 'mask')?.source.node_id).toBe('image_alpha_to_mask');

    // infill → i2l.image
    expect(getEdge(backendGraph, 'canvas_i2l', 'image')?.source.node_id).toBe('infill');

    // Composite-back blend claims canvas_output.
    expect(backendGraph.nodes.canvas_output?.type).toBe('invokeai_img_blend');
    expect(getNodeByType(backendGraph, 'core_metadata')?.generation_mode).toBe(
      txt2imgMode.replace('txt2img', 'outpaint')
    );
  });

  it('combines the inpaint mask with the raster alpha when a mask image is present', () => {
    const { backendGraph } = compile(sd1Model, 'outpaint', { maskImageName: 'canvas-mask.png' });
    const combine = backendGraph.nodes.mask_combine;
    expect(combine?.type).toBe('mask_combine');
    expect(combine?.mask1).toEqual({ image_name: 'canvas-mask.png' });
    expect(getEdge(backendGraph, 'mask_combine', 'mask2')?.source.node_id).toBe('image_alpha_to_mask');
    expect(getEdge(backendGraph, 'create_gradient_mask', 'mask')?.source.node_id).toBe('mask_combine');
  });

  it.each([
    ['patchmatch', 'infill_patchmatch'],
    ['lama', 'infill_lama'],
    ['cv2', 'infill_cv2'],
    ['tile', 'infill_tile'],
    ['color', 'infill_rgba'],
  ] as const)('uses the %s infill node', (method, nodeType) => {
    const { backendGraph } = compile(sd1Model, 'outpaint', {
      maskImageName: null,
      compositing: { infillMethod: method },
    });
    expect(backendGraph.nodes.infill?.type).toBe(nodeType);
  });

  it('threads infill sub-params (tile size, patchmatch downscale, color)', () => {
    const tile = compile(sd1Model, 'outpaint', {
      maskImageName: null,
      compositing: { infillMethod: 'tile', infillTileSize: 64 },
    }).backendGraph;
    expect(tile.nodes.infill?.tile_size).toBe(64);

    const patch = compile(sd1Model, 'outpaint', {
      maskImageName: null,
      compositing: { infillMethod: 'patchmatch', infillPatchmatchDownscaleSize: 3 },
    }).backendGraph;
    expect(patch.nodes.infill?.downscale).toBe(3);

    const color = compile(sd1Model, 'outpaint', {
      maskImageName: null,
      compositing: { infillMethod: 'color', infillColorValue: { r: 10, g: 20, b: 30, a: 1 } },
    }).backendGraph;
    expect(color.nodes.infill?.color).toEqual({ r: 10, g: 20, b: 30, a: 255 });
  });

  it('threads coherence + mask-blur settings into the outpaint gradient/expand nodes', () => {
    const { backendGraph } = compile(sd1Model, 'outpaint', {
      maskImageName: null,
      compositing: { coherenceMode: 'Staged', coherenceEdgeSize: 8, coherenceMinDenoise: 0.3, maskBlur: 24 },
    });
    expect(backendGraph.nodes.create_gradient_mask?.coherence_mode).toBe('Staged');
    expect(backendGraph.nodes.create_gradient_mask?.edge_radius).toBe(8);
    expect(backendGraph.nodes.create_gradient_mask?.minimum_denoise).toBe(0.3);
    expect(backendGraph.nodes.expand_mask?.fade_size_px).toBe(24);
  });

  it('rejects an external model for every image mode', () => {
    expect(() => compile(externalModel, 'outpaint', { maskImageName: null })).toThrow(
      'does not support canvas generation'
    );
  });
});

// Exercise control grafting through the real compiler entry point.
describe('compileCanvasGraph — control layers (integration)', () => {
  const controlNetLayer: ControlLayerGraphInput = {
    beginEndStepPct: [0.1, 0.85],
    controlMode: 'more_control',
    id: 'control-layer-1',
    imageName: 'control-composite.png',
    kind: 'controlnet',
    model: { base: 'sd-1', key: 'canny-controlnet', name: 'Canny ControlNet', type: 'controlnet' },
    weight: 0.65,
  };

  it('grafts an enabled control layer through a real compiled canvas graph: control node args, collector → denoise.control wiring, base graph intact', () => {
    const { backendGraph, graph, mode } = compile(sd1Model, 'img2img', { controlLayers: [controlNetLayer] });

    expect(mode).toBe('img2img');

    // The control node carries the resolved model/weight/begin-end/mode args.
    const controlNode = backendGraph.nodes['control_net_control-layer-1'];
    expect(controlNode?.type).toBe('controlnet');
    expect(controlNode?.control_model).toEqual(controlNetLayer.model);
    expect(controlNode?.control_weight).toBe(0.65);
    expect(controlNode?.begin_step_percent).toBe(0.1);
    expect(controlNode?.end_step_percent).toBe(0.85);
    expect(controlNode?.control_mode).toBe('more_control');
    expect(controlNode?.image).toEqual({ image_name: 'control-composite.png' });

    expect(backendGraph.nodes.control_net_collector?.type).toBe('collect');
    expect(getEdge(backendGraph, 'control_net_collector', 'item')?.source.node_id).toBe('control_net_control-layer-1');
    expect(getEdge(backendGraph, 'denoise_latents', 'control')?.source.node_id).toBe('control_net_collector');

    // Canvas grafting must preserve unaffected base-graph wiring.
    expect(backendGraph.nodes.positive_prompt).toBeDefined();
    expect(backendGraph.nodes.negative_prompt).toBeDefined();
    expect(backendGraph.nodes.seed).toBeDefined();
    expect(backendGraph.nodes.canvas_i2l?.type).toBe('i2l');
    expect(getEdge(backendGraph, 'denoise_latents', 'latents')?.source.node_id).toBe('canvas_i2l');
    expect(backendGraph.nodes.canvas_output).toBeDefined();
    expect(graph.label).toBe(`${sd1Model.name} img2img`);
  });

  it('grafts a Z-Image control into the real Z-Image base graph', () => {
    const layer: ControlLayerGraphInput = {
      beginEndStepPct: [0, 1],
      controlMode: null,
      id: 'z-control',
      imageName: 'z-control.png',
      kind: 'z_image_control',
      model: { base: 'z-image', key: 'z-control-model', name: 'Z Control', type: 'controlnet' },
      weight: 0.75,
    };

    const { backendGraph } = compile(zImageModel, 'txt2img', { controlLayers: [layer] });

    expect(backendGraph.nodes.denoise_latents?.type).toBe('z_image_denoise');
    expect(backendGraph.nodes['z_image_control_z-control']).toMatchObject({
      control_context_scale: 0.75,
      control_model: layer.model,
      image: { image_name: 'z-control.png' },
      type: 'z_image_control',
    });
    expect(getEdge(backendGraph, 'denoise_latents', 'control')?.source).toEqual({
      field: 'control',
      node_id: 'z_image_control_z-control',
    });
  });
});

describe('compileCanvasGraph — Anima ControlNet-LLLite (integration)', () => {
  const lliteLayer = (id: string, key: string, weight = 1): ControlLayerGraphInput => ({
    beginEndStepPct: [0, 0.9],
    controlMode: null,
    id,
    imageName: `${id}-composite.png`,
    kind: 'anima_lllite',
    model: { base: 'anima', hash: `blake3:${key}`, key, name: key, type: 'controlnet' },
    modelCondInChannels: 3,
    weight,
  });
  const lliteNodes = (graph: BackendGraphContract) =>
    Object.values(graph.nodes).filter((node) => node.type === 'anima_lllite');

  it('leaves control_lllite unset without control layers', () => {
    const { backendGraph } = compile(animaModel, 'txt2img');
    expect(lliteNodes(backendGraph)).toEqual([]);
    expect(getEdge(backendGraph, 'denoise_latents', 'control_lllite')).toBeUndefined();
  });

  it('feeds every adapter through one collector into anima_denoise during inpainting, keeping the mask on denoise', () => {
    const { backendGraph } = compile(animaModel, 'inpaint', {
      controlLayers: [lliteLayer('sketch-layer', 'sketch', 0.8), lliteLayer('depth-layer', 'depth')],
    });

    expect(backendGraph.nodes.denoise_latents?.type).toBe('anima_denoise');
    expect(getEdge(backendGraph, 'denoise_latents', 'control_lllite')?.source).toEqual({
      field: 'collection',
      node_id: 'control_lllite_collector',
    });
    expect(lliteNodes(backendGraph).map((node) => [node.id, node.weight, node.image])).toEqual([
      ['anima_lllite_sketch-layer', 0.8, { image_name: 'sketch-layer-composite.png' }],
      ['anima_lllite_depth-layer', 1, { image_name: 'depth-layer-composite.png' }],
    ]);
    // The canvas mask limits denoising; it is not an LLLite inpaint mask, which control layers never send.
    expect(getEdge(backendGraph, 'denoise_latents', 'denoise_mask')).toBeDefined();
    expect(lliteNodes(backendGraph).every((node) => !('mask' in node))).toBe(true);
    expect(backendGraph.edges.some((edge) => edge.destination.node_id.startsWith('anima_lllite_'))).toBe(false);
  });

  const inpaintAdapter: ComponentModelConfig = {
    base: 'anima',
    cond_in_channels: 4,
    key: 'anima-inpaint',
    name: 'Anima LLLite Inpainting',
    type: 'controlnet',
  };
  const withAdapter = (weight = 1): Partial<GenerateSettings> => ({
    animaInpaintAdapterModel: inpaintAdapter,
    animaInpaintAdapterWeight: weight,
  });
  const edgesInto = (graph: BackendGraphContract, nodeId: string) =>
    graph.edges
      .filter((edge) => edge.destination.node_id === nodeId)
      .map((edge) => [edge.destination.field, edge.source]);

  it('inpaint: the adapter reads the raster image and the inverted mask, through the collector (legacy addInpaint.ts)', () => {
    const { backendGraph } = compile(animaModel, 'inpaint', { settings: withAdapter(0.8) });

    // The canvas mask is white = keep; img_lerp 255 → 0 makes it white = inpaint for the adapter.
    expect(backendGraph.nodes.invert_anima_inpaint_adapter_mask).toEqual({
      id: 'invert_anima_inpaint_adapter_mask',
      image: { image_name: 'canvas-mask.png' },
      is_intermediate: true,
      max: 0,
      min: 255,
      type: 'img_lerp',
      use_cache: true,
    });
    expect(backendGraph.nodes.anima_inpaint_adapter).toEqual({
      control_model: inpaintAdapter,
      id: 'anima_inpaint_adapter',
      image: { image_name: 'canvas-composite.png' },
      is_intermediate: true,
      type: 'anima_lllite',
      use_cache: true,
      weight: 0.8,
    });
    expect(edgesInto(backendGraph, 'anima_inpaint_adapter')).toEqual([
      ['mask', { field: 'image', node_id: 'invert_anima_inpaint_adapter_mask' }],
    ]);
    expect(getEdge(backendGraph, 'control_lllite_collector', 'item')?.source).toEqual({
      field: 'control',
      node_id: 'anima_inpaint_adapter',
    });
    expect(getEdge(backendGraph, 'denoise_latents', 'control_lllite')?.source).toEqual({
      field: 'collection',
      node_id: 'control_lllite_collector',
    });
    // The denoise mask keeps limiting the denoise; the adapter is extra conditioning.
    expect(getEdge(backendGraph, 'denoise_latents', 'denoise_mask')?.source.node_id).toBe('create_gradient_mask');
    const metadata = getNodeByType(backendGraph, 'core_metadata');
    expect(metadata?.anima_inpaint_adapter).toEqual(inpaintAdapter);
    expect(metadata?.anima_inpaint_adapter_weight).toBe(0.8);
  });

  it('inpaint with Scale Before Processing: the adapter reads the resized image and mask', () => {
    const { backendGraph } = compile(animaModel, 'inpaint', {
      scaling: { height: 1024, method: 'manual', width: 1024 },
      settings: withAdapter(),
    });

    expect(backendGraph.nodes.invert_anima_inpaint_adapter_mask?.image).toBeUndefined();
    expect(backendGraph.nodes.anima_inpaint_adapter?.image).toBeUndefined();
    expect(getEdge(backendGraph, 'invert_anima_inpaint_adapter_mask', 'image')?.source.node_id).toBe(
      'canvas_resize_mask_to_processing'
    );
    expect(getEdge(backendGraph, 'anima_inpaint_adapter', 'image')?.source.node_id).toBe(
      'canvas_resize_initial_to_processing'
    );
  });

  it('shares the collector with control layers, each adapter once', () => {
    const { backendGraph } = compile(animaModel, 'inpaint', {
      controlLayers: [lliteLayer('sketch-layer', 'sketch')],
      settings: withAdapter(),
    });

    expect(
      Object.values(backendGraph.nodes)
        .filter((node) => node.type === 'collect')
        .map((node) => node.id)
    ).toEqual(['pos_cond_collect', 'neg_cond_collect', 'control_lllite_collector']);
    expect(
      backendGraph.edges
        .filter((edge) => edge.destination.node_id === 'control_lllite_collector')
        .map((edge) => edge.source.node_id)
    ).toEqual(['anima_inpaint_adapter', 'anima_lllite_sketch-layer']);
    expect(backendGraph.edges.filter((edge) => edge.destination.field === 'control_lllite')).toHaveLength(1);
  });

  it('outpaint: the adapter reads the infilled image and the alpha-derived or combined mask (legacy addOutpaint.ts)', () => {
    const alphaOnly = compile(animaModel, 'outpaint', { maskImageName: null, settings: withAdapter() }).backendGraph;
    expect(getEdge(alphaOnly, 'anima_inpaint_adapter', 'image')?.source.node_id).toBe('infill');
    expect(getEdge(alphaOnly, 'invert_anima_inpaint_adapter_mask', 'image')?.source.node_id).toBe(
      'image_alpha_to_mask'
    );
    expect(alphaOnly.nodes.anima_inpaint_adapter?.image).toBeUndefined();
    expect(getNodeByType(alphaOnly, 'core_metadata')?.anima_inpaint_adapter_weight).toBe(1);

    const combined = compile(animaModel, 'outpaint', {
      maskImageName: 'canvas-mask.png',
      settings: withAdapter(),
    }).backendGraph;
    expect(getEdge(combined, 'invert_anima_inpaint_adapter_mask', 'image')?.source.node_id).toBe('mask_combine');
    expect(getEdge(combined, 'anima_inpaint_adapter', 'mask')?.source.node_id).toBe(
      'invert_anima_inpaint_adapter_mask'
    );
  });

  it.each(['inpaint', 'outpaint'] as const)(
    'adds no adapter nodes, edges or metadata to the %s graph without one',
    (mode) => {
      const { backendGraph } = compile(animaModel, mode, { settings: { animaInpaintAdapterModel: null } });

      expect(backendGraph.nodes.anima_inpaint_adapter).toBeUndefined();
      expect(backendGraph.nodes.invert_anima_inpaint_adapter_mask).toBeUndefined();
      expect(backendGraph.nodes.control_lllite_collector).toBeUndefined();
      expect(getEdge(backendGraph, 'denoise_latents', 'control_lllite')).toBeUndefined();
      expect(getNodeByType(backendGraph, 'core_metadata')).not.toHaveProperty('anima_inpaint_adapter');
      expect(getNodeByType(backendGraph, 'core_metadata')).not.toHaveProperty('anima_inpaint_adapter_weight');
    }
  );

  it('outpaint with Scale Before Processing: the adapter reads the resized combined mask (legacy addOutpaint.ts:182)', () => {
    const { backendGraph } = compile(animaModel, 'outpaint', {
      maskImageName: 'canvas-mask.png',
      scaling: { height: 1024, method: 'manual', width: 1024 },
      settings: withAdapter(),
    });

    expect(getEdge(backendGraph, 'canvas_resize_outpaint_mask_to_processing', 'image')?.source.node_id).toBe(
      'mask_combine'
    );
    expect(getEdge(backendGraph, 'invert_anima_inpaint_adapter_mask', 'image')?.source.node_id).toBe(
      'canvas_resize_outpaint_mask_to_processing'
    );
    expect(getEdge(backendGraph, 'anima_inpaint_adapter', 'image')?.source.node_id).toBe('infill');
    expect(backendGraph.nodes.invert_anima_inpaint_adapter_mask?.image).toBeUndefined();
  });

  it('inpaint with a noise mask: the adapter reads the image before noise is added (legacy addInpaint.ts:223/316)', () => {
    const unscaled = compile(animaModel, 'inpaint', {
      noiseMaskImageName: 'noise-mask.png',
      settings: withAdapter(),
    }).backendGraph;
    expect(getEdge(unscaled, 'canvas_i2l', 'image')?.source.node_id).toBe('add_inpaint_noise');
    expect(unscaled.nodes.anima_inpaint_adapter?.image).toEqual({ image_name: 'canvas-composite.png' });
    expect(getEdge(unscaled, 'anima_inpaint_adapter', 'image')).toBeUndefined();

    const scaled = compile(animaModel, 'inpaint', {
      noiseMaskImageName: 'noise-mask.png',
      scaling: { height: 1024, method: 'manual', width: 1024 },
      settings: withAdapter(),
    }).backendGraph;
    expect(getEdge(scaled, 'add_inpaint_noise', 'image')?.source.node_id).toBe('canvas_resize_initial_to_processing');
    expect(getEdge(scaled, 'anima_inpaint_adapter', 'image')?.source.node_id).toBe(
      'canvas_resize_initial_to_processing'
    );
  });

  it('ignores a picked adapter outside inpaint/outpaint and outside Anima', () => {
    for (const mode of ['txt2img', 'img2img'] as const) {
      const { backendGraph } = compile(animaModel, mode, { settings: withAdapter() });
      expect(backendGraph.nodes.anima_inpaint_adapter).toBeUndefined();
      expect(getNodeByType(backendGraph, 'core_metadata')?.anima_inpaint_adapter).toBeUndefined();
    }
    const { backendGraph } = compile(sdxlModel, 'inpaint', { settings: withAdapter() });
    expect(backendGraph.nodes.anima_inpaint_adapter).toBeUndefined();
    expect(getNodeByType(backendGraph, 'anima_lllite')).toBeUndefined();
  });

  it('rejects a control (3-channel) adapter and an out-of-range weight', () => {
    expect(() =>
      compile(animaModel, 'inpaint', {
        settings: { animaInpaintAdapterModel: { ...inpaintAdapter, cond_in_channels: 3 } },
      })
    ).toThrow('The Anima inpaint adapter must be a 4-channel Anima ControlNet-LLLite model.');
    expect(() => compile(animaModel, 'outpaint', { settings: withAdapter(10.5) })).toThrow(
      'Anima inpaint adapter weight must be between -10 and 10.'
    );
    // Installs that predate `cond_in_channels` are inpaint adapters (legacy isAnimaInpaintControlNetModelConfig).
    expect(() =>
      compile(animaModel, 'inpaint', {
        settings: { animaInpaintAdapterModel: { ...inpaintAdapter, cond_in_channels: null } },
      })
    ).not.toThrow();
  });

  it('writes the graphs the backend validates against its invocations', async () => {
    // Ids minted from the clock and Math.random are pinned so the fixture changes only when the graph does.
    const clock = vi.spyOn(Date, 'now').mockReturnValue(0);
    const random = vi.spyOn(Math, 'random').mockReturnValue(0.5);
    // Installed models always carry a hash, which the backend's model identifier requires.
    const hashed = <T extends object>(config: T, hash: string): T => ({ ...config, hash });
    const settings = {
      qwen3EncoderModel: hashed(qwen3Encoder, 'blake3:qwen3'),
      vae: hashed(qwenImageVae, 'blake3:vae'),
    };
    let threeAdapters: BackendGraphContract;
    let inpaintAdapterGraph: BackendGraphContract;
    let outpaintAdapterGraph: BackendGraphContract;
    try {
      // Inpainting exercises the whole denoise wiring the adapters join; one collector serves any adapter count.
      threeAdapters = compile(hashed(animaModel, 'blake3:anima'), 'inpaint', {
        controlLayers: [
          lliteLayer('sketch-layer', 'sketch', 0.8),
          lliteLayer('depth-layer', 'depth', -1),
          lliteLayer('pose-layer', 'pose', 2),
        ],
        settings,
      }).backendGraph;
      // The inpaint adapter alone: it reads the composite and the uploaded mask as fixed image fields.
      inpaintAdapterGraph = compile(hashed(animaModel, 'blake3:anima'), 'inpaint', {
        settings: {
          ...settings,
          animaInpaintAdapterModel: hashed(inpaintAdapter, 'blake3:inpaint'),
          animaInpaintAdapterWeight: 0.75,
        },
      }).backendGraph;
      // The inpaint adapter beside a control layer: the masked adapter and the unmasked one share the collector.
      outpaintAdapterGraph = compile(hashed(animaModel, 'blake3:anima'), 'outpaint', {
        controlLayers: [lliteLayer('sketch-layer', 'sketch', 0.8)],
        maskImageName: 'canvas-mask.png',
        settings: {
          ...settings,
          animaInpaintAdapterModel: hashed(inpaintAdapter, 'blake3:inpaint'),
          animaInpaintAdapterWeight: 0.75,
        },
      }).backendGraph;
    } finally {
      clock.mockRestore();
      random.mockRestore();
    }
    const fixture = {
      _comment:
        'Generated by compileCanvasGraph.test.ts; regenerate with `vitest -u`. Consumed by tests/app/invocations/test_frontend_anima_lllite_graphs.py, which validates the graphs against the backend invocations.',
      inpaint_inpaint_adapter: { ...inpaintAdapterGraph, id: 'anima_lllite_canvas_inpaint_adapter' },
      inpaint_three_adapters: { ...threeAdapters, id: 'anima_lllite_canvas' },
      outpaint_inpaint_adapter_and_control_layer: {
        ...outpaintAdapterGraph,
        id: 'anima_lllite_canvas_outpaint_adapter',
      },
    };
    await expect(`${JSON.stringify(fixture, null, 2)}\n`).toMatchFileSnapshot(
      '../__snapshots__/animaLLLiteCanvasGraphs.json'
    );
  });
});

describe('compileCanvasGraph — PiD decode', () => {
  const gemma2Encoder: ComponentModelConfig = { base: 'any', key: 'gemma2', name: 'Gemma 2', type: 'gemma2_encoder' };
  const pidSettings = (model: MainModelConfig, pidMode: 'fit' | 'native'): Partial<GenerateSettings> => ({
    gemma2EncoderModel: gemma2Encoder,
    pidDecoderModel: {
      base: getPidDecoderBaseForMainBase(model.base) as string,
      key: `pid-${model.base}`,
      name: `PiD ${model.base}`,
      type: 'pid_decoder',
    },
    pidMode,
    pidSteps: 3,
  });
  const pidDecodeNode = (graph: BackendGraphContract) =>
    Object.values(graph.nodes).find((node) => isPidDecodeNodeType(node.type));
  const edgeSources = (graph: BackendGraphContract, targetNodeId: string) =>
    Object.fromEntries(
      graph.edges
        .filter((edge) => edge.destination.node_id === targetNodeId)
        .map((edge) => [edge.destination.field, [graph.nodes[edge.source.node_id]?.type, edge.source.field]])
    );
  const PID_BASES = [sdxlModel, sd3Model, fluxModel, flux2Model, qwenImageModel, zImageModel];

  it('txt2img fit: the PiD chain replaces the VAE decode and a resize back to the bbox owns canvas_output', () => {
    const { backendGraph } = compile(sdxlModel, 'txt2img', { settings: pidSettings(sdxlModel, 'fit') });

    expect(getNodeByType(backendGraph, 'l2i')).toBeUndefined();
    const decode = pidDecodeNode(backendGraph);
    expect(decode?.type).toBe('sdxl_pid_decode');
    expect(decode?.num_inference_steps).toBe(3);
    expect(decode?.is_intermediate).toBe(true);
    expect(edgeSources(backendGraph, decode!.id)).toEqual({
      gemma2_encoder: ['gemma2_encoder_loader', 'gemma2_encoder'],
      latents: ['denoise_latents', 'latents'],
      pid_decoder: ['pid_decoder_loader', 'pid_decoder'],
      prompt: ['string', 'value'],
      seed: ['integer', 'value'],
      vae: ['sdxl_model_loader', 'vae'],
    });
    expect(getNodeByType(backendGraph, 'gemma2_encoder_loader')?.gemma2_model).toEqual(gemma2Encoder);
    // Fit generates at the bbox; the 4x decode is scaled back to it.
    expect(backendGraph.nodes.noise?.width).toBe(bbox.width);
    expect(backendGraph.nodes.noise?.height).toBe(bbox.height);
    expect(backendGraph.nodes.canvas_output?.type).toBe('img_resize');
    expect(backendGraph.nodes.canvas_output?.width).toBe(bbox.width);
    expect(backendGraph.nodes.canvas_output?.height).toBe(bbox.height);
    expect(getEdge(backendGraph, 'canvas_output', 'image')?.source.node_id).toBe(decode?.id);
    const metadata = getNodeByType(backendGraph, 'core_metadata');
    expect(metadata?.pid_mode).toBe('fit');
    expect(metadata?.pid_steps).toBe(3);
    expect(metadata?.width).toBe(bbox.width);
    expect(metadata?.height).toBe(bbox.height);
  });

  it('txt2img fit with an off-grid bbox: resizes the 4x decode once, straight to the bbox', () => {
    // SDXL snaps 1000x700 to 1000x704 for denoising; legacy resizes the decode once to the final size.
    const { backendGraph } = compile(sdxlModel, 'txt2img', {
      bbox: { height: 700, width: 1000, x: 0, y: 0 },
      settings: pidSettings(sdxlModel, 'fit'),
    });

    expect(backendGraph.nodes.noise).toMatchObject({ height: 704, width: 1000 });
    const resizes = Object.values(backendGraph.nodes).filter((node) => node.type === 'img_resize');
    expect(resizes).toHaveLength(1);
    expect(backendGraph.nodes.canvas_output).toMatchObject({ height: 700, type: 'img_resize', width: 1000 });
    expect(getEdge(backendGraph, 'canvas_output', 'image')?.source.node_id).toBe(pidDecodeNode(backendGraph)?.id);
    expect(backendGraph.nodes.canvas_processing_output).toBeUndefined();
    const metadata = getNodeByType(backendGraph, 'core_metadata');
    expect(metadata).toMatchObject({ height: 700, width: 1000 });
    expect(getEdge(backendGraph, 'canvas_output', 'metadata')?.source.node_id).toBe(metadata?.id);
  });

  it('txt2img native: denoises at a quarter of the bbox and the PiD decode owns canvas_output unresized', () => {
    const { backendGraph } = compile(fluxModel, 'txt2img', { settings: pidSettings(fluxModel, 'native') });

    expect(backendGraph.nodes.denoise_latents?.width).toBe(bbox.width / 4);
    expect(backendGraph.nodes.denoise_latents?.height).toBe(bbox.height / 4);
    expect(backendGraph.nodes.canvas_output?.type).toBe('flux_pid_decode');
    expect(backendGraph.nodes.canvas_output?.is_intermediate).toBe(true);
    expect(getNodeByType(backendGraph, 'img_resize')).toBeUndefined();
    expect(getNodeByType(backendGraph, 'core_metadata')?.pid_mode).toBe('native');
    expect(getNodeByType(backendGraph, 'core_metadata')?.width).toBe(bbox.width);
  });

  it('img2img fit: encodes the composite at the bbox with the VAE the PiD decode reads', () => {
    const { backendGraph } = compile(sdxlModel, 'img2img', { settings: pidSettings(sdxlModel, 'fit'), strength: 0.5 });

    const encode = backendGraph.nodes.canvas_i2l;
    expect(encode?.type).toBe('i2l');
    expect(encode?.image).toEqual({ image_name: 'canvas-composite.png' });
    expect(getEdge(backendGraph, 'canvas_i2l', 'vae')?.source).toEqual({ field: 'vae', node_id: 'model_loader' });
    expect(getEdge(backendGraph, pidDecodeNode(backendGraph)!.id, 'vae')?.source.node_id).toBe('model_loader');
    expect(getEdge(backendGraph, 'denoise_latents', 'latents')?.source.node_id).toBe('canvas_i2l');
    expect(backendGraph.nodes.denoise_latents?.denoising_start).toBe(0.5);
    expect(backendGraph.nodes.canvas_output?.type).toBe('img_resize');
    expect(getNodeByType(backendGraph, 'core_metadata')?.generation_mode).toBe('sdxl_img2img');
    expect(getNodeByType(backendGraph, 'core_metadata')?.strength).toBe(0.5);
  });

  it('img2img native: downscales the composite to the denoise size before encoding and decodes straight up to the bbox (legacy addPidImageToImageNative)', () => {
    const { backendGraph } = compile(fluxModel, 'img2img', { settings: pidSettings(fluxModel, 'native') });

    const resizeIn = backendGraph.nodes.canvas_resize_initial_to_processing;
    expect(resizeIn?.type).toBe('img_resize');
    expect(resizeIn?.image).toEqual({ image_name: 'canvas-composite.png' });
    expect(resizeIn?.width).toBe(bbox.width / 4);
    expect(resizeIn?.height).toBe(bbox.height / 4);
    expect(getEdge(backendGraph, 'canvas_i2l', 'image')?.source.node_id).toBe('canvas_resize_initial_to_processing');
    expect(backendGraph.nodes.canvas_i2l?.type).toBe('flux_vae_encode');
    // flux_pid_decode takes no VAE; the encode reads the model loader's, as the FLUX builder decodes with it.
    expect(getEdge(backendGraph, 'canvas_i2l', 'vae')?.source).toEqual({ field: 'vae', node_id: 'model_loader' });
    expect(backendGraph.nodes.denoise_latents?.width).toBe(bbox.width / 4);
    expect(backendGraph.nodes.canvas_output?.type).toBe('flux_pid_decode');
    expect(backendGraph.nodes.canvas_resize_generated_to_bbox).toBeUndefined();
    expect(backendGraph.nodes.canvas_processing_output).toBeUndefined();
  });

  it('img2img native (SDXL): downscales the composite to the quarter size the noise node denoises at', () => {
    const { backendGraph } = compile(sdxlModel, 'img2img', { settings: pidSettings(sdxlModel, 'native') });

    expect(backendGraph.nodes.noise).toMatchObject({ height: bbox.height / 4, width: bbox.width / 4 });
    expect(backendGraph.nodes.canvas_resize_initial_to_processing).toMatchObject({
      height: bbox.height / 4,
      image: { image_name: 'canvas-composite.png' },
      type: 'img_resize',
      width: bbox.width / 4,
    });
    expect(getEdge(backendGraph, 'canvas_i2l', 'image')?.source.node_id).toBe('canvas_resize_initial_to_processing');
    expect(backendGraph.nodes.canvas_output?.type).toBe('sdxl_pid_decode');
    expect(backendGraph.nodes.canvas_processing_output).toBeUndefined();
  });

  it.each(PID_BASES)('resolves the encode VAE for a $base native img2img', (model) => {
    const { backendGraph } = compile(model, 'img2img', {
      settings: { ...COMPONENT_OVERRIDES[model.base], ...pidSettings(model, 'native') },
    });

    const vaeSource = getEdge(backendGraph, 'canvas_i2l', 'vae')?.source;
    expect(vaeSource?.field).toBe('vae');
    expect(backendGraph.nodes[vaeSource!.node_id]?.type).toMatch(/model_loader$/);
    expect(backendGraph.nodes.canvas_output?.type).toBe(pidDecodeNode(backendGraph)?.type);
  });

  it('rejects inpaint and outpaint with the legacy message (buildSDXLGraph.ts:197)', () => {
    for (const mode of ['inpaint', 'outpaint'] as const) {
      expect(() => compile(sdxlModel, mode, { settings: pidSettings(sdxlModel, 'fit') })).toThrow(
        'PiD decode currently supports Text to Image and Image to Image only. Disable PiD for Inpaint/Outpaint.'
      );
    }
  });

  it('rejects Scale Before Processing when it changes the processing size (buildSDXLGraph.ts:203)', () => {
    expect(() =>
      compile(sdxlModel, 'txt2img', {
        scaling: { height: null, method: 'auto', width: null },
        settings: pidSettings(sdxlModel, 'fit'),
      })
    ).toThrow('Turn off Scale Before Processing (set it to None) to use PiD decode.');
    expect(() =>
      compile(sdxlModel, 'img2img', {
        scaling: { height: 512, method: 'manual', width: 512 },
        settings: pidSettings(sdxlModel, 'native'),
      })
    ).toThrow('Turn off Scale Before Processing (set it to None) to use PiD decode.');
    // A manual size equal to the bbox leaves the generation size alone, which legacy accepted.
    expect(() =>
      compile(sdxlModel, 'img2img', {
        scaling: { height: bbox.height, method: 'manual', width: bbox.width },
        settings: pidSettings(sdxlModel, 'native'),
      })
    ).not.toThrow();
  });
});

describe('compileCanvasGraph — regional guidance', () => {
  const region = (id: string, overrides: Partial<RegionalGuidanceInput> = {}) => ({
    autoNegative: false,
    id,
    maskImageName: `${id}-mask.png`,
    negativePrompt: null,
    positivePrompt: 'a cat',
    referenceImages: [],
    ...overrides,
  });

  it('grafts SD1 regional conditioning into the pos/neg collectors', () => {
    const { backendGraph } = compile(sd1Model, 'txt2img', {
      regionalGuidance: [region('r1', { autoNegative: true })],
    });
    expect(backendGraph.nodes.rg_mask_to_tensor_r1).toBeDefined();
    expect(backendGraph.nodes.rg_pos_cond_r1?.prompt).toBe('a cat');
    // autoNegative inverted-mask node feeding the negative collector.
    expect(backendGraph.nodes.rg_invert_mask_r1).toBeDefined();
    expect(backendGraph.nodes.rg_pos_cond_inverted_r1).toBeDefined();
  });

  it('grafts SD2 regional conditioning through compel like SD1', () => {
    const { backendGraph } = compile(sd2Model, 'txt2img', {
      regionalGuidance: [region('r1', { negativePrompt: 'blurry' })],
    });
    expect(backendGraph.nodes.rg_pos_cond_r1).toMatchObject({ prompt: 'a cat', type: 'compel' });
    expect(backendGraph.nodes.rg_neg_cond_r1).toMatchObject({ prompt: 'blurry', type: 'compel' });
    expect(getEdge(backendGraph, 'rg_pos_cond_r1', 'clip')?.source.node_id).toBe(
      getEdge(backendGraph, 'pos_cond', 'clip')?.source.node_id
    );
  });

  it('is a no-op for a base with no regional path (sd-3)', () => {
    const { backendGraph } = compile(sd3Model, 'txt2img', { regionalGuidance: [region('r1')] });
    expect(backendGraph.nodes.rg_mask_to_tensor_r1).toBeUndefined();
  });

  it.each([
    ['Z-Image', zImageModel, 'z_image_text_encoder'],
    ['Anima', animaModel, 'anima_text_encoder'],
  ] as const)('grafts %s regions through the shared Qwen3 encoder', (_label, model, encoderType) => {
    const { backendGraph } = compile(model, 'txt2img', {
      regionalGuidance: [region('r1'), region('r2', { positivePrompt: 'a dog' })],
      settings: { cfgScale: 4 },
    });

    expect(backendGraph.nodes.rg_pos_cond_r1).toMatchObject({ prompt: 'a cat', type: encoderType });
    expect(backendGraph.nodes.rg_pos_cond_r2).toMatchObject({ prompt: 'a dog', type: encoderType });
    expect(getEdge(backendGraph, 'rg_pos_cond_r1', 'qwen3_encoder')?.source.node_id).toBe('model_loader');
    expect(getEdge(backendGraph, 'rg_pos_cond_r1', 'mask')?.source.node_id).toBe('rg_mask_to_tensor_r1');
    const collected = backendGraph.edges
      .filter((edge) => edge.destination.node_id === 'pos_cond_collect')
      .map((edge) => edge.source.node_id);
    expect(collected).toEqual(['pos_cond', 'rg_pos_cond_r1', 'rg_pos_cond_r2']);
    expect(getEdge(backendGraph, 'neg_cond_collect', 'item')?.source.node_id).toBe('neg_cond');
  });

  it('coexists with control layers in one graph', () => {
    const control: ControlLayerGraphInput = {
      beginEndStepPct: [0, 0.75],
      controlMode: 'balanced',
      id: 'c1',
      imageName: 'control.png',
      kind: 'controlnet',
      model: { base: 'sd-1', key: 'cn', name: 'CN', type: 'controlnet' },
      weight: 1,
    };
    const { backendGraph } = compile(sd1Model, 'txt2img', {
      controlLayers: [control],
      regionalGuidance: [region('r1')],
    });
    expect(backendGraph.nodes.control_net_c1).toBeDefined();
    expect(backendGraph.nodes.rg_pos_cond_r1).toBeDefined();
  });

  it('grafts positive-only regional conditioning into a FLUX.2 graph', () => {
    const { backendGraph } = compile(flux2Model, 'txt2img', { regionalGuidance: [region('r1')] });

    expect(backendGraph.nodes.rg_pos_cond_r1).toMatchObject({ prompt: 'a cat', type: 'flux2_klein_text_encoder' });
    expect(getEdge(backendGraph, 'rg_pos_cond_r1', 'qwen3_encoder')?.source.node_id).toBe('model_loader');
    expect(getEdge(backendGraph, 'pos_cond_collect', 'item')?.source.node_id).toBe('pos_cond');
  });

  it('grafts FLUX.2 [dev] regions through Mistral conditioning', () => {
    const { backendGraph } = compile(flux2DevModel, 'txt2img', { regionalGuidance: [region('r1')] });

    expect(backendGraph.nodes.rg_pos_cond_r1).toMatchObject({ prompt: 'a cat', type: 'flux2_dev_text_encoder' });
    expect(getEdge(backendGraph, 'rg_pos_cond_r1', 'mistral_encoder')?.source.node_id).toBe('model_loader');
    expect(getEdge(backendGraph, 'rg_pos_cond_r1', 'max_seq_len')?.source.node_id).toBe('model_loader');
  });

  it('grafts Krea-2 regions through uniquely named conditioning enhancers into the shared collector', () => {
    const { backendGraph } = compile(krea2Model, 'txt2img', {
      regionalGuidance: [region('r1'), region('r2', { positivePrompt: 'a dog' })],
      settings: {
        krea2RebalanceEnabled: true,
        krea2SeedVarianceEnabled: true,
        krea2SeedVarianceStrength: 0.2,
      },
    });

    expect(backendGraph.nodes.rg_pos_cond_r1).toMatchObject({ prompt: 'a cat', type: 'krea2_text_encoder' });
    expect(backendGraph.nodes.rg_pos_cond_r2).toMatchObject({ prompt: 'a dog', type: 'krea2_text_encoder' });
    expect(getEdge(backendGraph, 'rg_pos_cond_r1', 'qwen3_vl_encoder')?.source.node_id).toBe('model_loader');
    expect(backendGraph.nodes.rg_krea2_r1_rebalance?.type).toBe('krea2_conditioning_rebalance');
    expect(backendGraph.nodes.rg_krea2_r2_rebalance?.type).toBe('krea2_conditioning_rebalance');
    expect(backendGraph.nodes.rg_krea2_r1_seed_variance?.type).toBe('krea2_seed_variance');
    expect(backendGraph.nodes.rg_krea2_r2_seed_variance?.type).toBe('krea2_seed_variance');
    expect(getEdge(backendGraph, 'rg_krea2_r1_seed_variance', 'variance_seed')?.source.node_id).toBe('seed');
    expect(getEdge(backendGraph, 'denoise_latents', 'positive_conditioning')?.source.node_id).toBe('pos_cond_collect');

    const collectedSources = backendGraph.edges
      .filter((edge) => edge.destination.node_id === 'pos_cond_collect' && edge.destination.field === 'item')
      .map((edge) => edge.source.node_id);
    expect(collectedSources).toEqual(['krea2_seed_variance', 'rg_krea2_r1_seed_variance', 'rg_krea2_r2_seed_variance']);
  });
});

describe('compileCanvasGraph — SDXL Refiner', () => {
  const refinerModel: MainModelConfig = {
    base: 'sdxl-refiner',
    key: 'sdxl-refiner-model',
    name: 'SDXL Refiner',
    type: 'main',
  };
  const refinerSettings: Partial<GenerateSettings> = { refinerModel, refinerStart: 0.8, refinerSteps: 10 };

  it('hands txt2img latents to the refiner before the resize to the bbox', () => {
    const { backendGraph } = compile(sdxlModel, 'txt2img', { settings: refinerSettings });

    expect(backendGraph.nodes.denoise_latents).toMatchObject({ denoising_end: 0.8, denoising_start: 0 });
    expect(backendGraph.nodes.refiner_denoise_latents).toMatchObject({ denoising_end: 1, denoising_start: 0.8 });
    expect(getEdge(backendGraph, 'refiner_denoise_latents', 'latents')?.source.node_id).toBe('denoise_latents');
    expect(getEdge(backendGraph, 'canvas_output', 'latents')?.source.node_id).toBe('refiner_denoise_latents');
  });

  it('stops the img2img base pass at the hand-over and keeps its start below it', () => {
    const { backendGraph } = compile(sdxlModel, 'img2img', { settings: refinerSettings, strength: 0.6 });

    // 1 - 0.6 = 0.4 is below the 0.8 hand-over, so the base pass runs 0.4 → 0.8.
    expect(backendGraph.nodes.denoise_latents?.denoising_start).toBeCloseTo(0.4);
    expect(backendGraph.nodes.denoise_latents?.denoising_end).toBe(0.8);
    expect(backendGraph.nodes.refiner_denoise_latents).toMatchObject({ denoising_end: 1, denoising_start: 0.8 });
    expect(getEdge(backendGraph, 'denoise_latents', 'latents')?.source.node_id).toBe('canvas_i2l');
    expect(getEdge(backendGraph, 'refiner_denoise_latents', 'latents')?.source.node_id).toBe('denoise_latents');
    expect(getNodeByType(backendGraph, 'core_metadata')).toMatchObject({
      generation_mode: 'sdxl_img2img',
      refiner_model: { key: 'sdxl-refiner-model' },
      refiner_start: 0.8,
      strength: 0.6,
    });
  });

  it('caps a low-strength img2img start at the hand-over so the base pass is not inverted', () => {
    const { backendGraph } = compile(sdxlModel, 'img2img', { settings: refinerSettings, strength: 0.1 });

    expect(backendGraph.nodes.denoise_latents).toMatchObject({ denoising_end: 0.8, denoising_start: 0.8 });
  });

  it.each(['inpaint', 'outpaint'] as const)(
    'masks the base pass and constrains the decoded refiner with pasteback for %s',
    (mode) => {
      const { backendGraph } = compile(sdxlModel, mode, { settings: refinerSettings, strength: 0.5 });

      expect(backendGraph.nodes.denoise_latents).toMatchObject({ denoising_end: 0.8, denoising_start: 0.5 });
      expect(getEdge(backendGraph, 'denoise_latents', 'denoise_mask')?.source.node_id).toBe('create_gradient_mask');
      expect(getEdge(backendGraph, 'refiner_denoise_latents', 'denoise_mask')).toBeUndefined();
      expect(backendGraph.nodes.canvas_output).toMatchObject({ type: 'invokeai_img_blend' });
      expect(getEdge(backendGraph, 'canvas_output', 'layer_upper')?.source.node_id).toBe('canvas_l2i');
      expect(getEdge(backendGraph, 'canvas_output', 'mask')?.source.node_id).toBe('expand_mask');
      expect(getEdge(backendGraph, 'expand_mask', 'mask')?.source).toEqual({
        field: 'expanded_mask_area',
        node_id: 'create_gradient_mask',
      });
      expect(getEdge(backendGraph, 'refiner_denoise_latents', 'latents')?.source.node_id).toBe('denoise_latents');
      expect(getEdge(backendGraph, 'canvas_l2i', 'latents')?.source.node_id).toBe('refiner_denoise_latents');
      expect(getNodeByType(backendGraph, 'core_metadata')?.generation_mode).toBe(`sdxl_${mode}`);
    }
  );

  it.each(['txt2img', 'img2img', 'inpaint', 'outpaint'] as const)(
    'never lets the %s graft restore the full base schedule while a refiner is active',
    (mode) => {
      const { backendGraph } = compile(sdxlModel, mode, { settings: refinerSettings, strength: 0.9 });

      // The canvas re-derives the base range from the strength after the refiner stage is grafted.
      expect(backendGraph.nodes.denoise_latents?.denoising_end).toBe(0.8);
      expect(backendGraph.nodes.refiner_denoise_latents?.denoising_start).toBe(0.8);
    }
  );

  it('keeps regional guidance on the base pass', () => {
    const { backendGraph } = compile(sdxlModel, 'txt2img', {
      regionalGuidance: [
        {
          autoNegative: false,
          id: 'r1',
          maskImageName: 'r1-mask.png',
          negativePrompt: null,
          positivePrompt: 'a cat',
          referenceImages: [],
        },
      ],
      settings: refinerSettings,
    });

    const collected = backendGraph.edges
      .filter((edge) => edge.destination.node_id === 'pos_cond_collect')
      .map((edge) => edge.source.node_id);
    expect(collected).toEqual(['pos_cond', 'rg_pos_cond_r1']);
    expect(getEdge(backendGraph, 'denoise_latents', 'positive_conditioning')?.source.node_id).toBe('pos_cond_collect');
    expect(getEdge(backendGraph, 'refiner_denoise_latents', 'positive_conditioning')?.source.node_id).toBe(
      'refiner_pos_cond'
    );
    expect(
      backendGraph.edges
        .filter((edge) => edge.destination.node_id === 'refiner_denoise_latents')
        .map((edge) => edge.destination.field)
    ).toEqual(['unet', 'positive_conditioning', 'negative_conditioning', 'latents']);
  });

  it('attaches control layers to the base pass, not the refiner', () => {
    const { backendGraph } = compile(sdxlModel, 'txt2img', {
      controlLayers: [
        {
          beginEndStepPct: [0, 1],
          controlMode: 'balanced',
          id: 'control-1',
          imageName: 'control.png',
          kind: 'controlnet',
          model: { base: 'sdxl', key: 'sdxl-controlnet', name: 'SDXL ControlNet', type: 'controlnet' },
          weight: 1,
        },
      ],
      settings: refinerSettings,
    });

    expect(getEdge(backendGraph, 'denoise_latents', 'control')).toBeDefined();
    expect(getEdge(backendGraph, 'refiner_denoise_latents', 'control')).toBeUndefined();
  });

  it('refuses PiD on the canvas when a refiner is selected', () => {
    expect(() =>
      compile(sdxlModel, 'img2img', {
        settings: {
          ...refinerSettings,
          gemma2EncoderModel: { base: 'any', key: 'g', name: 'Gemma 2', type: 'gemma2_encoder' },
          pidDecoderModel: { base: 'sdxl', key: 'pid', name: 'PiD SDXL', type: 'pid_decoder' },
          pidMode: 'fit',
        },
      })
    ).toThrow('PiD decode is not compatible with the SDXL Refiner. Turn PiD off or clear the refiner model.');
  });
});

describe('compileCanvasGraph — output filters', () => {
  const metadataTarget = (graph: BackendGraphContract) =>
    graph.edges.find((edge) => edge.destination.field === 'metadata')?.destination.node_id;

  it('filters the resized txt2img output, keeping the resize intermediate and the metadata on the result', () => {
    const { backendGraph } = compile(sd1Model, 'txt2img', {
      // Off the model grid, so the base decode is resized back to the bbox before the filters run.
      bbox: { height: 1000, width: 700, x: 0, y: 0 },
      destination: 'gallery',
      projectSettings: { useNsfwChecker: true, useWatermark: true },
    });

    expect(backendGraph.nodes.canvas_unfiltered_output).toMatchObject({ is_intermediate: true, type: 'img_resize' });
    expect(backendGraph.nodes.canvas_img_nsfw?.type).toBe('img_nsfw');
    expect(backendGraph.nodes.canvas_output).toMatchObject({ is_intermediate: false, type: 'img_watermark' });
    expect(metadataTarget(backendGraph)).toBe('canvas_output');
  });

  it('filters the composited inpaint result rather than the decode behind it', () => {
    const { backendGraph } = compile(sd1Model, 'inpaint', { projectSettings: { useNsfwChecker: true } });

    expect(backendGraph.nodes.canvas_unfiltered_output?.type).toBe('invokeai_img_blend');
    expect(backendGraph.nodes.canvas_output).toMatchObject({ is_intermediate: true, type: 'img_nsfw' });
    expect(
      backendGraph.edges.find(
        (edge) => edge.destination.node_id === 'canvas_output' && edge.destination.field === 'image'
      )?.source.node_id
    ).toBe('canvas_unfiltered_output');
    expect(metadataTarget(backendGraph)).toBe('canvas_output');
  });

  it('runs a lone watermark as the staged outpaint result', () => {
    const { backendGraph } = compile(sd1Model, 'outpaint', {
      destination: 'canvas',
      projectSettings: { useWatermark: true },
    });

    expect(backendGraph.nodes.canvas_unfiltered_output?.type).toBe('invokeai_img_blend');
    expect(backendGraph.nodes.canvas_img_nsfw).toBeUndefined();
    expect(backendGraph.nodes.canvas_output).toMatchObject({ is_intermediate: true, type: 'img_watermark' });
    expect(
      backendGraph.edges.find(
        (edge) => edge.destination.node_id === 'canvas_output' && edge.destination.field === 'image'
      )?.source.node_id
    ).toBe('canvas_unfiltered_output');
    expect(metadataTarget(backendGraph)).toBe('canvas_output');
  });

  it('leaves every mode untouched while both filters are off', () => {
    for (const mode of ['txt2img', 'img2img', 'inpaint', 'outpaint'] as const) {
      const { backendGraph } = compile(sd1Model, mode, {
        projectSettings: { useNsfwChecker: false, useWatermark: false },
      });

      expect(backendGraph.nodes.canvas_unfiltered_output).toBeUndefined();
      expect(getNodeByType(backendGraph, 'img_nsfw')).toBeUndefined();
      expect(getNodeByType(backendGraph, 'img_watermark')).toBeUndefined();
    }
  });
});

describe('compileCanvasGraph — Krea-2 style reference', () => {
  it('encodes the reference at the processing size, not the bbox', () => {
    const { backendGraph } = compile(krea2Model, 'img2img', {
      // 700x1000 is off the 16px grid, so processing rounds it and resizes back.
      bbox: { height: 1000, width: 700, x: 0, y: 0 },
      settings: {
        referenceImages: [
          {
            config: {
              image: { original: { image: { height: 512, image_name: 'style.png', width: 512 } } },
              styleStrength: 1.2,
              type: 'krea2_reference_image',
            },
            id: 'style',
            isEnabled: true,
          },
        ],
      },
      strength: 0.5,
    });
    const styleReference = getNodeByType(backendGraph, 'krea2_style_reference');
    const denoise = backendGraph.nodes.denoise_latents;

    expect(styleReference).toMatchObject({ image: { image_name: 'style.png' }, style_strength: 1.2 });
    expect(styleReference?.width).toBe(denoise?.width);
    expect(styleReference?.height).toBe(denoise?.height);
    expect(denoise?.width).not.toBe(700);
    expect(backendGraph.edges.find((edge) => edge.destination.field === 'style_reference')?.source.node_id).toBe(
      styleReference?.id
    );
  });
});
