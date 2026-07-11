import type {
  ComponentModelConfig,
  GenerateModelConfig,
  GenerateSettings,
  MainModelConfig,
  VaeModelConfig,
} from '@workbench/generation/types';
import type { BackendGraphContract, ProjectSettings } from '@workbench/types';
import type { CanvasCompositingSettings } from '@workbench/widgets/canvas/invoke/canvasCompositing';

import { getDefaultGenerateSettings } from '@workbench/generation/baseGenerationPolicies';
import { DEFAULT_CANVAS_COMPOSITING } from '@workbench/widgets/canvas/invoke/canvasCompositing';
import { describe, expect, it } from 'vitest';

import type { ControlLayerGraphInput } from './addControlLayers';
import type { RegionalGuidanceInput } from './addRegionalGuidance';
import type { CanvasCompileMode, Rect } from './types';

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

const fluxVae: VaeModelConfig = { base: 'flux', key: 'flux-vae', name: 'FLUX VAE', type: 'vae' };
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

const PROJECT_SETTINGS: Pick<ProjectSettings, 'useCpuNoise'> = { useCpuNoise: true };

const settingsFor = (model: GenerateModelConfig, overrides: Partial<GenerateSettings> = {}): GenerateSettings => ({
  ...getDefaultGenerateSettings(model),
  seed: 1,
  shouldRandomizeSeed: false,
  ...overrides,
});

/** Per-base overrides needed to pass component validation. */
const COMPONENT_OVERRIDES: Partial<Record<string, Partial<GenerateSettings>>> = {
  flux: { clipEmbedModel: clipEmbed, t5EncoderModel: t5Encoder, vae: fluxVae },
  anima: { qwen3EncoderModel: qwen3Encoder, vae: qwenImageVae },
};

const bbox: Rect = { height: 1024, width: 768, x: 128, y: 64 };

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
    strength?: number;
    destination?: 'canvas' | 'gallery';
    controlLayers?: readonly ControlLayerGraphInput[];
    regionalGuidance?: readonly RegionalGuidanceInput[];
  } = {}
) =>
  compileCanvasGraph({
    bbox: overrides.bbox ?? bbox,
    compositeImageName:
      'compositeImageName' in overrides ? (overrides.compositeImageName ?? null) : 'canvas-composite.png',
    compositing: overrides.compositing
      ? { ...DEFAULT_CANVAS_COMPOSITING, ...overrides.compositing }
      : DEFAULT_CANVAS_COMPOSITING,
    controlLayers: overrides.controlLayers,
    regionalGuidance: overrides.regionalGuidance,
    destination: overrides.destination ?? 'canvas',
    maskImageName:
      'maskImageName' in overrides ? overrides.maskImageName : mode === 'inpaint' ? 'canvas-mask.png' : null,
    mode,
    model,
    noiseMaskImageName: overrides.noiseMaskImageName ?? null,
    projectSettings: PROJECT_SETTINGS,
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
];

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

        // denoising_start follows the base's strength curve (linear, or the
        // exponent-0.2 optimized curve for sd-3 / flux / flux2).
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

    it('rejects off-grid bbox dimensions', () => {
      expect(() => compile(flux2Model, 'txt2img', { bbox: { height: 888, width: 1024, x: 0, y: 0 } })).toThrow(
        'Generate height must be a multiple of 16.'
      );
    });

    it('surfaces missing-component validation for the bbox-sized settings', () => {
      // FLUX with no component overrides fails the shared generate validation.
      expect(() =>
        compileCanvasGraph({
          bbox,
          compositeImageName: null,
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

// Review fix (Task 38, finding 2): `addControlLayers` has its own isolated unit
// coverage (addControlLayers.test.ts), but nothing previously drove control
// layers through the REAL `compileCanvasGraph` entry point — so a wiring
// regression at the seam between the two (e.g. the wrong denoise node id, or a
// base graph that no longer exposes `denoise_latents`) could pass every
// existing test here while breaking a real canvas invoke.
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

    // The control node feeds the collector, whose own output is what
    // `denoise_latents.control` actually reads.
    expect(backendGraph.nodes.control_net_collector?.type).toBe('collect');
    expect(getEdge(backendGraph, 'control_net_collector', 'item')?.source.node_id).toBe('control_net_control-layer-1');
    expect(getEdge(backendGraph, 'denoise_latents', 'control')?.source.node_id).toBe('control_net_collector');

    // The base img2img graph is untouched by the control graft: prompts, seed,
    // the encode → denoise plumbing, and the composite-back output are all
    // still present and wired exactly as they would be with no control layers.
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

  it('is a no-op for an unsupported base (sd-2)', () => {
    const { backendGraph } = compile(sd2Model, 'txt2img', { regionalGuidance: [region('r1')] });
    expect(backendGraph.nodes.rg_mask_to_tensor_r1).toBeUndefined();
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
});
