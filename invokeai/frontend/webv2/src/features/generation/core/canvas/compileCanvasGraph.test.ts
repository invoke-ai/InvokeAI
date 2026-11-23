import type { BackendGraphContract } from '@features/generation/core/contracts';
import type {
  ComponentModelConfig,
  GenerateModelConfig,
  GenerateSettings,
  MainModelConfig,
} from '@features/generation/core/types';

import { getPidDecoderBaseForMainBase } from '@features/generation/core/pid';
import { isPidDecodeNodeType } from '@features/generation/core/pidGraph';
import { seedArchitectureCapabilities } from '@features/models/core/architectureCapabilities.testing';
import { describe, expect, it, vi } from 'vitest';

import type { ControlLayerGraphInput } from './addControlLayers';
import type { RegionalGuidanceInput } from './addRegionalGuidance';
import type { Rect } from './types';

import { bbox, compile, qwen3Encoder, qwenImageVae } from './compileCanvasGraph.testing';

const sd1Model: MainModelConfig = { base: 'sd-1', key: 'sd1-model', name: 'SD 1.5', type: 'main' };
const sd2Model: MainModelConfig = { base: 'sd-2', key: 'sd2-model', name: 'SD 2', type: 'main' };
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
const flux2DevModel: MainModelConfig = {
  ...flux2Model,
  key: 'flux2-dev-model',
  name: 'FLUX.2 [dev]',
  variant: 'dev',
};
const flux3Model: MainModelConfig = {
  base: 'flux3',
  format: 'diffusers',
  key: 'flux3-model',
  name: 'FLUX.3 Klein',
  type: 'main',
  variant: 'flux3_klein',
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
const mossImageModel: MainModelConfig = {
  base: 'moss-image',
  format: 'diffusers',
  key: 'moss-image-model',
  name: 'MOSS-Image',
  type: 'main',
};
const ideogram4Model: MainModelConfig = {
  base: 'ideogram-4',
  format: 'diffusers',
  key: 'ideogram4-model',
  name: 'Ideogram 4',
  type: 'main',
};
const ernieImageModel: MainModelConfig = {
  base: 'ernie-image',
  format: 'diffusers',
  key: 'ernie-image-model',
  name: 'ERNIE-Image',
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
  optimizedDenoising: boolean;
}

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
    model: flux3Model,
    encodeType: 'flux2_vae_encode',
    outputType: 'flux2_vae_decode',
    denoiseType: 'flux3_denoise',
    txt2imgMode: 'flux3_txt2img',
    img2imgMode: 'flux3_img2img',
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
  {
    model: mossImageModel,
    encodeType: 'flux2_vae_encode',
    outputType: 'flux2_vae_decode',
    denoiseType: 'moss_image_denoise',
    txt2imgMode: 'moss_image_txt2img',
    img2imgMode: 'moss_image_img2img',
    optimizedDenoising: true,
  },
];

seedArchitectureCapabilities();

describe('compileCanvasGraph', () => {
  describe('txt2img per base', () => {
    it.each(BASE_CASES)(
      'builds a $model.base txt2img graph sized to the bbox',
      ({ model, outputType, txt2imgMode }) => {
        const { backendGraph, graph, mode } = compile(model, 'txt2img');

        expect(getNodeByType(backendGraph, 'i2l')).toBeUndefined();
        expect(getNodeByType(backendGraph, `${model.base}_i2l`)).toBeUndefined();
        expect(getNodeByType(backendGraph, `${model.base}_vae_encode`)).toBeUndefined();

        expect(backendGraph.nodes.canvas_output?.type).toBe(outputType);
        expect(backendGraph.nodes.canvas_output?.is_intermediate).toBe(true);

        expect(backendGraph.nodes.positive_prompt).toBeDefined();
        expect(backendGraph.nodes.seed).toBeDefined();

        expect(getNodeByType(backendGraph, 'core_metadata')?.width).toBe(bbox.width);
        expect(getNodeByType(backendGraph, 'core_metadata')?.height).toBe(bbox.height);
        expect(getNodeByType(backendGraph, 'core_metadata')?.generation_mode).toBe(txt2imgMode);

        expect(mode).toBe('txt2img');
        expect(graph.label).toBe(`${model.name} txt2img`);
      }
    );

    it('produces a durable (non-intermediate) output for a Gallery destination', () => {
      expect(compile(sd1Model, 'txt2img').backendGraph.nodes.canvas_output?.is_intermediate).toBe(true);
      expect(
        compile(sd1Model, 'txt2img', { destination: 'gallery' }).backendGraph.nodes.canvas_output?.is_intermediate
      ).toBe(false);
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

        expect(getEdge(backendGraph, encode!.id, 'vae')).toBeDefined();
        expect(getEdge(backendGraph, 'denoise_latents', 'latents')?.source.node_id).toBe(encode?.id);

        expect(backendGraph.nodes.denoise_latents?.type).toBe(denoiseType);
        expect(backendGraph.nodes.denoise_latents?.denoising_start).toBeCloseTo(
          expectedDenoisingStart(0.6, optimizedDenoising),
          10
        );
        expect(backendGraph.nodes.denoise_latents?.denoising_end).toBe(1);

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

    it('reflects strength in denoising_start', () => {
      const { backendGraph } = compile(sd1Model, 'img2img', { strength: 0.25 });

      expect(backendGraph.nodes.denoise_latents?.denoising_start).toBeCloseTo(0.75, 10);
    });

    it.each([ernieImageModel, ideogram4Model])('rejects img2img for $base, which has no latent space', (model) => {
      expect(() => compile(model, 'img2img')).toThrow(`Canvas generation is not supported for ${model.name}.`);
    });
  });

  describe('external image generators', () => {
    const editModel: GenerateModelConfig = {
      ...externalModel,
      capabilities: { modes: ['txt2img', 'img2img', 'inpaint'], supports_seed: true },
    };
    const smallBbox: Rect = { height: 512, width: 640, x: 10, y: 20 };
    const gptImage: GenerateModelConfig = {
      ...editModel,
      capabilities: {
        allowed_aspect_ratios: ['1:1', '3:2', '2:3'],
        aspect_ratio_sizes: {
          '1:1': { height: 1024, width: 1024 },
          '3:2': { height: 1024, width: 1536 },
          '2:3': { height: 1536, width: 1024 },
        },
        modes: ['txt2img', 'img2img'],
        supports_seed: true,
      },
      provider_model_id: 'gpt-image-1',
    };

    it('passes the Canvas composite and up to 15 additional references to a MiniMax edit', () => {
      const minimax: GenerateModelConfig = {
        ...editModel,
        capabilities: {
          max_image_size: { height: 4096, width: 4096 },
          max_reference_images: 16,
          modes: ['txt2img', 'img2img'],
          supports_reference_images: true,
        },
        key: 'minimax-h3-image',
        name: 'MiniMax H3-Image',
        provider_id: 'minimax',
        provider_model_id: 'MiniMax-H3-Image',
      };
      const references = Array.from({ length: 16 }, (_, index) => ({
        config: {
          image: { original: { image: { height: 512, image_name: `ref-${index}.png`, width: 512 } } },
          type: 'external_reference_image' as const,
        },
        id: `reference-${index}`,
        isEnabled: true,
      }));
      const { backendGraph } = compile(minimax, 'img2img', {
        settings: { referenceImages: references.slice(0, 15) },
      });

      expect(backendGraph.nodes.canvas_processing_output).toMatchObject({
        height: bbox.height,
        init_image: { image_name: 'canvas-composite.png' },
        mode: 'img2img',
        reference_images: references.slice(0, 15).map(({ config }) => ({
          image_name: config.image.original.image.image_name,
        })),
        type: 'minimax_image_generation',
        width: bbox.width,
      });
      expect(backendGraph.nodes.canvas_processing_output).not.toHaveProperty('mask_image');
      expect(() => compile(minimax, 'img2img', { settings: { referenceImages: references } })).toThrow(
        'supports at most 15 reference images'
      );
      const textGraph = compile(minimax, 'txt2img', { settings: { referenceImages: references } }).backendGraph;
      expect(textGraph.nodes.canvas_processing_output?.reference_images).toHaveLength(16);
      expect(textGraph.nodes.canvas_processing_output).not.toHaveProperty('init_image');
    });

    it('writes hosted canvas graphs for backend invocation contract validation', async () => {
      const clock = vi.spyOn(Date, 'now').mockReturnValue(0);
      const random = vi.spyOn(Math, 'random').mockReturnValue(0.5);
      const graphs: Record<string, BackendGraphContract> = {};
      try {
        for (const mode of ['txt2img', 'img2img'] as const) {
          graphs[mode] = compile({ ...gptImage, hash: 'blake3:external' }, mode, { bbox: smallBbox }).backendGraph;
        }
      } finally {
        clock.mockRestore();
        random.mockRestore();
      }
      const fixture = {
        _comment:
          'Generated by compileCanvasGraph.test.ts; regenerate with vitest -u. Validated by tests/app/invocations/test_frontend_external_canvas_graphs.py.',
        ...graphs,
      };
      await expect(`${JSON.stringify(fixture, null, 2)}\n`).toMatchFileSnapshot(
        '../__snapshots__/externalCanvasGraphs.json'
      );
    });

    it.each(['canvas', 'gallery'] as const)('restores provider output to the exact bbox for %s', (destination) => {
      const { backendGraph, mode } = compile(externalModel, 'txt2img', {
        bbox: smallBbox,
        compositeImageName: null,
        destination,
        scaling: { height: null, method: 'auto', width: null },
      });
      const output = backendGraph.nodes.canvas_processing_output;

      expect(mode).toBe('txt2img');
      expect(output).toMatchObject({
        height: 512,
        is_intermediate: true,
        mode: 'txt2img',
        type: 'openai_image_generation',
        width: 640,
      });
      expect(output).not.toHaveProperty('init_image');
      expect(output).not.toHaveProperty('mask_image');
      expect(getEdge(backendGraph, 'canvas_processing_output', 'prompt')?.source.node_id).toBe('positive_prompt');
      expect(getEdge(backendGraph, 'canvas_processing_output', 'seed')?.source.node_id).toBe('seed');
      expect(backendGraph.nodes.canvas_i2l).toBeUndefined();
      expect(backendGraph.nodes.canvas_output).toMatchObject({
        type: 'img_resize',
        height: 512,
        width: 640,
        is_intermediate: destination === 'canvas',
      });
      expect(backendGraph.nodes.canvas_provider_image.type).toBe('iterate');
      expect(getEdge(backendGraph, 'canvas_provider_image', 'collection')?.source).toEqual({
        node_id: 'canvas_processing_output',
        field: 'collection',
      });
      expect(getEdge(backendGraph, 'canvas_output', 'image')?.source).toEqual({
        node_id: 'canvas_provider_image',
        field: 'item',
      });
      expect(backendGraph.nodes.canvas_provider_metadata.type).toBe('metadata_from_image');
      expect(getEdge(backendGraph, 'canvas_provider_metadata', 'image')?.source).toEqual({
        node_id: 'canvas_provider_image',
        field: 'item',
      });
      expect(getEdge(backendGraph, 'canvas_output', 'metadata')?.source.node_id).toBe('canvas_provider_metadata');
      expect(getEdge(backendGraph, 'canvas_processing_output', 'metadata')).toBeDefined();
      expect(getNodeByType(backendGraph, 'core_metadata')).toMatchObject({ height: 512, width: 640 });
    });

    it('asks a fixed-size provider for its nearest-ratio size and center-crops a text-to-image result to the bbox', () => {
      const { backendGraph } = compile(gptImage, 'txt2img', { bbox: smallBbox, compositeImageName: null });

      // 640×512 is nearest 3:2 among the table's ratios; the provider receives that exact entry.
      expect(backendGraph.nodes.canvas_processing_output).toMatchObject({ height: 1024, width: 1536 });
      // 1536×1024 scaled to cover 640×512 is 768×512, then 64 px come off each side.
      expect(backendGraph.nodes.canvas_provider_cover).toMatchObject({
        height: 512,
        is_intermediate: true,
        type: 'img_resize',
        width: 768,
      });
      expect(backendGraph.nodes.canvas_output).toMatchObject({
        height: 512,
        type: 'img_crop',
        width: 640,
        x: 64,
        y: 0,
      });
      expect(getEdge(backendGraph, 'canvas_provider_cover', 'image')?.source).toEqual({
        node_id: 'canvas_provider_image',
        field: 'item',
      });
      expect(getEdge(backendGraph, 'canvas_output', 'image')?.source.node_id).toBe('canvas_provider_cover');
      expect(getEdge(backendGraph, 'canvas_provider_metadata', 'image')?.source.node_id).toBe('canvas_provider_image');
      expect(getEdge(backendGraph, 'canvas_output', 'metadata')?.source.node_id).toBe('canvas_provider_metadata');
      expect(getNodeByType(backendGraph, 'core_metadata')).toMatchObject({ height: 512, width: 640 });
    });

    it('sends the composited bbox as the init image for img2img when the model advertises it', () => {
      const { backendGraph } = compile({ ...editModel, capabilities: { modes: ['img2img'] } }, 'img2img', {
        compositeImageName: 'bbox.png',
      });

      expect(backendGraph.nodes.canvas_processing_output).toMatchObject({
        height: 1024,
        init_image: { image_name: 'bbox.png' },
        is_intermediate: true,
        mode: 'img2img',
        width: 768,
      });
      expect(backendGraph.nodes.canvas_processing_output).not.toHaveProperty('mask_image');
      expect(backendGraph.nodes.canvas_output).toMatchObject({ height: 1024, type: 'img_resize', width: 768 });
      expect(getEdge(backendGraph, 'canvas_output', 'image')?.source).toEqual({
        node_id: 'canvas_provider_image',
        field: 'item',
      });
      expect(getNodeByType(backendGraph, 'core_metadata')).toMatchObject({ height: 1024, width: 768 });
      expect(backendGraph.nodes.denoise_latents).toBeUndefined();
      expect(backendGraph.nodes.create_gradient_mask).toBeUndefined();
    });

    it('rejects provider inpaint until Canvas mask semantics and compositing are supported', () => {
      expect(() =>
        compile(editModel, 'inpaint', {
          compositeImageName: 'bbox.png',
          maskImageName: 'mask.png',
        })
      ).toThrow('OpenAI Image does not support inpainting on the canvas yet. Remove or disable the inpaint mask.');
    });
  });

  describe('validation', () => {
    it('rejects img2img without a composite image', () => {
      expect(() => compile(sd1Model, 'img2img', { compositeImageName: null })).toThrow(
        'Canvas generation requires a composited source image.'
      );
    });

    it.each([0, 1.5])('rejects a strength of %s', (strength) => {
      expect(() => compile(sd1Model, 'img2img', { strength })).toThrow(
        'Canvas denoising strength must be greater than 0 and at most 1.'
      );
    });

    it('rejects a zero-area bbox', () => {
      expect(() => compile(sd1Model, 'txt2img', { bbox: { height: 0, width: 512, x: 0, y: 0 } })).toThrow(
        'Canvas frame must have a positive area.'
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

    it('processes at the manual size and restores the bbox footprint', () => {
      const { backendGraph } = compile(flux2Model, 'img2img', {
        bbox: { height: 512, width: 512, x: 0, y: 0 },
        scaling: { height: 768, method: 'manual', width: 1024 },
      });

      expect(backendGraph.nodes.canvas_resize_initial_to_processing).toMatchObject({ height: 768, width: 1024 });
      expect(backendGraph.nodes.canvas_output).toMatchObject({ height: 512, type: 'img_resize', width: 512 });
    });
  });
});

describe('compileCanvasGraph — inpaint per base', () => {
  it.each(BASE_CASES)(
    'grafts a $model.base inpaint pipeline (encode, gradient mask, blend composite-back)',
    ({ model, encodeType, outputType, denoiseType, txt2imgMode, optimizedDenoising }) => {
      const { backendGraph, mode } = compile(model, 'inpaint', { strength: 0.6 });
      expect(mode).toBe('inpaint');

      const encode = getNodeByType(backendGraph, encodeType);
      expect(encode).toBeDefined();
      expect(encode?.image).toEqual({ image_name: 'canvas-composite.png' });
      expect(getEdge(backendGraph, 'denoise_latents', 'latents')?.source.node_id).toBe(encode?.id);

      const gradient = backendGraph.nodes.create_gradient_mask;
      expect(gradient?.type).toBe('create_gradient_mask');
      expect(gradient?.coherence_mode).toBe('Gaussian Blur');
      expect(gradient?.edge_radius).toBe(16);
      expect(gradient?.minimum_denoise).toBe(0);
      expect(gradient?.image).toEqual({ image_name: 'canvas-composite.png' });
      expect(gradient?.mask).toEqual({ image_name: 'canvas-mask.png' });
      expect(getEdge(backendGraph, 'denoise_latents', 'denoise_mask')?.source.node_id).toBe('create_gradient_mask');

      expect(gradient?.fp32).toBe(encodeType === 'i2l');

      expect(backendGraph.nodes.canvas_l2i?.type).toBe(outputType);
      expect(backendGraph.nodes.canvas_l2i?.is_intermediate).toBe(true);
      const blend = backendGraph.nodes.canvas_output;
      expect(blend?.type).toBe('invokeai_img_blend');
      expect(blend?.is_intermediate).toBe(true);
      expect(blend?.layer_base).toEqual({ image_name: 'canvas-composite.png' });
      expect(getEdge(backendGraph, 'canvas_output', 'layer_upper')?.source.node_id).toBe('canvas_l2i');

      expect(backendGraph.nodes.expand_mask?.type).toBe('expand_mask_with_fade');
      expect(backendGraph.nodes.expand_mask?.fade_size_px).toBe(16);
      expect(getEdge(backendGraph, 'expand_mask', 'mask')?.source.node_id).toBe('create_gradient_mask');
      expect(getEdge(backendGraph, 'canvas_output', 'mask')?.source.node_id).toBe('expand_mask');

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

  it('inserts an img_noise node before encode when a noise mask is present', () => {
    const { backendGraph } = compile(sd1Model, 'inpaint', { noiseMaskImageName: 'noise.png' });
    const noise = backendGraph.nodes.add_inpaint_noise;
    expect(noise?.type).toBe('img_noise');
    expect(noise?.image).toEqual({ image_name: 'canvas-composite.png' });
    expect(noise?.mask).toEqual({ image_name: 'noise.png' });
    expect(getEdge(backendGraph, 'add_inpaint_noise', 'seed')?.source.node_id).toBe('seed');
    expect(getEdge(backendGraph, 'canvas_i2l', 'image')?.source.node_id).toBe('add_inpaint_noise');
  });

  it('rejects inpaint without a mask image', () => {
    expect(() => compile(sd1Model, 'inpaint', { maskImageName: null })).toThrow(
      'Canvas inpainting requires an inpaint mask.'
    );
  });
});

describe('compileCanvasGraph — outpaint per base', () => {
  it.each(BASE_CASES)('grafts a $model.base outpaint pipeline (infill + alpha mask)', ({ model, txt2imgMode }) => {
    const { backendGraph, mode } = compile(model, 'outpaint', { maskImageName: null });
    expect(mode).toBe('outpaint');

    const infill = backendGraph.nodes.infill;
    expect(infill?.type).toBe('infill_lama');
    expect(infill?.image).toEqual({ image_name: 'canvas-composite.png' });

    expect(backendGraph.nodes.image_alpha_to_mask?.type).toBe('tomask');
    expect(getEdge(backendGraph, 'create_gradient_mask', 'mask')?.source.node_id).toBe('image_alpha_to_mask');

    expect(getEdge(backendGraph, 'canvas_i2l', 'image')?.source.node_id).toBe('infill');

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
});

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

    expect(backendGraph.nodes.positive_prompt).toBeDefined();
    expect(backendGraph.nodes.negative_prompt).toBeDefined();
    expect(backendGraph.nodes.seed).toBeDefined();
    expect(backendGraph.nodes.canvas_i2l?.type).toBe('i2l');
    expect(getEdge(backendGraph, 'denoise_latents', 'latents')?.source.node_id).toBe('canvas_i2l');
    expect(backendGraph.nodes.canvas_output).toBeDefined();
    expect(graph.label).toBe(`${sd1Model.name} img2img`);
  });
});

describe('compileCanvasGraph — Qwen-Image ControlNet (integration)', () => {
  const layer: ControlLayerGraphInput = {
    beginEndStepPct: [0, 1],
    controlMode: null,
    id: 'q-control',
    imageName: 'q-control.png',
    kind: 'qwen_image_control',
    model: { base: 'qwen-image', key: 'qwen-union', name: 'Qwen Image ControlNet Union', type: 'controlnet' },
    weight: 0.85,
  };
  const instantxLayer: ControlLayerGraphInput = {
    ...layer,
    model: {
      base: 'qwen-image',
      key: 'qwen-instantx',
      name: 'Qwen Image InstantX ControlNet Union',
      type: 'controlnet',
    },
    modelControlArchitecture: 'instantx_union',
  };
  it('writes the graphs the backend validates against its invocations', async () => {
    // Ids minted from the clock and Math.random are pinned so the fixture changes only when the graph does.
    const clock = vi.spyOn(Date, 'now').mockReturnValue(0);
    const random = vi.spyOn(Math, 'random').mockReturnValue(0.5);
    // Installed models always carry a hash, which the backend's model identifier requires.
    const hashed = <T extends object>(config: T, hash: string): T => ({ ...config, hash });
    const model = hashed(qwenImageModel, 'blake3:qwen');
    const controlLayers = [{ ...layer, model: hashed(layer.model, 'blake3:union') }];
    const instantxLayers = [{ ...instantxLayer, model: hashed(instantxLayer.model, 'blake3:instantx') }];
    let txt2img: BackendGraphContract;
    let inpaint: BackendGraphContract;
    let outpaint: BackendGraphContract;
    let instantxInpaint: BackendGraphContract;
    try {
      txt2img = compile(model, 'txt2img', { controlLayers }).backendGraph;
      inpaint = compile(model, 'inpaint', { controlLayers }).backendGraph;
      outpaint = compile(model, 'outpaint', { controlLayers, maskImageName: 'canvas-mask.png' }).backendGraph;
      instantxInpaint = compile(model, 'inpaint', { controlLayers: instantxLayers }).backendGraph;
    } finally {
      clock.mockRestore();
      random.mockRestore();
    }
    const fixture = {
      _comment:
        'Generated by compileCanvasGraph.test.ts; regenerate with `vitest -u`. Consumed by tests/app/invocations/test_frontend_fun_control_graphs.py, which validates the graphs against the backend invocations.',
      inpaint_control_layer: { ...inpaint, id: 'qwen_image_control_canvas_inpaint' },
      inpaint_instantx_control_layer: { ...instantxInpaint, id: 'qwen_image_control_canvas_inpaint_instantx' },
      outpaint_control_layer: { ...outpaint, id: 'qwen_image_control_canvas_outpaint' },
      txt2img_control_layer: { ...txt2img, id: 'qwen_image_control_canvas_txt2img' },
    };
    await expect(`${JSON.stringify(fixture, null, 2)}\n`).toMatchFileSnapshot(
      '../__snapshots__/qwenImageControlCanvasGraphs.json'
    );
  });
});

describe('compileCanvasGraph — FLUX.2 [dev] ControlNet (integration)', () => {
  const layer: ControlLayerGraphInput = {
    beginEndStepPct: [0, 1],
    controlMode: null,
    id: 'f-control',
    imageName: 'f-control.png',
    kind: 'flux2_control',
    model: { base: 'flux2', key: 'flux2-union', name: 'FLUX.2 [dev] ControlNet Union', type: 'controlnet' },
    weight: 0.75,
  };
  it('writes the graphs the backend validates against its invocations', async () => {
    const clock = vi.spyOn(Date, 'now').mockReturnValue(0);
    const random = vi.spyOn(Math, 'random').mockReturnValue(0.5);
    const hashed = <T extends object>(config: T, hash: string): T => ({ ...config, hash });
    const model = hashed(flux2DevModel, 'blake3:flux2-dev');
    const controlLayers = [{ ...layer, model: hashed(layer.model, 'blake3:union') }];
    let txt2img: BackendGraphContract;
    let inpaint: BackendGraphContract;
    let outpaint: BackendGraphContract;
    try {
      txt2img = compile(model, 'txt2img', { controlLayers }).backendGraph;
      inpaint = compile(model, 'inpaint', { controlLayers }).backendGraph;
      outpaint = compile(model, 'outpaint', { controlLayers, maskImageName: 'canvas-mask.png' }).backendGraph;
    } finally {
      clock.mockRestore();
      random.mockRestore();
    }
    const fixture = {
      _comment:
        'Generated by compileCanvasGraph.test.ts; regenerate with `vitest -u`. Consumed by tests/app/invocations/test_frontend_fun_control_graphs.py, which validates the graphs against the backend invocations.',
      inpaint_control_layer: { ...inpaint, id: 'flux2_control_canvas_inpaint' },
      outpaint_control_layer: { ...outpaint, id: 'flux2_control_canvas_outpaint' },
      txt2img_control_layer: { ...txt2img, id: 'flux2_control_canvas_txt2img' },
    };
    await expect(`${JSON.stringify(fixture, null, 2)}
`).toMatchFileSnapshot('../__snapshots__/flux2ControlCanvasGraphs.json');
  });
});

describe('compileCanvasGraph — Krea-2 Control LoRA (integration)', () => {
  const layer: ControlLayerGraphInput = {
    beginEndStepPct: [0, 1],
    controlMode: null,
    id: 'k-control',
    imageName: 'k-control.png',
    kind: 'control_lora',
    model: { base: 'krea-2', key: 'krea2-depth', name: 'Krea-2 Depth Control LoRA', type: 'control_lora' },
    weight: 0.8,
  };

  it('encodes the control image with the graph VAE at the size the denoise runs at, not the bbox', () => {
    // 700x1000 is off the 16px grid, so processing rounds it.
    const { backendGraph } = compile(krea2Model, 'img2img', {
      bbox: { height: 1000, width: 700, x: 0, y: 0 },
      controlLayers: [layer],
    });
    const denoise = backendGraph.nodes.denoise_latents;
    const encode = backendGraph.nodes['control_lora_i2l_k-control'];

    expect(denoise?.type).toBe('krea2_denoise');
    expect(denoise?.width).not.toBe(700);
    expect(encode).toMatchObject({
      height: denoise?.height,
      image: { image_name: 'k-control.png' },
      type: 'qwen_image_i2l',
      width: denoise?.width,
    });
    expect(getEdge(backendGraph, 'control_lora_i2l_k-control', 'vae')?.source).toEqual(
      getEdge(backendGraph, 'canvas_i2l', 'vae')?.source
    );
    expect(getEdge(backendGraph, 'control_lora_k-control', 'control_latents')?.source.node_id).toBe(
      'control_lora_i2l_k-control'
    );
    expect(getEdge(backendGraph, 'denoise_latents', 'control_lora')?.source).toEqual({
      field: 'control_lora',
      node_id: 'control_lora_k-control',
    });
  });

  it('writes the graphs the backend validates against its invocations', async () => {
    // Ids minted from the clock and Math.random are pinned so the fixture changes only when the graph does.
    const clock = vi.spyOn(Date, 'now').mockReturnValue(0);
    const random = vi.spyOn(Math, 'random').mockReturnValue(0.5);
    // Installed models always carry a hash, which the backend's model identifier requires.
    const hashed = <T extends object>(config: T, hash: string): T => ({ ...config, hash });
    const model = hashed(krea2Model, 'blake3:krea2');
    const controlLayers = [{ ...layer, model: hashed(layer.model, 'blake3:depth') }];
    let txt2img: BackendGraphContract;
    let inpaint: BackendGraphContract;
    try {
      txt2img = compile(model, 'txt2img', { controlLayers }).backendGraph;
      inpaint = compile(model, 'inpaint', { controlLayers }).backendGraph;
    } finally {
      clock.mockRestore();
      random.mockRestore();
    }
    const fixture = {
      _comment:
        'Generated by compileCanvasGraph.test.ts; regenerate with `vitest -u`. Consumed by tests/app/invocations/test_frontend_krea2_control_lora_graphs.py, which validates the graphs against the backend invocations.',
      inpaint_control_lora: { ...inpaint, id: 'krea2_control_lora_canvas_inpaint' },
      txt2img_control_lora: { ...txt2img, id: 'krea2_control_lora_canvas_txt2img' },
    };
    await expect(`${JSON.stringify(fixture, null, 2)}\n`).toMatchFileSnapshot(
      '../__snapshots__/krea2ControlLoraCanvasGraphs.json'
    );
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
  it('rejects a control (3-channel) adapter and an out-of-range weight', () => {
    expect(() =>
      compile(animaModel, 'inpaint', {
        settings: { animaInpaintAdapterModel: { ...inpaintAdapter, cond_in_channels: 3 } },
      })
    ).toThrow('The Anima inpaint adapter must be a 4-channel Anima ControlNet-LLLite model.');
    expect(() => compile(animaModel, 'outpaint', { settings: withAdapter(10.5) })).toThrow(
      'Anima inpaint adapter weight must be between -10 and 10.'
    );
    expect(() =>
      compile(animaModel, 'inpaint', {
        settings: {
          animaInpaintAdapterEnabled: false,
          animaInpaintAdapterModel: { ...inpaintAdapter, cond_in_channels: 3 },
          animaInpaintAdapterWeight: 10.5,
        },
      })
    ).not.toThrow();
    expect(() =>
      compile(animaModel, 'inpaint', {
        settings: { animaInpaintAdapterModel: { ...inpaintAdapter, cond_in_channels: null } },
      })
    ).not.toThrow();
  });

  it('writes the graphs the backend validates against its invocations', async () => {
    const clock = vi.spyOn(Date, 'now').mockReturnValue(0);
    const random = vi.spyOn(Math, 'random').mockReturnValue(0.5);
    const hashed = <T extends object>(config: T, hash: string): T => ({ ...config, hash });
    const settings = {
      qwen3EncoderModel: hashed(qwen3Encoder, 'blake3:qwen3'),
      vae: hashed(qwenImageVae, 'blake3:vae'),
    };
    let threeAdapters: BackendGraphContract;
    let inpaintAdapterGraph: BackendGraphContract;
    let outpaintAdapterGraph: BackendGraphContract;
    try {
      threeAdapters = compile(hashed(animaModel, 'blake3:anima'), 'inpaint', {
        controlLayers: [
          lliteLayer('sketch-layer', 'sketch', 0.8),
          lliteLayer('depth-layer', 'depth', -1),
          lliteLayer('pose-layer', 'pose', 2),
        ],
        settings,
      }).backendGraph;
      inpaintAdapterGraph = compile(hashed(animaModel, 'blake3:anima'), 'inpaint', {
        settings: {
          ...settings,
          animaInpaintAdapterModel: hashed(inpaintAdapter, 'blake3:inpaint'),
          animaInpaintAdapterWeight: 0.75,
        },
      }).backendGraph;
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

  it('img2img fit: encodes the composite at the bbox with the VAE the PiD decode records', () => {
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

  it('rejects inpaint and outpaint with the legacy message (buildSDXLGraph.ts:197)', () => {
    for (const mode of ['inpaint', 'outpaint'] as const) {
      expect(() => compile(sdxlModel, mode, { settings: pidSettings(sdxlModel, 'fit') })).toThrow(
        'PiD decode currently supports Text to Image and Image to Image only. Disable PiD for Inpaint/Outpaint.'
      );
    }
  });
});

describe('compileCanvasGraph — regional guidance', () => {
  const region = (id: string, overrides: Partial<RegionalGuidanceInput> = {}): RegionalGuidanceInput => ({
    autoNegative: false,
    id,
    maskBounds: { height: 256, width: 192, x: 64, y: 128 },
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
    expect(backendGraph.nodes.rg_invert_mask_r1).toBeDefined();
    expect(backendGraph.nodes.rg_pos_cond_inverted_r1).toBeDefined();
  });

  describe.each([
    ['Z-Image', zImageModel, 'z_image_text_encoder', 'qwen3_encoder'],
    ['Anima', animaModel, 'anima_text_encoder', 'qwen3_encoder'],
    ['Krea-2', krea2Model, 'krea2_text_encoder', 'qwen3_vl_encoder'],
  ] as const)('%s regional negatives', (_label, model, encoderType, encoderField) => {
    it('collects masked negatives and inverted positives alongside the global negative', () => {
      const { backendGraph: graph } = compile(model, 'txt2img', {
        regionalGuidance: [
          region('r1', { autoNegative: true, negativePrompt: 'blurry' }),
          region('r2', { negativePrompt: 'text', positivePrompt: null }),
        ],
        settings: { cfgScale: 4 },
      });

      expect(graph.nodes.rg_neg_cond_r1).toMatchObject({ prompt: 'blurry', type: encoderType });
      expect(graph.nodes.rg_neg_cond_r2).toMatchObject({ prompt: 'text', type: encoderType });
      expect(graph.nodes.rg_pos_cond_r2).toBeUndefined();
      expect(graph.nodes.rg_pos_cond_inverted_r1).toMatchObject({ prompt: 'a cat', type: encoderType });
      expect(getEdge(graph, 'rg_neg_cond_r1', 'mask')?.source.node_id).toBe('rg_mask_to_tensor_r1');
      expect(getEdge(graph, 'rg_neg_cond_r2', 'mask')?.source.node_id).toBe('rg_mask_to_tensor_r2');
      expect(getEdge(graph, 'rg_invert_mask_r1', 'mask')?.source.node_id).toBe('rg_mask_to_tensor_r1');
      expect(getEdge(graph, 'rg_pos_cond_inverted_r1', 'mask')?.source.node_id).toBe('rg_invert_mask_r1');
      for (const nodeId of ['rg_neg_cond_r1', 'rg_neg_cond_r2', 'rg_pos_cond_inverted_r1']) {
        expect(getEdge(graph, nodeId, encoderField)?.source.node_id).toBe('model_loader');
      }
      expect(
        graph.edges.filter((edge) => edge.destination.node_id === 'neg_cond_collect').map((edge) => edge.source.node_id)
      ).toEqual(['neg_cond', 'rg_neg_cond_r1', 'rg_pos_cond_inverted_r1', 'rg_neg_cond_r2']);
      expect(getEdge(graph, 'denoise_latents', 'negative_conditioning')?.source).toEqual({
        field: 'collection',
        node_id: 'neg_cond_collect',
      });
    });
  });
  it('grafts positive-only regional conditioning into a FLUX.2 graph', () => {
    const { backendGraph } = compile(flux2Model, 'txt2img', { regionalGuidance: [region('r1')] });

    expect(backendGraph.nodes.rg_pos_cond_r1).toMatchObject({ prompt: 'a cat', type: 'flux2_klein_text_encoder' });
    expect(getEdge(backendGraph, 'rg_pos_cond_r1', 'qwen3_encoder')?.source.node_id).toBe('model_loader');
    expect(getEdge(backendGraph, 'pos_cond_collect', 'item')?.source.node_id).toBe('pos_cond');
  });

  it('writes the MOSS-Image layout canvas graphs the backend validates', async () => {
    const clock = vi.spyOn(Date, 'now').mockReturnValue(0);
    const random = vi.spyOn(Math, 'random').mockReturnValue(0.5);
    const model = { ...mossImageModel, hash: 'blake3:moss' };
    const regions = [
      region('sign', {
        layout: { kind: 'text', text: '一杯温暖\n一刻从容' },
        maskBounds: { height: 256, width: 512, x: 0, y: 0 },
        positivePrompt: 'a hand-lettered cafe sign',
      }),
      region('barista', { layout: { kind: 'character' }, positivePrompt: 'a barista pouring latte art' }),
      region('cup', {
        layout: { kind: 'object', text: 'kept for a switch back' },
        maskBounds: { height: 384, width: 256, x: 512, y: 640 },
        positivePrompt: 'a steaming ceramic cup',
      }),
    ];
    let txt2img: BackendGraphContract;
    let inpaint: BackendGraphContract;
    try {
      const settings = { positivePrompt: 'a cozy cafe interior, morning light' };
      txt2img = compile(model, 'txt2img', { regionalGuidance: regions, settings }).backendGraph;
      inpaint = compile(model, 'inpaint', { regionalGuidance: regions, settings }).backendGraph;
    } finally {
      clock.mockRestore();
      random.mockRestore();
    }
    const fixture = {
      _comment:
        'Generated by compileCanvasGraph.test.ts; regenerate with `vitest -u`. Validated against the backend invocations by the MOSS-Image layout graph tests.',
      inpaint: { ...inpaint, id: 'moss_image_layout_canvas_inpaint' },
      txt2img: { ...txt2img, id: 'moss_image_layout_canvas_txt2img' },
    };
    await expect(`${JSON.stringify(fixture, null, 2)}\n`).toMatchFileSnapshot(
      '../__snapshots__/mossImageLayoutCanvasGraphs.json'
    );
  });

  it('writes the Ideogram 4 regional canvas graph the backend validates', async () => {
    const clock = vi.spyOn(Date, 'now').mockReturnValue(0);
    const random = vi.spyOn(Math, 'random').mockReturnValue(0.5);
    const model = { ...ideogram4Model, hash: 'blake3:ideogram4' };
    const regions = [
      region('kite', { maskBounds: { height: 256, width: 384, x: 192, y: 64 }, positivePrompt: 'a red kite' }),
      region('dog', {
        maskBounds: { height: 384, width: 256, x: 512, y: 640 },
        positivePrompt: 'a dog digging in the sand',
      }),
    ];
    let txt2img: BackendGraphContract;
    try {
      txt2img = compile(model, 'txt2img', {
        regionalGuidance: regions,
        settings: { positivePrompt: 'a beach at noon' },
      }).backendGraph;
    } finally {
      clock.mockRestore();
      random.mockRestore();
    }
    const fixture = {
      _comment:
        'Generated by compileCanvasGraph.test.ts; regenerate with `vitest -u`. Validated against the backend invocations by the Ideogram 4 regional graph tests.',
      txt2img: { ...txt2img, id: 'ideogram4_regional_canvas_txt2img' },
    };
    await expect(`${JSON.stringify(fixture, null, 2)}\n`).toMatchFileSnapshot(
      '../__snapshots__/ideogram4RegionalCanvasGraphs.json'
    );
  });
  it('writes the FLUX.3 layout canvas graphs the backend validates', async () => {
    const clock = vi.spyOn(Date, 'now').mockReturnValue(0);
    const random = vi.spyOn(Math, 'random').mockReturnValue(0.5);
    const model = { ...flux3Model, hash: 'blake3:flux3' };
    const regions = [
      region('kite', {
        maskBounds: { height: 256, width: 384, x: 192, y: 64 },
        positivePrompt: 'a red kite reading "Éole"',
      }),
      region('dog', {
        maskBounds: { height: 384, width: 256, x: 512, y: 640 },
        positivePrompt: 'a dog digging in the sand',
      }),
    ];
    const reference = {
      config: {
        image: { original: { image: { height: 768, image_name: 'beach.png', width: 512 } } },
        type: 'flux2_reference_image' as const,
      },
      id: 'beach',
      isEnabled: true,
    };
    let txt2img: BackendGraphContract;
    let inpaint: BackendGraphContract;
    let edit: BackendGraphContract;
    let upsampledTxt2img: BackendGraphContract;
    let upsampledEdit: BackendGraphContract;
    try {
      const settings = { positivePrompt: 'a beach at noon' };
      const upsampled = {
        ...settings,
        flux3PromptUpsamplerEnabled: true,
        flux3PromptUpsamplerModel: {
          base: 'any',
          hash: 'blake3:pup',
          key: 'pup',
          name: 'FLUX.3 Prompt Upsampler',
          supports_images: true,
          type: 'text_llm',
        },
      };
      upsampledTxt2img = compile(model, 'txt2img', { regionalGuidance: regions, settings: upsampled }).backendGraph;
      upsampledEdit = compile(model, 'txt2img', {
        regionalGuidance: regions,
        settings: { ...upsampled, referenceImages: [reference] },
      }).backendGraph;
      txt2img = compile(model, 'txt2img', { regionalGuidance: regions, settings }).backendGraph;
      inpaint = compile(model, 'inpaint', { regionalGuidance: regions, settings }).backendGraph;
      edit = compile(model, 'txt2img', {
        regionalGuidance: regions,
        settings: { ...settings, referenceImages: [reference] },
      }).backendGraph;
    } finally {
      clock.mockRestore();
      random.mockRestore();
    }
    const fixture = {
      _comment:
        'Generated by compileCanvasGraph.test.ts; regenerate with `vitest -u`. Validated against the backend invocations by the FLUX.3 layout graph tests.',
      edit: { ...edit, id: 'flux3_layout_canvas_edit' },
      inpaint: { ...inpaint, id: 'flux3_layout_canvas_inpaint' },
      txt2img: { ...txt2img, id: 'flux3_layout_canvas_txt2img' },
      upsampled_edit: { ...upsampledEdit, id: 'flux3_upsampled_canvas_edit' },
      upsampled_txt2img: { ...upsampledTxt2img, id: 'flux3_upsampled_canvas_txt2img' },
    };
    await expect(`${JSON.stringify(fixture, null, 2)}\n`).toMatchFileSnapshot(
      '../__snapshots__/flux3LayoutCanvasGraphs.json'
    );
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
});

describe('compileCanvasGraph — output filters', () => {
  const metadataTarget = (graph: BackendGraphContract) =>
    graph.edges.find((edge) => edge.destination.field === 'metadata')?.destination.node_id;

  it('filters the resized txt2img output, keeping the resize intermediate and the metadata on the result', () => {
    const { backendGraph } = compile(sd1Model, 'txt2img', {
      bbox: { height: 1000, width: 700, x: 0, y: 0 },
      destination: 'gallery',
      projectSettings: { useNsfwChecker: true, useWatermark: true },
    });

    expect(backendGraph.nodes.canvas_unfiltered_output).toMatchObject({ is_intermediate: true, type: 'img_resize' });
    expect(backendGraph.nodes.canvas_img_nsfw?.type).toBe('img_nsfw');
    expect(backendGraph.nodes.canvas_output).toMatchObject({ is_intermediate: false, type: 'img_watermark' });
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

describe('compileCanvasGraph — Krea-2 edit references', () => {
  const editReference = (imageName: string, isEnabled = true) => ({
    config: {
      image: { original: { image: { height: 768, image_name: imageName, width: 512 } } },
      type: 'krea2_edit_reference_image' as const,
    },
    id: imageName,
    isEnabled,
  });
  it('inpaints with the edit references alongside the encoded canvas image', () => {
    const { backendGraph } = compile(krea2Model, 'inpaint', {
      settings: { referenceImages: [editReference('pose.png'), editReference('depth.png')] },
    });
    const referenceEncoders = Object.values(backendGraph.nodes).filter(
      (node) => node.type === 'qwen_image_i2l' && node.reference === true
    );

    expect(getEdge(backendGraph, 'denoise_latents', 'latents')?.source.node_id).toBe('canvas_i2l');
    expect(backendGraph.nodes.canvas_i2l).not.toHaveProperty('reference');
    expect(getEdge(backendGraph, 'denoise_latents', 'denoise_mask')).toBeDefined();
    expect(referenceEncoders.map((node) => node.image)).toEqual([
      { image_name: 'pose.png' },
      { image_name: 'depth.png' },
    ]);
    expect(getEdge(backendGraph, 'denoise_latents', 'reference_latents')).toBeDefined();
    expect(backendGraph.nodes.pos_cond?.reference_images).toEqual([
      { image_name: 'pose.png' },
      { image_name: 'depth.png' },
    ]);
  });
});
