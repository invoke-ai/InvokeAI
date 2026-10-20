import type { GenerateLora, VaeModelConfig } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';

import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { describe, expect, it } from 'vitest';

import type { TileControlNetModelConfig, UpscaleWidgetValues } from './types';

import {
  compileModelOnlyUpscaleGraph,
  compileUpscaleGraph,
  getUpscaleControlNetValues,
  getUpscaleDenoisingStart,
  getZImageTileControlValues,
} from './graph';
import { createDefaultUpscaleWidgetValues, normalizeUpscaleWidgetValues } from './settings';

const model = (key: string, type: string, base: string, name = key): ModelConfig => ({
  base,
  file_size: 1,
  format: 'checkpoint',
  hash: `${key}-hash`,
  key,
  name,
  path: key,
  source: key,
  source_type: 'path',
  type,
  // Every installed CLIP Embed carries its variant; FLUX.1 takes only CLIP-L.
  ...(type === 'clip_embed' ? { variant: 'large' } : {}),
});

const createValues = (base: 'sd-1' | 'sdxl') => {
  const models = [
    model('main', 'main', base),
    model('spandrel', 'spandrel_image_to_image', 'any'),
    model('tile', 'controlnet', base, 'Tile ControlNet'),
    model('lora', 'lora', base),
    model('vae', 'vae', base),
  ];
  const values = createDefaultUpscaleWidgetValues(models);

  return {
    ...values,
    inputImage: { height: 101, image_name: 'input.png', width: 203 },
    loras: [{ isEnabled: true, model: models[3] as GenerateLora['model'], weight: 0.7 }],
    negativePrompt: 'blur',
    positivePrompt: 'detail',
    vae: models[4] as VaeModelConfig,
  };
};

const hasEdge = (
  edges: ReturnType<typeof compileUpscaleGraph>['backendGraph']['edges'],
  source: string,
  sourceField: string,
  destination: string,
  destinationField: string
) =>
  edges.some(
    (edge) =>
      edge.source.node_id === source &&
      edge.source.field === sourceField &&
      edge.destination.node_id === destination &&
      edge.destination.field === destinationField
  );

seedArchitectureCapabilities();

describe('compileUpscaleGraph', () => {
  it('preserves the exact legacy creativity and structure formulas', () => {
    expect(getUpscaleDenoisingStart(0)).toBeCloseTo(0.499);
    expect(getUpscaleDenoisingStart(10)).toBe(0);
    expect(getUpscaleControlNetValues(0)).toEqual({
      first: { beginStepPercent: 0, controlWeight: 0.625, endStepPercent: 0.55 },
      second: { beginStepPercent: 0.55, controlWeight: 0.21375, endStepPercent: 0.85 },
    });
  });

  it.each(['sd-1', 'sdxl'] as const)('compiles the legacy %s topology with fixed batch ids', (base) => {
    const compiled = compileUpscaleGraph(createValues(base), 'gallery', { useCpuNoise: true });
    const { edges, nodes } = compiled.backendGraph;

    expect(compiled.positivePromptNodeId).toBe('positive_prompt');
    expect(compiled.negativePromptNodeId).toBe('negative_prompt');
    expect(compiled.seedNodeId).toBe('seed');
    expect(compiled.outputNodeId).toBe('upscale_output');
    expect(nodes.spandrel_autoscale).toMatchObject({
      fit_to_multiple_of_8: true,
      image: { image_name: 'input.png' },
      scale: 4,
      type: 'spandrel_image_to_image_autoscale',
    });
    expect(nodes.tiled_multidiffusion_denoise_latents).toMatchObject({
      cfg_scale: 2,
      scheduler: 'kdpm_2',
      steps: 30,
      tile_height: 1024,
      tile_overlap: 128,
      tile_width: 1024,
    });
    expect(nodes.tiled_multidiffusion_denoise_latents?.denoising_start).toBeCloseTo(0.499);
    expect(nodes.upscale_output).toMatchObject({ is_intermediate: false, tiled: true, type: 'l2i' });
    expect(nodes.noise?.use_cpu).toBe(true);
    expect(nodes.controlnet_1).toMatchObject({ control_weight: 0.625, end_step_percent: 0.55 });
    expect(nodes.controlnet_2).toMatchObject({ begin_step_percent: 0.55, control_weight: 0.21375 });
    expect(nodes.core_metadata).toMatchObject({
      creativity: 0,
      structure: 0,
      tile_overlap: 128,
      tile_size: 1024,
      upscale_scale: 4,
    });
    expect(
      hasEdge(edges, 'controlnet_collector', 'collection', 'tiled_multidiffusion_denoise_latents', 'control')
    ).toBe(true);
    expect(hasEdge(edges, 'positive_prompt', 'value', 'pos_cond', 'prompt')).toBe(true);
    expect(hasEdge(edges, 'vae_loader', 'vae', 'upscale_output', 'vae')).toBe(true);
    expect(Object.values(nodes).some((node) => node.type.includes('lora_collection_loader'))).toBe(true);
    expect(base === 'sdxl' ? nodes.clip_skip : nodes.clip_skip?.type).toBe(base === 'sdxl' ? undefined : 'clip_skip');
  });

  it('marks only Canvas-destination output intermediate', () => {
    expect(
      compileUpscaleGraph(createValues('sd-1'), 'canvas', { useCpuNoise: false }).backendGraph.nodes.upscale_output
    ).toMatchObject({ is_intermediate: true, use_cache: false });
  });

  it('records the accelerator supplied by the orchestration boundary in metadata', () => {
    const graph = compileUpscaleGraph(createValues('sd-1'), 'gallery', { useCpuNoise: false }, 'xpu').backendGraph;

    expect(graph.nodes.core_metadata?.rand_device).toBe('xpu');
  });

  it('refines values saved before modes existed, whatever source metadata is offered', () => {
    const { mode: _mode, ...saved } = createValues('sd-1');
    const values = normalizeUpscaleWidgetValues(saved) as UpscaleWidgetValues;
    const nodes = compileUpscaleGraph(values, 'gallery', { useCpuNoise: false }, 'cuda', { prompt: 'source' })
      .backendGraph.nodes;

    expect(nodes.tiled_multidiffusion_denoise_latents).toBeDefined();
    expect(nodes.core_metadata).toMatchObject({ creativity: 0, type: 'core_metadata' });
    expect(nodes.core_metadata).not.toHaveProperty('prompt');
  });
});

describe('model-only upscale graph', () => {
  const image = { height: 101, image_name: 'input.png', width: 203 };
  const upscaler = (scale: number | null) => ({
    ...model('ultrasharp', 'spandrel_image_to_image', 'any', '4x-UltraSharp'),
    scale,
    type: 'spandrel_image_to_image' as const,
  });
  const UPSCALER_ID = {
    base: 'any',
    hash: 'ultrasharp-hash',
    key: 'ultrasharp',
    name: '4x-UltraSharp',
    type: 'spandrel_image_to_image',
  };
  const SOURCE_METADATA = {
    height: 512,
    metadata_version: 1,
    model: { key: 'legacy-main', model_name: 'old shape' },
    positive_prompt: 'a lighthouse',
    seed: 42,
    upscale_initial_image: { height: 1, image_name: 'older.png', width: 1 },
    upscale_model: { key: 'older-upscaler' },
    upscale_scale: 8,
    width: 768,
  };

  it('runs the upscaler alone at the widget Scale, with no main model', () => {
    const values = {
      ...createDefaultUpscaleWidgetValues(),
      inputImage: image,
      mode: 'model-only' as const,
      scale: 2,
      upscaleModel: upscaler(4),
    };

    for (const destination of ['gallery', 'canvas'] as const) {
      const compiled = compileUpscaleGraph(values, destination, { useCpuNoise: false }, 'cuda', null);
      const { edges, nodes } = compiled.backendGraph;

      expect(Object.keys(nodes).sort()).toEqual(['negative_prompt', 'positive_prompt', 'seed', 'upscale_output']);
      expect(edges).toEqual([]);
      expect(compiled.outputNodeId).toBe('upscale_output');
      expect(nodes.upscale_output).toMatchObject({
        fit_to_multiple_of_8: true,
        image: { image_name: 'input.png' },
        image_to_image_model: UPSCALER_ID,
        is_intermediate: destination === 'canvas',
        scale: 2,
        type: 'spandrel_image_to_image_autoscale',
        use_cache: false,
      });
    }
  });

  it('runs the image menu upscale once, at the model native scale', () => {
    const output = compileModelOnlyUpscaleGraph({
      destination: 'gallery',
      image,
      scale: 'native',
      sourceMetadata: null,
      upscaleModel: upscaler(4),
    }).backendGraph.nodes.upscale_output;

    expect(output).toMatchObject({ is_intermediate: false, type: 'spandrel_image_to_image', use_cache: false });
    expect(output).not.toHaveProperty('scale');
    expect(output).not.toHaveProperty('fit_to_multiple_of_8');
  });

  it("copies the source image's metadata verbatim, replacing only what it says about an earlier upscale", () => {
    const metadata = compileModelOnlyUpscaleGraph({
      destination: 'gallery',
      image,
      scale: 2,
      sourceMetadata: SOURCE_METADATA,
      upscaleModel: upscaler(4),
    }).backendGraph.nodes.upscale_output?.metadata;

    expect(metadata).toEqual({
      height: 512,
      metadata_version: 1,
      model: { key: 'legacy-main', model_name: 'old shape' },
      positive_prompt: 'a lighthouse',
      seed: 42,
      upscale_initial_image: { height: 101, image_name: 'input.png', width: 203 },
      upscale_model: UPSCALER_ID,
      upscale_scale: 2,
      width: 768,
    });
  });

  it('records only the upscale for a source with no metadata, and no scale when the model never recorded one', () => {
    const metadata = (scale: number | null) =>
      compileModelOnlyUpscaleGraph({
        destination: 'gallery',
        image,
        scale: 'native',
        sourceMetadata: scale === null ? SOURCE_METADATA : null,
        upscaleModel: upscaler(scale),
      }).backendGraph.nodes.upscale_output?.metadata;

    expect(metadata(4)).toEqual({
      upscale_initial_image: { height: 101, image_name: 'input.png', width: 203 },
      upscale_model: UPSCALER_ID,
      upscale_scale: 4,
    });
    expect(metadata(null)).not.toHaveProperty('upscale_scale');
    expect(metadata(null)).toMatchObject({ positive_prompt: 'a lighthouse', upscale_model: UPSCALER_ID });
  });
});

const createFluxValues = (options: { withEncoders?: boolean } = {}) => {
  const { withEncoders = true } = options;
  const models = [
    model('main', 'main', 'flux'),
    model('spandrel', 'spandrel_image_to_image', 'any'),
    model('lora', 'lora', 'flux'),
    model('fluxvae', 'vae', 'flux'),
    ...(withEncoders ? [model('t5', 't5_encoder', 'any'), model('clip', 'clip_embed', 'any')] : []),
  ];
  const values = createDefaultUpscaleWidgetValues(models);

  return {
    ...values,
    inputImage: { height: 101, image_name: 'input.png', width: 203 },
    loras: [{ isEnabled: true, model: models[2] as GenerateLora['model'], weight: 0.7 }],
    positivePrompt: 'detail',
  };
};

describe('compileUpscaleGraph for FLUX.1', () => {
  it('drives the second pass with flux nodes and tiles both VAE ends', () => {
    const { edges, nodes } = compileUpscaleGraph(createFluxValues(), 'gallery', { useCpuNoise: true }).backendGraph;

    expect(nodes.model_loader).toMatchObject({ type: 'flux_model_loader' });
    expect(nodes.model_loader?.t5_encoder_model).toMatchObject({ key: 't5' });
    expect(nodes.model_loader?.clip_embed_model).toMatchObject({ key: 'clip' });
    expect(nodes.i2l).toMatchObject({ tile_size: 1024, tiled: true, type: 'flux_vae_encode' });
    expect(nodes.upscale_output).toMatchObject({ tile_size: 1024, tiled: true, type: 'flux_vae_decode' });
    expect(nodes.denoise_latents).toMatchObject({ guidance: 2, num_steps: 30, type: 'flux_denoise' });
    expect(nodes.denoise_latents?.denoising_start).toBeCloseTo(0.499);
    expect(hasEdge(edges, 'i2l', 'latents', 'denoise_latents', 'latents')).toBe(true);
    expect(hasEdge(edges, 'pos_cond_collect', 'collection', 'denoise_latents', 'positive_text_conditioning')).toBe(
      true
    );
    expect(Object.values(nodes).some((node) => node.type.includes('lora_collection_loader'))).toBe(true);
  });

  it('fits the frame to the grid flux_denoise accepts, instead of whatever Spandrel rounded to', () => {
    // Spandrel's autoscale floors to a multiple of 8; `flux_denoise` declares multiple_of=16 and
    // raises on anything else -- after the upscale has already been paid for. The 203x101 fixture
    // at 4x is exactly that case: Spandrel gives 808x400, and 808 is not a multiple of 16.
    const { edges, nodes } = compileUpscaleGraph(createFluxValues(), 'gallery', { useCpuNoise: true }).backendGraph;

    expect(nodes.fit_to_grid).toMatchObject({ height: 400, type: 'img_resize', width: 800 });
    expect(nodes.denoise_latents).toMatchObject({ height: 400, width: 800 });
    expect(hasEdge(edges, 'unsharp_2', 'image', 'fit_to_grid', 'image')).toBe(true);
    expect(hasEdge(edges, 'fit_to_grid', 'image', 'i2l', 'image')).toBe(true);
    expect(Number(nodes.denoise_latents?.width) % 16).toBe(0);
    expect(Number(nodes.denoise_latents?.height) % 16).toBe(0);
  });

  it('leaves the SD path on the size Spandrel produced, which is already the grid it needs', () => {
    const { nodes } = compileUpscaleGraph(createValues('sd-1'), 'gallery', { useCpuNoise: true }).backendGraph;

    expect(nodes.fit_to_grid).toBeUndefined();
  });

  it('resizes a frame the scale already puts on the grid, because a 1x model makes Spandrel ignore the scale', () => {
    // The literal 2048 on the denoise node is a promise only the resize keeps: a restoration model
    // that does not enlarge leaves the frame at 1024, and the latents would not match the node.
    const { edges, nodes } = compileUpscaleGraph(
      { ...createFluxValues(), inputImage: { height: 1024, image_name: 'input.png', width: 1024 }, scale: 2 },
      'gallery',
      { useCpuNoise: true }
    ).backendGraph;

    expect(nodes.fit_to_grid).toMatchObject({ height: 2048, width: 2048 });
    expect(nodes.denoise_latents).toMatchObject({ height: 2048, width: 2048 });
    expect(hasEdge(edges, 'fit_to_grid', 'image', 'i2l', 'image')).toBe(true);
  });

  it('wires no ControlNet, because FLUX.1 has no tile model to anchor with', () => {
    const { nodes } = compileUpscaleGraph(createFluxValues(), 'gallery', { useCpuNoise: true }).backendGraph;

    expect(Object.values(nodes).some((node) => node.type === 'controlnet')).toBe(false);
    expect(nodes.controlnet_collector).toBeUndefined();
  });

  it('names the pass without claiming a multi-diffusion that does not run', () => {
    const flux = compileUpscaleGraph(createFluxValues(), 'gallery', { useCpuNoise: true });
    const sd = compileUpscaleGraph(createValues('sd-1'), 'gallery', { useCpuNoise: true });

    expect(flux.graph.label).not.toContain('multi-diffusion');
    expect(sd.graph.label).toContain('multi-diffusion');
  });

  it('refuses to compile without the text encoders the loader demands', () => {
    expect(() =>
      compileUpscaleGraph(createFluxValues({ withEncoders: false }), 'gallery', { useCpuNoise: true })
    ).toThrow(/T5 encoder/);
  });
});

const withVariant = (config: ModelConfig, variant: string): ModelConfig => ({ ...config, variant }) as ModelConfig;

const createZImageValues = (options: { tileControlNet?: boolean; cfgScale?: number } = {}) => {
  const models = [
    model('main', 'main', 'z-image'),
    model('spandrel', 'spandrel_image_to_image', 'any'),
    model('lora', 'lora', 'z-image'),
    // Z-Image decodes with a FLUX VAE, and reads the 4B Qwen3, never Anima's 0.6B.
    model('fluxvae', 'vae', 'flux'),
    withVariant(model('qwen3-06b', 'qwen3_encoder', 'any'), 'qwen3_06b'),
    withVariant(model('qwen3-4b', 'qwen3_encoder', 'any'), 'qwen3_4b'),
    model('tile', 'controlnet', 'z-image', 'Z-Image ControlNet Tile'),
  ];
  const values = createDefaultUpscaleWidgetValues(models);

  return {
    ...values,
    cfgScale: options.cfgScale ?? 1,
    inputImage: { height: 101, image_name: 'input.png', width: 203 },
    loras: [{ isEnabled: true, model: models[2] as GenerateLora['model'], weight: 0.7 }],
    negativePrompt: 'blur',
    positivePrompt: 'detail',
    tileControlnetModel: options.tileControlNet ? (models[6] as TileControlNetModelConfig) : null,
  };
};

describe('compileUpscaleGraph for Z-Image', () => {
  it('drives the second pass with z-image nodes, tiles both VAE ends and fits the 16px grid', () => {
    const { edges, nodes } = compileUpscaleGraph(createZImageValues(), 'gallery', { useCpuNoise: true }).backendGraph;

    expect(nodes.model_loader).toMatchObject({ type: 'z_image_model_loader' });
    expect(nodes.model_loader?.qwen3_encoder_model).toMatchObject({ key: 'qwen3-4b' });
    expect(nodes.model_loader?.vae_model).toMatchObject({ key: 'fluxvae' });
    expect(nodes.model_loader?.qwen3_source_model).toBeUndefined();
    expect(nodes.i2l).toMatchObject({ tile_size: 1024, tiled: true, type: 'z_image_i2l' });
    expect(nodes.upscale_output).toMatchObject({ tile_size: 1024, tiled: true, type: 'z_image_l2i' });
    // 203x101 at 4x: Spandrel gives 808x400, and z_image_denoise takes multiples of 16 only.
    expect(nodes.fit_to_grid).toMatchObject({ height: 400, width: 800 });
    expect(nodes.denoise_latents).toMatchObject({
      guidance_scale: 1,
      height: 400,
      steps: 30,
      type: 'z_image_denoise',
      width: 800,
    });
    expect(hasEdge(edges, 'fit_to_grid', 'image', 'i2l', 'image')).toBe(true);
    expect(hasEdge(edges, 'i2l', 'latents', 'denoise_latents', 'latents')).toBe(true);
    expect(hasEdge(edges, 'pos_cond_collect', 'collection', 'denoise_latents', 'positive_conditioning')).toBe(true);
    expect(Object.values(nodes).some((node) => node.type === 'z_image_lora_collection_loader')).toBe(true);
  });

  it('gives the negative prompt a branch exactly where CFG runs an unconditional pass, as Generate does', () => {
    const turbo = compileUpscaleGraph(createZImageValues(), 'gallery', { useCpuNoise: true }).backendGraph;
    const withCfg = compileUpscaleGraph(createZImageValues({ cfgScale: 4 }), 'gallery', { useCpuNoise: true });
    // Switched off, the prompt still gets its branch and the queue fills it with an empty string.
    // Leaving the branch out would turn CFG off while the slider and the metadata still say 4.
    const switchedOff = compileUpscaleGraph(
      { ...createZImageValues({ cfgScale: 4 }), negativePromptEnabled: false },
      'gallery',
      { useCpuNoise: true }
    ).backendGraph;

    expect(turbo.nodes.neg_cond).toBeUndefined();
    expect(switchedOff.nodes.neg_cond).toMatchObject({ type: 'z_image_text_encoder' });
    expect(
      hasEdge(withCfg.backendGraph.edges, 'neg_cond_collect', 'collection', 'denoise_latents', 'negative_conditioning')
    ).toBe(true);
    expect(hasEdge(withCfg.backendGraph.edges, 'negative_prompt', 'value', 'core_metadata', 'negative_prompt')).toBe(
      true
    );
    expect(hasEdge(turbo.edges, 'negative_prompt', 'value', 'core_metadata', 'negative_prompt')).toBe(false);
  });

  it('anchors to the enlarged frame with a tiled control encode when a tile ControlNet is chosen', () => {
    const { edges, nodes } = compileUpscaleGraph(
      { ...createZImageValues({ tileControlNet: true }), structure: 0 },
      'gallery',
      { useCpuNoise: true }
    ).backendGraph;

    // The control image is encoded inside the denoise at the frame's full size; untiled, that is
    // the encode the i2l node is tiled to avoid.
    expect(nodes.tile_control).toMatchObject({
      begin_step_percent: 0,
      control_model: { key: 'tile' },
      tile_size: 1024,
      tiled: true,
      type: 'z_image_control',
    });
    expect(nodes.tile_control?.control_context_scale).toBeCloseTo(0.65);
    expect(nodes.tile_control?.end_step_percent).toBeCloseTo(0.75);
    expect(hasEdge(edges, 'fit_to_grid', 'image', 'tile_control', 'image')).toBe(true);
    expect(hasEdge(edges, 'tile_control', 'control', 'denoise_latents', 'control')).toBe(true);
    expect(hasEdge(edges, 'model_loader', 'vae', 'denoise_latents', 'vae')).toBe(true);
  });

  it('runs without a ControlNet when none is chosen', () => {
    const { nodes } = compileUpscaleGraph(createZImageValues(), 'gallery', { useCpuNoise: true }).backendGraph;

    expect(nodes.tile_control).toBeUndefined();
    expect(Object.values(nodes).some((node) => node.type === 'z_image_control')).toBe(false);
  });

  it('spans Structure from a light early nudge to holding the whole pass', () => {
    expect(getZImageTileControlValues(-10)).toEqual({
      beginStepPercent: 0,
      controlContextScale: expect.closeTo(0.3),
      endStepPercent: expect.closeTo(0.5),
    });
    expect(getZImageTileControlValues(10)).toEqual({
      beginStepPercent: 0,
      controlContextScale: expect.closeTo(1),
      endStepPercent: expect.closeTo(1),
    });
  });

  it('keeps an encoder the architecture does not read out of the graph and its metadata', () => {
    // A T5 picked while FLUX was selected stays in the widget for when FLUX comes back.
    const staleT5 = { base: 'any', key: 'stale-t5', name: 'stale-t5', type: 't5_encoder' } as never;
    const { nodes } = compileUpscaleGraph({ ...createZImageValues(), t5EncoderModel: staleT5 }, 'gallery', {
      useCpuNoise: true,
    }).backendGraph;

    expect(JSON.stringify(nodes)).not.toContain('stale-t5');
  });

  it('lets a Diffusers install supply its own encoder and VAE', () => {
    const diffusers = { ...model('main', 'main', 'z-image'), format: 'diffusers' } as ModelConfig;
    const values = {
      ...createDefaultUpscaleWidgetValues([diffusers, model('spandrel', 'spandrel_image_to_image', 'any')]),
      inputImage: { height: 64, image_name: 'input.png', width: 64 },
    };
    const { nodes } = compileUpscaleGraph(values, 'gallery', { useCpuNoise: true }).backendGraph;

    expect(nodes.model_loader?.qwen3_source_model).toMatchObject({ key: 'main' });
    expect(nodes.model_loader?.qwen3_encoder_model).toBeUndefined();
    expect(nodes.model_loader?.vae_model).toBeUndefined();
  });
});

const createAnimaValues = (variant = 'anima_qwen3') => {
  const models = [
    withVariant(model('main', 'main', 'anima'), variant),
    model('spandrel', 'spandrel_image_to_image', 'any'),
    model('qwenvae', 'vae', 'qwen-image'),
    withVariant(model('qwen3-4b', 'qwen3_encoder', 'any'), 'qwen3_4b'),
    withVariant(model('qwen3-06b', 'qwen3_encoder', 'any'), 'qwen3_06b'),
    withVariant(model('qwen35', 'qwen3_5_encoder', 'any'), 'qwen3_5_4b'),
  ];

  return {
    ...createDefaultUpscaleWidgetValues(models),
    cfgScale: 1,
    inputImage: { height: 101, image_name: 'input.png', width: 203 },
    positivePrompt: 'detail',
  };
};

describe('compileUpscaleGraph for Anima', () => {
  it('drives the second pass with anima nodes at the size it states', () => {
    const { edges, nodes } = compileUpscaleGraph(createAnimaValues(), 'gallery', { useCpuNoise: true }).backendGraph;

    expect(nodes.model_loader).toMatchObject({ type: 'anima_model_loader' });
    expect(nodes.model_loader?.qwen3_encoder_model).toMatchObject({ key: 'qwen3-06b' });
    expect(nodes.model_loader?.vae_model).toMatchObject({ key: 'qwenvae' });
    expect(nodes.model_loader?.qwen3_5_encoder_model).toBeUndefined();
    expect(nodes.i2l).toMatchObject({ tile_size: 1024, tiled: true, type: 'anima_i2l' });
    expect(nodes.upscale_output).toMatchObject({ type: 'anima_l2i' });
    // anima_denoise takes multiples of 8, so 808x400 needs no rounding -- but the size is a literal.
    expect(nodes.fit_to_grid).toMatchObject({ height: 400, width: 808 });
    expect(nodes.denoise_latents).toMatchObject({ height: 400, type: 'anima_denoise', width: 808 });
    expect(hasEdge(edges, 'fit_to_grid', 'image', 'i2l', 'image')).toBe(true);
    expect(Object.values(nodes).some((node) => node.type.includes('control'))).toBe(false);
  });

  it('hands Anima-3.8B the Qwen3.5 encoder its connector reads', () => {
    const { edges, nodes } = compileUpscaleGraph(createAnimaValues('anima_qwen35'), 'gallery', {
      useCpuNoise: true,
    }).backendGraph;

    expect(nodes.model_loader?.qwen3_5_encoder_model).toMatchObject({ key: 'qwen35' });
    expect(hasEdge(edges, 'model_loader', 'qwen3_5_encoder', 'pos_cond', 'qwen3_5_encoder')).toBe(true);
    expect(nodes.core_metadata?.qwen3_5_encoder).toMatchObject({ key: 'qwen35' });
  });
});

const createKrea2Values = (variant: 'krea2_turbo' | 'krea2_base', format = 'gguf_quantized') => {
  const models = [
    { ...withVariant(model('main', 'main', 'krea-2'), variant), format } as ModelConfig,
    model('spandrel', 'spandrel_image_to_image', 'any'),
    model('qwenvae', 'vae', 'qwen-image'),
    // Ideogram 4's 8B is the same model type and sorts first; `krea2_model_loader` refuses it.
    withVariant(model('qwen3vl-8b', 'qwen3_vl_encoder', 'any'), 'qwen3_vl_8b'),
    withVariant(model('qwen3vl-4b', 'qwen3_vl_encoder', 'any'), 'qwen3_vl_4b'),
  ];

  return {
    ...createDefaultUpscaleWidgetValues(models),
    cfgScale: variant === 'krea2_base' ? 5.5 : 1,
    inputImage: { height: 101, image_name: 'input.png', width: 203 },
    positivePrompt: 'detail',
  };
};

describe('compileUpscaleGraph for Krea-2', () => {
  it('drives the second pass with krea2 nodes and the Qwen-Image VAE, tiled at both ends', () => {
    const { edges, nodes } = compileUpscaleGraph(createKrea2Values('krea2_base'), 'gallery', {
      useCpuNoise: true,
    }).backendGraph;

    expect(nodes.model_loader).toMatchObject({ type: 'krea2_model_loader' });
    expect(nodes.model_loader?.qwen3_vl_encoder_model).toMatchObject({ key: 'qwen3vl-4b' });
    expect(nodes.model_loader?.vae_model).toMatchObject({ key: 'qwenvae' });
    expect(nodes.i2l).toMatchObject({ tile_size: 1024, tiled: true, type: 'qwen_image_i2l' });
    expect(nodes.upscale_output).toMatchObject({ tile_size: 1024, tiled: true, type: 'qwen_image_l2i' });
    // 203x101 at 4x: Spandrel gives 808x400, and krea2_denoise takes multiples of 16 only.
    expect(nodes.denoise_latents).toMatchObject({
      cfg_scale: 5.5,
      height: 400,
      steps: 30,
      type: 'krea2_denoise',
      width: 800,
    });
    // The node has no scheduler, and picks its own shift: fixed for Turbo, by size for Raw.
    expect(nodes.denoise_latents).not.toHaveProperty('scheduler');
    expect(nodes.denoise_latents).not.toHaveProperty('shift');
    expect(nodes.core_metadata).toMatchObject({ qwen3_vl_encoder: { key: 'qwen3vl-4b' }, scheduler: 'euler' });
    expect(hasEdge(edges, 'fit_to_grid', 'image', 'i2l', 'image')).toBe(true);
    expect(hasEdge(edges, 'i2l', 'latents', 'denoise_latents', 'latents')).toBe(true);
    expect(hasEdge(edges, 'neg_cond_collect', 'collection', 'denoise_latents', 'negative_conditioning')).toBe(true);
    expect(hasEdge(edges, 'model_loader', 'qwen3_vl_encoder', 'pos_cond', 'qwen3_vl_encoder')).toBe(true);
  });

  it('runs Turbo at CFG 1 without the unconditional branch it would not use', () => {
    const { nodes } = compileUpscaleGraph(createKrea2Values('krea2_turbo'), 'gallery', {
      useCpuNoise: true,
    }).backendGraph;

    expect(nodes.denoise_latents?.cfg_scale).toBe(1);
    expect(nodes.neg_cond).toBeUndefined();
  });

  it('lets a Diffusers install supply its own encoder and VAE', () => {
    const values = { ...createKrea2Values('krea2_turbo', 'diffusers'), vae: null };
    const { nodes } = compileUpscaleGraph(values, 'gallery', { useCpuNoise: true }).backendGraph;

    expect(nodes.model_loader?.qwen3_vl_encoder_model).toBeUndefined();
    expect(nodes.model_loader?.vae_model).toBeUndefined();
  });
});

const createFlux2Values = (variant: string, format = 'checkpoint') => {
  const models = [
    { ...withVariant(model('main', 'main', 'flux2'), variant), format } as ModelConfig,
    model('spandrel', 'spandrel_image_to_image', 'any'),
    model('lora', 'lora', 'flux2'),
    model('flux2vae', 'vae', 'flux2'),
    // The 8B and the Ministral sort ahead of the encoders Klein 4B and [dev] take.
    withVariant(model('qwen3-8b', 'qwen3_encoder', 'any'), 'qwen3_8b'),
    withVariant(model('qwen3-4b', 'qwen3_encoder', 'any'), 'qwen3_4b'),
    withVariant(model('ministral', 'mistral_encoder', 'any'), 'ministral3_3b'),
    withVariant(model('mistral', 'mistral_encoder', 'any'), 'mistral3_24b'),
  ];

  return {
    ...createDefaultUpscaleWidgetValues(models),
    cfgScale: 4,
    inputImage: { height: 101, image_name: 'input.png', width: 203 },
    loras: [{ isEnabled: true, model: models[2] as GenerateLora['model'], weight: 0.7 }],
    negativePrompt: 'blur',
    positivePrompt: 'detail',
  };
};

describe('compileUpscaleGraph for FLUX.2', () => {
  it('drives Klein with its own nodes, the guidance and the VAE statistics, tiled at both ends', () => {
    const { edges, nodes } = compileUpscaleGraph(createFlux2Values('klein_4b'), 'gallery', {
      useCpuNoise: true,
    }).backendGraph;

    expect(nodes.model_loader).toMatchObject({
      qwen3_encoder_model: { key: 'qwen3-4b' },
      type: 'flux2_klein_model_loader',
      vae_model: { key: 'flux2vae' },
    });
    expect(nodes.i2l).toMatchObject({ tile_size: 1024, tiled: true, type: 'flux2_vae_encode' });
    expect(nodes.upscale_output).toMatchObject({ tile_size: 1024, tiled: true, type: 'flux2_vae_decode' });
    // 203x101 at 4x: Spandrel gives 808x400, and flux2_denoise takes multiples of 16 only. A new Klein runs at
    // the 4 steps it is served with.
    expect(nodes.denoise_latents).toMatchObject({
      guidance: 4,
      height: 400,
      num_steps: 4,
      scheduler: 'euler',
      type: 'flux2_denoise',
      width: 800,
    });
    // CFG stays at the node's 1, as in Generate: the slider is guidance here, and there is no negative branch.
    expect(nodes.denoise_latents).not.toHaveProperty('cfg_scale');
    expect(nodes.neg_cond).toBeUndefined();
    expect(nodes.core_metadata).toMatchObject({ guidance: 4, qwen3_encoder: { key: 'qwen3-4b' } });
    expect(nodes.core_metadata).not.toHaveProperty('cfg_scale');
    expect(hasEdge(edges, 'negative_prompt', 'value', 'core_metadata', 'negative_prompt')).toBe(false);
    // The denoise normalizes the encoded latents with the VAE's BN statistics.
    expect(hasEdge(edges, 'model_loader', 'vae', 'denoise_latents', 'vae')).toBe(true);
    expect(hasEdge(edges, 'model_loader', 'max_seq_len', 'pos_cond', 'max_seq_len')).toBe(true);
    expect(hasEdge(edges, 'pos_cond_collect', 'collection', 'denoise_latents', 'positive_text_conditioning')).toBe(
      true
    );
    expect(hasEdge(edges, 'fit_to_grid', 'image', 'i2l', 'image')).toBe(true);
    expect(nodes.pos_cond).toMatchObject({ type: 'flux2_klein_text_encoder' });
    expect(Object.values(nodes).some((node) => node.type === 'flux2_klein_lora_collection_loader')).toBe(true);
  });

  it('hands Klein 9B the 8B encoder it was trained on', () => {
    const { nodes } = compileUpscaleGraph(createFlux2Values('klein_9b_base'), 'gallery', {
      useCpuNoise: true,
    }).backendGraph;

    expect(nodes.model_loader?.qwen3_encoder_model).toMatchObject({ key: 'qwen3-8b' });
  });

  it('drives [dev] with its Mistral encoder, never the Ministral, and records it', () => {
    const { edges, nodes } = compileUpscaleGraph(createFlux2Values('dev', 'gguf_quantized'), 'gallery', {
      useCpuNoise: true,
    }).backendGraph;
    const loraLoader = Object.entries(nodes).find(([, node]) => node.type === 'flux2_dev_lora_collection_loader');

    expect(nodes.model_loader).toMatchObject({
      mistral_encoder_model: { key: 'mistral' },
      type: 'flux2_dev_model_loader',
    });
    expect(nodes.model_loader).not.toHaveProperty('qwen3_encoder_model');
    expect(nodes.pos_cond).toMatchObject({ type: 'flux2_dev_text_encoder' });
    expect(loraLoader).toBeDefined();
    expect(hasEdge(edges, loraLoader![0], 'mistral_encoder', 'pos_cond', 'mistral_encoder')).toBe(true);
    expect(hasEdge(edges, 'model_loader', 'max_seq_len', 'pos_cond', 'max_seq_len')).toBe(true);
    expect(nodes.core_metadata).toMatchObject({ mistral_encoder: { key: 'mistral' } });
    expect(nodes.core_metadata).not.toHaveProperty('qwen3_encoder');
  });

  it('lets a Diffusers install supply its own encoder and VAE', () => {
    const values = { ...createFlux2Values('klein_4b', 'diffusers'), vae: null };
    const { nodes } = compileUpscaleGraph(values, 'gallery', { useCpuNoise: true }).backendGraph;

    expect(nodes.model_loader?.qwen3_encoder_model).toBeUndefined();
    expect(nodes.model_loader?.vae_model).toBeUndefined();
  });
});

const createQwenImageValues = (options: { cfgScale?: number; format?: string } = {}) => {
  const models = [
    { ...withVariant(model('main', 'main', 'qwen-image'), 'generate'), format: options.format ?? 'gguf_quantized' },
    model('spandrel', 'spandrel_image_to_image', 'any'),
    model('lora', 'lora', 'qwen-image'),
    model('qwenvae', 'vae', 'qwen-image'),
    model('qwen-vl', 'qwen_vl_encoder', 'any'),
  ] as ModelConfig[];

  return {
    ...createDefaultUpscaleWidgetValues(models),
    cfgScale: options.cfgScale ?? 4,
    inputImage: { height: 101, image_name: 'input.png', width: 203 },
    loras: [{ isEnabled: true, model: models[2] as GenerateLora['model'], weight: 0.7 }],
    negativePrompt: 'blur',
    positivePrompt: 'detail',
  };
};

describe('compileUpscaleGraph for Qwen Image', () => {
  it('drives the second pass with qwen_image nodes, tiled at both ends, on the shift the node picks', () => {
    const { edges, nodes } = compileUpscaleGraph(createQwenImageValues(), 'gallery', {
      useCpuNoise: true,
    }).backendGraph;

    expect(nodes.model_loader).toMatchObject({
      qwen_vl_encoder_model: { key: 'qwen-vl' },
      type: 'qwen_image_model_loader',
      vae_model: { key: 'qwenvae' },
    });
    expect(nodes.i2l).toMatchObject({ tile_size: 1024, tiled: true, type: 'qwen_image_i2l' });
    expect(nodes.upscale_output).toMatchObject({ tile_size: 1024, tiled: true, type: 'qwen_image_l2i' });
    expect(nodes.denoise_latents).toMatchObject({
      cfg_scale: 4,
      height: 400,
      steps: 30,
      type: 'qwen_image_denoise',
      width: 800,
    });
    // A shift would replace the held dynamic one that Creativity is translated against.
    expect(nodes.denoise_latents).not.toHaveProperty('shift');
    expect(nodes.denoise_latents).not.toHaveProperty('scheduler');
    // The denoise takes one conditioning per branch, not a collected list.
    expect(hasEdge(edges, 'pos_cond', 'conditioning', 'denoise_latents', 'positive_conditioning')).toBe(true);
    expect(hasEdge(edges, 'neg_cond', 'conditioning', 'denoise_latents', 'negative_conditioning')).toBe(true);
    expect(nodes.pos_cond_collect).toBeUndefined();
    expect(hasEdge(edges, 'model_loader', 'qwen_vl_encoder', 'neg_cond', 'qwen_vl_encoder')).toBe(true);
    expect(Object.values(nodes).some((node) => node.type === 'qwen_image_lora_collection_loader')).toBe(true);
    expect(nodes.core_metadata).toMatchObject({
      cfg_scale: 4,
      qwen_image_qwen_vl_encoder: { key: 'qwen-vl' },
      scheduler: 'euler',
    });
  });

  it('wires the negative prompt only above CFG 1, where the node runs an unconditional pass', () => {
    const { edges, nodes } = compileUpscaleGraph(createQwenImageValues({ cfgScale: 1 }), 'gallery', {
      useCpuNoise: true,
    }).backendGraph;

    expect(nodes.neg_cond).toBeUndefined();
    expect(hasEdge(edges, 'negative_prompt', 'value', 'core_metadata', 'negative_prompt')).toBe(false);
  });

  it('lets a Diffusers install supply its own encoder and VAE', () => {
    const values = { ...createQwenImageValues({ format: 'diffusers' }), vae: null };
    const { nodes } = compileUpscaleGraph(values, 'gallery', { useCpuNoise: true }).backendGraph;

    expect(nodes.model_loader?.qwen_vl_encoder_model).toBeUndefined();
    expect(nodes.model_loader?.vae_model).toBeUndefined();
  });
});

describe('Creativity on index-clipping architectures', () => {
  // Expected starts come from the backend's own schedules (`ZImageDenoiseInvocation._get_sigmas`
  // with `Z_IMAGE_SHIFT_FIT`, `AnimaDenoiseInvocation._get_sigmas`) and the nodes' clipping line,
  // `int(start * steps)`: each enters within one step of the sigma FLUX enters at, 1 - start.
  const startFor = (values: UpscaleWidgetValues, creativity: number, inputSide: number) =>
    compileUpscaleGraph(
      { ...values, creativity, inputImage: { height: inputSide, image_name: 'in.png', width: inputSide }, scale: 2 },
      'gallery',
      { useCpuNoise: true }
    ).backendGraph.nodes.denoise_latents?.denoising_start;

  it.each([
    [-5, 27.5 / 30],
    [0, 23.5 / 30],
    [5, 15.5 / 30],
    [10, 0],
  ])('starts Z-Image at 2048px where FLUX starts for creativity %i', (creativity, expected) => {
    // Today's 0.499 for balanced would enter Z-Image at sigma 0.783; this enters at 0.490.
    expect(startFor(createZImageValues(), creativity, 1024)).toBeCloseTo(expected, 6);
  });

  it("follows Z-Image's resolution-dependent shift below one megapixel", () => {
    expect(startFor(createZImageValues(), 0, 256)).toBeCloseTo(20.5 / 30, 6);
  });

  it.each([
    [0, 22.5 / 30],
    [8, 7.5 / 30],
  ])('starts Anima, whose shift is fixed, where FLUX starts for creativity %i', (creativity, expected) => {
    expect(startFor(createAnimaValues(), creativity, 1024)).toBeCloseTo(expected, 6);
  });

  // `krea2_denoise` rounds `start * steps` instead, so the start is the step itself. Expected steps were
  // worked out from the node's schedule: `exp(mu) * t / (1 + (exp(mu) - 1) * t)` at `t = 1 - i / 30`.
  it.each([
    ['Turbo', 'krea2_turbo', 1024, 0, 23 / 30],
    ['Turbo', 'krea2_turbo', 1024, 5, 15 / 30],
    // 1024px: Raw's fit gives mu 0.906, a flatter schedule than Turbo's 1.15.
    ['Raw', 'krea2_base', 512, 0, 21 / 30],
    // 2048px: past 1280x1280 Raw is held at mu 1.15, Turbo's schedule.
    ['Raw', 'krea2_base', 1024, 0, 23 / 30],
  ] as const)('starts Krea-2 %s from a %ipx input where FLUX starts', (_label, variant, side, creativity, expected) => {
    expect(startFor(createKrea2Values(variant), creativity, side)).toBeCloseTo(expected, 6);
  });

  // `qwen_image_denoise` rounds as well. Its scheduler shifts by `exp(mu)`, mu = 0.5 + 0.4 * (tokens - 256) / 7936
  // held at 0.9 past 8192 tokens, then stretches every sigma to `1 - (1 - s) / k`, `k = (1 - s(1/30)) / 0.98`, so
  // the last step lands on 0.02. Expected steps were worked out by hand from that schedule at `t = 1 - i / 30`.
  it.each([
    // 1024px result, 4096 tokens: mu 0.694, k 0.9545. Step 19 sits at 0.515 and step 20 at 0.476; unstretched
    // they would be 0.537 and 0.500, and the pass would start a step later.
    [512, 19 / 30],
    // 4096px result: held at mu 0.9, k 0.9406. Step 20 sits at 0.523, step 21 at 0.482.
    [2048, 21 / 30],
  ])('starts Qwen Image from a %ipx input where FLUX starts', (side, expected) => {
    expect(startFor(createQwenImageValues(), 0, side)).toBeCloseTo(expected, 6);
  });

  it.each([
    ['FLUX.1', () => createFluxValues()],
    // `flux2_denoise` clips by value too, then re-spaces the window to the requested steps.
    ['FLUX.2', () => createFlux2Values('klein_4b')],
  ] as const)('leaves the value-clipping %s pass on the legacy formula', (_label, create) => {
    expect(startFor(create(), 0, 1024)).toBeCloseTo(0.499, 6);
  });
});

describe('schedulers on the upscale pass', () => {
  // A stored Heun or LCM on a flow model -- left from SD, recalled, or picked before the list was
  // narrowed -- would start the pass from pure noise, or on Anima below the blend.
  it.each([
    ['flux', () => createFluxValues(), 'lcm', 'euler'],
    ['flux2', () => createFlux2Values('klein_4b'), 'heun', 'euler'],
    ['z-image', () => createZImageValues(), 'heun', 'euler'],
    ['anima', () => createAnimaValues(), 'heun', 'euler'],
    ['anima', () => createAnimaValues(), 'dpmpp_2m', 'dpmpp_2m'],
    ['sd-1', () => createValues('sd-1'), 'heun', 'heun'],
  ] as const)('runs %s with %s stored as %s, and records what ran', (_base, createBaseValues, stored, runs) => {
    const { nodes } = compileUpscaleGraph({ ...createBaseValues(), scheduler: stored }, 'gallery', {
      useCpuNoise: true,
    }).backendGraph;
    const denoise = nodes.denoise_latents ?? nodes.tiled_multidiffusion_denoise_latents;

    expect([denoise?.scheduler, nodes.core_metadata?.scheduler]).toEqual([runs, runs]);
  });
});

describe('upscale metadata', () => {
  it.each([
    ['sd-1', () => createValues('sd-1')],
    ['flux', () => createFluxValues()],
    ['z-image', () => createZImageValues()],
    ['anima', () => createAnimaValues()],
    ['krea-2', () => createKrea2Values('krea2_base')],
    ['flux2', () => createFlux2Values('klein_4b')],
    ['qwen-image', () => createQwenImageValues()],
  ] as const)('records the size of the image the %s pass encoded, which is the size of the result', (_base, create) => {
    // Where a pass resizes Spandrel's frame, the result does not have Spandrel's size: a 1x
    // restoration model leaves the frame unscaled, and its multiple of 8 is not FLUX's or Z-Image's
    // 16. Recall restores the size recorded here.
    const { edges } = compileUpscaleGraph(create(), 'gallery', { useCpuNoise: true }).backendGraph;
    const sourceOf = (nodeId: string, field: string) =>
      edges.find((edge) => edge.destination.node_id === nodeId && edge.destination.field === field)?.source;
    const encoded = sourceOf('i2l', 'image')?.node_id;

    expect(encoded).toBeDefined();
    expect(sourceOf('core_metadata', 'width')).toEqual({ field: 'width', node_id: encoded });
    expect(sourceOf('core_metadata', 'height')).toEqual({ field: 'height', node_id: encoded });
  });

  it.each([
    ['sd-1', () => createValues('sd-1')],
    ['sdxl', () => createValues('sdxl')],
    ['flux', () => createFluxValues()],
    ['z-image', () => createZImageValues()],
  ] as const)('records exactly the concepts the %s pass loads', (_base, createBaseValues) => {
    const values = createBaseValues();
    const { nodes } = compileUpscaleGraph(
      {
        ...values,
        loras: [
          ...values.loras,
          { isEnabled: false, model: { ...values.loras[0]!.model, key: 'disabled' }, weight: 1 },
          { isEnabled: true, model: { ...model('other', 'lora', 'sd-3'), type: 'lora' }, weight: 1 },
        ],
      },
      'gallery',
      { useCpuNoise: true }
    ).backendGraph;
    const loaded = Object.values(nodes)
      .filter((node) => node.type === 'lora_selector')
      .map((node) => [(node.lora as { key: string }).key, node.weight]);
    const recordedLoras = (nodes.core_metadata?.loras ?? []) as { model: { key: string }; weight: number }[];
    const recorded = recordedLoras.map((entry) => [entry.model.key, entry.weight]);

    expect(loaded).toEqual([[values.loras[0]!.model.key, 0.7]]);
    expect(recorded).toEqual(loaded);
  });
});
