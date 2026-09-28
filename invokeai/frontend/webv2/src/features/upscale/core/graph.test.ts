import type { GenerateLora, VaeModelConfig } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';

import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { describe, expect, it } from 'vitest';

import { compileUpscaleGraph, getUpscaleControlNetValues, getUpscaleDenoisingStart } from './graph';
import { createDefaultUpscaleWidgetValues } from './settings';

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
    expect(hasEdge(edges, 'spandrel_autoscale', 'width', 'core_metadata', 'width')).toBe(true);
    expect(hasEdge(edges, 'spandrel_autoscale', 'height', 'core_metadata', 'height')).toBe(true);
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
