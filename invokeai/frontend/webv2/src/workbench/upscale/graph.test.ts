import type { GenerateLora, VaeModelConfig } from '@workbench/generation/types';
import type { ModelConfig } from '@workbench/models/types';

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
});
