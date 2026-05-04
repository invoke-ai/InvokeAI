import type { RootState } from 'app/store/store';
import type { HrfMethod } from 'features/controlLayers/store/types';
import { describe, expect, it } from 'vitest';

import { addHighResFix } from './addHighResFix';
import { Graph } from './Graph';

const buildState = (overrides?: {
  base?: string;
  hrfEnabled?: boolean;
  hrfMethod?: HrfMethod;
  refinerModel?: unknown;
}): RootState =>
  ({
    ui: { activeTab: 'generate' },
    params: {
      model: { key: 'model', hash: 'model-hash', name: 'model', base: overrides?.base ?? 'sdxl', type: 'main' },
      dimensions: { width: 512, height: 512 },
      hrfEnabled: overrides?.hrfEnabled ?? true,
      hrfMethod: overrides?.hrfMethod ?? 'latent',
      hrfScale: 2,
      hrfStrength: 0.35,
      hrfLatentInterpolationMode: 'bilinear',
      hrfUpscaleModel: {
        key: 'upscale',
        hash: 'upscale-hash',
        name: 'upscale',
        base: 'any',
        type: 'spandrel_image_to_image',
      },
      hrfTileControlNetModel: {
        key: 'tile',
        hash: 'tile-hash',
        name: 'tile',
        base: overrides?.base ?? 'sdxl',
        type: 'controlnet',
      },
      hrfStructure: 0,
      hrfTileControlEnd: 0.2,
      hrfTileSize: 1024,
      hrfTileOverlap: 128,
      optimizedDenoisingEnabled: true,
      refinerModel: overrides?.refinerModel ?? null,
    },
  }) as unknown as RootState;

const buildClassicGraph = () => {
  const g = new Graph('test_graph');
  const seed = g.addNode({ id: 'seed', type: 'integer' });
  const noise = g.addNode({ id: 'noise', type: 'noise', use_cpu: true, width: 512, height: 512 });
  const modelLoader = g.addNode({
    id: 'model_loader',
    type: 'sdxl_model_loader',
    model: { key: 'model', hash: 'model-hash', name: 'model', base: 'sdxl', type: 'main' },
  });
  const denoise = g.addNode({
    id: 'denoise',
    type: 'denoise_latents',
    cfg_scale: 7.5,
    scheduler: 'euler',
    steps: 30,
  });
  const l2i = g.addNode({ id: 'l2i', type: 'l2i', fp32: true });

  g.addEdge(seed, 'value', noise, 'seed');
  g.addEdge(noise, 'noise', denoise, 'noise');
  g.addEdge(modelLoader, 'unet', denoise, 'unet');
  g.addEdge(modelLoader, 'vae', l2i, 'vae');
  g.addEdge(denoise, 'latents', l2i, 'latents');

  return { g, seed, noise, modelLoader, denoise, l2i };
};

const addSDXLConditioning = (g: Graph, denoise: ReturnType<typeof buildClassicGraph>['denoise']) => {
  const posCond = g.addNode({
    id: 'pos_cond',
    type: 'sdxl_compel_prompt',
    original_width: 512,
    original_height: 512,
    target_width: 512,
    target_height: 512,
  });
  const posCollect = g.addNode({ id: 'pos_collect', type: 'collect' });

  g.addEdge(posCond, 'conditioning', posCollect, 'item');
  g.addEdge(posCollect, 'collection', denoise, 'positive_conditioning');

  return { posCond, posCollect };
};

const buildTransformerGraph = () => {
  const g = new Graph('test_transformer_graph');
  const seed = g.addNode({ id: 'seed', type: 'integer' });
  const denoise = g.addNode({
    id: 'sd3_denoise',
    type: 'sd3_denoise',
    cfg_scale: 4,
    width: 512,
    height: 512,
    steps: 20,
    denoising_start: 0,
    denoising_end: 1,
  });
  const l2i = g.addNode({ id: 'sd3_l2i', type: 'sd3_l2i' });

  g.addEdge(seed, 'value', denoise, 'seed');
  g.addEdge(denoise, 'latents', l2i, 'latents');

  return { g, seed, denoise, l2i };
};

describe('addHighResFix', () => {
  it('reroutes classic txt2img graphs through latent resize and a second denoise pass', () => {
    const { g, seed, noise, denoise, l2i } = buildClassicGraph();

    addHighResFix({ g, state: buildState(), generationMode: 'txt2img', denoise, l2i, noise, seed });

    const graph = g.getGraph();
    const nodes = Object.values(graph.nodes);
    const resize = nodes.find((node) => node.type === 'lresize');
    const hrfDenoise = nodes.find((node) => node.id.startsWith('hrf_denoise_latents'));
    const hrfNoise = nodes.find((node) => node.id.startsWith('hrf_noise'));

    expect(resize).toMatchObject({ type: 'lresize', width: 1024, height: 1024, mode: 'bilinear' });
    expect(hrfDenoise).toMatchObject({ type: 'denoise_latents', denoising_start: 0.65, denoising_end: 1 });
    expect(hrfNoise).toMatchObject({ type: 'noise', width: 1024, height: 1024, use_cpu: true });
    expect(graph.edges).not.toContainEqual({
      source: { node_id: 'denoise', field: 'latents' },
      destination: { node_id: 'l2i', field: 'latents' },
    });
    expect(g.getMetadataNode()).toMatchObject({
      width: 1024,
      height: 1024,
      hrf_enabled: true,
      hrf_method: 'latent',
      hrf_strength: 0.35,
      hrf_scale: 2,
      hrf_latent_interpolation_mode: 'bilinear',
    });
  });

  it('preserves the original graph and writes disabled metadata when HRF is off', () => {
    const { g, seed, noise, denoise, l2i } = buildClassicGraph();

    addHighResFix({
      g,
      state: buildState({ hrfEnabled: false }),
      generationMode: 'txt2img',
      denoise,
      l2i,
      noise,
      seed,
    });

    const graph = g.getGraph();
    expect(Object.values(graph.nodes).some((node) => node.type === 'lresize')).toBe(false);
    expect(graph.edges).toContainEqual({
      source: { node_id: 'denoise', field: 'latents' },
      destination: { node_id: 'l2i', field: 'latents' },
    });
    expect(g.getMetadataNode()).toMatchObject({ hrf_enabled: false });
  });

  it('reroutes transformer txt2img graphs through latent resize and a final-size second denoise pass', () => {
    const { g, seed, denoise, l2i } = buildTransformerGraph();

    addHighResFix({ g, state: buildState({ base: 'sd-3' }), generationMode: 'txt2img', denoise, l2i, seed });

    const graph = g.getGraph();
    const nodes = Object.values(graph.nodes);
    const resize = nodes.find((node) => node.type === 'lresize');
    const hrfDenoise = nodes.find((node) => node.id.startsWith('hrf_sd3_denoise'));

    if (!hrfDenoise) {
      throw new Error('Expected HRF SD3 denoise node');
    }

    expect(resize).toMatchObject({ type: 'lresize', width: 1024, height: 1024, mode: 'bilinear' });
    expect(hrfDenoise).toMatchObject({ type: 'sd3_denoise', width: 1024, height: 1024, denoising_end: 1 });
    expect((hrfDenoise as { denoising_start: number }).denoising_start).toBeCloseTo(1 - 0.35 ** 0.2);
    expect(nodes.some((node) => node.id.startsWith('hrf_noise'))).toBe(false);
    expect(graph.edges).toContainEqual({
      source: { node_id: 'seed', field: 'value' },
      destination: { node_id: hrfDenoise.id, field: 'seed' },
    });
    expect(graph.edges).not.toContainEqual({
      source: { node_id: 'sd3_denoise', field: 'latents' },
      destination: { node_id: 'sd3_l2i', field: 'latents' },
    });
  });

  it('clones SDXL conditioning with final HRF dimensions for the second pass', () => {
    const { g, seed, noise, denoise, l2i } = buildClassicGraph();
    const { posCond, posCollect } = addSDXLConditioning(g, denoise);

    addHighResFix({ g, state: buildState(), generationMode: 'txt2img', denoise, l2i, noise, seed });

    const graph = g.getGraph();
    const nodes = Object.values(graph.nodes);
    const hrfDenoise = nodes.find((node) => node.id.startsWith('hrf_denoise_latents'));
    const hrfPosCond = nodes.find((node) => node.id.startsWith('hrf_pos_cond'));
    const hrfPosCollect = nodes.find((node) => node.id.startsWith('hrf_sdxl_conditioning_collect'));

    if (!hrfDenoise || !hrfPosCond || !hrfPosCollect) {
      throw new Error('Expected HRF denoise and cloned SDXL conditioning nodes');
    }

    expect(posCond).toMatchObject({ original_width: 512, original_height: 512 });
    expect(hrfPosCond).toMatchObject({
      type: 'sdxl_compel_prompt',
      original_width: 1024,
      original_height: 1024,
      target_width: 1024,
      target_height: 1024,
    });
    expect(graph.edges).toContainEqual({
      source: { node_id: hrfPosCond.id, field: 'conditioning' },
      destination: { node_id: hrfPosCollect.id, field: 'item' },
    });
    expect(graph.edges).toContainEqual({
      source: { node_id: hrfPosCollect.id, field: 'collection' },
      destination: { node_id: hrfDenoise.id, field: 'positive_conditioning' },
    });
    expect(graph.edges).not.toContainEqual({
      source: { node_id: posCollect.id, field: 'collection' },
      destination: { node_id: hrfDenoise.id, field: 'positive_conditioning' },
    });
  });

  it('reroutes SDXL txt2img graphs through an upscale model, tiled encode, and tiled second pass', () => {
    const { g, seed, noise, denoise, l2i } = buildClassicGraph();
    addSDXLConditioning(g, denoise);

    addHighResFix({
      g,
      state: buildState({ hrfMethod: 'upscale_model' }),
      generationMode: 'txt2img',
      denoise,
      l2i,
      noise,
      seed,
    });

    const graph = g.getGraph();
    const nodes = Object.values(graph.nodes);
    const intermediateL2i = nodes.find((node) => node.id.startsWith('hrf_intermediate_l2i'));
    const spandrel = nodes.find((node) => node.type === 'spandrel_image_to_image_autoscale');
    const unsharp = nodes.find((node) => node.type === 'unsharp_mask');
    const i2l = nodes.find((node) => node.id.startsWith('hrf_i2l'));
    const tiledDenoise = nodes.find((node) => node.type === 'tiled_multi_diffusion_denoise_latents');
    const tileControlNet = nodes.find((node) => node.id.startsWith('hrf_controlnet'));

    if (!intermediateL2i || !spandrel || !unsharp || !i2l || !tiledDenoise || !tileControlNet) {
      throw new Error('Expected upscale-model HRF nodes');
    }

    expect(nodes.some((node) => node.type === 'lresize')).toBe(false);
    expect(intermediateL2i).toMatchObject({ type: 'l2i', is_intermediate: true });
    expect(spandrel).toMatchObject({
      type: 'spandrel_image_to_image_autoscale',
      image_to_image_model: { key: 'upscale' },
      scale: 2,
      tile_size: 1024,
      fit_to_multiple_of_8: true,
    });
    expect(i2l).toMatchObject({ type: 'i2l', tiled: true, tile_size: 1024 });
    expect(l2i).toMatchObject({ type: 'l2i', tiled: true, tile_size: 1024 });
    expect(tiledDenoise).toMatchObject({
      type: 'tiled_multi_diffusion_denoise_latents',
      tile_height: 1024,
      tile_width: 1024,
      tile_overlap: 128,
      denoising_start: 0.65,
      denoising_end: 1,
    });
    expect(tileControlNet).toMatchObject({
      type: 'controlnet',
      begin_step_percent: 0,
      end_step_percent: 0.2,
    });
    expect(graph.edges).toContainEqual({
      source: { node_id: tileControlNet.id, field: 'control' },
      destination: { node_id: tiledDenoise.id, field: 'control' },
    });
    const positiveConditioningEdge = graph.edges.find(
      (edge) => edge.destination.node_id === tiledDenoise.id && edge.destination.field === 'positive_conditioning'
    );
    if (!positiveConditioningEdge) {
      throw new Error('Expected positive conditioning edge');
    }
    expect(graph.nodes[positiveConditioningEdge.source.node_id]?.type).not.toBe('collect');
    expect(graph.edges).not.toContainEqual({
      source: { node_id: 'denoise', field: 'latents' },
      destination: { node_id: 'l2i', field: 'latents' },
    });
    expect(graph.edges).toContainEqual({
      source: { node_id: 'denoise', field: 'latents' },
      destination: { node_id: intermediateL2i.id, field: 'latents' },
    });
    expect(graph.edges).toContainEqual({
      source: { node_id: tiledDenoise.id, field: 'latents' },
      destination: { node_id: 'l2i', field: 'latents' },
    });
    expect(g.getMetadataNode()).toMatchObject({
      hrf_enabled: true,
      hrf_method: 'upscale_model',
      hrf_upscale_model: { key: 'upscale' },
      hrf_tile_controlnet_model: { key: 'tile' },
      hrf_structure: 0,
      hrf_tile_control_end: 0.2,
      hrf_tile_size: 1024,
      hrf_tile_overlap: 128,
    });
  });

  it('uses a regular second denoise for upscale-model HRF when reference image adapters are connected', () => {
    const { g, seed, noise, denoise, l2i } = buildClassicGraph();
    addSDXLConditioning(g, denoise);

    const ipAdapter = g.addNode({
      id: 'ip_adapter',
      type: 'ip_adapter',
      weight: 1,
      method: 'full',
      ip_adapter_model: { key: 'ip', hash: 'ip-hash', name: 'ip', base: 'sdxl', type: 'ip_adapter' },
      clip_vision_model: 'ViT-H',
      begin_step_percent: 0,
      end_step_percent: 1,
      image: { image_name: 'test' },
    });
    const ipAdapterCollector = g.addNode({ id: 'ip_adapter_collector', type: 'collect' });
    g.addEdge(ipAdapter, 'ip_adapter', ipAdapterCollector, 'item');
    g.addEdge(ipAdapterCollector, 'collection', denoise, 'ip_adapter');

    addHighResFix({
      g,
      state: buildState({ hrfMethod: 'upscale_model' }),
      generationMode: 'txt2img',
      denoise,
      l2i,
      noise,
      seed,
    });

    const graph = g.getGraph();
    const nodes = Object.values(graph.nodes);
    const classicHrfDenoise = nodes.find((node) => node.id.startsWith('hrf_denoise_latents'));

    if (!classicHrfDenoise) {
      throw new Error('Expected classic HRF denoise node');
    }

    expect(nodes.some((node) => node.type === 'tiled_multi_diffusion_denoise_latents')).toBe(false);
    expect(graph.edges).toContainEqual({
      source: { node_id: ipAdapterCollector.id, field: 'collection' },
      destination: { node_id: classicHrfDenoise.id, field: 'ip_adapter' },
    });
    expect(graph.edges).toContainEqual({
      source: { node_id: classicHrfDenoise.id, field: 'latents' },
      destination: { node_id: 'l2i', field: 'latents' },
    });
  });
});
