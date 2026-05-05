import type { RootState } from 'app/store/store';
import type { HrfMethod } from 'features/controlLayers/store/types';
import { describe, expect, it } from 'vitest';

import { addHighResFix } from './addHighResFix';
import { Graph } from './Graph';

const buildState = (overrides?: {
  base?: string;
  hrfEnabled?: boolean;
  hrfMethod?: HrfMethod;
  hrfSteps?: number | null;
  hrfModel?: unknown;
  hrfLoraMode?: 'reuse_generate' | 'none' | 'dedicated';
  hrfLoras?: unknown[];
  hrfTileSize?: number;
  loras?: unknown[];
  refinerModel?: unknown;
}): RootState =>
  ({
    ui: { activeTab: 'generate' },
    loras: { loras: overrides?.loras ?? [] },
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
      hrfTileControlWeight: 0.625,
      hrfTileControlEnd: 0.2,
      hrfTileSize: overrides?.hrfTileSize ?? 1024,
      hrfTileOverlap: 128,
      hrfSteps: overrides?.hrfSteps ?? null,
      hrfModel: overrides?.hrfModel ?? null,
      hrfLoraMode: overrides?.hrfLoraMode ?? 'reuse_generate',
      hrfLoras: overrides?.hrfLoras ?? [],
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
  const negCond = g.addNode({
    id: 'neg_cond',
    type: 'sdxl_compel_prompt',
    prompt: 'negative',
    style: 'negative',
    original_width: 512,
    original_height: 512,
    target_width: 512,
    target_height: 512,
  });
  const negCollect = g.addNode({ id: 'neg_collect', type: 'collect' });

  g.addEdge(posCond, 'conditioning', posCollect, 'item');
  g.addEdge(posCollect, 'collection', denoise, 'positive_conditioning');
  g.addEdge(negCond, 'conditioning', negCollect, 'item');
  g.addEdge(negCollect, 'collection', denoise, 'negative_conditioning');

  return { posCond, posCollect, negCond, negCollect };
};

const addSDXLRegionalConditioning = (g: Graph, conditioning: ReturnType<typeof addSDXLConditioning>) => {
  const mask = g.addNode({
    id: 'region_mask',
    type: 'alpha_mask_to_tensor',
    image: { image_name: 'region-mask' },
  });
  const regionPrompt = g.addNode({ id: 'region_prompt', type: 'string', value: 'regional prompt' });
  const regionPosCond = g.addNode({
    id: 'region_pos_cond',
    type: 'sdxl_compel_prompt',
    style: 'regional positive style',
    original_width: 512,
    original_height: 512,
    target_width: 512,
    target_height: 512,
  });
  const regionNegCond = g.addNode({
    id: 'region_neg_cond',
    type: 'sdxl_compel_prompt',
    prompt: 'regional negative',
    style: 'regional negative style',
    original_width: 512,
    original_height: 512,
    target_width: 512,
    target_height: 512,
  });

  g.addEdge(regionPrompt, 'value', regionPosCond, 'prompt');
  g.addEdgeFromObj({
    source: { node_id: mask.id, field: 'mask' },
    destination: { node_id: regionPosCond.id, field: 'mask' },
  });
  g.addEdgeFromObj({
    source: { node_id: mask.id, field: 'mask' },
    destination: { node_id: regionNegCond.id, field: 'mask' },
  });
  g.addEdge(regionPosCond, 'conditioning', conditioning.posCollect, 'item');
  g.addEdge(regionNegCond, 'conditioning', conditioning.negCollect, 'item');

  return { mask, regionPrompt, regionPosCond, regionNegCond };
};

const addSeamlessToClassicGraph = ({
  g,
  modelLoader,
  denoise,
}: Pick<ReturnType<typeof buildClassicGraph>, 'g' | 'modelLoader' | 'denoise'>) => {
  const seamless = g.addNode({
    id: 'seamless',
    type: 'seamless',
    seamless_x: true,
    seamless_y: false,
  });

  g.deleteEdgesTo(denoise, ['unet']);
  g.addEdge(modelLoader, 'unet', seamless, 'unet');
  g.addEdge(modelLoader, 'vae', seamless, 'vae');
  g.addEdge(seamless, 'unet', denoise, 'unet');

  return seamless;
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
    expect(l2i).toMatchObject({ type: 'l2i', tiled: true, tile_size: 1024 });
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

  it('ignores stale upscale-model-only settings when latent HRF is selected', () => {
    const { g, seed, noise, denoise, l2i } = buildClassicGraph();

    addHighResFix({
      g,
      state: buildState({
        hrfMethod: 'latent',
        hrfSteps: 12,
        hrfModel: { key: 'hrf-model', hash: 'hrf-hash', name: 'HRF Model', base: 'sdxl', type: 'main' },
        hrfLoraMode: 'dedicated',
        hrfTileSize: 1536,
        hrfLoras: [
          {
            id: 'hrf-lora-id',
            isEnabled: true,
            model: { key: 'hrf-lora', hash: 'hrf-lora-hash', name: 'HRF LoRA', base: 'sdxl', type: 'lora' },
            weight: 0.6,
          },
        ],
      }),
      generationMode: 'txt2img',
      denoise,
      l2i,
      noise,
      seed,
    });

    const nodes = Object.values(g.getGraph().nodes);
    const hrfDenoise = nodes.find((node) => node.id.startsWith('hrf_denoise_latents'));

    expect(nodes.some((node) => node.type === 'lresize')).toBe(true);
    expect(nodes.some((node) => node.type === 'spandrel_image_to_image_autoscale')).toBe(false);
    expect(nodes.some((node) => node.id.startsWith('hrf_sdxl_model_loader'))).toBe(false);
    expect(nodes.some((node) => node.type === 'sdxl_lora_collection_loader')).toBe(false);
    expect(hrfDenoise).toMatchObject({ steps: 30 });
    expect(l2i).toMatchObject({ type: 'l2i', tiled: true, tile_size: 1024 });
    expect(g.getMetadataNode()).toMatchObject({ hrf_method: 'latent' });
    expect(g.getMetadataNode().hrf_steps).toBeUndefined();
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

  it('applies custom HRF steps only to the second pass', () => {
    const { g, seed, noise, denoise, l2i } = buildClassicGraph();
    addSDXLConditioning(g, denoise);

    addHighResFix({
      g,
      state: buildState({ hrfMethod: 'upscale_model', hrfSteps: 12 }),
      generationMode: 'txt2img',
      denoise,
      l2i,
      noise,
      seed,
    });

    const hrfDenoise = Object.values(g.getGraph().nodes).find(
      (node) => node.type === 'tiled_multi_diffusion_denoise_latents'
    );

    expect(denoise.steps).toBe(30);
    expect(hrfDenoise).toMatchObject({ steps: 12 });
    expect(g.getMetadataNode()).toMatchObject({ hrf_steps: 12 });
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
    const hrfPosCollect = nodes.find((node) => node.id.startsWith('hrf_pos_collect_conditioning_collect'));

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
      hrf_tile_control_weight: 0.625,
      hrf_tile_control_end: 0.2,
      hrf_tile_size: 1024,
      hrf_tile_overlap: 128,
      hrf_lora_mode: 'reuse_generate',
    });
  });

  it('uses a dedicated SDXL HRF model and custom steps for the second pass only', () => {
    const { g, seed, noise, denoise, l2i } = buildClassicGraph();
    addSDXLConditioning(g, denoise);

    addHighResFix({
      g,
      state: buildState({
        hrfMethod: 'upscale_model',
        hrfSteps: 14,
        hrfModel: { key: 'hrf-model', hash: 'hrf-hash', name: 'HRF Model', base: 'sdxl', type: 'main' },
        hrfLoraMode: 'none',
      }),
      generationMode: 'txt2img',
      denoise,
      l2i,
      noise,
      seed,
    });

    const graph = g.getGraph();
    const nodes = Object.values(graph.nodes);
    const hrfModelLoader = nodes.find((node) => node.id.startsWith('hrf_sdxl_model_loader'));
    const hrfPosCond = nodes.find((node) => node.id.startsWith('hrf_pos_cond'));
    const tiledDenoise = nodes.find((node) => node.type === 'tiled_multi_diffusion_denoise_latents');
    const i2l = nodes.find((node) => node.id.startsWith('hrf_i2l'));

    if (!hrfModelLoader || !hrfPosCond || !tiledDenoise || !i2l) {
      throw new Error('Expected dedicated HRF model graph nodes');
    }

    expect(hrfModelLoader).toMatchObject({ type: 'sdxl_model_loader', model: { key: 'hrf-model' } });
    expect(tiledDenoise).toMatchObject({ steps: 14 });
    expect(denoise.steps).toBe(30);
    expect(graph.edges).toContainEqual({
      source: { node_id: hrfModelLoader.id, field: 'clip' },
      destination: { node_id: hrfPosCond.id, field: 'clip' },
    });
    expect(graph.edges).toContainEqual({
      source: { node_id: hrfModelLoader.id, field: 'vae' },
      destination: { node_id: i2l.id, field: 'vae' },
    });
    expect(graph.edges).toContainEqual({
      source: { node_id: hrfModelLoader.id, field: 'vae' },
      destination: { node_id: 'l2i', field: 'vae' },
    });
    expect(nodes.some((node) => node.type === 'sdxl_lora_collection_loader')).toBe(false);
    expect(g.getMetadataNode()).toMatchObject({
      hrf_steps: 14,
      hrf_model: { key: 'hrf-model' },
      hrf_lora_mode: 'none',
    });
  });

  it('preserves regional SDXL conditioning when a dedicated HRF model uses classic fallback', () => {
    const { g, seed, noise, denoise, l2i } = buildClassicGraph();
    const conditioning = addSDXLConditioning(g, denoise);
    const { mask, regionPrompt } = addSDXLRegionalConditioning(g, conditioning);

    addHighResFix({
      g,
      state: buildState({
        hrfMethod: 'upscale_model',
        hrfModel: { key: 'hrf-model', hash: 'hrf-hash', name: 'HRF Model', base: 'sdxl', type: 'main' },
        hrfLoraMode: 'none',
      }),
      generationMode: 'txt2img',
      denoise,
      l2i,
      noise,
      seed,
    });

    const graph = g.getGraph();
    const nodes = Object.values(graph.nodes);
    const classicHrfDenoise = nodes.find((node) => node.id.startsWith('hrf_denoise_latents'));
    const hrfRegionPosCond = nodes.find((node) => node.id.startsWith('hrf_region_pos_cond'));
    const hrfPositiveCollect = nodes.find((node) => node.id.startsWith('hrf_positive_conditioning_collect'));

    if (!classicHrfDenoise || !hrfRegionPosCond || !hrfPositiveCollect) {
      throw new Error('Expected classic HRF denoise and cloned regional conditioning nodes');
    }

    expect(nodes.some((node) => node.type === 'tiled_multi_diffusion_denoise_latents')).toBe(false);
    expect(hrfRegionPosCond).toMatchObject({
      type: 'sdxl_compel_prompt',
      original_width: 1024,
      original_height: 1024,
      target_width: 1024,
      target_height: 1024,
    });
    expect(graph.edges).toContainEqual({
      source: { node_id: regionPrompt.id, field: 'value' },
      destination: { node_id: hrfRegionPosCond.id, field: 'prompt' },
    });
    expect(graph.edges).toContainEqual({
      source: { node_id: mask.id, field: 'mask' },
      destination: { node_id: hrfRegionPosCond.id, field: 'mask' },
    });
    expect(graph.edges).toContainEqual({
      source: { node_id: hrfRegionPosCond.id, field: 'conditioning' },
      destination: { node_id: hrfPositiveCollect.id, field: 'item' },
    });
    expect(graph.edges).toContainEqual({
      source: { node_id: hrfPositiveCollect.id, field: 'collection' },
      destination: { node_id: classicHrfDenoise.id, field: 'positive_conditioning' },
    });
  });

  it('applies dedicated HRF LoRA CLIP outputs to all regional conditioning clones', () => {
    const { g, seed, noise, denoise, l2i } = buildClassicGraph();
    const conditioning = addSDXLConditioning(g, denoise);
    addSDXLRegionalConditioning(g, conditioning);

    addHighResFix({
      g,
      state: buildState({
        hrfMethod: 'upscale_model',
        hrfLoraMode: 'dedicated',
        hrfLoras: [
          {
            id: 'hrf-lora-id',
            isEnabled: true,
            model: { key: 'hrf-lora', hash: 'hrf-lora-hash', name: 'HRF LoRA', base: 'sdxl', type: 'lora' },
            weight: 0.6,
          },
        ],
      }),
      generationMode: 'txt2img',
      denoise,
      l2i,
      noise,
      seed,
    });

    const graph = g.getGraph();
    const nodes = Object.values(graph.nodes);
    const loraLoader = nodes.find((node) => node.type === 'sdxl_lora_collection_loader');
    const classicHrfDenoise = nodes.find((node) => node.id.startsWith('hrf_denoise_latents'));
    const hrfPosCond = nodes.find((node) => node.id.startsWith('hrf_pos_cond'));
    const hrfRegionPosCond = nodes.find((node) => node.id.startsWith('hrf_region_pos_cond'));
    const hrfRegionNegCond = nodes.find((node) => node.id.startsWith('hrf_region_neg_cond'));

    if (!loraLoader || !classicHrfDenoise || !hrfPosCond || !hrfRegionPosCond || !hrfRegionNegCond) {
      throw new Error('Expected dedicated HRF LoRA and cloned regional conditioning nodes');
    }

    for (const cond of [hrfPosCond, hrfRegionPosCond, hrfRegionNegCond]) {
      expect(graph.edges).toContainEqual({
        source: { node_id: loraLoader.id, field: 'clip' },
        destination: { node_id: cond.id, field: 'clip' },
      });
      expect(graph.edges).toContainEqual({
        source: { node_id: loraLoader.id, field: 'clip2' },
        destination: { node_id: cond.id, field: 'clip2' },
      });
    }
    expect(graph.edges).toContainEqual({
      source: { node_id: loraLoader.id, field: 'unet' },
      destination: { node_id: classicHrfDenoise.id, field: 'unet' },
    });
  });

  it('applies only dedicated HRF LoRAs when dedicated LoRA mode is selected', () => {
    const { g, seed, noise, denoise, l2i } = buildClassicGraph();
    addSDXLConditioning(g, denoise);

    addHighResFix({
      g,
      state: buildState({
        hrfMethod: 'upscale_model',
        hrfLoraMode: 'dedicated',
        hrfLoras: [
          {
            id: 'hrf-lora-id',
            isEnabled: true,
            model: { key: 'hrf-lora', hash: 'hrf-lora-hash', name: 'HRF LoRA', base: 'sdxl', type: 'lora' },
            weight: 0.6,
          },
        ],
      }),
      generationMode: 'txt2img',
      denoise,
      l2i,
      noise,
      seed,
    });

    const graph = g.getGraph();
    const nodes = Object.values(graph.nodes);
    const loraLoader = nodes.find((node) => node.type === 'sdxl_lora_collection_loader');
    const tiledDenoise = nodes.find((node) => node.type === 'tiled_multi_diffusion_denoise_latents');

    if (!loraLoader || !tiledDenoise) {
      throw new Error('Expected dedicated HRF LoRA graph nodes');
    }

    expect(graph.edges).toContainEqual({
      source: { node_id: loraLoader.id, field: 'unet' },
      destination: { node_id: tiledDenoise.id, field: 'unet' },
    });
    expect(g.getMetadataNode()).toMatchObject({
      hrf_lora_mode: 'dedicated',
      hrf_loras: [{ model: { key: 'hrf-lora' }, weight: 0.6 }],
    });
  });

  it('ignores incompatible stale dedicated HRF LoRAs in the graph and metadata', () => {
    const { g, seed, noise, denoise, l2i } = buildClassicGraph();
    addSDXLConditioning(g, denoise);

    addHighResFix({
      g,
      state: buildState({
        hrfMethod: 'upscale_model',
        hrfLoraMode: 'dedicated',
        hrfLoras: [
          {
            id: 'stale-sd1-lora-id',
            isEnabled: true,
            model: { key: 'stale-sd1-lora', hash: 'stale-hash', name: 'Stale SD1 LoRA', base: 'sd-1', type: 'lora' },
            weight: 0.6,
          },
        ],
      }),
      generationMode: 'txt2img',
      denoise,
      l2i,
      noise,
      seed,
    });

    const nodes = Object.values(g.getGraph().nodes);

    expect(nodes.some((node) => node.type === 'sdxl_lora_collection_loader')).toBe(false);
    expect(nodes.some((node) => node.type === 'lora_selector')).toBe(false);
    expect(g.getMetadataNode()).toMatchObject({ hrf_lora_mode: 'dedicated' });
    expect(g.getMetadataNode().hrf_loras).toEqual([]);
  });

  it('preserves seamless routing when a dedicated HRF model is selected', () => {
    const { g, seed, noise, modelLoader, denoise, l2i } = buildClassicGraph();
    addSDXLConditioning(g, denoise);
    addSeamlessToClassicGraph({ g, modelLoader, denoise });

    addHighResFix({
      g,
      state: buildState({
        hrfMethod: 'upscale_model',
        hrfModel: { key: 'hrf-model', hash: 'hrf-hash', name: 'HRF Model', base: 'sdxl', type: 'main' },
        hrfLoraMode: 'none',
      }),
      generationMode: 'txt2img',
      denoise,
      l2i,
      noise,
      seed,
    });

    const graph = g.getGraph();
    const nodes = Object.values(graph.nodes);
    const hrfModelLoader = nodes.find((node) => node.id.startsWith('hrf_sdxl_model_loader'));
    const hrfSeamless = nodes.find((node) => node.id.startsWith('hrf_seamless'));
    const tiledDenoise = nodes.find((node) => node.type === 'tiled_multi_diffusion_denoise_latents');
    const i2l = nodes.find((node) => node.id.startsWith('hrf_i2l'));

    if (!hrfModelLoader || !hrfSeamless || !tiledDenoise || !i2l) {
      throw new Error('Expected dedicated HRF seamless graph nodes');
    }

    expect(hrfSeamless).toMatchObject({ type: 'seamless', seamless_x: true, seamless_y: false });
    expect(graph.edges).toContainEqual({
      source: { node_id: hrfModelLoader.id, field: 'unet' },
      destination: { node_id: hrfSeamless.id, field: 'unet' },
    });
    expect(graph.edges).toContainEqual({
      source: { node_id: hrfModelLoader.id, field: 'vae' },
      destination: { node_id: hrfSeamless.id, field: 'vae' },
    });
    expect(graph.edges).toContainEqual({
      source: { node_id: hrfSeamless.id, field: 'unet' },
      destination: { node_id: tiledDenoise.id, field: 'unet' },
    });
    expect(graph.edges).toContainEqual({
      source: { node_id: hrfSeamless.id, field: 'vae' },
      destination: { node_id: i2l.id, field: 'vae' },
    });
    expect(graph.edges).toContainEqual({
      source: { node_id: hrfSeamless.id, field: 'vae' },
      destination: { node_id: 'l2i', field: 'vae' },
    });
  });

  it('preserves seamless routing before dedicated HRF LoRAs', () => {
    const { g, seed, noise, modelLoader, denoise, l2i } = buildClassicGraph();
    addSDXLConditioning(g, denoise);
    addSeamlessToClassicGraph({ g, modelLoader, denoise });

    addHighResFix({
      g,
      state: buildState({
        hrfMethod: 'upscale_model',
        hrfLoraMode: 'dedicated',
        hrfLoras: [
          {
            id: 'hrf-lora-id',
            isEnabled: true,
            model: { key: 'hrf-lora', hash: 'hrf-lora-hash', name: 'HRF LoRA', base: 'sdxl', type: 'lora' },
            weight: 0.6,
          },
        ],
      }),
      generationMode: 'txt2img',
      denoise,
      l2i,
      noise,
      seed,
    });

    const graph = g.getGraph();
    const nodes = Object.values(graph.nodes);
    const hrfSeamless = nodes.find((node) => node.id.startsWith('hrf_seamless'));
    const loraLoader = nodes.find((node) => node.type === 'sdxl_lora_collection_loader');
    const tiledDenoise = nodes.find((node) => node.type === 'tiled_multi_diffusion_denoise_latents');

    if (!hrfSeamless || !loraLoader || !tiledDenoise) {
      throw new Error('Expected HRF seamless and dedicated LoRA graph nodes');
    }

    expect(graph.edges).toContainEqual({
      source: { node_id: 'model_loader', field: 'vae' },
      destination: { node_id: hrfSeamless.id, field: 'vae' },
    });
    expect(graph.edges).toContainEqual({
      source: { node_id: hrfSeamless.id, field: 'unet' },
      destination: { node_id: loraLoader.id, field: 'unet' },
    });
    expect(graph.edges).toContainEqual({
      source: { node_id: loraLoader.id, field: 'unet' },
      destination: { node_id: tiledDenoise.id, field: 'unet' },
    });
    expect(graph.edges).not.toContainEqual({
      source: { node_id: hrfSeamless.id, field: 'unet' },
      destination: { node_id: tiledDenoise.id, field: 'unet' },
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
