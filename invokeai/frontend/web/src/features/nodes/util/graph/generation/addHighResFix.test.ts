import type { RootState } from 'app/store/store';
import { describe, expect, it } from 'vitest';

import { addHighResFix } from './addHighResFix';
import { Graph } from './Graph';

const buildState = (overrides?: { base?: string; hrfEnabled?: boolean; refinerModel?: unknown }): RootState =>
  ({
    ui: { activeTab: 'generate' },
    params: {
      model: { key: 'model', name: 'model', base: overrides?.base ?? 'sdxl', type: 'main' },
      dimensions: { width: 512, height: 512 },
      hrfEnabled: overrides?.hrfEnabled ?? true,
      hrfScale: 2,
      hrfStrength: 0.35,
      hrfLatentInterpolationMode: 'bilinear',
      optimizedDenoisingEnabled: true,
      refinerModel: overrides?.refinerModel ?? null,
    },
  }) as unknown as RootState;

const buildClassicGraph = () => {
  const g = new Graph('test_graph');
  const seed = g.addNode({ id: 'seed', type: 'integer' });
  const noise = g.addNode({ id: 'noise', type: 'noise', use_cpu: true, width: 512, height: 512 });
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
  g.addEdge(denoise, 'latents', l2i, 'latents');

  return { g, seed, noise, denoise, l2i };
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
});
