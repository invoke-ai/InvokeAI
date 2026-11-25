import type { GenerateLora, VaeModelConfig } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';

import { seedArchitectureCapabilities } from '@features/models/core/architectureCapabilities.testing';
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
import { CASES, valuesForCase } from './upscaleGraphCoverage.testing';

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

  it.each(['sd-1', 'sdxl'] as const)(
    'runs the %s pass in tiles behind the Spandrel prefix, anchored by the ControlNet pair, with fixed batch ids',
    (base) => {
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
      expect(nodes.unsharp_2).toMatchObject({ radius: 2, strength: 60, type: 'unsharp_mask' });
      expect(hasEdge(edges, 'spandrel_autoscale', 'image', 'unsharp_2', 'image')).toBe(true);
      expect(hasEdge(edges, 'unsharp_2', 'image', 'fit_to_grid', 'image')).toBe(true);
      expect(nodes.denoise_latents).toMatchObject({
        cfg_scale: 2,
        scheduler: 'kdpm_2',
        steps: 30,
        tile_height: 1024,
        tile_overlap: 128,
        tile_width: 1024,
        type: 'tiled_multi_diffusion_denoise_latents',
      });
      expect(nodes.denoise_latents?.denoising_start).toBeCloseTo(0.499);
      expect(nodes.i2l).toMatchObject({ fp32: true, tile_size: 1024, tiled: true, type: 'i2l' });
      expect(hasEdge(edges, 'fit_to_grid', 'image', 'i2l', 'image')).toBe(true);
      expect(hasEdge(edges, 'i2l', 'latents', 'denoise_latents', 'latents')).toBe(true);
      expect(nodes.upscale_output).toMatchObject({ is_intermediate: false, tile_size: 1024, tiled: true, type: 'l2i' });
      expect(nodes.noise?.use_cpu).toBe(true);
      expect(nodes.controlnet_1).toMatchObject({ control_weight: 0.625, end_step_percent: 0.55 });
      expect(nodes.controlnet_2).toMatchObject({ begin_step_percent: 0.55, control_weight: 0.21375 });
      expect(hasEdge(edges, 'fit_to_grid', 'image', 'controlnet_1', 'image')).toBe(true);
      expect(hasEdge(edges, 'controlnet_collector', 'collection', 'denoise_latents', 'control')).toBe(true);
      expect(nodes.core_metadata).toMatchObject({
        creativity: 0,
        structure: 0,
        tile_overlap: 128,
        tile_size: 1024,
        upscale_scale: 4,
      });
    }
  );

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

    expect(nodes.denoise_latents?.type).toBe('tiled_multi_diffusion_denoise_latents');
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

  it('runs an upscaler of unknown native scale once, as a plain pass', () => {
    const output = compileModelOnlyUpscaleGraph({
      destination: 'gallery',
      image,
      scale: 'native',
      sourceMetadata: null,
      upscaleModel: upscaler(null),
    }).backendGraph.nodes.upscale_output;

    expect(output).toMatchObject({ is_intermediate: false, type: 'spandrel_image_to_image', use_cache: false });
    expect(output).not.toHaveProperty('scale');
    expect(output).not.toHaveProperty('fit_to_multiple_of_8');
  });

  it('sizes a recorded native scale as that Scale, so the menu and the widget give the same result', () => {
    const compile = (scale: number | 'native') =>
      compileModelOnlyUpscaleGraph({
        destination: 'gallery',
        image,
        scale,
        sourceMetadata: null,
        upscaleModel: upscaler(4),
      }).backendGraph.nodes.upscale_output;

    expect(compile('native')).toEqual(compile(4));
    expect(compile('native')).toMatchObject({
      fit_to_multiple_of_8: true,
      scale: 4,
      type: 'spandrel_image_to_image_autoscale',
    });
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

const withVariant = (config: ModelConfig, variant: string): ModelConfig => ({ ...config, variant }) as ModelConfig;

const createZImageValues = (options: { tileControlNet?: boolean; cfgScale?: number } = {}) => {
  const models = [
    model('main', 'main', 'z-image'),
    model('spandrel', 'spandrel_image_to_image', 'any'),
    model('lora', 'lora', 'z-image'),
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

const createKrea2Values = (variant: 'krea2_turbo' | 'krea2_base', format = 'gguf_quantized') => {
  const models = [
    { ...withVariant(model('main', 'main', 'krea-2'), variant), format } as ModelConfig,
    model('spandrel', 'spandrel_image_to_image', 'any'),
    model('qwenvae', 'vae', 'qwen-image'),
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

const createFlux2Values = (variant: string, format = 'checkpoint') => {
  const models = [
    { ...withVariant(model('main', 'main', 'flux2'), variant), format } as ModelConfig,
    model('spandrel', 'spandrel_image_to_image', 'any'),
    model('lora', 'lora', 'flux2'),
    model('flux2vae', 'vae', 'flux2'),
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

const compileCoverageCase = (label: string) => {
  const testCase = CASES.find((candidate) => candidate.label === label);

  if (!testCase) {
    throw new Error(`No coverage case ${label}`);
  }

  return compileUpscaleGraph(valuesForCase(testCase).values, 'gallery', { useCpuNoise: true }).backendGraph;
};

describe('the frame the pass encodes', () => {
  it('fits the frame to the grid flux_denoise accepts, instead of whatever Spandrel rounded to', () => {
    // Spandrel's autoscale floors to a multiple of 8; `flux_denoise` declares multiple_of=16 and
    // raises on anything else -- after the upscale has already been paid for. The 203x101 fixture
    // at 4x is exactly that case: Spandrel gives 808x400, and 808 is not a multiple of 16.
    const { edges, nodes } = compileUpscaleGraph(createFluxValues(), 'gallery', { useCpuNoise: true }).backendGraph;

    expect(nodes.fit_to_grid).toMatchObject({ height: 400, type: 'img_resize', width: 800 });
    expect(nodes.denoise_latents).toMatchObject({ height: 400, width: 800 });
    expect(hasEdge(edges, 'unsharp_2', 'image', 'fit_to_grid', 'image')).toBe(true);
    expect(hasEdge(edges, 'fit_to_grid', 'image', 'i2l', 'image')).toBe(true);
  });

  it.each([
    ['FLUX.1', () => createFluxValues(), 'denoise_latents'],
    // SD's noise is sized as a literal too, so a frame the upscaler left small would not match it.
    ['SD1.5', () => createValues('sd-1'), 'noise'],
  ] as const)(
    'resizes a %s frame the scale already puts on the grid, because a 1x model makes Spandrel ignore the scale',
    (_label, create, sizedNode) => {
      const { edges, nodes } = compileUpscaleGraph(
        { ...create(), inputImage: { height: 1024, image_name: 'input.png', width: 1024 }, scale: 2 },
        'gallery',
        { useCpuNoise: true }
      ).backendGraph;

      expect(nodes.fit_to_grid).toMatchObject({ height: 2048, width: 2048 });
      expect(nodes[sizedNode]).toMatchObject({ height: 2048, width: 2048 });
      expect(hasEdge(edges, 'fit_to_grid', 'image', 'i2l', 'image')).toBe(true);
    }
  );

  it.each(CASES.map(({ label }) => label))(
    'tiles both VAE ends of the %s pass where its nodes have the switch',
    (label) => {
      const { nodes } = compileCoverageCase(label);

      expect(nodes.i2l).toMatchObject({ tile_size: 1024, tiled: true });
      // `anima_l2i` has no switch: it tiles on its own once an untiled decode would take most of the card.
      if (nodes.upscale_output?.type === 'anima_l2i') {
        expect(nodes.upscale_output).not.toHaveProperty('tiled');
      } else {
        expect(nodes.upscale_output).toMatchObject({ tile_size: 1024, tiled: true });
      }
    }
  );

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

describe("the family pipeline Upscale runs Generation's way", () => {
  // Upscale keeps its own persisted field names; each must reach the loader input the family reads, and its record.
  it.each([
    ['flux', 't5_encoder_model', 't5', 't5_encoder'],
    ['flux', 'clip_embed_model', 'clip', 'clip_embed_model'],
    ['z-image', 'qwen3_encoder_model', 'qwen3-4b', 'qwen3_encoder'],
    ['anima-3.8b', 'qwen3_encoder_model', 'qwen3-06b', 'qwen3_encoder'],
    ['anima-3.8b', 'qwen3_5_encoder_model', 'qwen35', 'qwen3_5_encoder'],
    ['krea-2-raw', 'qwen3_vl_encoder_model', 'qwen3vl-4b', 'qwen3_vl_encoder'],
    ['flux2-dev', 'mistral_encoder_model', 'mistral', 'mistral_encoder'],
    ['flux2-klein-9b', 'qwen3_encoder_model', 'qwen3-8b', 'qwen3_encoder'],
    ['moss-image', 'qwen3_5_encoder_model', 'qwen35-9b', 'qwen3_5_encoder'],
    ['qwen-image', 'qwen_vl_encoder_model', 'qwen-vl', 'qwen_image_qwen_vl_encoder'],
  ])('hands the %s loader its %s, and records it', (label, loaderField, key, metadataKey) => {
    const { nodes } = compileCoverageCase(label);

    expect(nodes.model_loader?.[loaderField]).toMatchObject({ key });
    expect(nodes.core_metadata?.[metadataKey]).toMatchObject({ key });
  });

  it('records the VAE the loader used, not one a self-contained SDNQ FLUX.1 ignores', () => {
    const values = createFluxValues();
    const submodels = Object.fromEntries(
      ['transformer', 'vae', 'text_encoder', 'tokenizer', 'text_encoder_2', 'tokenizer_2'].map((name) => [name, {}])
    );
    const sdnq = { ...values.model, format: 'sdnq_quantized', submodels } as UpscaleWidgetValues['model'];
    const separate = compileUpscaleGraph(values, 'gallery', { useCpuNoise: true }).backendGraph.nodes;
    const bundled = compileUpscaleGraph({ ...values, model: sdnq }, 'gallery', { useCpuNoise: true }).backendGraph
      .nodes;

    expect(values.vae).toMatchObject({ key: 'fluxvae' });
    expect(separate.model_loader?.vae_model).toMatchObject({ key: 'fluxvae' });
    expect(separate.core_metadata?.vae).toMatchObject({ key: 'fluxvae' });
    expect(bundled.model_loader?.vae_model).toBeUndefined();
    expect(bundled.core_metadata?.vae).toBeUndefined();
  });

  it('keeps an encoder the architecture does not read out of the graph and its metadata', () => {
    // A T5 picked while FLUX was selected stays in the widget for when FLUX comes back.
    const staleT5 = { base: 'any', key: 'stale-t5', name: 'stale-t5', type: 't5_encoder' } as never;
    const { nodes } = compileUpscaleGraph({ ...createZImageValues(), t5EncoderModel: staleT5 }, 'gallery', {
      useCpuNoise: true,
    }).backendGraph;

    expect(JSON.stringify(nodes)).not.toContain('stale-t5');
  });

  it.each(['z-image', 'qwen-image'])(
    'sends the %s pass no schedule shift, which would move the schedule Creativity is translated against',
    (label) => {
      expect(compileCoverageCase(label).nodes.denoise_latents).not.toHaveProperty('shift');
    }
  );

  it('loads an enabled MOSS-Image LoRA into the transformer the pass denoises with, and records it', () => {
    const { edges, nodes } = compileCoverageCase('moss-image');
    const [loaderId] =
      Object.entries(nodes).find(([, node]) => node.type === 'moss_image_lora_collection_loader') ?? [];

    expect(loaderId).toBeDefined();
    expect(hasEdge(edges, 'model_loader', 'transformer', loaderId ?? '', 'transformer')).toBe(true);
    expect(hasEdge(edges, loaderId ?? '', 'transformer', 'denoise_latents', 'transformer')).toBe(true);
    expect(nodes.core_metadata?.loras).toEqual([{ model: expect.objectContaining({ key: 'moss-lora' }), weight: 1 }]);
  });
});

describe('tile control', () => {
  it('wires no ControlNet where the architecture has none', () => {
    const { nodes } = compileUpscaleGraph(createFluxValues(), 'gallery', { useCpuNoise: true }).backendGraph;

    expect(Object.values(nodes).some((node) => node.type === 'controlnet')).toBe(false);
    expect(nodes.controlnet_collector).toBeUndefined();
  });

  it('anchors Z-Image to the enlarged frame with a tiled control encode when a tile ControlNet is chosen', () => {
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
    // The denoise encodes the control image itself, with the VAE the pass decodes with.
    expect(hasEdge(edges, 'model_loader', 'vae', 'denoise_latents', 'vae')).toBe(true);
  });

  it('runs Z-Image without a ControlNet when none is chosen', () => {
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
});

describe('Creativity on index-clipping architectures', () => {
  // Expected starts come from the backend's own schedules and `int(start * steps)`: within a step of FLUX's 1 - start.
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

  // `krea2_denoise` rounds `start * steps`; steps from `exp(mu) * t / (1 + (exp(mu) - 1) * t)` at t = 1 - i / 30.
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

  // `qwen_image_denoise`: mu = 0.5 + 0.4 * (tokens - 256) / 7936, held at 0.9, sigmas stretched by
  // `k = (1 - s(1/30)) / 0.98` so the last lands on 0.02; steps worked out by hand at `t = 1 - i / 30`.
  it.each([
    // 4096 tokens: mu 0.694, k 0.9545; step 19 at 0.515, step 20 at 0.476 (unstretched 0.537 and 0.500).
    [512, 19 / 30],
    // 4096px result: held at mu 0.9, k 0.9406. Step 20 sits at 0.523, step 21 at 0.482.
    [2048, 21 / 30],
  ])('starts Qwen Image from a %ipx input where FLUX starts', (side, expected) => {
    expect(startFor(createQwenImageValues(), 0, side)).toBeCloseTo(expected, 6);
  });

  it.each([
    ['FLUX.1', () => createFluxValues()],
    ['FLUX.2', () => createFlux2Values('klein_4b')],
  ] as const)('leaves the value-clipping %s pass on the legacy formula', (_label, create) => {
    expect(startFor(create(), 0, 1024)).toBeCloseTo(0.499, 6);
  });
});

describe('schedulers on the upscale pass', () => {
  it.each([
    ['flux', () => createFluxValues(), 'lcm', 'euler'],
    ['flux', () => createFluxValues(), 'heun', 'heun'],
    ['flux2', () => createFlux2Values('klein_4b'), 'heun', 'euler'],
    ['z-image', () => createZImageValues(), 'lcm', 'euler'],
    ['z-image', () => createZImageValues(), 'heun', 'heun'],
    ['anima', () => createAnimaValues(), 'heun', 'euler'],
    ['anima', () => createAnimaValues(), 'dpmpp_2m', 'dpmpp_2m'],
    ['sd-1', () => createValues('sd-1'), 'heun', 'heun'],
  ] as const)('runs %s with %s stored as %s, and records what ran', (_base, createBaseValues, stored, runs) => {
    const { nodes } = compileUpscaleGraph({ ...createBaseValues(), scheduler: stored }, 'gallery', {
      useCpuNoise: true,
    }).backendGraph;
    expect([nodes.denoise_latents?.scheduler, nodes.core_metadata?.scheduler]).toEqual([runs, runs]);
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

  it.each([
    ['Z-Image at CFG 1', () => createZImageValues(), false],
    ['Z-Image at CFG 4', () => createZImageValues({ cfgScale: 4 }), true],
    // Switched off, the prompt still gets its branch and the queue fills it with an empty string, so CFG runs at
    // the value the slider and the metadata show.
    [
      'Z-Image at CFG 4 with the negative prompt off',
      () => ({ ...createZImageValues({ cfgScale: 4 }), negativePromptEnabled: false }),
      true,
    ],
    ['Qwen Image at CFG 1', () => createQwenImageValues({ cfgScale: 1 }), false],
    ['Krea-2 Raw at CFG 5.5', () => createKrea2Values('krea2_base'), true],
    ['FLUX.2', () => createFlux2Values('klein_4b'), false],
    ['SD1.5', () => createValues('sd-1'), true],
  ] as const)('records the negative prompt of %s exactly when the pass encodes one', (_label, create, reads) => {
    // The pipeline builds the negative branch by its own rule; the metadata must record the prompt on the same terms.
    const { edges, nodes } = compileUpscaleGraph(create(), 'gallery', { useCpuNoise: true }).backendGraph;

    expect(nodes.neg_cond !== undefined).toBe(reads);
    expect(hasEdge(edges, 'negative_prompt', 'value', 'core_metadata', 'negative_prompt')).toBe(reads);
  });

  it('records the slider as guidance for a family that reads no negative prompt, and as CFG elsewhere', () => {
    const metadataFor = (values: UpscaleWidgetValues) =>
      compileUpscaleGraph(values, 'gallery', { useCpuNoise: true }).backendGraph.nodes.core_metadata;

    expect(metadataFor(createFlux2Values('klein_4b'))).toMatchObject({ guidance: 4 });
    expect(metadataFor(createFlux2Values('klein_4b'))).not.toHaveProperty('cfg_scale');
    expect(metadataFor(createQwenImageValues())).toMatchObject({ cfg_scale: 4 });
  });
});

describe('compileUpscaleGraph for MOSS-Image', () => {
  const createMossImageValues = () => {
    const models = [
      { ...model('main', 'main', 'moss-image'), format: 'diffusers' },
      model('spandrel', 'spandrel_image_to_image', 'any'),
      model('lora', 'lora', 'moss-image'),
    ] as ModelConfig[];

    return {
      ...createDefaultUpscaleWidgetValues(models),
      inputImage: { height: 101, image_name: 'input.png', width: 203 },
      loras: [{ isEnabled: true, model: models[2] as GenerateLora['model'], weight: 0.7 }],
      positivePrompt: 'detail',
    };
  };

  it('patches the refine pass transformer with the active LoRAs and leaves the encoder and VAE on the loader', () => {
    const { edges, nodes } = compileUpscaleGraph(createMossImageValues(), 'gallery', {
      useCpuNoise: true,
    }).backendGraph;
    const loraLoaderId = Object.entries(nodes).find(
      ([, node]) => node.type === 'moss_image_lora_collection_loader'
    )?.[0];

    expect(loraLoaderId).toBeDefined();
    expect(hasEdge(edges, 'model_loader', 'transformer', loraLoaderId ?? '', 'transformer')).toBe(true);
    expect(hasEdge(edges, loraLoaderId ?? '', 'transformer', 'denoise_latents', 'transformer')).toBe(true);
    expect(hasEdge(edges, 'model_loader', 'transformer', 'denoise_latents', 'transformer')).toBe(false);
    expect(hasEdge(edges, 'model_loader', 'qwen3_5_encoder', 'pos_cond', 'qwen3_5_encoder')).toBe(true);
    expect(hasEdge(edges, 'model_loader', 'vae', 'denoise_latents', 'vae')).toBe(true);
  });

  it('takes the transformer straight from the model loader without LoRAs', () => {
    const { edges, nodes } = compileUpscaleGraph({ ...createMossImageValues(), loras: [] }, 'gallery', {
      useCpuNoise: true,
    }).backendGraph;

    expect(Object.values(nodes).some((node) => node.type === 'moss_image_lora_collection_loader')).toBe(false);
    expect(hasEdge(edges, 'model_loader', 'transformer', 'denoise_latents', 'transformer')).toBe(true);
  });
});
