import type { BackendGraphContract, MainModelConfig } from '@features/generation/contracts';

import { describe, expect, it } from 'vitest';

import type { VideoSettings } from './types';

import { compileVideoGraph } from './graph';
import { getDefaultVideoSettings } from './videoPolicies';

const wanModel = (variant: string, format = 'diffusers', key = `wan-${variant}-${format}`): MainModelConfig => ({
  base: 'wan',
  format,
  key,
  name: `Wan 2.2 ${variant}`,
  type: 'main',
  variant,
});

const h3Model = (key = 'h3-main'): MainModelConfig => ({
  base: 'minimax-h3',
  format: 'diffusers',
  key,
  name: 'MiniMax H3',
  type: 'main',
  variant: 'fl2va',
});
const FIRST_FRAME = { height: 1080, image_name: 'first.png', width: 1920 };
const SOURCE_VIDEO = {
  endFrame: 79,
  fps: 16,
  height: 480,
  numFrames: 81,
  startFrame: 0,
  video_name: 'clip.mp4',
  width: 832,
};

const settingsFor = (model: MainModelConfig, overrides: Partial<VideoSettings> = {}): VideoSettings => ({
  ...getDefaultVideoSettings(model),
  positivePrompt: 'a red fox',
  ...overrides,
});

const nodesOfType = (graph: BackendGraphContract, type: string) =>
  Object.values(graph.nodes).filter((node) => node.type === type);

const nodeOfType = (graph: BackendGraphContract, type: string) => {
  const nodes = nodesOfType(graph, type);

  expect(nodes, `expected exactly one ${type} node`).toHaveLength(1);

  return nodes[0] as Record<string, unknown> & { id: string; type: string };
};

const hasEdge = (
  graph: BackendGraphContract,
  sourceId: string,
  sourceField: string,
  targetId: string,
  targetField: string
) =>
  graph.edges.some(
    (edge) =>
      edge.source.node_id === sourceId &&
      edge.source.field === sourceField &&
      edge.destination.node_id === targetId &&
      edge.destination.field === targetField
  );

describe('compileVideoGraph — Wan 2.2', () => {
  it('builds a text-to-video graph ending in wan_l2v', () => {
    const model = wanModel('t2v_a14b');
    const settings = settingsFor(model, { aspectRatioId: '16:9', targetResolution: '720p' });
    const { backendGraph, negativePromptNodeId, positivePromptNodeId, seedNodeId } = compileVideoGraph(settings, model);

    expect(positivePromptNodeId).toBe('positive_prompt');
    expect(negativePromptNodeId).toBe('negative_prompt');
    expect(seedNodeId).toBe('seed');

    const denoise = nodeOfType(backendGraph, 'wan_video_denoise');

    expect(denoise).toMatchObject({
      guidance_scale: settings.cfgScale,
      height: 720,
      num_frames: settings.numFrames,
      steps: settings.steps,
      width: 1280,
    });
    expect(denoise.guidance_scale_low_noise).toBe(settings.cfgScaleLowNoise ?? settings.cfgScale);

    const output = nodeOfType(backendGraph, 'wan_l2v');

    expect(output).toMatchObject({ fps: settings.fps, id: 'video_output', is_intermediate: false });
    expect(hasEdge(backendGraph, 'model_loader', 'vae', 'video_output', 'vae')).toBe(true);
    expect(hasEdge(backendGraph, 'denoise_latents', 'latents', 'video_output', 'latents')).toBe(true);
    expect(hasEdge(backendGraph, 'seed', 'value', 'denoise_latents', 'seed')).toBe(true);
    expect(hasEdge(backendGraph, 'positive_prompt', 'value', 'pos_cond', 'prompt')).toBe(true);
    expect(hasEdge(backendGraph, 'negative_prompt', 'value', 'neg_cond', 'prompt')).toBe(true);

    expect(nodesOfType(backendGraph, 'wan_ref_image_encoder')).toHaveLength(0);
    expect(nodeOfType(backendGraph, 'core_metadata')).toMatchObject({
      generation_mode: 'wan_t2v',
      height: 720,
      num_frames: settings.numFrames,
      width: 1280,
    });
    expect(hasEdge(backendGraph, 'core_metadata', 'metadata', 'video_output', 'metadata')).toBe(true);
    expect(hasEdge(backendGraph, 'negative_prompt', 'value', 'core_metadata', 'negative_prompt')).toBe(true);
  });

  it('chains wan_lora_loader nodes with expert targets and preserves target/low_noise_weight in metadata', () => {
    const model = wanModel('t2v_a14b');
    const highLora = {
      isEnabled: true,
      model: { base: 'wan', key: 'lit-high', name: 'Wan High', type: 'lora' as const, variant: 'a14b' },
      target: 'high' as const,
      weight: 0.9,
    };
    const lowLora = {
      isEnabled: true,
      model: { base: 'wan', key: 'lit-low', name: 'Wan Low', type: 'lora' as const, variant: 'a14b' },
      target: 'low' as const,
      weight: 0.5,
    };
    const dualLora = {
      isEnabled: true,
      lowNoiseWeight: 0.3,
      model: { base: 'wan', key: 'lit-dual', name: 'Wan Dual', type: 'lora' as const, variant: 'a14b' },
      target: 'both' as const,
      weight: 0.8,
    };
    const untargetedLora = {
      isEnabled: true,
      model: { base: 'wan', key: 'lit-auto', name: 'Wan Auto', type: 'lora' as const, variant: 'a14b' },
      weight: 0.7,
    };

    const { backendGraph } = compileVideoGraph(
      settingsFor(model, { loras: [highLora, lowLora, dualLora, untargetedLora] }),
      model
    );

    expect(nodesOfType(backendGraph, 'wan_lora_collection_loader')).toHaveLength(0);
    const loaders = nodesOfType(backendGraph, 'wan_lora_loader');
    expect(loaders).toHaveLength(5);
    expect(
      loaders.map((node) => ({ key: (node.lora as { key: string }).key, target: node.target, weight: node.weight }))
    ).toEqual([
      { key: 'lit-high', target: 'high', weight: 0.9 },
      { key: 'lit-low', target: 'low', weight: 0.5 },
      { key: 'lit-dual', target: 'high', weight: 0.8 },
      { key: 'lit-dual', target: 'low', weight: 0.3 },
      { key: 'lit-auto', target: 'auto', weight: 0.7 },
    ]);
    expect(hasEdge(backendGraph, 'model_loader', 'transformer', loaders[0]!.id, 'transformer')).toBe(true);
    expect(hasEdge(backendGraph, loaders[4]!.id, 'transformer', 'denoise_latents', 'transformer')).toBe(true);
    expect(nodeOfType(backendGraph, 'core_metadata').loras).toEqual([
      { model: expect.objectContaining({ key: 'lit-high' }), target: 'high', weight: 0.9 },
      { model: expect.objectContaining({ key: 'lit-low' }), target: 'low', weight: 0.5 },
      { low_noise_weight: 0.3, model: expect.objectContaining({ key: 'lit-dual' }), target: 'both', weight: 0.8 },
      { model: expect.objectContaining({ key: 'lit-auto' }), weight: 0.7 },
    ]);

    const equalLowNoise = compileVideoGraph(
      settingsFor(model, {
        loras: [{ ...dualLora, lowNoiseWeight: 0.8, target: 'both', weight: 0.8 }],
      }),
      model
    );
    const equalLoaders = nodesOfType(equalLowNoise.backendGraph, 'wan_lora_loader');
    expect(equalLoaders).toHaveLength(1);
    expect(equalLoaders[0]).toMatchObject({ target: 'both', weight: 0.8 });
    expect(nodeOfType(equalLowNoise.backendGraph, 'core_metadata').loras).toEqual([
      { model: expect.objectContaining({ key: 'lit-dual' }), target: 'both', weight: 0.8 },
    ]);
  });

  it('emits a single wan_lora_loader node and omits low_noise_weight on single-transformer TI2V-5B even with lowNoiseWeight set', () => {
    const model = wanModel('ti2v_5b');
    const dualLora5b = {
      isEnabled: true,
      lowNoiseWeight: 0.4,
      model: { base: 'wan', key: 'lit-5b-dual', name: 'Wan 5B Dual', type: 'lora' as const, variant: '5b' },
      target: 'both' as const,
      weight: 0.8,
    };
    const { backendGraph } = compileVideoGraph(settingsFor(model, { loras: [dualLora5b] }), model);
    const loaders = nodesOfType(backendGraph, 'wan_lora_loader');

    expect(nodesOfType(backendGraph, 'wan_lora_collection_loader')).toHaveLength(0);
    expect(loaders).toHaveLength(1);
    expect(loaders[0]).toMatchObject({
      lora: expect.objectContaining({ key: 'lit-5b-dual' }),
      target: 'both',
      weight: 0.8,
    });
    expect(hasEdge(backendGraph, 'model_loader', 'transformer', loaders[0]!.id, 'transformer')).toBe(true);
    expect(hasEdge(backendGraph, loaders[0]!.id, 'transformer', 'denoise_latents', 'transformer')).toBe(true);
    expect(nodeOfType(backendGraph, 'core_metadata').loras).toEqual([
      { model: expect.objectContaining({ key: 'lit-5b-dual' }), target: 'both', weight: 0.8 },
    ]);
  });

  it('wires first-frame conditioning through wan_ref_image_encoder with matching canvas and frames', () => {
    const model = wanModel('i2v_a14b');
    const settings = settingsFor(model, { firstFrameImage: FIRST_FRAME, targetResolution: '480p' });
    const { backendGraph } = compileVideoGraph(settings, model);
    const refEncoder = nodeOfType(backendGraph, 'wan_ref_image_encoder');
    const denoise = nodeOfType(backendGraph, 'wan_video_denoise');

    expect(refEncoder).toMatchObject({
      height: denoise.height,
      image: { image_name: 'first.png' },
      num_frames: settings.numFrames,
      width: denoise.width,
    });
    expect(refEncoder.end_image).toBeUndefined();
    expect(hasEdge(backendGraph, 'model_loader', 'vae', 'ref_image_encoder', 'vae')).toBe(true);
    expect(hasEdge(backendGraph, 'ref_image_encoder', 'ref_image', 'denoise_latents', 'ref_image')).toBe(true);
    // 1920x1080 at 480p on the ×16 grid.
    expect(denoise).toMatchObject({ height: 480, width: 848 });
    expect(nodeOfType(backendGraph, 'core_metadata')).toMatchObject({
      first_frame_image: { image_name: 'first.png' },
      generation_mode: 'wan_i2v',
    });
  });

  it('builds the extend graph: trim, last-frame conditioning, concat at the source fps', () => {
    const model = wanModel('i2v_a14b');
    const settings = settingsFor(model, { sourceVideo: SOURCE_VIDEO });
    const { backendGraph } = compileVideoGraph(settings, model);

    expect(nodeOfType(backendGraph, 'extract_video_range')).toMatchObject({
      end_frame: -2,
      is_intermediate: true,
      start_frame: 0,
      video: { video_name: 'clip.mp4' },
    });
    expect(nodeOfType(backendGraph, 'video_frame_extract')).toMatchObject({ frame_index: -1, is_intermediate: true });
    expect(hasEdge(backendGraph, 'source_last_frame', 'image', 'ref_image_encoder', 'image')).toBe(true);
    expect(nodeOfType(backendGraph, 'wan_ref_image_encoder').image).toBeUndefined();

    expect(nodeOfType(backendGraph, 'wan_l2v')).toMatchObject({ id: 'extension_clip', is_intermediate: true });
    expect(nodeOfType(backendGraph, 'video_concat')).toMatchObject({
      id: 'video_output',
      is_intermediate: false,
      size_mismatch: 'match_first',
      transition: 'crossfade',
      transition_frames: 2,
    });

    expect(hasEdge(backendGraph, 'source_video', 'video', 'source_clip_collect', 'item')).toBe(true);
    expect(hasEdge(backendGraph, 'source_clip_collect', 'collection', 'clips_to_join', 'collection')).toBe(true);
    expect(hasEdge(backendGraph, 'extension_clip', 'video', 'clips_to_join', 'item')).toBe(true);
    expect(hasEdge(backendGraph, 'clips_to_join', 'collection', 'video_output', 'videos')).toBe(true);

    expect(hasEdge(backendGraph, 'source_video', 'fps', 'source_fps', 'value')).toBe(true);
    expect(hasEdge(backendGraph, 'source_fps', 'value', 'extension_clip', 'fps')).toBe(true);
    expect(hasEdge(backendGraph, 'source_fps', 'value', 'video_output', 'fps')).toBe(true);

    expect(nodeOfType(backendGraph, 'core_metadata')).toMatchObject({
      generation_mode: 'wan_extend_video',
      source_video: { video_name: 'clip.mp4' },
    });
    expect(hasEdge(backendGraph, 'core_metadata', 'metadata', 'video_output', 'metadata')).toBe(true);
    expect(hasEdge(backendGraph, 'core_metadata', 'metadata', 'extension_clip', 'metadata')).toBe(true);
    expect(hasEdge(backendGraph, 'source_last_frame', 'image', 'core_metadata', 'first_frame_image')).toBe(true);
  });

  it('refuses to compile invalid settings', () => {
    const t2v = wanModel('t2v_a14b');

    expect(() => compileVideoGraph(settingsFor(t2v, { firstFrameImage: FIRST_FRAME }), t2v)).toThrow(
      /does not support/
    );
    expect(() => compileVideoGraph(settingsFor(t2v, { numFrames: 80 }), t2v)).toThrow(/Frame count/);
  });
});

describe('compileVideoGraph — MiniMax H3', () => {
  const model = h3Model();

  it('builds a text-to-video graph with audio, no negative prompt, and string frame counts', () => {
    const settings = settingsFor(model, { aspectRatioId: '16:9' });
    const { backendGraph } = compileVideoGraph(settings, model);
    const denoise = nodeOfType(backendGraph, 'minimax_h3_denoise');

    // Node steps include one terminal sigma point beyond the model evaluations.
    expect(denoise).toMatchObject({ height: 768, num_frames: '124', steps: settings.steps + 1, width: 1344 });

    const output = nodeOfType(backendGraph, 'minimax_h3_latents_to_video');

    expect(output.id).toBe('video_output');
    expect(hasEdge(backendGraph, 'denoise_latents', 'video_latents', 'video_output', 'video_latents')).toBe(true);
    expect(hasEdge(backendGraph, 'denoise_latents', 'audio_latents', 'video_output', 'audio_latents')).toBe(true);
    expect(hasEdge(backendGraph, 'model_loader', 'vae', 'video_output', 'vae')).toBe(true);
    expect(hasEdge(backendGraph, 'model_loader', 'audio_vae', 'video_output', 'audio_vae')).toBe(true);

    expect(nodesOfType(backendGraph, 'minimax_h3_frame_conditioning')).toHaveLength(0);
    expect(nodeOfType(backendGraph, 'core_metadata')).toMatchObject({ fps: 24, generation_mode: 'minimax_h3_t2v' });
    expect(hasEdge(backendGraph, 'negative_prompt', 'value', 'core_metadata', 'negative_prompt')).toBe(false);
  });

  it('builds the extend graph with the source resampled to 24 fps', () => {
    const settings = settingsFor(model, { sourceVideo: { ...SOURCE_VIDEO, endFrame: 80 } });
    const { backendGraph } = compileVideoGraph(settings, model);

    expect(nodeOfType(backendGraph, 'extract_video_range')).toMatchObject({ fps: 24 });
    expect(hasEdge(backendGraph, 'source_last_frame', 'image', 'pos_cond', 'first_image')).toBe(true);
    expect(hasEdge(backendGraph, 'source_last_frame', 'image', 'frame_conditioning', 'first_image')).toBe(true);
    expect(nodeOfType(backendGraph, 'minimax_h3_latents_to_video')).toMatchObject({
      id: 'extension_clip',
      is_intermediate: true,
    });
    expect(nodeOfType(backendGraph, 'video_concat').id).toBe('video_output');
    expect(nodesOfType(backendGraph, 'float_to_int')).toHaveLength(0);
    expect(nodeOfType(backendGraph, 'core_metadata').generation_mode).toBe('minimax_h3_extend_video');
  });

  it('maps a single-file main onto the loader: component source as model, checkpoint as override', () => {
    const checkpoint = { ...h3Model('h3-int8'), format: 'checkpoint' };
    const encoder = { base: 'minimax-h3', key: 'h3-te', name: 'H3 TE int8', type: 'qwen3_vl_encoder' as const };
    const settings = settingsFor(checkpoint, { componentSourceModel: model, h3TextEncoderModel: encoder });
    const { backendGraph } = compileVideoGraph(settings, checkpoint);

    expect(nodeOfType(backendGraph, 'minimax_h3_model_loader')).toMatchObject({
      model: { key: model.key },
      text_encoder_model: { key: 'h3-te' },
      transformer_model: { key: 'h3-int8' },
    });
    expect(nodeOfType(backendGraph, 'core_metadata')).toMatchObject({
      minimax_h3_component_source: { key: model.key },
      minimax_h3_text_encoder_model: { key: 'h3-te' },
      model: { key: 'h3-int8' },
    });
  });
});

describe('compileVideoGraph — MiniMax H3 Ref2VA', () => {
  const componentSource = h3Model();
  const model: MainModelConfig = {
    base: 'minimax-h3',
    format: 'checkpoint',
    key: 'h3-ref2va-ckpt',
    name: 'MiniMax H3 Ref2VA Transformer (int8, pruned)',
    type: 'main',
    variant: 'ref2va',
  };
  const referenceSettings = settingsFor(model, {
    componentSourceModel: componentSource,
    references: [
      {
        clip: { endFrame: 47, fps: 24, height: 480, numFrames: 48, startFrame: 2, video_name: 'ref.mp4', width: 832 },
        conditioning: 'video_audio',
        kind: 'video',
      },
      { detail: 'match', image: { height: 512, image_name: 'ref.png', width: 512 }, kind: 'image' },
    ],
  });

  it('builds the reference graph: ordered chained collect into conditioning AND prompt, no frame conditioning', () => {
    const { backendGraph } = compileVideoGraph(referenceSettings, model);

    const video = nodeOfType(backendGraph, 'minimax_h3_video_reference');
    const image = nodeOfType(backendGraph, 'minimax_h3_image_reference');

    expect(video.id).toBe('reference_1');
    expect(video.conditioning).toBe('video_audio');
    expect(video.start_frame).toBe(2);
    expect(video.end_frame).toBe(-1);
    expect(image.id).toBe('reference_2');
    expect(image.detail).toBe('match');

    expect(hasEdge(backendGraph, 'reference_1', 'reference', 'reference_collect_1', 'item')).toBe(true);
    expect(hasEdge(backendGraph, 'reference_collect_1', 'collection', 'reference_collect_2', 'collection')).toBe(true);
    expect(hasEdge(backendGraph, 'reference_2', 'reference', 'reference_collect_2', 'item')).toBe(true);

    expect(hasEdge(backendGraph, 'reference_collect_2', 'collection', 'reference_conditioning', 'references')).toBe(
      true
    );
    expect(hasEdge(backendGraph, 'reference_collect_2', 'collection', 'pos_cond', 'references')).toBe(true);
    expect(
      hasEdge(
        backendGraph,
        'reference_conditioning',
        'reference_conditioning',
        'denoise_latents',
        'reference_conditioning'
      )
    ).toBe(true);
    expect(hasEdge(backendGraph, 'model_loader', 'vae', 'reference_conditioning', 'vae')).toBe(true);
    expect(hasEdge(backendGraph, 'model_loader', 'audio_vae', 'reference_conditioning', 'audio_vae')).toBe(true);

    const posCond = nodeOfType(backendGraph, 'minimax_h3_text_encoder');
    const referenceConditioning = nodeOfType(backendGraph, 'minimax_h3_reference_conditioning');

    expect(posCond.num_frames).toBe(referenceSettings.numFrames);
    expect(referenceConditioning.num_frames).toBe(referenceSettings.numFrames);
    expect(nodesOfType(backendGraph, 'minimax_h3_frame_conditioning')).toHaveLength(0);

    const metadata = nodeOfType(backendGraph, 'core_metadata');

    expect(metadata.generation_mode).toBe('minimax_h3_ref2v');
    expect(metadata.minimax_h3_references).toEqual([
      { conditioning: 'video_audio', end_frame: 47, kind: 'video', start_frame: 2, video_name: 'ref.mp4' },
      { detail: 'match', image_name: 'ref.png', kind: 'image' },
    ]);
  });

  it('hybrid: loads the FL2VA base, overlays the Ref2VA main from the start block, and records both', () => {
    const fl2vaBase: MainModelConfig = {
      base: 'minimax-h3',
      format: 'checkpoint',
      key: 'h3-fl2va-ckpt',
      name: 'MiniMax H3 FL2VA Transformer (int8, pruned)',
      type: 'main',
      variant: 'fl2va',
    };
    const turbo = {
      base: 'minimax-h3',
      key: 'ref2v-turbo',
      name: 'MiniMax H3 Ref2V Turbo LoRA',
      type: 'lora' as const,
    };
    const settings: VideoSettings = {
      ...referenceSettings,
      h3HybridBaseModel: fl2vaBase,
      h3HybridStartBlock: 30,
      loras: [{ isEnabled: true, model: turbo, weight: 1 }],
    };
    const { backendGraph } = compileVideoGraph(settings, model);

    expect(nodeOfType(backendGraph, 'minimax_h3_model_loader')).toMatchObject({
      model: { key: componentSource.key },
      transformer_model: { key: fl2vaBase.key },
    });
    expect(nodeOfType(backendGraph, 'minimax_h3_hybrid_overlay')).toMatchObject({
      end_block: 49,
      id: 'hybrid_overlay',
      include_final_layer: false,
      overlay_model: { key: model.key },
      start_block: 30,
    });
    expect(hasEdge(backendGraph, 'model_loader', 'transformer', 'hybrid_overlay', 'transformer')).toBe(true);

    const loraLoader = nodeOfType(backendGraph, 'minimax_h3_lora_collection_loader');

    expect(hasEdge(backendGraph, 'hybrid_overlay', 'transformer', loraLoader.id, 'transformer')).toBe(true);
    expect(hasEdge(backendGraph, loraLoader.id, 'transformer', 'denoise_latents', 'transformer')).toBe(true);
    expect(hasEdge(backendGraph, 'model_loader', 'transformer', 'denoise_latents', 'transformer')).toBe(false);

    expect(nodeOfType(backendGraph, 'core_metadata')).toMatchObject({
      minimax_h3_hybrid_base_model: { key: fl2vaBase.key },
      minimax_h3_hybrid_start_block: 30,
      model: { key: model.key },
    });
  });

  it('reference-extend: appends the new clip to the initial video without frame conditioning', () => {
    const initialVideo = {
      endFrame: 400,
      fps: 24,
      height: 480,
      numFrames: 402,
      startFrame: 10,
      video_name: 'long.mp4',
      width: 832,
    };
    const settings = {
      ...referenceSettings,
      references: [
        {
          clip: { ...initialVideo, endFrame: 400, startFrame: 260 },
          conditioning: 'video_audio' as const,
          fromSourceVideo: true,
          kind: 'video' as const,
        },
        ...referenceSettings.references,
      ],
      sourceVideo: initialVideo,
    };
    const { backendGraph } = compileVideoGraph(settings, model);

    expect(nodeOfType(backendGraph, 'minimax_h3_latents_to_video')).toMatchObject({
      id: 'extension_clip',
      is_intermediate: true,
    });
    expect(nodeOfType(backendGraph, 'video_concat')).toMatchObject({ id: 'video_output', transition: 'crossfade' });
    expect(nodeOfType(backendGraph, 'extract_video_range')).toMatchObject({ end_frame: -2, fps: 24, start_frame: 10 });
    expect(hasEdge(backendGraph, 'source_video', 'video', 'source_clip_collect', 'item')).toBe(true);
    expect(hasEdge(backendGraph, 'extension_clip', 'video', 'clips_to_join', 'item')).toBe(true);

    expect(nodesOfType(backendGraph, 'minimax_h3_frame_conditioning')).toHaveLength(0);
    expect(nodesOfType(backendGraph, 'video_frame_extract')).toHaveLength(0);

    const videoReferences = nodesOfType(backendGraph, 'minimax_h3_video_reference');

    expect(videoReferences[0]).toMatchObject({ end_frame: -2, id: 'reference_1', start_frame: -142 });
    expect((videoReferences[0].end_frame as number) - (videoReferences[0].start_frame as number)).toBe(140);
    const metadata = nodeOfType(backendGraph, 'core_metadata');

    expect(metadata.generation_mode).toBe('minimax_h3_ref2v');
    expect(metadata).toMatchObject({
      source_video: { video_name: 'long.mp4' },
      source_video_end_frame: 400,
      source_video_start_frame: 10,
    });
    expect((metadata.minimax_h3_references as Record<string, unknown>[])[0]).toEqual({
      conditioning: 'video_audio',
      end_frame: 260 + 140,
      kind: 'video',
      start_frame: 260,
      video_name: 'long.mp4',
    });
  });
});

const ltx2Model = (variant: string, format = 'checkpoint', key = `ltx2-${variant}-${format}`): MainModelConfig => ({
  base: 'ltx-2',
  format,
  key,
  name: `LTX-2 ${variant}`,
  type: 'main',
  variant,
});

const LTX2_COMPONENTS: MainModelConfig = {
  base: 'ltx-2',
  format: 'diffusers',
  key: 'ltx2-components',
  name: 'LTX-2.5 Components',
  type: 'main',
  variant: 'ltx2_dev',
};
const LTX2_ENCODER = { base: 'ltx-2', key: 'gemma4', name: 'LTX-2.5 Text Encoder', type: 'gemma4_encoder' as const };

const LTX2_DURATION_HEAD = {
  base: 'ltx-2',
  key: 'duration',
  name: 'LTX-2.5 Duration Head',
  type: 'ltx2_duration_head' as const,
};

const LTX2_SOURCE_CLIP = {
  endFrame: 94,
  fps: 24,
  height: 704,
  numFrames: 96,
  startFrame: 0,
  video_name: 'source.mp4',
  width: 1248,
};

/** A four-second 24 fps clip: 96 frames, which snaps DOWN to 89 on the VAE's 8n + 1 grid. */
const LTX2_CLIP = { fps: 24, height: 704, numFrames: 96, video_name: 'clip.mp4', width: 1248 };

const ltx2SettingsFor = (model: MainModelConfig, overrides: Partial<VideoSettings> = {}): VideoSettings =>
  settingsFor(model, {
    componentSourceModel: model.format === 'diffusers' ? null : LTX2_COMPONENTS,
    ltx2TextEncoderModel: LTX2_ENCODER,
    ...overrides,
  });

describe('compileVideoGraph — LTX-2', () => {
  it('assembles the generation from the transformer, the component folder and the Gemma-4 encoder', () => {
    const model = ltx2Model('ltx2_dev');
    const { backendGraph } = compileVideoGraph(ltx2SettingsFor(model), model);
    const loader = nodeOfType(backendGraph, 'ltx2_model_loader');

    expect(loader.model).toEqual(model);
    expect(loader.component_source).toEqual(LTX2_COMPONENTS);
    expect(loader.text_encoder_model).toEqual(LTX2_ENCODER);
  });

  it('generates at half the canvas and refines the upscaled latent', () => {
    const model = ltx2Model('ltx2_dev');
    const { backendGraph } = compileVideoGraph(
      ltx2SettingsFor(model, { aspectRatioId: '16:9', targetResolution: '1024p' }),
      model
    );
    const base = backendGraph.nodes.denoise_latents;
    const refine = backendGraph.nodes.refine_latents;

    // The base canvas is the final one halved.
    expect({ height: base?.height, width: base?.width }).toEqual({ height: 512, width: 896 });
    expect({ height: refine?.height, width: refine?.width }).toEqual({ height: 1024, width: 1792 });
    expect(hasEdge(backendGraph, 'denoise_latents', 'video_latents', 'latent_upsample', 'video_latents')).toBe(true);
    expect(hasEdge(backendGraph, 'latent_upsample', 'latents', 'refine_latents', 'latents')).toBe(true);
    expect(hasEdge(backendGraph, 'model_loader', 'latent_upsampler', 'latent_upsample', 'latent_upsampler')).toBe(true);
    expect(hasEdge(backendGraph, 'denoise_latents', 'audio_latents', 'refine_latents', 'audio_latents')).toBe(true);
    expect(hasEdge(backendGraph, 'latent_upsample', 'latents', 'video_output', 'video_latents')).toBe(false);
    expect(hasEdge(backendGraph, 'refine_latents', 'video_latents', 'video_output', 'video_latents')).toBe(true);
    expect(hasEdge(backendGraph, 'refine_latents', 'audio_latents', 'video_output', 'audio_latents')).toBe(true);
  });

  it("conditions on a clip's soundtrack and takes the length from the encoder, not the panel", () => {
    const model = ltx2Model('ltx2_dev');
    const { backendGraph } = compileVideoGraph(
      ltx2SettingsFor(model, { conditioningClip: { clip: LTX2_CLIP, fpsKnown: true, role: 'audio' }, numFrames: 121 }),
      model
    );
    const conditioning = nodeOfType(backendGraph, 'ltx2_audio_conditioning');
    const output = nodeOfType(backendGraph, 'ltx2_latents_to_video');

    expect(conditioning.video).toEqual({ video_name: LTX2_CLIP.video_name });
    expect(hasEdge(backendGraph, 'model_loader', 'audio_vae', conditioning.id, 'audio_vae')).toBe(true);
    expect(hasEdge(backendGraph, 'model_loader', 'vocoder', conditioning.id, 'vocoder')).toBe(true);
    expect(hasEdge(backendGraph, conditioning.id, 'audio_conditioning', 'denoise_latents', 'audio_conditioning')).toBe(
      true
    );
    // The clip's 89 frames, not the panel's 121, with the encoder's own count wired over that.
    expect(hasEdge(backendGraph, conditioning.id, 'num_frames', 'denoise_latents', 'num_frames')).toBe(true);
    expect(backendGraph.nodes.denoise_latents).toMatchObject({ num_frames: 89 });
    expect(hasEdge(backendGraph, conditioning.id, 'audio_conditioning', output.id, 'source_audio')).toBe(true);
  });

  it('records the conditioning clip and the frames that ran, for recall', () => {
    const model = ltx2Model('ltx2_dev');
    const { backendGraph } = compileVideoGraph(
      ltx2SettingsFor(model, { conditioningClip: { clip: LTX2_CLIP, fpsKnown: true, role: 'audio' }, numFrames: 121 }),
      model
    );

    expect(backendGraph.nodes.core_metadata).toMatchObject({
      generation_mode: 'ltx2_a2v',
      ltx2_conditioning_role: 'audio',
      ltx2_conditioning_video: { video_name: LTX2_CLIP.video_name },
      num_frames: 89,
    });
  });

  it('continues a clip from its own tail and consumes the overlap in the join', () => {
    const model = ltx2Model('ltx2_dev');
    const { backendGraph } = compileVideoGraph(ltx2SettingsFor(model, { sourceVideo: LTX2_SOURCE_CLIP }), model);
    const extend = nodeOfType(backendGraph, 'ltx2_extend_conditioning');
    const concat = nodeOfType(backendGraph, 'video_concat');

    expect(hasEdge(backendGraph, extend.id, 'video_conditioning', 'denoise_latents', 'video_conditioning')).toBe(true);
    expect(hasEdge(backendGraph, extend.id, 'context_frames', concat.id, 'transition_frames')).toBe(true);
    expect(concat).toMatchObject({ transition: 'crossfade' });
    const extract = nodeOfType(backendGraph, 'extract_video_range');

    expect(hasEdge(backendGraph, extract.id, 'fps', 'denoise_latents', 'fps')).toBe(true);
    expect(hasEdge(backendGraph, extract.id, 'fps', backendGraph.nodes.extension_clip!.id, 'fps')).toBe(true);
    expect(hasEdge(backendGraph, extract.id, 'fps', concat.id, 'fps')).toBe(false);
    expect(concat).not.toHaveProperty('fps');
    expect(extract).not.toHaveProperty('fps');

    expect(hasEdge(backendGraph, extract.id, 'video', extend.id, 'video')).toBe(true);
    expect(backendGraph.nodes.extension_clip).toMatchObject({ is_intermediate: true });
    expect(concat.is_intermediate).toBe(false);
    expect(backendGraph.nodes.core_metadata).toMatchObject({ generation_mode: 'ltx2_extend_video' });
  });

  it('records what recall needs and nothing it can derive', () => {
    const model = ltx2Model('ltx2_dev');
    const { backendGraph } = compileVideoGraph(ltx2SettingsFor(model), model);
    const metadata = nodeOfType(backendGraph, 'core_metadata');

    expect(metadata.generation_mode).toBe('ltx2_t2v');
    expect(metadata.ltx2_component_source).toEqual(LTX2_COMPONENTS);
    expect(metadata.ltx2_text_encoder_model).toEqual(LTX2_ENCODER);
    expect(metadata.ltx2_audio_cfg_scale).toBe(7);
    expect(metadata.ltx2_stg_scale).toBe(1);
    expect(metadata.ltx2_modality_scale).toBe(3);
    expect(metadata.fps).toBe(24);
  });

  it('refuses to compile without the Gemma-4 encoder no LTX-2 model carries', () => {
    const model = ltx2Model('ltx2_dev');

    expect(() => compileVideoGraph(ltx2SettingsFor(model, { ltx2TextEncoderModel: null }), model)).toThrow(
      /Gemma-4 text encoder/
    );
  });
});

describe('compileVideoGraph — LTX-2 auto duration', () => {
  const autoSettings = (model: MainModelConfig, overrides: Partial<VideoSettings> = {}) =>
    ltx2SettingsFor(model, { autoDuration: true, ltx2DurationHeadModel: LTX2_DURATION_HEAD, ...overrides });

  it('lets the duration head set the frame count on both the denoise and the recorded metadata', () => {
    const model = ltx2Model('ltx2_dev');
    const { backendGraph } = compileVideoGraph(autoSettings(model), model);
    const duration = nodeOfType(backendGraph, 'ltx2_duration');
    const denoise = nodeOfType(backendGraph, 'ltx2_denoise');
    const metadata = nodeOfType(backendGraph, 'core_metadata');

    expect(duration.duration_head).toEqual(LTX2_DURATION_HEAD);
    expect(hasEdge(backendGraph, duration.id, 'num_frames', denoise.id, 'num_frames')).toBe(true);
    expect(hasEdge(backendGraph, duration.id, 'num_frames', metadata.id, 'num_frames')).toBe(true);
  });
});
