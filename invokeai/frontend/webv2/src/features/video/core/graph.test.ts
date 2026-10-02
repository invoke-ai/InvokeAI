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

const WAN_VAE = { base: 'wan', key: 'wan-vae', latent_channels: 16, name: 'Wan 2.1 VAE', type: 'vae' as const };
const WAN_T5 = { base: 'any', key: 'umt5', name: 'UMT5-XXL', type: 'wan_t5_encoder' as const };

const FIRST_FRAME = { height: 1080, image_name: 'first.png', width: 1920 };
const LAST_FRAME = { height: 1080, image_name: 'last.png', width: 1920 };
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
    // A14B: null low-noise CFG falls back to the primary, not the node default.
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

  it('splices the transformer LoRA collection loader when LoRAs are active', () => {
    const model = wanModel('t2v_a14b');
    const lightning = {
      isEnabled: true,
      model: { base: 'wan', key: 'lit-high', name: 'Wan Lightning High Noise', type: 'lora' as const, variant: 'a14b' },
      weight: 1,
    };
    const { backendGraph } = compileVideoGraph(settingsFor(model, { loras: [lightning] }), model);
    const loader = nodeOfType(backendGraph, 'wan_lora_collection_loader');

    expect(hasEdge(backendGraph, 'model_loader', 'transformer', loader.id, 'transformer')).toBe(true);
    expect(hasEdge(backendGraph, loader.id, 'transformer', 'denoise_latents', 'transformer')).toBe(true);
    expect(nodeOfType(backendGraph, 'core_metadata').loras).toEqual([
      { model: expect.objectContaining({ key: 'lit-high' }), weight: 1 },
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

  it('adds the end frame for first-to-last interpolation', () => {
    const model = wanModel('i2v_a14b');
    const settings = settingsFor(model, { firstFrameImage: FIRST_FRAME, lastFrameImage: LAST_FRAME });
    const { backendGraph } = compileVideoGraph(settings, model);

    expect(nodeOfType(backendGraph, 'wan_ref_image_encoder')).toMatchObject({
      end_image: { image_name: 'last.png' },
      image: { image_name: 'first.png' },
    });
    expect(nodeOfType(backendGraph, 'core_metadata')).toMatchObject({
      generation_mode: 'wan_interpolate',
      last_frame_image: { image_name: 'last.png' },
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
    // The extracted last frame conditions the new clip via an edge, not a literal.
    expect(hasEdge(backendGraph, 'source_last_frame', 'image', 'ref_image_encoder', 'image')).toBe(true);
    expect(nodeOfType(backendGraph, 'wan_ref_image_encoder').image).toBeUndefined();

    // The freshly generated clip stays out of the gallery; the concat is the output.
    expect(nodeOfType(backendGraph, 'wan_l2v')).toMatchObject({ id: 'extension_clip', is_intermediate: true });
    expect(nodeOfType(backendGraph, 'video_concat')).toMatchObject({
      id: 'video_output',
      is_intermediate: false,
      size_mismatch: 'match_first',
      transition: 'crossfade',
      transition_frames: 2,
    });

    // Join order: [trimmed source, new clip] via the chained collectors.
    expect(hasEdge(backendGraph, 'source_video', 'video', 'source_clip_collect', 'item')).toBe(true);
    expect(hasEdge(backendGraph, 'source_clip_collect', 'collection', 'clips_to_join', 'collection')).toBe(true);
    expect(hasEdge(backendGraph, 'extension_clip', 'video', 'clips_to_join', 'item')).toBe(true);
    expect(hasEdge(backendGraph, 'clips_to_join', 'collection', 'video_output', 'videos')).toBe(true);

    // The extension inherits the source clip's frame rate.
    expect(hasEdge(backendGraph, 'source_video', 'fps', 'source_fps', 'value')).toBe(true);
    expect(hasEdge(backendGraph, 'source_fps', 'value', 'extension_clip', 'fps')).toBe(true);
    expect(hasEdge(backendGraph, 'source_fps', 'value', 'video_output', 'fps')).toBe(true);

    expect(nodeOfType(backendGraph, 'core_metadata')).toMatchObject({
      generation_mode: 'wan_extend_video',
      source_video: { video_name: 'clip.mp4' },
    });
    // Record metadata on both clips and wire runtime conditioning-frame identity through an edge.
    expect(hasEdge(backendGraph, 'core_metadata', 'metadata', 'video_output', 'metadata')).toBe(true);
    expect(hasEdge(backendGraph, 'core_metadata', 'metadata', 'extension_clip', 'metadata')).toBe(true);
    expect(hasEdge(backendGraph, 'source_last_frame', 'image', 'core_metadata', 'first_frame_image')).toBe(true);
  });

  it('refuses fractional trim bounds — extract_video_range takes integers', () => {
    const model = wanModel('i2v_a14b');
    const settings = settingsFor(model, { sourceVideo: { ...SOURCE_VIDEO, endFrame: -1.5 } });

    expect(() => compileVideoGraph(settings, model)).toThrow(/whole frame numbers/);
  });

  it('compiles ceiling-touching trim ends as negative indices (estimate-proof)', () => {
    const model = wanModel('i2v_a14b');
    // endFrame == numFrames - 2 (the default trim): compiled as -2 so the
    // backend resolves it against the clip's REAL frame count.
    const atDefault = compileVideoGraph(settingsFor(model, { sourceVideo: SOURCE_VIDEO }), model).backendGraph;

    expect(nodeOfType(atDefault, 'extract_video_range').end_frame).toBe(-2);

    // A mid-clip pick stays positive.
    const midClip = compileVideoGraph(
      settingsFor(model, { sourceVideo: { ...SOURCE_VIDEO, endFrame: 40 } }),
      model
    ).backendGraph;

    expect(nodeOfType(midClip, 'extract_video_range').end_frame).toBe(40);
  });

  it('compiles ceiling-touching trim STARTS as negative indices too — a keep-the-tail trim survives estimate overshoot', () => {
    const model = wanModel('i2v_a14b');
    // Near-tail bounds convert together to negative offsets while preserving order.
    const tailTrim = compileVideoGraph(
      settingsFor(model, { sourceVideo: { ...SOURCE_VIDEO, endFrame: 79, startFrame: 77 } }),
      model
    ).backendGraph;

    expect(nodeOfType(tailTrim, 'extract_video_range')).toMatchObject({ end_frame: -2, start_frame: -4 });

    // Bounds outside tail slop remain absolute to avoid drifting mid-clip picks.
    const boundary = compileVideoGraph(
      settingsFor(model, { sourceVideo: { ...SOURCE_VIDEO, endFrame: 78, startFrame: 76 } }),
      model
    ).backendGraph;

    expect(nodeOfType(boundary, 'extract_video_range')).toMatchObject({ end_frame: -3, start_frame: 76 });
  });

  it('records the delivered fps in Wan metadata — the panel fps normally, the source clip fps for extend', () => {
    const t2vModel = wanModel('t2v_a14b');
    const t2v = compileVideoGraph(settingsFor(t2vModel, { fps: 20 }), t2vModel).backendGraph;

    expect(nodeOfType(t2v, 'core_metadata')).toMatchObject({ fps: 20 });

    // Extend inherits the source clip's (rounded) rate, not the panel setting.
    const extendModel = wanModel('i2v_a14b');
    const extend = compileVideoGraph(
      settingsFor(extendModel, { fps: 20, sourceVideo: { ...SOURCE_VIDEO, fps: 23.7 } }),
      extendModel
    ).backendGraph;

    expect(nodeOfType(extend, 'core_metadata')).toMatchObject({ fps: 24 });
  });

  it('refuses a trim shorter than two frames', () => {
    const model = wanModel('i2v_a14b');

    expect(() =>
      compileVideoGraph(settingsFor(model, { sourceVideo: { ...SOURCE_VIDEO, endFrame: 10, startFrame: 10 } }), model)
    ).toThrow(/at least two frames/);
    expect(() =>
      compileVideoGraph(settingsFor(model, { sourceVideo: { ...SOURCE_VIDEO, endFrame: 500 } }), model)
    ).toThrow(/at least two frames/);
  });

  it('extends toward a destination image via the FLF2V end-frame channel', () => {
    const model = wanModel('i2v_a14b');
    const settings = settingsFor(model, { lastFrameImage: LAST_FRAME, sourceVideo: SOURCE_VIDEO });
    const { backendGraph } = compileVideoGraph(settings, model);

    expect(nodeOfType(backendGraph, 'wan_ref_image_encoder')).toMatchObject({
      end_image: { image_name: 'last.png' },
    });
  });

  it('passes standalone components for single-file mains and omits the low-noise expert for Diffusers', () => {
    const gguf = wanModel('i2v_a14b', 'gguf_quantized');
    const lowExpert = wanModel('i2v_a14b', 'checkpoint', 'low-expert');
    const settings = settingsFor(gguf, {
      firstFrameImage: FIRST_FRAME,
      vae: WAN_VAE,
      wanLowNoiseModel: lowExpert,
      wanT5EncoderModel: WAN_T5,
    });
    const { backendGraph } = compileVideoGraph(settings, gguf);

    expect(nodeOfType(backendGraph, 'wan_model_loader')).toMatchObject({
      transformer_low_noise_model: { key: 'low-expert' },
      vae_model: { key: 'wan-vae' },
      wan_t5_encoder_model: { key: 'umt5' },
    });
    expect(nodeOfType(backendGraph, 'core_metadata')).toMatchObject({
      vae: { key: 'wan-vae' },
      wan_t5_encoder_model: { key: 'umt5' },
      wan_transformer_low_noise: { key: 'low-expert' },
    });

    const diffusers = wanModel('i2v_a14b', 'diffusers');
    const diffusersGraph = compileVideoGraph(
      settingsFor(diffusers, { firstFrameImage: FIRST_FRAME, wanLowNoiseModel: lowExpert }),
      diffusers
    ).backendGraph;

    // Diffusers mains bundle transformer_2; the loader input would be ignored.
    expect(nodeOfType(diffusersGraph, 'wan_model_loader').transformer_low_noise_model).toBeUndefined();
  });

  it('omits the low-noise guidance for TI2V-5B and snaps to its ×32 grid', () => {
    const model = wanModel('ti2v_5b');
    const settings = settingsFor(model, { aspectRatioId: '16:9', targetResolution: '720p' });
    const { backendGraph } = compileVideoGraph(settings, model);
    const denoise = nodeOfType(backendGraph, 'wan_video_denoise');

    expect(denoise.guidance_scale_low_noise).toBeUndefined();
    // 16:9 at 720p with banker's rounding on the ×32 grid: 1280×704.
    expect(denoise).toMatchObject({ height: 704, width: 1280 });
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

    // H3 canvas follows preset policy; node steps include one terminal sigma point beyond model evaluations.
    expect(denoise).toMatchObject({ height: 768, num_frames: '124', steps: settings.steps + 1, width: 1344 });

    const output = nodeOfType(backendGraph, 'minimax_h3_latents_to_video');

    expect(output.id).toBe('video_output');
    expect(hasEdge(backendGraph, 'denoise_latents', 'video_latents', 'video_output', 'video_latents')).toBe(true);
    expect(hasEdge(backendGraph, 'denoise_latents', 'audio_latents', 'video_output', 'audio_latents')).toBe(true);
    expect(hasEdge(backendGraph, 'model_loader', 'vae', 'video_output', 'vae')).toBe(true);
    expect(hasEdge(backendGraph, 'model_loader', 'audio_vae', 'video_output', 'audio_vae')).toBe(true);

    expect(nodesOfType(backendGraph, 'minimax_h3_frame_conditioning')).toHaveLength(0);
    expect(nodeOfType(backendGraph, 'core_metadata')).toMatchObject({ fps: 24, generation_mode: 'minimax_h3_t2v' });
    // H3 has no negative prompt; the metadata must not claim one.
    expect(hasEdge(backendGraph, 'negative_prompt', 'value', 'core_metadata', 'negative_prompt')).toBe(false);
  });

  it('mirrors keyframes into both the text encoder and the frame conditioning', () => {
    const settings = settingsFor(model, { firstFrameImage: FIRST_FRAME });
    const { backendGraph } = compileVideoGraph(settings, model);
    const posCond = nodeOfType(backendGraph, 'minimax_h3_text_encoder');
    const frameCond = nodeOfType(backendGraph, 'minimax_h3_frame_conditioning');
    const denoise = nodeOfType(backendGraph, 'minimax_h3_denoise');

    for (const node of [posCond, frameCond]) {
      expect(node).toMatchObject({
        first_image: { image_name: 'first.png' },
        height: denoise.height,
        width: denoise.width,
      });
      expect(node.last_image).toBeUndefined();
    }

    expect(hasEdge(backendGraph, 'model_loader', 'vae', 'frame_conditioning', 'vae')).toBe(true);
    expect(
      hasEdge(backendGraph, 'frame_conditioning', 'frame_conditioning', 'denoise_latents', 'frame_conditioning')
    ).toBe(true);
    expect(nodeOfType(backendGraph, 'core_metadata').generation_mode).toBe('minimax_h3_i2v');
  });

  it('supports last-frame-only conditioning', () => {
    const settings = settingsFor(model, { lastFrameImage: LAST_FRAME });
    const { backendGraph } = compileVideoGraph(settings, model);

    expect(nodeOfType(backendGraph, 'minimax_h3_frame_conditioning')).toMatchObject({
      last_image: { image_name: 'last.png' },
    });
    expect(nodeOfType(backendGraph, 'minimax_h3_frame_conditioning').first_image).toBeUndefined();
    expect(nodeOfType(backendGraph, 'core_metadata').generation_mode).toBe('minimax_h3_lf2v');
  });

  it('builds the extend graph with the source resampled to 24 fps', () => {
    const settings = settingsFor(model, { sourceVideo: { ...SOURCE_VIDEO, endFrame: 80 } });
    const { backendGraph } = compileVideoGraph(settings, model);

    // Retime source to H3's fixed 24 fps before concatenation inherits its rate.
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

  it('a full Diffusers main is the loader model directly, with no overrides recorded', () => {
    const { backendGraph } = compileVideoGraph(settingsFor(model), model);
    const loader = nodeOfType(backendGraph, 'minimax_h3_model_loader');
    const metadata = nodeOfType(backendGraph, 'core_metadata');

    expect(loader).toMatchObject({ model: { key: model.key } });
    expect(loader.transformer_model).toBeUndefined();
    expect(metadata.minimax_h3_component_source).toBeUndefined();
  });

  it('refuses to compile fractional or off-grid frame counts', () => {
    expect(() => compileVideoGraph(settingsFor(model, { numFrames: 100 }), model)).toThrow(/17·n \+ 5/);
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
    // Resolve tail-window bounds against actual backend frame counts to tolerate VFR estimate overshoot.
    expect(video.end_frame).toBe(-1);
    expect(image.id).toBe('reference_2');
    expect(image.detail).toBe('match');

    // Order is contractual: chained collectors, each appending its item after the
    // inherited collection.
    expect(hasEdge(backendGraph, 'reference_1', 'reference', 'reference_collect_1', 'item')).toBe(true);
    expect(hasEdge(backendGraph, 'reference_collect_1', 'collection', 'reference_collect_2', 'collection')).toBe(true);
    expect(hasEdge(backendGraph, 'reference_2', 'reference', 'reference_collect_2', 'item')).toBe(true);

    // The final collection fans out to BOTH consumers.
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

    // num_frames rides on prompt + reference conditioning; frame conditioning is absent.
    const posCond = nodeOfType(backendGraph, 'minimax_h3_text_encoder');
    const referenceConditioning = nodeOfType(backendGraph, 'minimax_h3_reference_conditioning');

    expect(posCond.num_frames).toBe(referenceSettings.numFrames);
    expect(referenceConditioning.num_frames).toBe(referenceSettings.numFrames);
    expect(nodesOfType(backendGraph, 'minimax_h3_frame_conditioning')).toHaveLength(0);

    // Metadata records the mode and the ordered reference payload for recall.
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

    // The loader loads the FL2VA base; the selected Ref2VA main becomes the overlay.
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

    // LoRAs apply on top of the hybrid, and the denoise reads the LoRA-patched hybrid.
    const loraLoader = nodeOfType(backendGraph, 'minimax_h3_lora_collection_loader');

    expect(hasEdge(backendGraph, 'hybrid_overlay', 'transformer', loraLoader.id, 'transformer')).toBe(true);
    expect(hasEdge(backendGraph, loraLoader.id, 'transformer', 'denoise_latents', 'transformer')).toBe(true);
    expect(hasEdge(backendGraph, 'model_loader', 'transformer', 'denoise_latents', 'transformer')).toBe(false);

    // Metadata keeps the Ref2VA main as the model and records the hybrid for recall.
    expect(nodeOfType(backendGraph, 'core_metadata')).toMatchObject({
      minimax_h3_hybrid_base_model: { key: fl2vaBase.key },
      minimax_h3_hybrid_start_block: 30,
      model: { key: model.key },
    });
  });

  it('hybrid: a stale quality base on an fl2va main is ignored', () => {
    const fl2vaMain: MainModelConfig = { ...h3Model('h3-fl2va-main'), format: 'checkpoint' };
    const settings: VideoSettings = {
      ...settingsFor(fl2vaMain, { componentSourceModel: componentSource }),
      h3HybridBaseModel: { ...h3Model('h3-other-fl2va'), format: 'checkpoint' },
      h3HybridStartBlock: 10,
    };
    const { backendGraph } = compileVideoGraph(settings, fl2vaMain);

    expect(nodesOfType(backendGraph, 'minimax_h3_hybrid_overlay')).toHaveLength(0);
    expect(nodeOfType(backendGraph, 'minimax_h3_model_loader')).toMatchObject({
      transformer_model: { key: fl2vaMain.key },
    });
    expect(hasEdge(backendGraph, 'model_loader', 'transformer', 'denoise_latents', 'transformer')).toBe(true);
    expect(nodeOfType(backendGraph, 'core_metadata').minimax_h3_hybrid_base_model).toBeUndefined();
  });

  it('refuses to compile references on an fl2va model', () => {
    expect(() => compileVideoGraph({ ...referenceSettings, componentSourceModel: null }, componentSource)).toThrow(
      /reference-conditioned/
    );
  });

  it('refuses to compile a single-file main with no component source', () => {
    expect(() => compileVideoGraph({ ...referenceSettings, componentSourceModel: null }, model)).toThrow(
      /Model Components/
    );
  });

  it('a window ending in the tail anchors BOTH bounds to the clip end', () => {
    const linked = (clip: Record<string, unknown>, flag = true) => ({
      clip: { fps: 24, height: 480, numFrames: 402, video_name: 'long.mp4', width: 832, ...clip },
      conditioning: 'video_audio' as const,
      ...(flag ? { fromSourceVideo: true } : {}),
      kind: 'video' as const,
    });
    const startOf = (reference: unknown) =>
      nodeOfType(
        compileVideoGraph({ ...referenceSettings, references: [reference] } as never, model).backendGraph,
        'minimax_h3_video_reference'
      ).start_frame;

    // Convert both reference bounds together; mixed absolute/relative indices distort length or reverse short
    // windows.
    expect(startOf(linked({ endFrame: 400, startFrame: 260 }, false))).toBe(-142);
    // The inversion that mixing produced: [398,399] of an estimated 402 emitted
    // `398 / -3`, which against a real 400 is start 398, end 397.
    expect(startOf(linked({ endFrame: 399, startFrame: 398 }, false))).toBe(-4);
    // Keep near-start bounds absolute because estimate overshoot could make a relative start negative and fail
    // extraction.
    expect(startOf(linked({ endFrame: 400, startFrame: 0 }))).toBe(0);
    expect(startOf(linked({ endFrame: 400, startFrame: 1 }))).toBe(1);
    expect(startOf(linked({ endFrame: 400, startFrame: 3 }))).toBe(3);
    // Clear of the slop, the window rides the negative anchor again.
    expect(startOf(linked({ endFrame: 400, startFrame: 4 }))).toBe(-398);
    // A cutpoint far enough from the end that BOTH bounds keep the estimate.
    expect(startOf(linked({ endFrame: 300, startFrame: 160 }))).toBe(160);
    // The anchor's tail window: end went negative, so the start follows it.
    expect(startOf(linked({ endFrame: 400, startFrame: 260 }))).toBe(-142);
  });

  it('fl2va graphs are unchanged by the ref2va machinery', () => {
    const { backendGraph } = compileVideoGraph(settingsFor(componentSource), componentSource);

    expect(nodesOfType(backendGraph, 'minimax_h3_reference_conditioning')).toHaveLength(0);
    expect(nodesOfType(backendGraph, 'collect')).toHaveLength(0);
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
        // The linked tail reference (as the setter derives it) plus a user reference.
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

    // Continuity comes from the references — no frame conditioning, no last-frame extraction.
    expect(nodesOfType(backendGraph, 'minimax_h3_frame_conditioning')).toHaveLength(0);
    expect(nodesOfType(backendGraph, 'video_frame_extract')).toHaveLength(0);

    // The linked reference is an ordinary first reference; the flag never reaches metadata.
    const videoReferences = nodesOfType(backendGraph, 'minimax_h3_video_reference');

    // Use one tail-relative anchor for both bounds to preserve window length despite frame-count estimation error.
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

  it('decodes through the video VAE, the audio VAE and the vocoder', () => {
    const model = ltx2Model('ltx2_dev');
    const { backendGraph } = compileVideoGraph(ltx2SettingsFor(model), model);
    const output = nodeOfType(backendGraph, 'ltx2_latents_to_video');

    // The soundtrack is generated with the picture, so its whole decode chain has to be wired or
    // the clip comes out silent.
    expect(hasEdge(backendGraph, 'denoise_latents', 'video_latents', output.id, 'video_latents')).toBe(true);
    expect(hasEdge(backendGraph, 'denoise_latents', 'audio_latents', output.id, 'audio_latents')).toBe(true);
    expect(hasEdge(backendGraph, 'model_loader', 'vae', output.id, 'vae')).toBe(true);
    expect(hasEdge(backendGraph, 'model_loader', 'audio_vae', output.id, 'audio_vae')).toBe(true);
    expect(hasEdge(backendGraph, 'model_loader', 'vocoder', output.id, 'vocoder')).toBe(true);
  });

  it('generates at half the canvas and refines the upscaled latent', () => {
    const model = ltx2Model('ltx2_dev');
    const { backendGraph } = compileVideoGraph(
      ltx2SettingsFor(model, { aspectRatioId: '16:9', targetResolution: '1024p' }),
      model
    );
    const base = backendGraph.nodes.denoise_latents;
    const refine = backendGraph.nodes.refine_latents;

    // The x2 upscaler doubles a latent grid exactly, so the base canvas is the final one halved --
    // not a separate resolution that happens to be smaller.
    expect({ height: base?.height, width: base?.width }).toEqual({ height: 512, width: 896 });
    expect({ height: refine?.height, width: refine?.width }).toEqual({ height: 1024, width: 1792 });
    expect(hasEdge(backendGraph, 'denoise_latents', 'video_latents', 'latent_upsample', 'video_latents')).toBe(true);
    expect(hasEdge(backendGraph, 'latent_upsample', 'latents', 'refine_latents', 'latents')).toBe(true);
    expect(hasEdge(backendGraph, 'model_loader', 'latent_upsampler', 'latent_upsample', 'latent_upsampler')).toBe(true);
    // Audio has no spatial extent, so it skips the upscaler -- but it still goes through the refine
    // denoise, which re-noises both modalities to one level.
    expect(hasEdge(backendGraph, 'denoise_latents', 'audio_latents', 'refine_latents', 'audio_latents')).toBe(true);
    expect(hasEdge(backendGraph, 'latent_upsample', 'latents', 'video_output', 'video_latents')).toBe(false);
    expect(hasEdge(backendGraph, 'refine_latents', 'video_latents', 'video_output', 'video_latents')).toBe(true);
    expect(hasEdge(backendGraph, 'refine_latents', 'audio_latents', 'video_output', 'audio_latents')).toBe(true);
  });

  it('coerces a preset the model does not offer instead of compiling a graph of NaNs', () => {
    // A record persisted under another family keeps its own preset, and only a model *selection*
    // re-coerces it. An unknown preset has no short edge, so every dimension would come back NaN --
    // and NaN compares unequal to itself, so a stage count recovered by comparing canvases would
    // have said "two". The graph must be the one the panel promised: a single pass at the default.
    const model = ltx2Model('ltx2_dev');
    const { backendGraph } = compileVideoGraph(ltx2SettingsFor(model, { targetResolution: '720p' as never }), model);
    const denoise = nodeOfType(backendGraph, 'ltx2_denoise');

    expect(Object.values(backendGraph.nodes).some((node) => node.type === 'ltx2_latent_upsample')).toBe(false);
    expect(denoise.width).toBe(1248);
    expect(denoise.height).toBe(704);
    for (const node of Object.values(backendGraph.nodes)) {
      for (const [field, value] of Object.entries(node)) {
        expect(Number.isNaN(value), `${String(node.type)}.${field} is NaN`).toBe(false);
      }
    }
  });

  it('never lets the refine pass outlast a base pass the user shortened', () => {
    // Cutting Steps for a quick probe must not leave the expensive half of the run longer than the
    // half that was just cut.
    const model = ltx2Model('ltx2_dev');
    const { backendGraph } = compileVideoGraph(ltx2SettingsFor(model, { steps: 4, targetResolution: '1024p' }), model);

    expect(backendGraph.nodes.denoise_latents?.steps).toBe(4);
    expect(backendGraph.nodes.refine_latents?.steps).toBe(4);
  });

  it('gives the refine pass its own step budget where the variant sets one', () => {
    // Dev pays four forwards a step; inheriting the base pass's 30 would be over an hour of refine
    // at this canvas. The distilled schedule is fixed, so it has no budget of its own to apply.
    const dev = ltx2Model('ltx2_dev');
    const devGraph = compileVideoGraph(ltx2SettingsFor(dev, { targetResolution: '1024p' }), dev).backendGraph;

    expect(devGraph.nodes.denoise_latents?.steps).toBe(30);
    expect(devGraph.nodes.refine_latents?.steps).toBe(8);

    const distilled = ltx2Model('ltx2_distilled');
    const distilledGraph = compileVideoGraph(
      ltx2SettingsFor(distilled, { targetResolution: '1024p' }),
      distilled
    ).backendGraph;

    expect(distilledGraph.nodes.refine_latents?.steps).toBe(distilledGraph.nodes.denoise_latents?.steps);
  });

  it("anchors a first frame on the base pass, at the base pass's canvas", () => {
    const model = ltx2Model('ltx2_dev');
    const { backendGraph } = compileVideoGraph(
      ltx2SettingsFor(model, {
        aspectRatioId: '16:9',
        firstFrameImage: { height: 1080, image_name: 'frame.png', width: 1920 },
        targetResolution: '1024p',
      }),
      model
    );
    const conditioning = backendGraph.nodes.image_conditioning;

    // Each pass is anchored at its own canvas. The refine pass re-noises every token, frame 0
    // included, so the base pass's anchor does not survive into it -- and that encode is half the
    // size, so it cannot be reused. Without a second encode, two-stage image-to-video would
    // regenerate the first frame from the prompt alone.
    expect({ height: conditioning?.height, width: conditioning?.width }).toEqual({ height: 512, width: 896 });
    expect(
      hasEdge(backendGraph, 'image_conditioning', 'video_conditioning', 'denoise_latents', 'video_conditioning')
    ).toBe(true);

    const refineConditioning = backendGraph.nodes.refine_image_conditioning;

    expect({ height: refineConditioning?.height, width: refineConditioning?.width }).toEqual({
      height: 1024,
      width: 1792,
    });
    expect(
      hasEdge(backendGraph, 'refine_image_conditioning', 'video_conditioning', 'refine_latents', 'video_conditioning')
    ).toBe(true);
  });

  it('adds no second image encode when there is no first frame to anchor', () => {
    const model = ltx2Model('ltx2_dev');
    const { backendGraph } = compileVideoGraph(ltx2SettingsFor(model, { targetResolution: '1024p' }), model);

    expect(backendGraph.nodes.refine_image_conditioning).toBeUndefined();
    expect(Object.values(backendGraph.nodes).some((node) => node.type === 'ltx2_image_conditioning')).toBe(false);
  });

  it('records the pair of canvases a two-stage run used', () => {
    const model = ltx2Model('ltx2_dev');
    const twoStage = compileVideoGraph(
      ltx2SettingsFor(model, { aspectRatioId: '16:9', targetResolution: '1024p' }),
      model
    ).backendGraph;
    const single = compileVideoGraph(
      ltx2SettingsFor(model, { aspectRatioId: '16:9', targetResolution: '704p' }),
      model
    ).backendGraph;

    expect(twoStage.nodes.core_metadata).toMatchObject({
      height: 1024,
      ltx2_base_height: 512,
      ltx2_base_width: 896,
      ltx2_two_stage: true,
      width: 1792,
    });
    // A single-stage run says nothing about stages rather than saying "one".
    expect(single.nodes.core_metadata).not.toHaveProperty('ltx2_two_stage');
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
    // The soundtrack's own length is authoritative: the panel's 121 gives way to the clip's 89,
    // and the encoder's own count is wired in over even that.
    expect(hasEdge(backendGraph, conditioning.id, 'num_frames', 'denoise_latents', 'num_frames')).toBe(true);
    expect(backendGraph.nodes.denoise_latents).toMatchObject({ num_frames: 89 });
    // The user's recording is muxed back in rather than a vocoder's copy of its own latents.
    expect(hasEdge(backendGraph, conditioning.id, 'audio_conditioning', output.id, 'source_audio')).toBe(true);
  });

  it('anchors the picture generated for a soundtrack on first and last frames', () => {
    const model = ltx2Model('ltx2_dev');
    const first = { height: 704, image_name: 'first.png', width: 1248 };
    const last = { height: 704, image_name: 'last.png', width: 1248 };
    const { backendGraph } = compileVideoGraph(
      ltx2SettingsFor(model, {
        conditioningClip: { clip: LTX2_CLIP, fpsKnown: true, role: 'audio' },
        firstFrameImage: first,
        lastFrameImage: last,
        numFrames: 121,
      }),
      model
    );
    const audio = nodeOfType(backendGraph, 'ltx2_audio_conditioning');

    expect(backendGraph.nodes.image_conditioning).toMatchObject({ frame_index: 0, image: { image_name: 'first.png' } });
    expect(backendGraph.nodes.last_frame_conditioning).toMatchObject({
      frame_index: -1,
      image: { image_name: 'last.png' },
    });
    expect(
      hasEdge(backendGraph, 'image_conditioning', 'video_conditioning', 'denoise_latents', 'video_conditioning')
    ).toBe(true);
    expect(
      hasEdge(backendGraph, 'last_frame_conditioning', 'video_conditioning', 'denoise_latents', 'keyframe_conditioning')
    ).toBe(true);
    // The soundtrack is still held and still decides the length the last frame's -1 resolves against.
    expect(hasEdge(backendGraph, audio.id, 'audio_conditioning', 'denoise_latents', 'audio_conditioning')).toBe(true);
    expect(hasEdge(backendGraph, audio.id, 'num_frames', 'denoise_latents', 'num_frames')).toBe(true);
    expect(backendGraph.nodes.core_metadata).toMatchObject({
      first_frame_image: { image_name: 'first.png' },
      generation_mode: 'ltx2_a2v',
      last_frame_image: { image_name: 'last.png' },
      ltx2_conditioning_role: 'audio',
    });
  });

  it("conditions on a clip's picture at the canvas its own ratio resolves to", () => {
    const model = ltx2Model('ltx2_dev');
    const { backendGraph } = compileVideoGraph(
      // A 4:3 clip against a 16:9 preset: the clip wins, so the conditioning encode and the
      // denoise have to agree on the canvas or `_load_image_latents`' geometry check fires.
      ltx2SettingsFor(model, {
        aspectRatioId: '16:9',
        conditioningClip: { clip: { ...LTX2_CLIP, height: 480, width: 640 }, fpsKnown: true, role: 'video' },
      }),
      model
    );
    const conditioning = nodeOfType(backendGraph, 'ltx2_video_conditioning');
    const denoise = nodeOfType(backendGraph, 'ltx2_denoise');

    // Both sides come from the same expression, so equality alone would still hold if the canvas
    // stopped following the clip -- they would simply both be 16:9. Pin the ratio itself.
    expect(Number(denoise.width) / Number(denoise.height)).toBeCloseTo(4 / 3, 1);
    expect(conditioning.width).toBe(denoise.width);
    expect(conditioning.height).toBe(denoise.height);
    expect(hasEdge(backendGraph, 'model_loader', 'vae', conditioning.id, 'vae')).toBe(true);
    expect(
      hasEdge(backendGraph, conditioning.id, 'video_conditioning', 'denoise_latents', 'full_video_conditioning')
    ).toBe(true);
    expect(hasEdge(backendGraph, conditioning.id, 'num_frames', 'denoise_latents', 'num_frames')).toBe(true);
    // The picture is the given one, so the run adopts the clip's rate, not the panel's.
    expect(denoise.fps).toBe(LTX2_CLIP.fps);
    // Its own soundtrack is the thing being generated, so nothing is muxed back in.
    const decode = nodeOfType(backendGraph, 'ltx2_latents_to_video');

    expect(decode.source_audio).toBeUndefined();
    // But the picture was the given half, so the user's own frames are written out rather than the
    // held latents being decoded into a cover-cropped copy of footage they already have.
    expect(hasEdge(backendGraph, conditioning.id, 'video_conditioning', decode.id, 'source_video')).toBe(true);
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

  it('holds a last frame as a keyframe, so it can coexist with a first one', () => {
    const model = ltx2Model('ltx2_dev');
    const { backendGraph } = compileVideoGraph(
      ltx2SettingsFor(model, { firstFrameImage: FIRST_FRAME, lastFrameImage: LAST_FRAME }),
      model
    );
    const first = backendGraph.nodes.image_conditioning;
    const last = backendGraph.nodes.last_frame_conditioning;

    // Index 0 overwrites the grid's opening tokens; -1 is appended to the sequence instead, which
    // is the only reason both can be held at once.
    expect(first).toMatchObject({ frame_index: 0 });
    expect(last).toMatchObject({ frame_index: -1 });
    expect(
      hasEdge(backendGraph, 'image_conditioning', 'video_conditioning', 'denoise_latents', 'video_conditioning')
    ).toBe(true);
    expect(
      hasEdge(backendGraph, 'last_frame_conditioning', 'video_conditioning', 'denoise_latents', 'keyframe_conditioning')
    ).toBe(true);
    expect(backendGraph.nodes.core_metadata).toMatchObject({ generation_mode: 'ltx2_flf2v' });
  });

  it('holds a last frame alone for last-frame-only generation', () => {
    const model = ltx2Model('ltx2_dev');
    const { backendGraph } = compileVideoGraph(ltx2SettingsFor(model, { lastFrameImage: LAST_FRAME }), model);

    expect(backendGraph.nodes.image_conditioning).toBeUndefined();
    expect(backendGraph.nodes.last_frame_conditioning).toMatchObject({ frame_index: -1 });
    expect(backendGraph.nodes.core_metadata).toMatchObject({ generation_mode: 'ltx2_lf2v' });
  });

  it('re-encodes every held frame at the refine canvas', () => {
    const model = ltx2Model('ltx2_dev');
    const { backendGraph } = compileVideoGraph(
      ltx2SettingsFor(model, {
        aspectRatioId: '16:9',
        firstFrameImage: FIRST_FRAME,
        lastFrameImage: LAST_FRAME,
        targetResolution: '1024p',
      }),
      model
    );

    // The refine pass re-noises every token, and appended keyframes never go through the upsampler
    // at all -- so both frames have to be encoded again, at this pass's own (doubled) canvas.
    expect(backendGraph.nodes.refine_image_conditioning).toMatchObject({ frame_index: 0, height: 1024, width: 1792 });
    expect(backendGraph.nodes.refine_last_frame_conditioning).toMatchObject({ frame_index: -1, height: 1024 });
    expect(backendGraph.nodes.image_conditioning).toMatchObject({ height: 512, width: 896 });
    expect(
      hasEdge(
        backendGraph,
        'refine_last_frame_conditioning',
        'video_conditioning',
        'refine_latents',
        'keyframe_conditioning'
      )
    ).toBe(true);
  });

  it('patches the distilled LoRA onto the transformer both passes read', () => {
    const model = ltx2Model('ltx2_dev');
    const distilled = { base: 'ltx-2', key: 'ltx2-distilled', name: 'LTX-2.5 Distilled LoRA', type: 'lora' as const };
    const { backendGraph } = compileVideoGraph(
      ltx2SettingsFor(model, {
        acceleratorEnabled: true,
        acceleratorLoraKeys: [distilled.key],
        aspectRatioId: '16:9',
        loras: [{ isEnabled: true, model: distilled, weight: 1 }],
        targetResolution: '1024p',
      }),
      model
    );
    const loraLoader = nodeOfType(backendGraph, 'ltx2_lora_collection_loader');

    expect(hasEdge(backendGraph, 'model_loader', 'transformer', loraLoader.id, 'transformer')).toBe(true);
    // Both passes share one transformer, so both must read the patched one -- a refine pass wired
    // straight to the loader would sample the second half of the run unpatched.
    expect(hasEdge(backendGraph, loraLoader.id, 'transformer', 'denoise_latents', 'transformer')).toBe(true);
    expect(hasEdge(backendGraph, loraLoader.id, 'transformer', 'refine_latents', 'transformer')).toBe(true);
    expect(hasEdge(backendGraph, 'model_loader', 'transformer', 'denoise_latents', 'transformer')).toBe(false);
    expect(hasEdge(backendGraph, 'model_loader', 'transformer', 'refine_latents', 'transformer')).toBe(false);
  });

  it('names the distilled schedule, which the checkpoint variant cannot', () => {
    // `auto` resolves the schedule off the transformer's VARIANT, which names the checkpoint and
    // not the patch. On a Dev checkpoint that is the guided ~30-step schedule, so an accelerated
    // run would take 8 steps of the wrong schedule and look like a broken model.
    const model = ltx2Model('ltx2_dev');
    const distilled = { base: 'ltx-2', key: 'ltx2-distilled', name: 'LTX-2.5 Distilled LoRA', type: 'lora' as const };
    const accelerated = compileVideoGraph(
      ltx2SettingsFor(model, {
        acceleratorEnabled: true,
        acceleratorLoraKeys: [distilled.key],
        aspectRatioId: '16:9',
        loras: [{ isEnabled: true, model: distilled, weight: 1 }],
        targetResolution: '1024p',
      }),
      model
    ).backendGraph;

    expect(accelerated.nodes.denoise_latents).toMatchObject({ schedule: 'distilled' });
    expect(accelerated.nodes.refine_latents).toMatchObject({ schedule: 'distilled' });

    // Off, the checkpoint stays the authority.
    const plain = compileVideoGraph(ltx2SettingsFor(model), model).backendGraph;

    expect(plain.nodes.denoise_latents).toMatchObject({ schedule: 'auto' });
    expect(nodeOfType(plain, 'ltx2_model_loader')).toBeDefined();
    expect(Object.values(plain.nodes).some((node) => node.type === 'ltx2_lora_collection_loader')).toBe(false);
  });

  it('continues a clip from its own tail and consumes the overlap in the join', () => {
    const model = ltx2Model('ltx2_dev');
    const { backendGraph } = compileVideoGraph(ltx2SettingsFor(model, { sourceVideo: LTX2_SOURCE_CLIP }), model);
    const extend = nodeOfType(backendGraph, 'ltx2_extend_conditioning');
    const concat = nodeOfType(backendGraph, 'video_concat');

    expect(hasEdge(backendGraph, extend.id, 'video_conditioning', 'denoise_latents', 'video_conditioning')).toBe(true);
    // The generated clip opens with the source's own tail, so the crossfade has to consume exactly
    // those frames -- and only the node that read the clip knows how many it got.
    expect(hasEdge(backendGraph, extend.id, 'context_frames', concat.id, 'transition_frames')).toBe(true);
    expect(concat).toMatchObject({ transition: 'crossfade' });
    // Both halves play at one speed, and the rate is read off the trimmed source at run time
    // rather than from the gallery's record of it -- a video row's fps is nullable.
    const extract = nodeOfType(backendGraph, 'extract_video_range');

    // The denoise and the decode take the clip's true float rate.
    expect(hasEdge(backendGraph, extract.id, 'fps', 'denoise_latents', 'fps')).toBe(true);
    expect(hasEdge(backendGraph, extract.id, 'fps', backendGraph.nodes.extension_clip!.id, 'fps')).toBe(true);
    // The join's fps is left unset on purpose: video_concat takes the first input's rate, which is
    // this same clip. Wiring it is float -> Optional[int], which the queue refuses at enqueue.
    expect(hasEdge(backendGraph, extract.id, 'fps', concat.id, 'fps')).toBe(false);
    expect(concat).not.toHaveProperty('fps');
    expect(extract).not.toHaveProperty('fps');

    // And the anchor is the TRIMMED clip, not the gallery file: the join's source half ends at the
    // user's trim, so anchoring past it would dissolve two unrelated moments together.
    expect(hasEdge(backendGraph, extract.id, 'video', extend.id, 'video')).toBe(true);
    // The join is the result; the generated half is kept as an intermediate beside it.
    expect(backendGraph.nodes.extension_clip).toMatchObject({ is_intermediate: true });
    expect(concat.is_intermediate).toBe(false);
    expect(backendGraph.nodes.core_metadata).toMatchObject({ generation_mode: 'ltx2_extend_video' });
  });

  it('holds the source\u2019s closing sound over the span the join crossfades', () => {
    // The join fades the held frames out of both halves. The picture survives it because both clips
    // render the same instant; the soundtrack only does if it is held over the same span. Left
    // generated, the blend fades invented audio in against the source\u2019s real audio and the new
    // soundtrack starts one overlap early -- an audible seam at the junction.
    const model = ltx2Model('ltx2_dev');
    const { backendGraph } = compileVideoGraph(ltx2SettingsFor(model, { sourceVideo: LTX2_SOURCE_CLIP }), model);
    const extend = nodeOfType(backendGraph, 'ltx2_extend_conditioning');

    expect(hasEdge(backendGraph, 'model_loader', 'audio_vae', extend.id, 'audio_vae')).toBe(true);
    expect(hasEdge(backendGraph, 'model_loader', 'vocoder', extend.id, 'vocoder')).toBe(true);
    // The rate the held frames are counted at has to be the clip\u2019s own, or the held sound spans a
    // different stretch of time than the held picture.
    expect(hasEdge(backendGraph, nodeOfType(backendGraph, 'extract_video_range').id, 'fps', extend.id, 'fps')).toBe(
      true
    );
    expect(hasEdge(backendGraph, extend.id, 'audio_conditioning', 'denoise_latents', 'audio_prefix_conditioning')).toBe(
      true
    );
    // Held, not muxed. The decode's source_audio channel replaces the generated soundtrack wholesale
    // with the source file's own -- right for audio-to-video, silently wrong here, where everything
    // past the overlap is meant to be new sound.
    expect(
      hasEdge(backendGraph, extend.id, 'audio_conditioning', backendGraph.nodes.extension_clip!.id, 'source_audio')
    ).toBe(false);
  });

  it('holds the closing sound through a two-stage continuation, encoding it once', () => {
    // Stage two re-noises every audio row, so a prefix wired only into stage one is gone by the
    // join and the seam is back with nothing else looking different. But the canvas never reaches
    // the audio path, so a second anchor would re-read the clip, load the audio VAE and vocoder,
    // and produce identical latents -- stage two is fed from stage one's encode instead.
    const model = ltx2Model('ltx2_dev');
    const { backendGraph } = compileVideoGraph(
      ltx2SettingsFor(model, { aspectRatioId: '16:9', sourceVideo: LTX2_SOURCE_CLIP, targetResolution: '1024p' }),
      model
    );

    expect(
      hasEdge(backendGraph, 'extend_conditioning', 'audio_conditioning', 'refine_latents', 'audio_prefix_conditioning')
    ).toBe(true);
    // The refine anchor holds picture only: no audio models, so it never opens the soundtrack.
    expect(hasEdge(backendGraph, 'model_loader', 'audio_vae', 'refine_extend_conditioning', 'audio_vae')).toBe(false);
    expect(hasEdge(backendGraph, 'model_loader', 'vocoder', 'refine_extend_conditioning', 'vocoder')).toBe(false);
  });

  it('re-anchors a two-stage continuation at the refine canvas', () => {
    const model = ltx2Model('ltx2_dev');
    const { backendGraph } = compileVideoGraph(
      ltx2SettingsFor(model, { aspectRatioId: '16:9', sourceVideo: LTX2_SOURCE_CLIP, targetResolution: '1024p' }),
      model
    );

    // Without this the refine pass continues from nothing and the join cuts between two unrelated
    // shots -- the same failure two-stage image-to-video had before it re-encoded its first frame.
    expect(backendGraph.nodes.refine_extend_conditioning).toMatchObject({ height: 1024, width: 1792 });
    expect(backendGraph.nodes.extend_conditioning).toMatchObject({ height: 512, width: 896 });
    expect(
      hasEdge(backendGraph, 'refine_extend_conditioning', 'video_conditioning', 'refine_latents', 'video_conditioning')
    ).toBe(true);
  });

  it('lands a continuation on a destination frame', () => {
    // The only path where both new mechanisms meet: a multi-frame leading anchor from the source's
    // tail, and an appended keyframe at the end. The panel offers it deliberately -- the Last Frame
    // field has its own copy for the extend case.
    const model = ltx2Model('ltx2_dev');
    const { backendGraph } = compileVideoGraph(
      ltx2SettingsFor(model, { lastFrameImage: LAST_FRAME, sourceVideo: LTX2_SOURCE_CLIP }),
      model
    );
    const extend = nodeOfType(backendGraph, 'ltx2_extend_conditioning');

    expect(hasEdge(backendGraph, extend.id, 'video_conditioning', 'denoise_latents', 'video_conditioning')).toBe(true);
    expect(backendGraph.nodes.last_frame_conditioning).toMatchObject({ frame_index: -1 });
    expect(
      hasEdge(backendGraph, 'last_frame_conditioning', 'video_conditioning', 'denoise_latents', 'keyframe_conditioning')
    ).toBe(true);
    // Still an extension: the join and its crossfade are unchanged by the destination frame.
    expect(
      hasEdge(
        backendGraph,
        extend.id,
        'context_frames',
        nodeOfType(backendGraph, 'video_concat').id,
        'transition_frames'
      )
    ).toBe(true);
  });

  it('encodes the negative prompt whenever either classifier-free scale will consume it', () => {
    const model = ltx2Model('ltx2_dev');
    const isWired = (settings: Partial<VideoSettings>) => {
      const graph = compileVideoGraph(ltx2SettingsFor(model, settings), model).backendGraph;

      return {
        denoise: nodeOfType(graph, 'ltx2_denoise'),
        encodes: nodeOfType(graph, 'ltx2_text_encoder').encode_negative,
        wired: hasEdge(graph, 'pos_cond', 'negative_conditioning', 'denoise_latents', 'negative_conditioning'),
      };
    };

    const guided = isWired({ cfgScale: 3, audioCfgScale: 7 });

    expect([guided.encodes, guided.wired]).toEqual([true, true]);

    // One unconditional pass serves both streams, so audio guidance alone still consumes the
    // negative prompt — and the node refuses an audio scale above 1 with nothing wired.
    const audioOnly = isWired({ cfgScale: 1, audioCfgScale: 7 });

    expect([audioOnly.encodes, audioOnly.wired]).toEqual([true, true]);

    // Both at 1: no unconditional pass, so a 12B encode of the negative prompt would be wasted.
    const unguided = isWired({ cfgScale: 1, audioCfgScale: 1 });

    expect([unguided.encodes, unguided.wired]).toEqual([false, false]);
  });

  it('holds both classifier-free scales at 1 when the negative prompt is switched off', () => {
    // Otherwise the panel's own switch queues a graph the denoise node refuses — after the 12B
    // prompt encode and the 22B transformer load.
    const model = ltx2Model('ltx2_dev');
    const settings = ltx2SettingsFor(model, { negativePromptEnabled: false, cfgScale: 3, audioCfgScale: 7 });
    const { backendGraph } = compileVideoGraph(settings, model);
    const denoise = nodeOfType(backendGraph, 'ltx2_denoise');

    expect(nodeOfType(backendGraph, 'ltx2_text_encoder').encode_negative).toBe(false);
    expect(denoise.cfg_scale).toBe(1);
    expect(denoise.audio_cfg_scale).toBe(1);
    // The other two passes steer against the positive conditioning, so they keep running.
    expect(denoise.stg_scale).toBe(1);
    expect(denoise.modality_scale).toBe(3);

    // Metadata records the run, not the panel: recalling this video must not restore a CFG of 3
    // that the generation never used.
    const metadata = nodeOfType(backendGraph, 'core_metadata');

    expect(metadata.cfg_scale).toBe(1);
    expect(metadata.ltx2_audio_cfg_scale).toBe(1);
  });

  it('writes the guidance the dev schedule runs and lets the backend resolve the schedule itself', () => {
    const model = ltx2Model('ltx2_dev');
    const settings = ltx2SettingsFor(model);
    const { backendGraph } = compileVideoGraph(settings, model);
    const denoise = nodeOfType(backendGraph, 'ltx2_denoise');

    expect(denoise.cfg_scale).toBe(3);
    expect(denoise.audio_cfg_scale).toBe(7);
    expect(denoise.stg_scale).toBe(1);
    expect(denoise.modality_scale).toBe(3);
    expect(denoise.steps).toBe(30);
    // The loader stamps the schedule from the checkpoint, which outranks the panel's own reading of
    // a variant it may not recognise.
    expect(denoise.schedule).toBe('auto');
  });

  it('collapses the guidance to its inert values on the distilled schedule', () => {
    const model = ltx2Model('ltx2_distilled');
    const { backendGraph } = compileVideoGraph(ltx2SettingsFor(model), model);
    const denoise = nodeOfType(backendGraph, 'ltx2_denoise');

    // The node ignores the scales on a distilled checkpoint; the graph should say what runs.
    expect(denoise.cfg_scale).toBe(1);
    expect(denoise.audio_cfg_scale).toBe(1);
    expect(denoise.stg_scale).toBe(0);
    expect(denoise.modality_scale).toBe(1);
    expect(denoise.steps).toBe(8);
    expect(nodesOfType(backendGraph, 'ltx2_text_encoder')[0]?.encode_negative).toBe(false);
  });

  it('conditions on a first frame through the image-conditioning node at the denoise canvas', () => {
    const model = ltx2Model('ltx2_dev');
    const settings = ltx2SettingsFor(model, { firstFrameImage: FIRST_FRAME });
    const { backendGraph } = compileVideoGraph(settings, model);
    const conditioning = nodeOfType(backendGraph, 'ltx2_image_conditioning');
    const denoise = nodeOfType(backendGraph, 'ltx2_denoise');

    expect(conditioning.image).toEqual({ image_name: FIRST_FRAME.image_name });
    // A canvas mismatch is what the denoise node refuses, so the same dimensions must reach both.
    expect(conditioning.width).toBe(denoise.width);
    expect(conditioning.height).toBe(denoise.height);
    expect(hasEdge(backendGraph, 'model_loader', 'vae', conditioning.id, 'vae')).toBe(true);
    expect(hasEdge(backendGraph, conditioning.id, 'video_conditioning', denoise.id, 'video_conditioning')).toBe(true);
    expect(nodeOfType(backendGraph, 'core_metadata').generation_mode).toBe('ltx2_i2v');
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

  it('needs no component folder when the model is a full install', () => {
    const model = ltx2Model('ltx2_dev', 'diffusers');
    const { backendGraph } = compileVideoGraph(ltx2SettingsFor(model), model);

    expect(nodeOfType(backendGraph, 'ltx2_model_loader').component_source).toBeUndefined();
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
    // Without this the clip's metadata records the stale panel value, and recall replays a length
    // the run never used.
    expect(hasEdge(backendGraph, duration.id, 'num_frames', metadata.id, 'num_frames')).toBe(true);
  });

  it('reads the prompt through the same conditioning the transformer will see', () => {
    const model = ltx2Model('ltx2_dev');
    const { backendGraph } = compileVideoGraph(autoSettings(model), model);
    const duration = nodeOfType(backendGraph, 'ltx2_duration');
    const encoder = nodeOfType(backendGraph, 'ltx2_text_encoder');

    expect(hasEdge(backendGraph, encoder.id, 'conditioning', duration.id, 'conditioning')).toBe(true);
  });

  it('stays out of a continuation, whose length the source clip already fixes', () => {
    const model = ltx2Model('ltx2_dev');
    const settings = autoSettings(model, { sourceVideo: LTX2_SOURCE_CLIP });
    const { backendGraph } = compileVideoGraph(settings, model);

    // Two edges into one `num_frames` is a malformed graph, not a preference that loses.
    expect(nodesOfType(backendGraph, 'ltx2_duration')).toHaveLength(0);
  });

  it('runs both passes of a two-stage preset at the length it chose', () => {
    const model = ltx2Model('ltx2_dev');
    const { backendGraph } = compileVideoGraph(autoSettings(model, { targetResolution: '1024p' }), model);
    const duration = nodeOfType(backendGraph, 'ltx2_duration');
    const denoises = nodesOfType(backendGraph, 'ltx2_denoise');

    expect(denoises).toHaveLength(2);
    // The refine pass validates the upsampled latents against its own num_frames, so a stale
    // literal here fails the run *after* the base pass and the upsampler have both done their work.
    for (const denoise of denoises) {
      expect(
        hasEdge(backendGraph, duration.id, 'num_frames', denoise.id, 'num_frames'),
        `${denoise.id} does not take its frame count from the duration head`
      ).toBe(true);
    }
  });

  it('lets the head choose at most the Frames value, at the rate the clip will play', () => {
    const model = ltx2Model('ltx2_dev');
    const { backendGraph } = compileVideoGraph(autoSettings(model, { fps: 60, numFrames: 241 }), model);
    const duration = nodeOfType(backendGraph, 'ltx2_duration');

    // Frames is the ceiling the run's memory was sized for; the head reads seconds at the run's rate.
    expect(duration.fps).toBe(60);
    expect(duration.max_seconds).toBe(241 / 60);
    expect(duration.min_seconds).toBe(1);
  });

  it('skips the head when the Frames ceiling leaves it no range to choose from', () => {
    const model = ltx2Model('ltx2_dev');
    // 17 frames at 24 fps is 0.7 s, under the head's 1 s floor: the clip runs at 17 frames.
    const { backendGraph } = compileVideoGraph(autoSettings(model, { fps: 24, numFrames: 17 }), model);

    expect(nodesOfType(backendGraph, 'ltx2_duration')).toHaveLength(0);
    expect(nodeOfType(backendGraph, 'ltx2_denoise').num_frames).toBe(17);
  });

  it.each(['audio' as const, 'video' as const])(
    'stays out of a %s-conditioned run, whose clip already drives the frame count',
    (role) => {
      const model = ltx2Model('ltx2_dev');
      const settings = autoSettings(model, {
        conditioningClip: { clip: { ...LTX2_SOURCE_CLIP, video_name: 'clip.mp4' }, fpsKnown: true, role },
      });
      const { backendGraph } = compileVideoGraph(settings, model);
      const denoise = nodeOfType(backendGraph, 'ltx2_denoise');

      // The conditioning node already edges into num_frames; a second edge is a malformed graph.
      expect(nodesOfType(backendGraph, 'ltx2_duration')).toHaveLength(0);
      expect(
        backendGraph.edges.filter(
          (edge) => edge.destination.node_id === denoise.id && edge.destination.field === 'num_frames'
        )
      ).toHaveLength(1);
    }
  );

  it('does nothing without a duration head selected', () => {
    const model = ltx2Model('ltx2_dev');
    const settings = ltx2SettingsFor(model, { autoDuration: true, ltx2DurationHeadModel: null });
    const { backendGraph } = compileVideoGraph(settings, model);
    const denoise = nodeOfType(backendGraph, 'ltx2_denoise');

    expect(nodesOfType(backendGraph, 'ltx2_duration')).toHaveLength(0);
    expect(denoise.num_frames).toBe(settings.numFrames);
  });
});
