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
    // Metadata rides both the final concat and the intermediate clip, and the
    // run-time conditioning frame is recorded via an edge (template behavior).
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
    // numFrames 81: startFrame 77 has tail offset 3 (→ -4), endFrame 79 has
    // tail offset 1 (→ -2). Both inside the tail window, order preserved.
    const tailTrim = compileVideoGraph(
      settingsFor(model, { sourceVideo: { ...SOURCE_VIDEO, endFrame: 79, startFrame: 77 } }),
      model
    ).backendGraph;

    expect(nodeOfType(tailTrim, 'extract_video_range')).toMatchObject({ end_frame: -2, start_frame: -4 });

    // Tail offset 4 is the first index OUTSIDE the window: it stays a
    // positive literal (converting it would drift mid-clip picks when the
    // estimate overshoots), even when the end still converts.
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
      transformer_low_noise: { key: 'low-expert' },
      vae: { key: 'wan-vae' },
      wan_t5_encoder: { key: 'umt5' },
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

    // The H3 canvas policy: 16:9 caps at 1344×768; frame counts are string literals.
    expect(denoise).toMatchObject({ height: 768, num_frames: '124', steps: settings.steps, width: 1344 });

    const output = nodeOfType(backendGraph, 'minimax_h3_latents_to_video');

    expect(output.id).toBe('video_output');
    expect(hasEdge(backendGraph, 'denoise_latents', 'video_latents', 'video_output', 'video_latents')).toBe(true);
    expect(hasEdge(backendGraph, 'denoise_latents', 'audio_latents', 'video_output', 'audio_latents')).toBe(true);
    expect(hasEdge(backendGraph, 'model_loader', 'vae', 'video_output', 'vae')).toBe(true);
    expect(hasEdge(backendGraph, 'model_loader', 'audio_vae', 'video_output', 'audio_vae')).toBe(true);

    expect(nodesOfType(backendGraph, 'minimax_h3_frame_conditioning')).toHaveLength(0);
    expect(nodeOfType(backendGraph, 'core_metadata').generation_mode).toBe('minimax_h3_t2v');
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

    // H3 renders at a fixed 24 fps; the source is retimed up front so the
    // concat (which inherits the first clip's rate) joins at one speed.
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

  it('passes the single-file overrides to the loader and the metadata', () => {
    const transformer = { ...h3Model('h3-int8'), format: 'checkpoint' };
    const encoder = { base: 'minimax-h3', key: 'h3-te', name: 'H3 TE int8', type: 'qwen3_vl_encoder' as const };
    const settings = settingsFor(model, { h3TextEncoderModel: encoder, h3TransformerModel: transformer });
    const { backendGraph } = compileVideoGraph(settings, model);

    expect(nodeOfType(backendGraph, 'minimax_h3_model_loader')).toMatchObject({
      text_encoder_model: { key: 'h3-te' },
      transformer_model: { key: 'h3-int8' },
    });
    expect(nodeOfType(backendGraph, 'core_metadata')).toMatchObject({
      minimax_h3_text_encoder_model: { key: 'h3-te' },
      minimax_h3_transformer_model: { key: 'h3-int8' },
    });
  });

  it('refuses to compile fractional or off-grid frame counts', () => {
    expect(() => compileVideoGraph(settingsFor(model, { numFrames: 100 }), model)).toThrow(/17·n \+ 5/);
  });
});
