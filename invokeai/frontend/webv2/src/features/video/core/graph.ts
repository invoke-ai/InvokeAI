import type {
  BackendGraphContract,
  BackendInvocationContract,
  GraphContract,
  MainModelConfig,
} from '@features/generation/contracts';

import {
  addEdge,
  addNode,
  addTransformerLoraCollectionLoader,
  createId,
  getActiveCompatibleLoras,
  toGraphContract,
  toModelIdentifier,
} from '@features/generation/graph';
import { getCompatibleDiffusersComponentSource } from '@features/generation/settings';

import type {
  Ltx2TargetResolution,
  VideoGenerationMode,
  VideoReferenceItem,
  VideoSettings,
  VideoSourceClip,
} from './types';
import type { SupportedVideoBase } from './videoPolicies';

import { getLtx2StageCanvases, isLtx2TwoStage, MINIMAX_H3_FPS } from './dimensions';
import { MINIMAX_H3_HYBRID_BLOCK_RANGE, resolveVideoMode } from './settings';
import {
  getEffectiveVideoTiming,
  getVideoDimensions,
  getVideoModelPolicy,
  getVideoTargetResolution,
  getAutoDurationBounds,
  getVideoValidationReasons,
} from './videoPolicies';

/** Mirror bundled template wiring per model family and expose fixed prompt/seed node IDs for submission batching. */

export interface CompiledVideoGraph {
  backendGraph: BackendGraphContract;
  graph: GraphContract;
  negativePromptNodeId: string;
  positivePromptNodeId: string;
  seedNodeId: string;
}

const WAN_GENERATION_MODES: Partial<Record<VideoGenerationMode, string>> = {
  extend: 'wan_extend_video',
  'first-frame': 'wan_i2v',
  'first-last': 'wan_interpolate',
  txt2vid: 'wan_t2v',
};

const LTX2_GENERATION_MODES: Partial<Record<VideoGenerationMode, string>> = {
  'audio-to-video': 'ltx2_a2v',
  extend: 'ltx2_extend_video',
  'first-frame': 'ltx2_i2v',
  'first-last': 'ltx2_flf2v',
  'last-frame': 'ltx2_lf2v',
  txt2vid: 'ltx2_t2v',
  'video-to-audio': 'ltx2_v2a',
};

const MINIMAX_H3_GENERATION_MODES: Partial<Record<VideoGenerationMode, string>> = {
  extend: 'minimax_h3_extend_video',
  'first-frame': 'minimax_h3_i2v',
  'first-last': 'minimax_h3_flf2v',
  'last-frame': 'minimax_h3_lf2v',
  reference: 'minimax_h3_ref2v',
  txt2vid: 'minimax_h3_t2v',
};

/**
 * Feed identical ordered references to conditioning and prompt nodes; chain collectors to preserve the request's
 * rotary order.
 */
const addReferenceNodes = (
  graph: BackendGraphContract,
  references: readonly VideoReferenceItem[]
): BackendInvocationContract => {
  let chain: BackendInvocationContract | null = null;
  references.forEach((reference, index) => {
    const node = addReferenceNode(graph, reference, index);
    const collect = addNode(graph, { id: `reference_collect_${index + 1}`, type: 'collect' });

    if (chain) {
      addEdge(graph, chain, 'collection', collect, 'collection');
    }
    addEdge(graph, node, 'reference', collect, 'item');
    chain = collect;
  });

  if (!chain) {
    throw new Error('Reference mode requires at least one reference.');
  }

  return chain;
};

const addReferenceNode = (
  graph: BackendGraphContract,
  reference: VideoReferenceItem,
  index: number
): BackendInvocationContract => {
  if (reference.kind === 'image') {
    return addNode(graph, {
      detail: reference.detail,
      id: `reference_${index + 1}`,
      image: toImageField(reference.image),
      type: 'minimax_h3_image_reference',
    });
  }
  // Compile tail-window bounds relative to the actual backend frame count, as in extension graphs.
  const endFrame = toTailAwareIndex(reference.clip.endFrame, reference.clip.numFrames);

  return addNode(graph, {
    conditioning: reference.conditioning,
    end_frame: endFrame,
    id: `reference_${index + 1}`,
    start_frame: toReferenceStartIndex(reference.clip, endFrame),
    type: 'minimax_h3_video_reference',
    video: { video_name: reference.clip.video_name },
  });
};

/**
 * Convert reference bounds together to preserve length near the tail. Keep starts within TAIL_INDEX_SLOP absolute
 * to avoid negative starts; larger estimate errors remain outside this guard.
 */
const toReferenceStartIndex = (clip: VideoSourceClip, endIndex: number): number => {
  if (endIndex >= 0 || clip.startFrame <= TAIL_INDEX_SLOP) {
    return toTailAwareIndex(clip.startFrame, clip.numFrames);
  }

  return endIndex - (clip.endFrame - clip.startFrame);
};

const addPromptAndSeedNodes = (graph: BackendGraphContract) => ({
  negativePrompt: addNode(graph, { id: 'negative_prompt', type: 'string' }),
  positivePrompt: addNode(graph, { id: 'positive_prompt', type: 'string' }),
  seed: addNode(graph, { id: 'seed', type: 'integer' }),
});

const toImageField = (image: { image_name: string }) => ({ image_name: image.image_name });

// Compile near-end trim bounds as negative indices against actual frame count; keep mid-clip bounds absolute to
// avoid estimate-induced drift.
/** Bound tolerated duration-times-fps overshoot; indices within this tail window resolve against the real end. */
const TAIL_INDEX_SLOP = 3;

const toTailAwareIndex = (frame: number, estimatedNumFrames: number): number => {
  const tailOffset = estimatedNumFrames - 1 - frame;

  return tailOffset <= TAIL_INDEX_SLOP ? -(tailOffset + 1) : frame;
};

/** Extract source trim and crossfade source/new clips; video_concat reconstructs audio from both inputs. */
const addSourceJoin = (
  graph: BackendGraphContract,
  sourceVideo: NonNullable<VideoSettings['sourceVideo']>,
  newClip: BackendInvocationContract,
  options: { extractFps?: number } = {}
) => {
  const extract = addNode(graph, {
    end_frame: toTailAwareIndex(sourceVideo.endFrame, sourceVideo.numFrames),
    id: 'source_video',
    is_intermediate: true,
    start_frame: toTailAwareIndex(sourceVideo.startFrame, sourceVideo.numFrames),
    type: 'extract_video_range',
    use_cache: false,
    video: { video_name: sourceVideo.video_name },
    ...(options.extractFps === undefined ? {} : { fps: options.extractFps }),
  });
  const sourceCollect = addNode(graph, { id: 'source_clip_collect', type: 'collect' });
  const clipsCollect = addNode(graph, { id: 'clips_to_join', type: 'collect' });
  const concat = addNode(graph, {
    id: 'video_output',
    is_intermediate: false,
    size_mismatch: 'match_first',
    transition: 'crossfade',
    transition_frames: 2,
    type: 'video_concat',
    use_cache: false,
  });

  // Chained collectors keep the join order deterministic: [trimmed source, new clip].
  addEdge(graph, extract, 'video', sourceCollect, 'item');
  addEdge(graph, sourceCollect, 'collection', clipsCollect, 'collection');
  addEdge(graph, newClip, 'video', clipsCollect, 'item');
  addEdge(graph, clipsCollect, 'collection', concat, 'videos');

  return { concat, extract };
};

const addExtendScaffolding = (
  graph: BackendGraphContract,
  sourceVideo: NonNullable<VideoSettings['sourceVideo']>,
  newClip: BackendInvocationContract,
  options: { extractFps?: number } = {}
) => {
  const { concat, extract } = addSourceJoin(graph, sourceVideo, newClip, options);
  const lastFrame = addNode(graph, {
    frame_index: -1,
    id: 'source_last_frame',
    is_intermediate: true,
    type: 'video_frame_extract',
    use_cache: false,
  });

  addEdge(graph, extract, 'video', lastFrame, 'video');

  return { concat, extract, lastFrame };
};

interface VideoMetadataInput {
  graph: BackendGraphContract;
  /** Every video-emitting node that should carry the metadata (in extend mode: the concat AND the intermediate clip). */
  outputs: BackendInvocationContract[];
  settings: VideoSettings;
  model: MainModelConfig;
  generationMode: string;
  width: number;
  height: number;
  /** Whether the negative prompt participates in this family's graphs at all. */
  negativeWired: boolean;
  extras?: Record<string, unknown>;
}

const addVideoMetadata = ({
  graph,
  outputs,
  settings,
  model,
  generationMode,
  width,
  height,
  negativeWired,
  extras = {},
}: VideoMetadataInput) => {
  const activeLoras = getActiveCompatibleLoras(settings, model);
  const metadata = addNode(graph, {
    generation_mode: generationMode,
    height,
    id: 'core_metadata',
    model,
    num_frames: settings.numFrames,
    steps: settings.steps,
    type: 'core_metadata',
    width,
    ...(settings.firstFrameImage ? { first_frame_image: toImageField(settings.firstFrameImage) } : {}),
    ...(settings.lastFrameImage ? { last_frame_image: toImageField(settings.lastFrameImage) } : {}),
    // Persist trim bounds as metadata extras so recall restores the actual extension cutpoint.
    ...(settings.sourceVideo
      ? {
          source_video: { video_name: settings.sourceVideo.video_name },
          source_video_end_frame: settings.sourceVideo.endFrame,
          source_video_start_frame: settings.sourceVideo.startFrame,
        }
      : {}),
    ...(activeLoras.length
      ? { loras: activeLoras.map((lora) => ({ model: toModelIdentifier(lora.model), weight: lora.weight })) }
      : {}),
    ...extras,
  });

  addEdge(graph, graph.nodes.seed, 'value', metadata, 'seed');
  addEdge(graph, graph.nodes.positive_prompt, 'value', metadata, 'positive_prompt');
  if (negativeWired) {
    addEdge(graph, graph.nodes.negative_prompt, 'value', metadata, 'negative_prompt');
  }
  for (const output of outputs) {
    addEdge(graph, metadata, 'metadata', output, 'metadata');
  }

  return metadata;
};

const buildWanVideoGraph = (settings: VideoSettings, model: MainModelConfig): BackendGraphContract => {
  const mode = resolveVideoMode(settings);
  const policy = getVideoModelPolicy(model, settings);
  const dimensions = getVideoDimensions(model, settings);

  if (!dimensions) {
    throw new Error('Video dimensions could not be derived from the current settings.');
  }

  // Standalone Wan checkpoints need external VAE/UMT5 components; Diffusers mains bundle them and the second
  // expert.
  const isSingleFileMain = model.format !== 'diffusers';
  const sourceModel = isSingleFileMain
    ? getCompatibleDiffusersComponentSource(model, settings.componentSourceModel)
    : undefined;

  const graph: BackendGraphContract = { edges: [], id: createId('wan_video_graph'), nodes: {} };
  const { negativePrompt, positivePrompt, seed } = addPromptAndSeedNodes(graph);
  const modelLoader = addNode(graph, {
    component_source: sourceModel,
    id: 'model_loader',
    model,
    transformer_low_noise_model: (isSingleFileMain ? settings.wanLowNoiseModel : null) ?? undefined,
    type: 'wan_model_loader',
    vae_model: settings.vae ?? undefined,
    wan_t5_encoder_model: settings.wanT5EncoderModel ?? undefined,
  });
  // Wan LoRAs patch transformers; auto routing uses probed high/low tags to place accelerator pairs correctly.
  const activeLoras = getActiveCompatibleLoras(settings, model);
  const loraSource = activeLoras.length
    ? addTransformerLoraCollectionLoader(graph, activeLoras, 'wan_lora_collection_loader', modelLoader, ['transformer'])
    : modelLoader;
  const posCond = addNode(graph, { id: 'pos_cond', type: 'wan_text_encoder' });
  const negCond = addNode(graph, { id: 'neg_cond', type: 'wan_text_encoder' });
  const denoise = addNode(graph, {
    guidance_scale: settings.cfgScale,
    // Only the A14B expert pairs run a low-noise phase; null falls back to the
    // primary guidance rather than the node's own 4.0 default.
    ...(policy.ui.cfgLowNoiseVisible
      ? { guidance_scale_low_noise: settings.cfgScaleLowNoise ?? settings.cfgScale }
      : {}),
    height: dimensions.height,
    id: 'denoise_latents',
    num_frames: settings.numFrames,
    steps: settings.steps,
    type: 'wan_video_denoise',
    width: dimensions.width,
  });

  addEdge(graph, loraSource, 'transformer', denoise, 'transformer');
  addEdge(graph, modelLoader, 'wan_t5_encoder', posCond, 'wan_t5_encoder');
  addEdge(graph, modelLoader, 'wan_t5_encoder', negCond, 'wan_t5_encoder');
  addEdge(graph, positivePrompt, 'value', posCond, 'prompt');
  addEdge(graph, negativePrompt, 'value', negCond, 'prompt');
  addEdge(graph, posCond, 'conditioning', denoise, 'positive_conditioning');
  addEdge(graph, negCond, 'conditioning', denoise, 'negative_conditioning');
  addEdge(graph, seed, 'value', denoise, 'seed');

  // Share canvas/frame literals between reference conditioning and denoise to keep their shapes identical.
  const addRefEncoder = (inputs: Record<string, unknown>) => {
    const refEncoder = addNode(graph, {
      height: dimensions.height,
      id: 'ref_image_encoder',
      num_frames: settings.numFrames,
      type: 'wan_ref_image_encoder',
      width: dimensions.width,
      ...inputs,
    });

    addEdge(graph, modelLoader, 'vae', refEncoder, 'vae');
    addEdge(graph, refEncoder, 'ref_image', denoise, 'ref_image');

    return refEncoder;
  };

  let output: BackendInvocationContract;
  let extendParts: { lastFrame: BackendInvocationContract; newClip: BackendInvocationContract } | null = null;

  if (mode === 'extend' && settings.sourceVideo) {
    const refEncoder = addRefEncoder(
      settings.lastFrameImage ? { end_image: toImageField(settings.lastFrameImage) } : {}
    );
    const newClip = addNode(graph, {
      id: 'extension_clip',
      is_intermediate: true,
      type: 'wan_l2v',
      use_cache: false,
    });
    const { concat, extract, lastFrame } = addExtendScaffolding(graph, settings.sourceVideo, newClip);
    // Inherit source fps so the extension and original play at one speed.
    const fpsToInt = addNode(graph, { id: 'source_fps', method: 'Nearest', multiple: 1, type: 'float_to_int' });

    addEdge(graph, lastFrame, 'image', refEncoder, 'image');
    addEdge(graph, modelLoader, 'vae', newClip, 'vae');
    addEdge(graph, denoise, 'latents', newClip, 'latents');
    addEdge(graph, extract, 'fps', fpsToInt, 'value');
    addEdge(graph, fpsToInt, 'value', newClip, 'fps');
    addEdge(graph, fpsToInt, 'value', concat, 'fps');
    output = concat;
    extendParts = { lastFrame, newClip };
  } else {
    if (mode === 'first-frame' || mode === 'first-last') {
      if (!settings.firstFrameImage) {
        throw new Error('A first frame is required for image-to-video generation.');
      }

      addRefEncoder({
        image: toImageField(settings.firstFrameImage),
        ...(mode === 'first-last' && settings.lastFrameImage
          ? { end_image: toImageField(settings.lastFrameImage) }
          : {}),
      });
    }

    output = addNode(graph, {
      fps: settings.fps,
      id: 'video_output',
      is_intermediate: false,
      type: 'wan_l2v',
      use_cache: false,
    });
    addEdge(graph, modelLoader, 'vae', output, 'vae');
    addEdge(graph, denoise, 'latents', output, 'latents');
  }

  const metadata = addVideoMetadata({
    extras: {
      cfg_scale: settings.cfgScale,
      // Record delivered fps: rounded source rate for extension, panel fps otherwise.
      fps: extendParts && settings.sourceVideo ? Math.round(settings.sourceVideo.fps) : settings.fps,
      ...(policy.ui.cfgLowNoiseVisible && settings.cfgScaleLowNoise !== null
        ? { wan_guidance_scale_low_noise: settings.cfgScaleLowNoise }
        : {}),
      ...(settings.vae ? { vae: settings.vae } : {}),
      ...(settings.wanT5EncoderModel ? { wan_t5_encoder_model: settings.wanT5EncoderModel } : {}),
      ...(sourceModel ? { wan_component_source: sourceModel } : {}),
      ...(isSingleFileMain && settings.wanLowNoiseModel
        ? { wan_transformer_low_noise: settings.wanLowNoiseModel }
        : {}),
    },
    generationMode: WAN_GENERATION_MODES[mode] ?? 'wan_t2v',
    graph,
    height: dimensions.height,
    model,
    negativeWired: true,
    outputs: extendParts ? [output, extendParts.newClip] : [output],
    settings,
    width: dimensions.width,
  });

  if (extendParts) {
    // Record runtime-extracted conditioning frames through an edge, matching bundled templates.
    addEdge(graph, extendParts.lastFrame, 'image', metadata, 'first_frame_image');
  }

  return graph;
};

const buildMiniMaxH3VideoGraph = (settings: VideoSettings, model: MainModelConfig): BackendGraphContract => {
  const mode = resolveVideoMode(settings);
  const dimensions = getVideoDimensions(model, settings);

  if (!dimensions) {
    throw new Error('Video dimensions could not be derived from the current settings.');
  }

  // H3 checkpoints override the transformer of a required Diffusers component model; direct callers still need the
  // missing-slot guard.
  const isSingleFileMain = model.format !== 'diffusers';
  const componentSource = isSingleFileMain ? settings.componentSourceModel : null;

  if (isSingleFileMain && !componentSource) {
    throw new Error('A single-file MiniMax H3 transformer needs a Diffusers install selected under Model Components.');
  }

  // Hybrid loading overlays Ref2VA AdaLN blocks onto FL2VA; guard stale settings so unsupported mains cannot
  // silently load the wrong file.
  const hybridBase = isSingleFileMain && model.variant === 'ref2va' ? settings.h3HybridBaseModel : null;

  const graph: BackendGraphContract = { edges: [], id: createId('minimax_h3_video_graph'), nodes: {} };
  const { positivePrompt, seed } = addPromptAndSeedNodes(graph);
  const modelLoader = addNode(graph, {
    id: 'model_loader',
    model: componentSource ?? model,
    text_encoder_model: settings.h3TextEncoderModel ?? undefined,
    transformer_model: isSingleFileMain ? (hybridBase ?? model) : undefined,
    type: 'minimax_h3_model_loader',
  });
  let transformerSource: BackendInvocationContract = modelLoader;

  if (hybridBase) {
    const hybridOverlay = addNode(graph, {
      end_block: MINIMAX_H3_HYBRID_BLOCK_RANGE.max,
      id: 'hybrid_overlay',
      include_final_layer: false,
      overlay_model: model,
      start_block: settings.h3HybridStartBlock,
      type: 'minimax_h3_hybrid_overlay',
    });

    addEdge(graph, modelLoader, 'transformer', hybridOverlay, 'transformer');
    transformerSource = hybridOverlay;
  }

  const activeLoras = getActiveCompatibleLoras(settings, model);

  if (activeLoras.length) {
    transformerSource = addTransformerLoraCollectionLoader(
      graph,
      activeLoras,
      'minimax_h3_lora_collection_loader',
      transformerSource,
      ['transformer']
    );
  }

  // Share canvas/keyframe inputs across text, frame conditioning, and denoise.
  const keyframeLiterals = {
    ...(mode !== 'extend' && settings.firstFrameImage ? { first_image: toImageField(settings.firstFrameImage) } : {}),
    ...(settings.lastFrameImage ? { last_image: toImageField(settings.lastFrameImage) } : {}),
  };
  const posCond = addNode(graph, {
    height: dimensions.height,
    id: 'pos_cond',
    type: 'minimax_h3_text_encoder',
    width: dimensions.width,
    ...keyframeLiterals,
    ...(mode === 'reference' ? { num_frames: settings.numFrames } : {}),
  });
  const denoise = addNode(graph, {
    height: dimensions.height,
    id: 'denoise_latents',
    // The node's frame counts are a string Literal choice list.
    num_frames: String(settings.numFrames),
    // Add the terminal sigma point to panel model-evaluation steps; metadata retains the panel value.
    steps: settings.steps + 1,
    type: 'minimax_h3_denoise',
    width: dimensions.width,
  });

  addEdge(graph, modelLoader, 'text_encoder', posCond, 'text_encoder');
  addEdge(graph, positivePrompt, 'value', posCond, 'prompt');
  addEdge(graph, transformerSource, 'transformer', denoise, 'transformer');
  addEdge(graph, posCond, 'conditioning', denoise, 'positive_conditioning');
  addEdge(graph, seed, 'value', denoise, 'seed');

  if (mode === 'reference') {
    const referenceChain = addReferenceNodes(graph, settings.references);
    const referenceConditioning = addNode(graph, {
      height: dimensions.height,
      id: 'reference_conditioning',
      num_frames: settings.numFrames,
      type: 'minimax_h3_reference_conditioning',
      width: dimensions.width,
    });

    addEdge(graph, referenceChain, 'collection', referenceConditioning, 'references');
    addEdge(graph, referenceChain, 'collection', posCond, 'references');
    addEdge(graph, modelLoader, 'vae', referenceConditioning, 'vae');
    addEdge(graph, modelLoader, 'audio_vae', referenceConditioning, 'audio_vae');
    addEdge(graph, referenceConditioning, 'reference_conditioning', denoise, 'reference_conditioning');
  }

  const needsFrameConditioning = mode !== 'txt2vid' && mode !== 'reference';
  let frameConditioning: BackendInvocationContract | null = null;

  if (needsFrameConditioning) {
    frameConditioning = addNode(graph, {
      height: dimensions.height,
      id: 'frame_conditioning',
      type: 'minimax_h3_frame_conditioning',
      width: dimensions.width,
      ...keyframeLiterals,
    });
    addEdge(graph, modelLoader, 'vae', frameConditioning, 'vae');
    addEdge(graph, frameConditioning, 'frame_conditioning', denoise, 'frame_conditioning');
  }

  const addLatentsToVideo = (id: string, isIntermediate: boolean) => {
    const node = addNode(graph, {
      id,
      is_intermediate: isIntermediate,
      type: 'minimax_h3_latents_to_video',
      use_cache: false,
    });

    addEdge(graph, denoise, 'video_latents', node, 'video_latents');
    addEdge(graph, denoise, 'audio_latents', node, 'audio_latents');
    addEdge(graph, modelLoader, 'vae', node, 'vae');
    addEdge(graph, modelLoader, 'audio_vae', node, 'audio_vae');

    return node;
  };

  let output: BackendInvocationContract;
  let extendParts: { lastFrame: BackendInvocationContract; newClip: BackendInvocationContract } | null = null;
  let referenceExtendClip: BackendInvocationContract | null = null;

  if (mode === 'extend' && settings.sourceVideo && frameConditioning) {
    const newClip = addLatentsToVideo('extension_clip', true);
    // Resample source to H3's fixed 24 fps before concat inherits its rate.
    const { concat, lastFrame } = addExtendScaffolding(graph, settings.sourceVideo, newClip, { extractFps: 24 });

    addEdge(graph, lastFrame, 'image', posCond, 'first_image');
    addEdge(graph, lastFrame, 'image', frameConditioning, 'first_image');
    output = concat;
    extendParts = { lastFrame, newClip };
  } else if (mode === 'reference' && settings.sourceVideo) {
    // Reference extension derives continuity from references, not extracted frame conditioning.
    const newClip = addLatentsToVideo('extension_clip', true);
    const { concat } = addSourceJoin(graph, settings.sourceVideo, newClip, { extractFps: 24 });

    output = concat;
    referenceExtendClip = newClip;
  } else {
    output = addLatentsToVideo('video_output', false);
  }

  const metadata = addVideoMetadata({
    extras: {
      // H3 always delivers its fixed rate; recording it keeps the record self-describing.
      fps: MINIMAX_H3_FPS,
      ...(componentSource ? { minimax_h3_component_source: componentSource } : {}),
      ...(settings.h3TextEncoderModel ? { minimax_h3_text_encoder_model: settings.h3TextEncoderModel } : {}),
      ...(hybridBase
        ? { minimax_h3_hybrid_base_model: hybridBase, minimax_h3_hybrid_start_block: settings.h3HybridStartBlock }
        : {}),
      ...(mode === 'reference'
        ? {
            minimax_h3_references: settings.references.map((reference) =>
              reference.kind === 'image'
                ? { detail: reference.detail, image_name: reference.image.image_name, kind: 'image' }
                : {
                    conditioning: reference.conditioning,
                    end_frame: reference.clip.endFrame,
                    kind: 'video',
                    start_frame: reference.clip.startFrame,
                    video_name: reference.clip.video_name,
                  }
            ),
          }
        : {}),
    },
    generationMode: MINIMAX_H3_GENERATION_MODES[mode] ?? 'minimax_h3_t2v',
    graph,
    height: dimensions.height,
    model,
    negativeWired: false,
    outputs: extendParts
      ? [output, extendParts.newClip]
      : referenceExtendClip
        ? [output, referenceExtendClip]
        : [output],
    settings,
    width: dimensions.width,
  });

  if (extendParts) {
    // Record runtime-extracted conditioning frames through an edge, matching bundled templates.
    addEdge(graph, extendParts.lastFrame, 'image', metadata, 'first_frame_image');
  }

  return graph;
};

/**
 * LTX-2: one transformer generates video and its soundtrack together, so there
 * is a single denoise and a single decode; what varies is how much guidance the
 * checkpoint's schedule wants. The dev checkpoint runs up to four forwards per
 * step (conditional, unconditional, spatio-temporal, modality-isolation); the
 * distilled one runs a fixed eight steps with none, which is why the guidance
 * literals below collapse to their inert values rather than being omitted — the
 * graph should state what will actually run.
 */
const buildLtx2VideoGraph = (settings: VideoSettings, model: MainModelConfig): BackendGraphContract => {
  const mode = resolveVideoMode(settings);
  const policy = getVideoModelPolicy(model, settings);
  const dimensions = getVideoDimensions(model, settings);
  // A conditioning clip owns the length, and in the video role the frame rate too; everywhere else
  // these are the panel's own numbers. Used for every literal below, and for the metadata, so all
  // three describe the same run.
  const timing = getEffectiveVideoTiming(model, settings);

  if (!dimensions) {
    throw new Error('Video dimensions could not be derived from the current settings.');
  }

  // No LTX-2 main carries text-encoder weights, and a single-file transformer
  // carries none of the VAEs, vocoder or connectors either. Validation requires
  // both slots; these throws are backstops for direct callers.
  const componentSource = model.format === 'diffusers' ? null : settings.componentSourceModel;

  if (model.format !== 'diffusers' && !componentSource) {
    throw new Error('A single-file LTX-2 transformer needs an LTX-2 components install under Model Components.');
  }
  if (!settings.ltx2TextEncoderModel) {
    throw new Error('LTX-2 needs its Gemma-4 text encoder selected under Model Components.');
  }

  // A two-stage preset generates at half the canvas and refines an upscaled latent, so the base
  // pass runs at `stages.base` and everything downstream of the upscaler at `stages.final`. Single
  // stage returns the same canvas twice, which is what lets one code path build both.
  // The same coercion `getVideoDimensions` applied: a settings record can hold a preset this model
  // does not offer, and reading the raw value here would describe a different run than the canvas
  // above was resolved from -- an unknown preset has no short edge, which makes every dimension NaN.
  const targetResolution = getVideoTargetResolution(model, settings.targetResolution) as Ltx2TargetResolution;
  const stages = getLtx2StageCanvases(dimensions.width, dimensions.height, targetResolution);

  if (!stages) {
    throw new Error('Video dimensions could not be derived from the current settings.');
  }

  // Asked of the preset rather than recovered by comparing the two canvases: equal widths would say
  // "one stage" for a preset that is two, and NaN widths compare unequal, so a degenerate canvas
  // would claim to be two.
  const twoStage = isLtx2TwoStage(targetResolution);
  const graph: BackendGraphContract = { edges: [], id: createId('ltx2_video_graph'), nodes: {} };
  const { negativePrompt, positivePrompt, seed } = addPromptAndSeedNodes(graph);
  const modelLoader = addNode(graph, {
    component_source: componentSource ?? undefined,
    id: 'model_loader',
    model,
    text_encoder_model: settings.ltx2TextEncoderModel,
    type: 'ltx2_model_loader',
  });

  // The negative prompt only reaches the model through classifier-free
  // guidance, so a schedule that runs none skips a 12B encode entirely.
  const negativeWired = policy.prompt.negativeUsedInGraph;
  const textEncoder = addNode(graph, {
    encode_negative: negativeWired,
    id: 'pos_cond',
    type: 'ltx2_text_encoder',
  });

  addEdge(graph, modelLoader, 'text_encoder', textEncoder, 'text_encoder');
  addEdge(graph, positivePrompt, 'value', textEncoder, 'prompt');
  if (negativeWired) {
    addEdge(graph, negativePrompt, 'value', textEncoder, 'negative_prompt');
  }

  // Both classifier-free scales are held at 1 without a negative prompt: they are the only terms
  // that consume one, and the node refuses a scale above 1 with nothing wired. The spatio-temporal
  // and modality passes steer against the *positive* conditioning, so they keep running — turning
  // the negative prompt off is not a request to stop guiding. Metadata records these rather than
  // the panel's own numbers, so a recall reproduces the run instead of the settings.
  const guidance = policy.ui.cfgVisible
    ? {
        audio_cfg_scale: negativeWired ? (settings.audioCfgScale ?? policy.defaults.audioCfgScale ?? 1) : 1,
        cfg_scale: negativeWired ? settings.cfgScale : 1,
        modality_scale: settings.modalityScale ?? policy.defaults.modalityScale ?? 1,
        stg_scale: settings.stgScale ?? policy.defaults.stgScale ?? 0,
      }
    : { audio_cfg_scale: 1, cfg_scale: 1, modality_scale: 1, stg_scale: 0 };

  const activeLoras = getActiveCompatibleLoras(settings, model);
  let transformerSource: BackendInvocationContract = modelLoader;

  if (activeLoras.length) {
    transformerSource = addTransformerLoraCollectionLoader(
      graph,
      activeLoras,
      'ltx2_lora_collection_loader',
      transformerSource,
      ['transformer']
    );
  }

  const denoise = addNode(graph, {
    fps: timing.fps,
    height: stages.base.height,
    id: 'denoise_latents',
    num_frames: timing.numFrames,
    // Normally 'auto': the loader stamps the schedule off the checkpoint itself, which is the
    // authority when a release the panel does not recognise falls back to the dev policy.
    //
    // The step-distillation LoRA is the exception, and it has to be named here. 'auto' resolves off
    // the transformer's *variant*, which names the checkpoint and not the patch, so a distilled
    // LoRA on a Dev checkpoint would still take the guided ~30-step schedule -- the accelerator
    // would set 8 steps and then sample them on the wrong schedule, which looks like a broken
    // model rather than a wiring mistake.
    schedule: settings.acceleratorEnabled ? 'distilled' : 'auto',
    steps: settings.steps,
    type: 'ltx2_denoise',
    width: stages.base.width,
    ...guidance,
  });

  addEdge(graph, transformerSource, 'transformer', denoise, 'transformer');
  addEdge(graph, textEncoder, 'conditioning', denoise, 'positive_conditioning');
  if (negativeWired) {
    addEdge(graph, textEncoder, 'negative_conditioning', denoise, 'negative_conditioning');
  }
  addEdge(graph, seed, 'value', denoise, 'seed');

  // The length the run will actually use, decided from the prompt the transformer will see.
  //
  // Only in the modes whose length is still open. A conditioning clip drives `num_frames` by its
  // own edge further down, and a second edge into one input is a malformed graph rather than a
  // fallback. An extension's prompt sizes the continuation; its rate and held context arrive over
  // edges once the join is laid out. The head chooses at most the Frames value, which under auto
  // duration is the ceiling the run's memory was sized for. It goes in frames only: the node turns it
  // into seconds at the rate it actually runs at, which for a continuation only the run knows.
  const durationBounds = getAutoDurationBounds(model, settings);
  const durationHead =
    durationBounds && settings.ltx2DurationHeadModel
      ? addNode(graph, {
          duration_head: settings.ltx2DurationHeadModel,
          ...(durationBounds.contextFrames ? {} : { fps: durationBounds.fps }),
          id: 'duration',
          type: 'ltx2_duration',
          max_num_frames: timing.numFrames,
          min_seconds: durationBounds.minSeconds,
        })
      : null;

  if (durationHead) {
    addEdge(graph, textEncoder, 'conditioning', durationHead, 'conditioning');
    addEdge(graph, durationHead, 'num_frames', denoise, 'num_frames');
  }

  let extendConditioning: BackendInvocationContract | null = null;

  /**
   * The tail of the clip being continued, encoded at one canvas and held as the opening of one
   * pass. Read from the TRIMMED source, not the gallery file: the join's source half ends at the
   * user's trim, so anchoring on the raw file's last frames would dissolve two unrelated moments
   * into each other wherever the trim cut something off. The literal is what keeps the node's
   * required field valid in the graph JSON; the edge is what the run actually reads.
   */
  const addExtendAnchor = (
    id: string,
    sourceVideo: NonNullable<VideoSettings['sourceVideo']>,
    extract: BackendInvocationContract,
    canvas: { width: number; height: number },
    pass: BackendInvocationContract,
    holdAudio: boolean
  ) => {
    const conditioning = addNode(graph, {
      // Sent explicitly rather than left to the node's own default, so the panel's number and the
      // run's are the same one. It still reaches the join over an edge from the node's OUTPUT: a
      // source shorter than this contributes fewer frames, and only the node that read it knows.
      context_frames: settings.ltx2ExtendContextFrames,
      height: canvas.height,
      id,
      type: 'ltx2_extend_conditioning',
      video: { video_name: sourceVideo.video_name },
      width: canvas.width,
    });

    addEdge(graph, extract, 'video', conditioning, 'video');
    addEdge(graph, modelLoader, 'vae', conditioning, 'vae');
    addEdge(graph, conditioning, 'video_conditioning', pass, 'video_conditioning');
    // The same span of the source's sound as the picture it holds. Without it the join fades newly
    // invented audio in against the source's real audio, and the new soundtrack starts one overlap
    // early -- audible as the cut-over arriving a fraction of a second too soon.
    //
    // Only one anchor encodes it. The canvas never reaches the audio path, so a two-stage run's two
    // anchors would read the same trimmed clip, load the audio VAE and vocoder, and produce byte-
    // identical latents twice; the second pass is fed from the first instead.
    if (holdAudio) {
      addEdge(graph, modelLoader, 'audio_vae', conditioning, 'audio_vae');
      addEdge(graph, modelLoader, 'vocoder', conditioning, 'vocoder');
      addEdge(graph, extract, 'fps', conditioning, 'fps');
      addEdge(graph, conditioning, 'audio_conditioning', pass, 'audio_prefix_conditioning');
    }

    return conditioning;
  };

  /**
   * A held frame, encoded at one canvas and wired into one pass. Index 0 overwrites the grid's
   * opening tokens; anything else is appended to the model's sequence as a keyframe, which is what
   * lets a last frame coexist with a first one. Each pass needs its own encode at its own canvas --
   * see the refine block below for why.
   */
  const addHeldFrame = (
    id: string,
    image: NonNullable<VideoSettings['firstFrameImage']>,
    canvas: { width: number; height: number },
    pass: BackendInvocationContract,
    frameIndex: number
  ) => {
    const conditioning = addNode(graph, {
      frame_index: frameIndex,
      height: canvas.height,
      id,
      image: toImageField(image),
      type: 'ltx2_image_conditioning',
      width: canvas.width,
    });

    addEdge(graph, modelLoader, 'vae', conditioning, 'vae');
    addEdge(
      graph,
      conditioning,
      'video_conditioning',
      pass,
      frameIndex === 0 ? 'video_conditioning' : 'keyframe_conditioning'
    );

    return conditioning;
  };

  // The generation opens either from a still or from the tail of the clip being continued; a last
  // frame rides on top of either as a keyframe. `-1` is resolved against the run's own frame count
  // by the denoise node, so the encode does not go stale when the length changes -- including the
  // length a held soundtrack wires in, since audio-to-video holds the other stream and leaves both
  // frame slots free to anchor the picture generated for it.
  // Keyed on the MODE rather than on whichever slot still holds something. Validation refuses every
  // bad combination before a graph is built, so this is the second line rather than the first -- but
  // a held frame reaching the graph unasked is both a silent change of output and, beside a
  // whole-clip conditioning, a shape failure inside the transformer, and the sibling MiniMax builder
  // guards the same way.
  const holdsFirstFrame =
    (mode === 'first-frame' || mode === 'first-last' || mode === 'audio-to-video') && settings.firstFrameImage;
  const holdsLastFrame =
    (mode === 'last-frame' || mode === 'first-last' || mode === 'extend' || mode === 'audio-to-video') &&
    settings.lastFrameImage;

  if (holdsFirstFrame) {
    addHeldFrame('image_conditioning', holdsFirstFrame, stages.base, denoise, 0);
  }
  if (holdsLastFrame) {
    addHeldFrame('last_frame_conditioning', holdsLastFrame, stages.base, denoise, -1);
  }

  // The conditioning clip, in whichever role it was given. Both nodes report the frame count they
  // actually encoded, and that count is wired into the denoise rather than the panel's prediction
  // of it: an audio track need not be exactly as long as the picture it shipped with, and the
  // gallery's own frame count is duration x fps rounded.
  const conditioningClip = settings.conditioningClip;
  let audioConditioning: BackendInvocationContract | null = null;
  let heldVideoConditioning: BackendInvocationContract | null = null;

  if (mode === 'audio-to-video' || mode === 'video-to-audio') {
    if (!conditioningClip) {
      throw new Error('A conditioning clip is required for audio-to-video and video-to-audio generation.');
    }

    const video = { video_name: conditioningClip.clip.video_name };

    if (conditioningClip.role === 'audio') {
      audioConditioning = addNode(graph, {
        fps: timing.fps,
        id: 'audio_conditioning',
        type: 'ltx2_audio_conditioning',
        video,
      });

      addEdge(graph, modelLoader, 'audio_vae', audioConditioning, 'audio_vae');
      addEdge(graph, modelLoader, 'vocoder', audioConditioning, 'vocoder');
      addEdge(graph, audioConditioning, 'audio_conditioning', denoise, 'audio_conditioning');
      addEdge(graph, audioConditioning, 'num_frames', denoise, 'num_frames');
    } else {
      // The base canvas, which is the only canvas: a conditioned run has no refine pass, so the
      // two stages are the same and this matches the denoise's own width and height.
      const videoConditioning = addNode(graph, {
        fps: timing.fps,
        height: stages.base.height,
        id: 'video_conditioning',
        type: 'ltx2_video_conditioning',
        video,
        width: stages.base.width,
      });

      addEdge(graph, modelLoader, 'vae', videoConditioning, 'vae');
      addEdge(graph, videoConditioning, 'video_conditioning', denoise, 'full_video_conditioning');
      addEdge(graph, videoConditioning, 'num_frames', denoise, 'num_frames');
      heldVideoConditioning = videoConditioning;
    }
  }

  // The pass whose latents are decoded: the refine pass when there is one. Audio bypasses the
  // upscaler -- it has no spatial extent -- but still goes through the refine denoise, which
  // re-noises both modalities to one level so the transformer reads a single pair of timesteps.
  let finalDenoise = denoise;

  if (twoStage) {
    const upsample = addNode(graph, { id: 'latent_upsample', type: 'ltx2_latent_upsample' });

    addEdge(graph, denoise, 'video_latents', upsample, 'video_latents');
    addEdge(graph, modelLoader, 'latent_upsampler', upsample, 'latent_upsampler');
    addEdge(graph, modelLoader, 'vae', upsample, 'vae');

    const refine = addNode(graph, {
      fps: timing.fps,
      height: stages.final.height,
      id: 'refine_latents',
      num_frames: timing.numFrames,
      // Same schedule as the base pass: the patch is on the transformer both passes share.
      schedule: settings.acceleratorEnabled ? 'distilled' : 'auto',
      // The refine pass enters the schedule partway down and samples its tail, so a variant that
      // pays four forwards a step gets a budget of its own rather than the base pass's. Never more
      // than the base pass, though: shortening Steps for a quick probe must not leave the expensive
      // half of the run longer than the half the user just cut.
      steps: Math.min(policy.refineSteps ?? settings.steps, settings.steps),
      type: 'ltx2_denoise',
      width: stages.final.width,
      ...guidance,
    });

    addEdge(graph, transformerSource, 'transformer', refine, 'transformer');
    addEdge(graph, textEncoder, 'conditioning', refine, 'positive_conditioning');
    if (negativeWired) {
      addEdge(graph, textEncoder, 'negative_conditioning', refine, 'negative_conditioning');
    }
    addEdge(graph, seed, 'value', refine, 'seed');
    addEdge(graph, upsample, 'latents', refine, 'latents');
    // Both passes must run at one frame count. Under auto duration the literal above is the
    // panel's stale number, and `build_refine_state` checks the upsampled latents against it --
    // so without this the run dies after the base pass and the upsampler, blaming the canvas.
    if (durationHead) {
      addEdge(graph, durationHead, 'num_frames', refine, 'num_frames');
    }
    addEdge(graph, denoise, 'audio_latents', refine, 'audio_latents');

    // A second encode of every held frame, at the refine canvas. The refine pass re-noises every
    // token -- frame 0 included, and appended keyframes are not carried through the upsampler at
    // all -- and the base pass's encodes are half this size, so they cannot be re-used. Without
    // this, a two-stage run would regenerate its held frames from the prompt alone and quietly
    // mean something different by "first frame" than a single-stage one.
    if (holdsFirstFrame) {
      addHeldFrame('refine_image_conditioning', holdsFirstFrame, stages.final, refine, 0);
    }
    if (holdsLastFrame) {
      addHeldFrame('refine_last_frame_conditioning', holdsLastFrame, stages.final, refine, -1);
    }

    finalDenoise = refine;
  }

  // An extension's own clip is an intermediate: what the user asked for is the joined result, and
  // `addSourceJoin` names the join `video_output`, so the decode has to give that id up or it would
  // be silently overwritten by it.
  const joining = mode === 'extend' && settings.sourceVideo !== null;
  const output = addNode(graph, {
    fps: timing.fps,
    id: joining ? 'extension_clip' : 'video_output',
    is_intermediate: joining,
    type: 'ltx2_latents_to_video',
    use_cache: false,
  });

  addEdge(graph, finalDenoise, 'video_latents', output, 'video_latents');
  addEdge(graph, finalDenoise, 'audio_latents', output, 'audio_latents');
  if (audioConditioning) {
    // The soundtrack the clip was generated for is muxed in as supplied. The model still denoises
    // its own audio latents -- they are the stream it holds clean -- but decoding those would
    // replace the user's recording with a vocoder's copy of itself.
    addEdge(graph, audioConditioning, 'audio_conditioning', output, 'source_audio');
  }
  if (heldVideoConditioning) {
    // And the mirror of it: the picture was the given half here, so the user's own frames are
    // written out rather than the held latents being decoded back into a cover-cropped copy.
    addEdge(graph, heldVideoConditioning, 'video_conditioning', output, 'source_video');
  }
  addEdge(graph, modelLoader, 'vae', output, 'vae');
  addEdge(graph, modelLoader, 'audio_vae', output, 'audio_vae');
  addEdge(graph, modelLoader, 'vocoder', output, 'vocoder');

  // An extension is joined back onto the clip it continues. The generated half opens with the
  // source's own tail -- held clean, so both clips render the same moment -- and the crossfade
  // consumes exactly those frames from each side, which is what stops them playing twice. Its
  // length comes over an edge rather than as a literal: a source shorter than the requested
  // context contributes fewer frames, and only the node that read it knows how many.
  const join = joining && settings.sourceVideo ? addSourceJoin(graph, settings.sourceVideo, output) : null;

  if (join && settings.sourceVideo) {
    // Built here rather than with the other conditioning because it reads the TRIMMED clip, which
    // only exists once the join has been laid out.
    extendConditioning = addExtendAnchor(
      'extend_conditioning',
      settings.sourceVideo,
      join.extract,
      stages.base,
      denoise,
      true
    );
    if (twoStage) {
      // Appended or not, a held anchor does not survive the refine pass's re-noise, and the
      // upsampler never sees it -- so stage two is given the tail again at its own canvas. Its
      // sound is not canvas-dependent, so it comes from the first anchor rather than a second
      // encode of the same audio.
      addExtendAnchor(
        'refine_extend_conditioning',
        settings.sourceVideo,
        join.extract,
        stages.final,
        finalDenoise,
        false
      );
      addEdge(graph, extendConditioning, 'audio_conditioning', finalDenoise, 'audio_prefix_conditioning');
    }

    addEdge(graph, extendConditioning, 'context_frames', join.concat, 'transition_frames');
    if (durationHead) {
      addEdge(graph, extendConditioning, 'context_frames', durationHead, 'context_frames');
      addEdge(graph, join.extract, 'fps', durationHead, 'fps');
    }
    // The continuation inherits the source's own frame rate, read off the trimmed clip at run time
    // rather than from the gallery's record of it -- a video row's fps is nullable, and a guess
    // here would play the two halves at different speeds and mistime the generated soundtrack.
    addEdge(graph, join.extract, 'fps', denoise, 'fps');
    addEdge(graph, join.extract, 'fps', output, 'fps');
    if (twoStage) {
      addEdge(graph, join.extract, 'fps', finalDenoise, 'fps');
    }
    // The join's own fps is deliberately left unset: `video_concat` then takes the first input's
    // rate, which is this same trimmed clip, and refuses the join outright if the two halves
    // disagree. Wiring it would be worse in three ways -- the field is `Optional[int]`, so the edge
    // is float -> int and the graph is refused at enqueue before anything runs; an int cannot carry
    // 23.976; and forcing a rate would retime rather than catch a mismatch.
  }

  const joined = join?.concat ?? null;

  const metadata = addVideoMetadata({
    extras: {
      cfg_scale: guidance.cfg_scale,
      // What will actually be delivered: a continuation runs at the source's rate, every other mode
      // at the panel's. Recorded rounded, mirroring what the clip reports back.
      // For a continuation this is the gallery's reading of the source, which is the best number
      // available when the graph is built -- the run itself takes the clip's true rate off the
      // extract node, so a fractional source is delivered at 23.976 and recorded as 24.
      fps: joined && settings.sourceVideo ? Math.round(settings.sourceVideo.fps) : timing.fps,
      ltx2_text_encoder_model: settings.ltx2TextEncoderModel,
      ...(componentSource ? { ltx2_component_source: componentSource } : {}),
      ...(policy.ui.audioCfgVisible ? { ltx2_audio_cfg_scale: guidance.audio_cfg_scale } : {}),
      ...(policy.ui.stgVisible ? { ltx2_stg_scale: guidance.stg_scale } : {}),
      ...(policy.ui.modalityVisible ? { ltx2_modality_scale: guidance.modality_scale } : {}),
      // Only a continuation has one, and it is not recoverable from anything else in the record:
      // the output length folds the source, the generated half and the crossfade together, so a
      // recall without this would silently reinstate the default context and a different run.
      ...(policy.ui.extendContext ? { ltx2_context_frames: settings.ltx2ExtendContextFrames } : {}),
      // Informational, not load-bearing: recall recovers the preset by matching the recorded
      // width/height against each one, and a two-stage preset's canvas is unique among them. These
      // are here for someone reading a clip's metadata, to whom "1792x1024" alone does not say that
      // it was reached by refining an 896x512 pass rather than generated at size.
      ...(twoStage
        ? { ltx2_base_height: stages.base.height, ltx2_base_width: stages.base.width, ltx2_two_stage: true }
        : {}),
      // The clip is the run's length and, in the video role, its frame rate -- so record the count
      // that ran rather than the panel's stored one, which these modes do not use.
      ...(conditioningClip
        ? {
            ltx2_conditioning_role: conditioningClip.role,
            ltx2_conditioning_video: { video_name: conditioningClip.clip.video_name },
            num_frames: timing.numFrames,
          }
        : {}),
    },
    generationMode: LTX2_GENERATION_MODES[mode] ?? 'ltx2_t2v',
    graph,
    height: dimensions.height,
    model,
    negativeWired,
    // The joined clip is the result; the generated half is kept as an intermediate, and both carry
    // the metadata so either can be recalled.
    outputs: joined ? [output, joined] : [output],
    settings,
    width: dimensions.width,
  });

  // The recorded length has to be the one that ran, not the one the panel was holding: under auto
  // duration the panel's number is stale by construction, and recall reads this back.
  if (durationHead) {
    addEdge(graph, durationHead, 'num_frames', metadata, 'num_frames');
  }

  return graph;
};

/**
 * One builder per supported family. A record rather than a conditional so a new
 * family cannot compile without one — the failure would otherwise be a Wan
 * graph built for someone else's model.
 */
const VIDEO_GRAPH_BUILDERS = {
  'ltx-2': buildLtx2VideoGraph,
  'minimax-h3': buildMiniMaxH3VideoGraph,
  wan: buildWanVideoGraph,
} satisfies Record<SupportedVideoBase, (settings: VideoSettings, model: MainModelConfig) => BackendGraphContract>;

export const compileVideoGraph = (settings: VideoSettings, model: MainModelConfig): CompiledVideoGraph => {
  const validationReasons = getVideoValidationReasons(model, settings);

  if (validationReasons.length > 0) {
    throw new Error(validationReasons[0]);
  }

  const builder = VIDEO_GRAPH_BUILDERS[model.base as SupportedVideoBase];

  if (!builder) {
    throw new Error(`No video graph builder for ${model.base}.`);
  }

  const backendGraph = builder(settings, model);

  return {
    backendGraph,
    graph: toGraphContract(backendGraph, `${model.name} video`),
    negativePromptNodeId: 'negative_prompt',
    positivePromptNodeId: 'positive_prompt',
    seedNodeId: 'seed',
  };
};
