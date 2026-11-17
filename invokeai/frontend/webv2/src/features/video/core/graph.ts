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

/** Feed identical ordered references to conditioning and prompt nodes; chained collectors keep the rotary order. */
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

/** Converts both bounds together to preserve length near the tail; starts within TAIL_INDEX_SLOP stay absolute. */
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

/** Bounds within this many frames of the end compile as negative indices; mid-clip bounds stay absolute. */
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

  // Standalone Wan checkpoints need external VAE/UMT5 components; Diffusers mains bundle them and the second expert.
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
    // Only A14B pairs run a low-noise phase; null means the primary guidance, not the node's 4.0 default.
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

  // H3 checkpoints override the transformer of a required Diffusers component model.
  const isSingleFileMain = model.format !== 'diffusers';
  const componentSource = isSingleFileMain ? settings.componentSourceModel : null;

  if (isSingleFileMain && !componentSource) {
    throw new Error('A single-file MiniMax H3 transformer needs a Diffusers install selected under Model Components.');
  }

  // Guard stale settings so an unsupported main cannot silently load the hybrid's wrong file.
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
    addEdge(graph, extendParts.lastFrame, 'image', metadata, 'first_frame_image');
  }

  return graph;
};

/** Inert guidance literals are sent rather than omitted, so the graph states what will actually run. */
const buildLtx2VideoGraph = (settings: VideoSettings, model: MainModelConfig): BackendGraphContract => {
  const mode = resolveVideoMode(settings);
  const policy = getVideoModelPolicy(model, settings);
  const dimensions = getVideoDimensions(model, settings);
  // A conditioning clip owns the length (and in the video role the rate); literals and metadata all read this.
  const timing = getEffectiveVideoTiming(model, settings);

  if (!dimensions) {
    throw new Error('Video dimensions could not be derived from the current settings.');
  }

  // Validation requires both slots; these throws are backstops for direct callers.
  const componentSource = model.format === 'diffusers' ? null : settings.componentSourceModel;

  if (model.format !== 'diffusers' && !componentSource) {
    throw new Error('A single-file LTX-2 transformer needs an LTX-2 components install under Model Components.');
  }
  if (!settings.ltx2TextEncoderModel) {
    throw new Error('LTX-2 needs its Gemma-4 text encoder selected under Model Components.');
  }

  // Coerced as in `getVideoDimensions`: a record can hold a preset this model lacks, whose canvas would be NaN.
  const targetResolution = getVideoTargetResolution(model, settings.targetResolution) as Ltx2TargetResolution;
  const stages = getLtx2StageCanvases(dimensions.width, dimensions.height, targetResolution);

  if (!stages) {
    throw new Error('Video dimensions could not be derived from the current settings.');
  }

  // Asked of the preset: comparing canvases misreads equal widths and NaN.
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

  // Without CFG the negative prompt never reaches the model, so skip its 12B encode.
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

  // CFG scales hold at 1 without a negative prompt (the node refuses more); STG and modality steer off the positive.
  // Metadata records these, so recall reproduces the run rather than the settings.
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
    // 'auto' resolves off the transformer variant, not the patch, so a distilled LoRA on Dev must be named here.
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

  // Only where the length is open: a conditioning clip drives `num_frames` by its own edge. The head picks at
  // most Frames, in frames; the node converts to seconds at the rate it actually runs.
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

  /** Reads the trimmed source, not the gallery file, so the join does not dissolve two unrelated moments. */
  const addExtendAnchor = (
    id: string,
    sourceVideo: NonNullable<VideoSettings['sourceVideo']>,
    extract: BackendInvocationContract,
    canvas: { width: number; height: number },
    pass: BackendInvocationContract,
    holdAudio: boolean
  ) => {
    const conditioning = addNode(graph, {
      // Sent explicitly so panel and run agree; the join still reads the node's output, as a short source gives fewer.
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
    // Holds the source's audio too, or the new soundtrack cuts over early. Only one anchor encodes it.
    if (holdAudio) {
      addEdge(graph, modelLoader, 'audio_vae', conditioning, 'audio_vae');
      addEdge(graph, modelLoader, 'vocoder', conditioning, 'vocoder');
      addEdge(graph, extract, 'fps', conditioning, 'fps');
      addEdge(graph, conditioning, 'audio_conditioning', pass, 'audio_prefix_conditioning');
    }

    return conditioning;
  };

  /** Index 0 overwrites the opening tokens; others append as keyframes. Each pass needs its own encode. */
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

  // Keyed on the mode, not the filled slot: an unasked held frame changes output or fails beside a whole-clip
  // conditioning. `-1` resolves against the run's frame count, so the encode does not go stale.
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

  // Wire the frame counts the nodes actually encoded, not the panel's prediction.
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
      // A conditioned run has no refine pass, so the base canvas is the only one.
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

  // Audio bypasses the upscaler but still goes through the refine denoise, so both modalities share one timestep.
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
      // Its own budget for expensive variants, never more than the base pass's.
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
    // Both passes need one frame count; `build_refine_state` would otherwise reject the upsampled latents late.
    if (durationHead) {
      addEdge(graph, durationHead, 'num_frames', refine, 'num_frames');
    }
    addEdge(graph, denoise, 'audio_latents', refine, 'audio_latents');

    // Re-encode held frames at the refine canvas: re-noise drops them, and the base encodes are half size.
    if (holdsFirstFrame) {
      addHeldFrame('refine_image_conditioning', holdsFirstFrame, stages.final, refine, 0);
    }
    if (holdsLastFrame) {
      addHeldFrame('refine_last_frame_conditioning', holdsLastFrame, stages.final, refine, -1);
    }

    finalDenoise = refine;
  }

  // `addSourceJoin` names the join `video_output`, so the intermediate decode must give that id up.
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
    // Mux the user's soundtrack as supplied, rather than a vocoder's copy of it.
    addEdge(graph, audioConditioning, 'audio_conditioning', output, 'source_audio');
  }
  if (heldVideoConditioning) {
    // Likewise write out the user's frames rather than a cover-cropped decode of them.
    addEdge(graph, heldVideoConditioning, 'video_conditioning', output, 'source_video');
  }
  addEdge(graph, modelLoader, 'vae', output, 'vae');
  addEdge(graph, modelLoader, 'audio_vae', output, 'audio_vae');
  addEdge(graph, modelLoader, 'vocoder', output, 'vocoder');

  // The crossfade consumes the held tail from each side, so it does not play twice; its length comes over an edge.
  const join = joining && settings.sourceVideo ? addSourceJoin(graph, settings.sourceVideo, output) : null;

  if (join && settings.sourceVideo) {
    // Built here because it reads the trimmed clip, which exists only once the join is laid out.
    extendConditioning = addExtendAnchor(
      'extend_conditioning',
      settings.sourceVideo,
      join.extract,
      stages.base,
      denoise,
      true
    );
    if (twoStage) {
      // Stage two gets the tail again at its canvas; its audio comes from the first anchor.
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
    // Read the source rate off the trimmed clip at run time; the gallery's fps is nullable.
    addEdge(graph, join.extract, 'fps', denoise, 'fps');
    addEdge(graph, join.extract, 'fps', output, 'fps');
    if (twoStage) {
      addEdge(graph, join.extract, 'fps', finalDenoise, 'fps');
    }
    // Join fps stays unset: `video_concat` takes the first input's rate. The field is Optional[int], so an edge
    // from a float is refused at enqueue, and an int cannot carry 23.976.
  }

  const joined = join?.concat ?? null;

  const metadata = addVideoMetadata({
    extras: {
      cfg_scale: guidance.cfg_scale,
      // Recorded rounded; the run takes the true rate off the extract node.
      fps: joined && settings.sourceVideo ? Math.round(settings.sourceVideo.fps) : timing.fps,
      ltx2_text_encoder_model: settings.ltx2TextEncoderModel,
      ...(componentSource ? { ltx2_component_source: componentSource } : {}),
      ...(policy.ui.audioCfgVisible ? { ltx2_audio_cfg_scale: guidance.audio_cfg_scale } : {}),
      ...(policy.ui.stgVisible ? { ltx2_stg_scale: guidance.stg_scale } : {}),
      ...(policy.ui.modalityVisible ? { ltx2_modality_scale: guidance.modality_scale } : {}),
      // Not recoverable from the output length, so recall needs it to reproduce the context.
      ...(policy.ui.extendContext ? { ltx2_context_frames: settings.ltx2ExtendContextFrames } : {}),
      // Informational: recall recovers the preset from width/height.
      ...(twoStage
        ? { ltx2_base_height: stages.base.height, ltx2_base_width: stages.base.width, ltx2_two_stage: true }
        : {}),
      // Record the clip-driven count that ran, not the panel's stored one.
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
    // Both outputs carry metadata so either can be recalled.
    outputs: joined ? [output, joined] : [output],
    settings,
    width: dimensions.width,
  });

  // Record the length that ran: under auto duration the panel's number is stale.
  if (durationHead) {
    addEdge(graph, durationHead, 'num_frames', metadata, 'num_frames');
  }

  return graph;
};

/** A record rather than a conditional, so a new family cannot compile without its builder. */
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
