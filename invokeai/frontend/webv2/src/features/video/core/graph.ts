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

import type { VideoGenerationMode, VideoSettings } from './types';

import { resolveVideoMode } from './settings';
import { getVideoDimensions, getVideoModelPolicy, getVideoValidationReasons } from './videoPolicies';

/**
 * Video graph compilation: one builder per model family, mirroring the shape
 * of the bundled video workflow templates (which are the reference wiring for
 * conditioning fan-out and metadata). Every graph exposes the same three fixed
 * node ids so the submit pipeline can inject expanded prompts and batch seeds
 * exactly as it does for image generation.
 */

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

const MINIMAX_H3_GENERATION_MODES: Record<VideoGenerationMode, string> = {
  extend: 'minimax_h3_extend_video',
  'first-frame': 'minimax_h3_i2v',
  'first-last': 'minimax_h3_flf2v',
  'last-frame': 'minimax_h3_lf2v',
  txt2vid: 'minimax_h3_t2v',
};

const addPromptAndSeedNodes = (graph: BackendGraphContract) => ({
  negativePrompt: addNode(graph, { id: 'negative_prompt', type: 'string' }),
  positivePrompt: addNode(graph, { id: 'positive_prompt', type: 'string' }),
  seed: addNode(graph, { id: 'seed', type: 'integer' }),
});

const toImageField = (image: { image_name: string }) => ({ image_name: image.image_name });

/**
 * The extend-mode front end shared by both families, mirroring the bundled
 * extend templates: trim the source clip, pull its last frame as the new
 * clip's conditioning frame, and set up the concat that joins the trimmed
 * source with the freshly generated clip (2-frame crossfade over the seam).
 * Both intermediate video artifacts stay out of the gallery.
 */
const addExtendScaffolding = (
  graph: BackendGraphContract,
  sourceVideo: NonNullable<VideoSettings['sourceVideo']>,
  newClip: BackendInvocationContract,
  options: { extractFps?: number } = {}
) => {
  // The panel's frame count is an estimate (duration × fps; the records store
  // no exact count, and VFR uploads can overshoot by a frame or two). A trim
  // end near the estimated ceiling therefore compiles as a NEGATIVE index —
  // resolved by the backend against the clip's REAL count — so "keep to the
  // end" can never land past it. Mid-clip picks stay positive: a negative
  // offset computed from an overshooting estimate would drift them instead.
  const tailOffset = sourceVideo.numFrames - 1 - sourceVideo.endFrame;
  const endFrameLiteral = tailOffset <= 3 ? -(tailOffset + 1) : sourceVideo.endFrame;
  const extract = addNode(graph, {
    end_frame: endFrameLiteral,
    id: 'source_video',
    is_intermediate: true,
    start_frame: sourceVideo.startFrame,
    type: 'extract_video_range',
    use_cache: false,
    video: { video_name: sourceVideo.video_name },
    ...(options.extractFps === undefined ? {} : { fps: options.extractFps }),
  });
  const lastFrame = addNode(graph, {
    frame_index: -1,
    id: 'source_last_frame',
    is_intermediate: true,
    type: 'video_frame_extract',
    use_cache: false,
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

  addEdge(graph, extract, 'video', lastFrame, 'video');
  // Chained collectors keep the join order deterministic: [trimmed source, new clip].
  addEdge(graph, extract, 'video', sourceCollect, 'item');
  addEdge(graph, sourceCollect, 'collection', clipsCollect, 'collection');
  addEdge(graph, newClip, 'video', clipsCollect, 'item');
  addEdge(graph, clipsCollect, 'collection', concat, 'videos');

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
    ...(settings.sourceVideo ? { source_video: { video_name: settings.sourceVideo.video_name } } : {}),
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

  // A GGUF/checkpoint Wan main carries only the transformer; the VAE and
  // UMT5-XXL encoder come from standalone models or a Diffusers component
  // source. A Diffusers main bundles everything, including the second expert.
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
  // Wan LoRAs patch only the transformer; expert routing ('auto') reads each
  // LoRA's probed high/low tag, which is how the Lightning pair lands on the
  // right experts.
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

  // The reference-image conditioning: the canvas and frame count must match
  // the denoise exactly, so the same literals fan out to both nodes.
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
    // The extension inherits the source clip's frame rate, template-style, so
    // the joined halves play at one speed.
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
      ...(policy.ui.cfgLowNoiseVisible && settings.cfgScaleLowNoise !== null
        ? { guidance_scale_low_noise: settings.cfgScaleLowNoise }
        : {}),
      ...(settings.vae ? { vae: settings.vae } : {}),
      ...(settings.wanT5EncoderModel ? { wan_t5_encoder: settings.wanT5EncoderModel } : {}),
      ...(sourceModel ? { wan_component_source: sourceModel } : {}),
      ...(isSingleFileMain && settings.wanLowNoiseModel ? { transformer_low_noise: settings.wanLowNoiseModel } : {}),
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
    // The extracted conditioning frame is only known at run time; record it
    // via an edge, like the bundled extend templates do.
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

  const graph: BackendGraphContract = { edges: [], id: createId('minimax_h3_video_graph'), nodes: {} };
  const { positivePrompt, seed } = addPromptAndSeedNodes(graph);
  const modelLoader = addNode(graph, {
    id: 'model_loader',
    model,
    text_encoder_model: settings.h3TextEncoderModel ?? undefined,
    transformer_model: settings.h3TransformerModel ?? undefined,
    type: 'minimax_h3_model_loader',
  });
  const activeLoras = getActiveCompatibleLoras(settings, model);
  const transformerSource = activeLoras.length
    ? addTransformerLoraCollectionLoader(graph, activeLoras, 'minimax_h3_lora_collection_loader', modelLoader, [
        'transformer',
      ])
    : modelLoader;

  // The canvas and the keyframes must match across text encoder, frame
  // conditioning, and denoise — the same literals (and, in extend mode, the
  // same extracted-frame edge) fan out to all of them.
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
  });
  const denoise = addNode(graph, {
    height: dimensions.height,
    id: 'denoise_latents',
    // The node's frame counts are a string Literal choice list.
    num_frames: String(settings.numFrames),
    steps: settings.steps,
    type: 'minimax_h3_denoise',
    width: dimensions.width,
  });

  addEdge(graph, modelLoader, 'text_encoder', posCond, 'text_encoder');
  addEdge(graph, positivePrompt, 'value', posCond, 'prompt');
  addEdge(graph, transformerSource, 'transformer', denoise, 'transformer');
  addEdge(graph, posCond, 'conditioning', denoise, 'positive_conditioning');
  addEdge(graph, seed, 'value', denoise, 'seed');

  const needsFrameConditioning = mode !== 'txt2vid';
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

  if (mode === 'extend' && settings.sourceVideo && frameConditioning) {
    const newClip = addLatentsToVideo('extension_clip', true);
    // H3 generates at a fixed 24 fps, so the source is resampled to 24 up
    // front (template behavior) and the concat inherits it from the first clip.
    const { concat, lastFrame } = addExtendScaffolding(graph, settings.sourceVideo, newClip, { extractFps: 24 });

    addEdge(graph, lastFrame, 'image', posCond, 'first_image');
    addEdge(graph, lastFrame, 'image', frameConditioning, 'first_image');
    output = concat;
    extendParts = { lastFrame, newClip };
  } else {
    output = addLatentsToVideo('video_output', false);
  }

  const metadata = addVideoMetadata({
    extras: {
      ...(settings.h3TransformerModel ? { minimax_h3_transformer_model: settings.h3TransformerModel } : {}),
      ...(settings.h3TextEncoderModel ? { minimax_h3_text_encoder_model: settings.h3TextEncoderModel } : {}),
    },
    generationMode: MINIMAX_H3_GENERATION_MODES[mode],
    graph,
    height: dimensions.height,
    model,
    negativeWired: false,
    outputs: extendParts ? [output, extendParts.newClip] : [output],
    settings,
    width: dimensions.width,
  });

  if (extendParts) {
    // The extracted conditioning frame is only known at run time; record it
    // via an edge, like the bundled extend templates do.
    addEdge(graph, extendParts.lastFrame, 'image', metadata, 'first_frame_image');
  }

  return graph;
};

export const compileVideoGraph = (settings: VideoSettings, model: MainModelConfig): CompiledVideoGraph => {
  const validationReasons = getVideoValidationReasons(model, settings);

  if (validationReasons.length > 0) {
    throw new Error(validationReasons[0]);
  }

  const backendGraph =
    model.base === 'minimax-h3' ? buildMiniMaxH3VideoGraph(settings, model) : buildWanVideoGraph(settings, model);

  return {
    backendGraph,
    graph: toGraphContract(backendGraph, `${model.name} video`),
    negativePromptNodeId: 'negative_prompt',
    positivePromptNodeId: 'positive_prompt',
    seedNodeId: 'seed',
  };
};
