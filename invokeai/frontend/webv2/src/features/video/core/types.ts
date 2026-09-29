import type {
  GenerateLora,
  ImageWithDims,
  MainModelConfig,
  ModelIdentifierConfig,
  VaeModelConfig,
} from '@features/generation/contracts';
import type { SeedMode } from '@platform/core/seed';

/** Infer conditioning mode from populated inputs through resolveVideoMode. */
export type VideoGenerationMode =
  | 'txt2vid'
  | 'first-frame'
  | 'last-frame'
  | 'first-last'
  | 'extend'
  | 'reference'
  | 'audio-to-video'
  | 'video-to-audio';

/** A gallery video's identity and geometry, as the panel stores it. */
export interface VideoClipRef {
  video_name: string;
  width: number;
  height: number;
  numFrames: number;
  fps: number;
}

/** A gallery video selected as the clip to extend, with the trim range to keep. */
export interface VideoSourceClip extends VideoClipRef {
  /** Inclusive trim bounds forwarded to `extract_video_range`; negative indices count from the end. */
  startFrame: number;
  endFrame: number;
}

/** Which stream of a conditioning clip the model is given, and which it therefore generates. */
export type VideoConditioningRole = 'audio' | 'video';

/**
 * A clip conditioning the opposite modality. One slot, because LTX-2 holds one modality clean and samples the
 * other, so a clip can only be given in one role at a time.
 */
export interface VideoConditioningClip {
  /** Whole-clip and untrimmed: the conditioning nodes consume the whole recording. */
  clip: VideoClipRef;
  role: VideoConditioningRole;
  /** False when the gallery record had no frame rate and `clip.fps` is the panel's guess. */
  fpsKnown: boolean;
}

/** Reference conditioning maps directly to graph literals; audio uses the clip's soundtrack without visual rows. */
export type VideoReferenceConditioning = 'video_audio' | 'video' | 'audio';

/** Ref2VA image-reference sizing: 'max' = 2048px short edge, 'match' = generation's pixel area. */
export type VideoReferenceImageDetail = 'max' | 'match';

/** Reference order affects generation; retain one ordered mixed-kind array with video trim bounds. */
export type VideoReferenceItem =
  | {
      kind: 'video';
      clip: VideoSourceClip;
      conditioning: VideoReferenceConditioning;
      /**
       * Panel-only marker for the Initial Video continuity anchor, pinned last. Its default trim follows source
       * cutpoints and generated frame count unless trimOverridden is set.
       */
      fromSourceVideo?: boolean;
      /**
       * Panel-only anchor override disables cutpoint/frame-count re-derivation after manual trim edits.
       * Clear/reset source removes it; meaningful only with fromSourceVideo.
       */
      trimOverridden?: boolean;
      /**
       * Panel-only requested length stays separate from clamped clip bounds so drags can restore it; read through
       * referenceSampleFrames.
       */
      sampleFrames?: number;
    }
  | { kind: 'image'; image: ImageWithDims; detail: VideoReferenceImageDetail };

export type WanTargetResolution = '480p' | '720p' | '1080p';
export type MiniMaxH3TargetResolution = '768 highres' | '768 lowres';
/** LTX-2 presets pin the canvas's SHORT edge; the long edge follows the aspect ratio. */
export type Ltx2TargetResolution = '512p' | '704p' | '768p' | '1024p' | '1536p';
export type VideoTargetResolution = WanTargetResolution | MiniMaxH3TargetResolution | Ltx2TargetResolution;

/** Derive dimensions from preset ratio/resolution or conditioning media; no free-size fields are offered. */
export type VideoAspectRatioId = '21:9' | '16:9' | '3:2' | '4:3' | '1:1' | '3:4' | '2:3' | '9:16' | '9:21';

/** Project-persisted settings owned by the Video widget. */
export interface VideoSettings {
  batchCount: number;
  modelKey: string;
  positivePrompt: string;
  positivePromptHeightPx: number;
  negativePromptEnabled: boolean;
  negativePrompt: string;
  negativePromptHeightPx: number;
  /** Image-to-video conditioning. Mutually exclusive with `sourceVideo`. */
  firstFrameImage: ImageWithDims | null;
  /**
   * Last frame supplies interpolation's endpoint or an extension destination, paired with first frame or source
   * video.
   */
  lastFrameImage: ImageWithDims | null;
  /** Source excludes first frame; FL2VA uses extend mode while Ref2VA appends using a linked tail reference. */
  sourceVideo: VideoSourceClip | null;
  /** A clip conditioning the opposite modality; null unless the family offers a2v/v2a. */
  conditioningClip: VideoConditioningClip | null;
  /** Ordered Ref2VA references exclude frame slots but may coexist with source video for reference extension. */
  references: VideoReferenceItem[];
  aspectRatioId: VideoAspectRatioId;
  targetResolution: VideoTargetResolution;
  numFrames: number;
  fps: number;
  steps: number;
  cfgScale: number;
  /** Guidance for the low-noise half of a Wan A14B schedule; null reuses `cfgScale`. */
  cfgScaleLowNoise: number | null;
  /**
   * LTX-2 guides audio separately from video and far harder (7 against 3). Null on a family with one guidance
   * scale, like `cfgScaleLowNoise` outside Wan A14B.
   */
  audioCfgScale: number | null;
  /** LTX-2 spatio-temporal guidance; 0 turns it off and saves a forward per step. Null on families without it. */
  stgScale: number | null;
  /** LTX-2 modality-isolation guidance; 1 turns it off. Null on families without it. */
  modalityScale: number | null;
  /**
   * Acceleration patches visible sampling settings and LoRAs; this flag records intent rather than hidden graph
   * state.
   */
  /**
   * Frames of the source an LTX-2 continuation opens with, held clean so the model reads the clip's
   * motion rather than just its last still. On the VAE's 8k + 1 grid, and it is spent twice over:
   * the generation reproduces these frames, and the join then crossfades exactly them out of both
   * halves — so raising it costs new material one frame for one against a fixed Frames budget.
   */
  ltx2ExtendContextFrames: number;
  acceleratorEnabled: boolean;
  /**
   * Track exactly the toggle-added LoRA keys for removal; never remove matching user-owned entries, and clear
   * enabled intent if they disappear.
   */
  acceleratorLoraKeys: string[];
  seed: number;
  seedMode: SeedMode;
  loras: GenerateLora[];
  /** Optional VAE override; null uses the VAE bundled with the main model or component source. */
  vae: VaeModelConfig | null;
  /** Wan 2.2's UMT5-XXL text encoder. */
  wanT5EncoderModel: ModelIdentifierConfig | null;
  /** The low-noise expert of a Wan 2.2 A14B mixture-of-experts pair. */
  wanLowNoiseModel: MainModelConfig | null;
  /** Diffusers component source supplies missing non-transformer components for standalone Wan/H3 mains. */
  componentSourceModel: MainModelConfig | null;
  /**
   * Legacy-only transformer override: reconciliation promotes it to model and retains the old main as
   * componentSourceModel; new writes omit it.
   */
  h3TransformerModel: MainModelConfig | null;
  /** Optional single-file MiniMax H3 Qwen3-VL text-encoder override. */
  h3TextEncoderModel: ModelIdentifierConfig | null;
  /**
   * LTX-2's Gemma-4 text encoder. Required, not an override: no LTX-2 main carries text-encoder weights.
   */
  ltx2TextEncoderModel: ModelIdentifierConfig | null;
  /**
   * LTX-2's duration head. Optional: without it the run uses the frame count the panel holds.
   */
  ltx2DurationHeadModel: ModelIdentifierConfig | null;
  /**
   * Let the duration head choose the clip's length instead of the Frames control.
   *
   * Healed against the head only: a stored `true` whose model is gone becomes `false`. It is
   * deliberately NOT healed against the mode, which is transient -- dropping in a source clip and
   * taking it out again would otherwise silently lose the setting. `isAutoDurationActive` is what
   * decides whether it applies to a given run.
   */
  autoDuration: boolean;
  /**
   * Hybrid loads FL2VA base weights and overlays selected Ref2VA AdaLN from h3HybridStartBlock onward while
   * retaining reference conditioning.
   */
  h3HybridBaseModel: MainModelConfig | null;
  /** First transformer block (0-49) whose AdaLN projection stays Ref2VA's under the hybrid. */
  h3HybridStartBlock: number;
}

export interface VideoWidgetValues extends VideoSettings {
  /** The selected main model; null until the user picks one (or none is installed). */
  model: MainModelConfig | null;
}
