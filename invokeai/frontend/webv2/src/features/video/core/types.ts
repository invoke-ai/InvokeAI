import type {
  DynamicPromptsSettings,
  GenerateLora,
  ImageWithDims,
  MainModelConfig,
  ModelIdentifierConfig,
  VaeModelConfig,
} from '@features/generation/contracts';
import type { SeedMode } from '@platform/core/seed';

export type VideoGenerationMode =
  | 'txt2vid'
  | 'first-frame'
  | 'last-frame'
  | 'first-last'
  | 'extend'
  | 'reference'
  | 'audio-to-video'
  | 'video-to-audio';

export interface VideoClipRef {
  video_name: string;
  width: number;
  height: number;
  numFrames: number;
  fps: number;
}

export interface VideoSourceClip extends VideoClipRef {
  /** Inclusive trim bounds forwarded to `extract_video_range`; negative indices count from the end. */
  startFrame: number;
  endFrame: number;
}

/** Which stream of a conditioning clip the model is given, and which it therefore generates. */
export type VideoConditioningRole = 'audio' | 'video';

/** One slot: LTX-2 holds one modality clean and samples the other. */
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
      /** Panel-only Initial Video anchor, pinned last; its default trim follows the source unless trimOverridden. */
      fromSourceVideo?: boolean;
      /** Panel-only: stops trim re-derivation after manual edits; meaningful only with fromSourceVideo. */
      trimOverridden?: boolean;
      /** Panel-only requested length, kept apart from clamped bounds; read through referenceSampleFrames. */
      sampleFrames?: number;
    }
  | { kind: 'image'; image: ImageWithDims; detail: VideoReferenceImageDetail };

export type WanTargetResolution = '480p' | '720p' | '1080p';
export type MiniMaxH3TargetResolution = '768 highres' | '768 lowres' | '480p' | '544p';
/** LTX-2 presets pin the canvas's SHORT edge; the long edge follows the aspect ratio. */
export type Ltx2TargetResolution = '512p' | '704p' | '768p' | '1024p' | '1536p';
export type VideoTargetResolution = WanTargetResolution | MiniMaxH3TargetResolution | Ltx2TargetResolution;

/** Derive dimensions from preset ratio/resolution or conditioning media; no free-size fields are offered. */
export type VideoAspectRatioId = '21:9' | '16:9' | '3:2' | '4:3' | '1:1' | '3:4' | '2:3' | '9:16' | '9:21';

/** Project-persisted settings owned by the Video widget; the prompt expands like Generate's. */
export interface VideoSettings extends DynamicPromptsSettings {
  batchCount: number;
  modelKey: string;
  positivePrompt: string;
  positivePromptHeightPx: number;
  negativePromptEnabled: boolean;
  negativePrompt: string;
  negativePromptHeightPx: number;
  /** Image-to-video conditioning. Mutually exclusive with `sourceVideo`. */
  firstFrameImage: ImageWithDims | null;
  /** Interpolation's endpoint or an extension's destination, paired with first frame or source video. */
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
  /** LTX-2 guides audio separately and harder; null on families with one guidance scale. */
  audioCfgScale: number | null;
  /** LTX-2 spatio-temporal guidance; 0 turns it off and saves a forward per step. Null on families without it. */
  stgScale: number | null;
  /** LTX-2 modality-isolation guidance; 1 turns it off. Null on families without it. */
  modalityScale: number | null;
  /** On the 8k + 1 grid; spent twice (reproduced, then crossfaded), so each frame costs one of new material. */
  ltx2ExtendContextFrames: number;
  /** Records intent: acceleration patches visible sampling settings and LoRAs, not hidden graph state. */
  acceleratorEnabled: boolean;
  /** Only toggle-added LoRA keys, so user-owned entries are never removed. */
  acceleratorLoraKeys: string[];
  seed: number;
  seedMode: SeedMode;
  loras: GenerateLora[];
  /** Optional VAE override; null uses the VAE bundled with the main model or component source. */
  vae: VaeModelConfig | null;
  wanT5EncoderModel: ModelIdentifierConfig | null;
  wanLowNoiseModel: MainModelConfig | null;
  /** Diffusers component source supplies missing non-transformer components for standalone Wan/H3 mains. */
  componentSourceModel: MainModelConfig | null;
  /** Legacy only: reconciliation promotes it to `model`, keeping the old main as componentSourceModel. */
  h3TransformerModel: MainModelConfig | null;
  /** Optional single-file MiniMax H3 Qwen3-VL text-encoder override. */
  h3TextEncoderModel: ModelIdentifierConfig | null;
  /** LTX-2's Gemma-4 text encoder. Required, not an override: no LTX-2 main carries text-encoder weights. */
  ltx2TextEncoderModel: ModelIdentifierConfig | null;
  /** LTX-2's duration head. Optional: without it the run uses the frame count the panel holds. */
  ltx2DurationHeadModel: ModelIdentifierConfig | null;
  /** Healed against the head only, not the transient mode, or a source clip round trip would lose it. */
  autoDuration: boolean;
  /** Hybrid: FL2VA base weights with Ref2VA AdaLN from h3HybridStartBlock onward. */
  h3HybridBaseModel: MainModelConfig | null;
  /** First transformer block (0-49) whose AdaLN projection stays Ref2VA's under the hybrid. */
  h3HybridStartBlock: number;
}

export interface VideoWidgetValues extends VideoSettings {
  /** The selected main model; null until the user picks one (or none is installed). */
  model: MainModelConfig | null;
}
