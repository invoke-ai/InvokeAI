import type {
  GenerateLora,
  ImageWithDims,
  MainModelConfig,
  ModelIdentifierConfig,
  VaeModelConfig,
} from '@features/generation/contracts';

/**
 * How a video generation is conditioned. There is no explicit mode selector:
 * the mode is inferred from which inputs are filled — see `resolveVideoMode`.
 */
export type VideoGenerationMode = 'txt2vid' | 'first-frame' | 'last-frame' | 'first-last' | 'extend';

/** A gallery video selected as the clip to extend, with the trim range to keep. */
export interface VideoSourceClip {
  video_name: string;
  width: number;
  height: number;
  numFrames: number;
  fps: number;
  /** Inclusive trim bounds forwarded to `extract_video_range`; negative indices count from the end. */
  startFrame: number;
  endFrame: number;
}

export type WanTargetResolution = '480p' | '720p' | '1080p';
export type MiniMaxH3TargetResolution = '768 highres' | '768 lowres';
export type VideoTargetResolution = WanTargetResolution | MiniMaxH3TargetResolution;

/**
 * The preset ratios the video panel offers. No `Free` and no width/height
 * fields: pixel dimensions are always derived — from this ratio plus the
 * target-resolution preset in text-to-video, or from the conditioning media's
 * own ratio once a frame or source video is set.
 */
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
   * The frame the clip should end on: FLF2V interpolation with a first frame,
   * or the destination image when extending a source video. Combines with
   * either `firstFrameImage` or `sourceVideo`.
   */
  lastFrameImage: ImageWithDims | null;
  /** The clip to extend. Mutually exclusive with `firstFrameImage`. */
  sourceVideo: VideoSourceClip | null;
  aspectRatioId: VideoAspectRatioId;
  targetResolution: VideoTargetResolution;
  numFrames: number;
  fps: number;
  steps: number;
  cfgScale: number;
  /** Guidance for the low-noise half of a Wan A14B schedule; null reuses `cfgScale`. */
  cfgScaleLowNoise: number | null;
  /**
   * The family's distillation fast path: the Lightning LoRA pair at 4 steps /
   * CFG 1 for Wan A14B, the Turbo LoRA at 6 steps for MiniMax H3. Toggling it
   * patches steps/CFG and the `loras` list — see `getAcceleratorToggleResult`
   * — so the flag records intent, not hidden state.
   */
  acceleratorEnabled: boolean;
  /**
   * The keys of the LoRA entries the accelerator toggle added. Turning the
   * fast path off (or switching families) removes exactly these — never a
   * user's own LoRA that happens to be named like one — and the enabled flag
   * cannot outlive them.
   */
  acceleratorLoraKeys: string[];
  seed: number;
  shouldRandomizeSeed: boolean;
  loras: GenerateLora[];
  /** Optional VAE override; null uses the VAE bundled with the main model or component source. */
  vae: VaeModelConfig | null;
  /** Wan 2.2's UMT5-XXL text encoder. */
  wanT5EncoderModel: ModelIdentifierConfig | null;
  /** The low-noise expert of a Wan 2.2 A14B mixture-of-experts pair. */
  wanLowNoiseModel: MainModelConfig | null;
  /** Optional Diffusers main model used as a component source for split/quantized Wan models. */
  componentSourceModel: MainModelConfig | null;
  /** Optional single-file MiniMax H3 transformer override (e.g. the pruned int8 repack). */
  h3TransformerModel: MainModelConfig | null;
  /** Optional single-file MiniMax H3 Qwen3-VL text-encoder override. */
  h3TextEncoderModel: ModelIdentifierConfig | null;
}

export interface VideoWidgetValues extends VideoSettings {
  model: MainModelConfig;
}
