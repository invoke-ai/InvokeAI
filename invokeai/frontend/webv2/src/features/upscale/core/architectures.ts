import type { MainModelConfig } from '@features/generation/contracts';

/**
 * Which main-model architectures the Upscale widget can drive, and what each one needs.
 *
 * This is the single source of truth: the graph compiler keys its builders off the same record
 * (`UPSCALE_PATH_BUILDERS`, enforced by `satisfies`), validation reads the requirements from here,
 * and the widget shows a control only where the architecture has something for it to steer. Adding
 * an architecture should mean adding one entry and one builder, not hunting for a base string.
 */
export interface UpscaleArchitecture {
  /**
   * A tile ControlNet anchors the second pass to the upscaled frame. Where one exists the widget
   * requires it and the Structure slider drives its weights; where it does not, Structure has
   * nothing to scale and is hidden rather than left inert.
   */
  readonly usesTileControlNet: boolean;
  /**
   * Whether the pipeline's parts must be picked separately. FLUX transformers ship as bare
   * checkpoints far more often than not, and `flux_model_loader` refuses those without an explicit
   * T5 encoder, CLIP Embed *and* VAE -- only a complete SDNQ pipeline install can supply its own.
   */
  readonly needsExplicitComponents: boolean;
  /**
   * The size granularity the architecture's denoise node demands. Spandrel's autoscale rounds to a
   * multiple of 8, which `flux_denoise` (multiple_of=16) rejects outright — after the upscale has
   * already run. Where this is coarser than 8 the compiler resizes the frame down to it and passes
   * the size as a literal, so the graph cannot be built with a size the backend will refuse.
   */
  readonly denoiseGrid: number;
  /** Guidance-distilled architectures ignore a negative prompt; showing one invites a silent no-op. */
  readonly usesNegativePrompt: boolean;
  /** Shown in validation messages, so a user reads "FLUX.1" rather than "flux". */
  readonly label: string;
}

export const UPSCALE_ARCHITECTURES = {
  'sd-1': {
    denoiseGrid: 8,
    label: 'SD1.5',
    needsExplicitComponents: false,
    usesNegativePrompt: true,
    usesTileControlNet: true,
  },
  sdxl: {
    denoiseGrid: 8,
    label: 'SDXL',
    needsExplicitComponents: false,
    usesNegativePrompt: true,
    usesTileControlNet: true,
  },
  flux: {
    denoiseGrid: 16,
    label: 'FLUX.1',
    needsExplicitComponents: true,
    usesNegativePrompt: false,
    usesTileControlNet: false,
  },
} as const satisfies Record<string, UpscaleArchitecture>;

export type UpscaleBase = keyof typeof UPSCALE_ARCHITECTURES;

export const isUpscaleBase = (base: unknown): base is UpscaleBase =>
  typeof base === 'string' && base in UPSCALE_ARCHITECTURES;

export const upscaleArchitectureFor = (model: MainModelConfig | null): UpscaleArchitecture | null =>
  model && isUpscaleBase(model.base) ? UPSCALE_ARCHITECTURES[model.base] : null;

/** "SD1.5, SDXL or FLUX.1" — built from the table so the copy cannot go stale. */
export const supportedUpscaleArchitectureLabels = (): string => {
  const labels = Object.values(UPSCALE_ARCHITECTURES).map((architecture) => architecture.label);
  const last = labels[labels.length - 1];

  return labels.length > 1 ? `${labels.slice(0, -1).join(', ')} or ${last}` : (last ?? '');
};

/**
 * A self-contained SDNQ pipeline carries its own transformer, CLIP, T5 and VAE, so asking the user
 * to pick them would be busywork. Every other FLUX format must have them chosen explicitly.
 *
 * The backend is stricter still -- it also checks the pipeline actually contains those submodels
 * (`is_self_contained_sdnq_flux1_pipeline`), which the model list does not expose. An SDNQ install
 * missing a submodel therefore passes here and is refused by the loader with a precise message,
 * which is the right way round.
 */
export const needsExplicitComponents = (model: MainModelConfig | null): boolean =>
  (upscaleArchitectureFor(model)?.needsExplicitComponents ?? false) && model?.format !== 'sdnq_quantized';

/** What Spandrel's autoscale produces: the scaled dimension floored to a multiple of 8. */
export const spandrelAutoscaleDimension = (dimension: number, scale: number): number =>
  Math.floor(Math.trunc(dimension * scale) / 8) * 8;

/** The same dimension floored to what the architecture's denoise node will accept. */
export const upscaleDenoiseDimension = (dimension: number, scale: number, grid: number): number =>
  Math.max(grid, Math.floor(spandrelAutoscaleDimension(dimension, scale) / grid) * grid);
