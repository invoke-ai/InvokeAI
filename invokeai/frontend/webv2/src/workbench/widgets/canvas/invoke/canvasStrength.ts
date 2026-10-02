/**
 * Share the persisted canvas denoising key/default/clamp between UI and invocation preparation. img2img consumes
 * it; txt2img ignores it.
 */

/** The persisted key inside the canvas widget's `state.values`. */
export const CANVAS_DENOISING_STRENGTH_KEY = 'denoisingStrength';

/** Default strength when unset — a moderate img2img denoise (legacy parity). */
export const DEFAULT_CANVAS_DENOISING_STRENGTH = 0.75;

/** Inclusive slider/value bounds. */
export const MIN_CANVAS_DENOISING_STRENGTH = 0.01;
export const MAX_CANVAS_DENOISING_STRENGTH = 1;

/** Clamps to `[MIN, MAX]`, snapping a non-finite value to the default. */
export const clampCanvasDenoisingStrength = (value: number): number =>
  Number.isFinite(value)
    ? Math.min(MAX_CANVAS_DENOISING_STRENGTH, Math.max(MIN_CANVAS_DENOISING_STRENGTH, value))
    : DEFAULT_CANVAS_DENOISING_STRENGTH;

export const readCanvasDenoisingStrength = (values: Record<string, unknown> | undefined): number => {
  const raw = values?.[CANVAS_DENOISING_STRENGTH_KEY];
  return typeof raw === 'number' && Number.isFinite(raw)
    ? clampCanvasDenoisingStrength(raw)
    : DEFAULT_CANVAS_DENOISING_STRENGTH;
};
