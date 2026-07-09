/**
 * Canvas denoising-strength: the single knob a canvas img2img invoke exposes.
 *
 * The value is persisted in the canvas widget's own state values
 * (`widgetInstances['canvas'].state.values.denoisingStrength`) so it survives
 * reloads and rides along in queue snapshots, and is read back — with the
 * default applied — by the invoke orchestrator. Only consulted for img2img
 * (txt2img ignores it), matching the graph compiler.
 *
 * Pure data + a reader; no React, no engine. Shared by the tool-options UI and
 * `prepareCanvasInvocation` so the storage key and clamp stay in one place.
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

/**
 * Reads the persisted canvas denoising strength from a widget's `state.values`,
 * applying the default when unset and clamping to the valid range.
 */
export const readCanvasDenoisingStrength = (values: Record<string, unknown> | undefined): number => {
  const raw = values?.[CANVAS_DENOISING_STRENGTH_KEY];
  return typeof raw === 'number' && Number.isFinite(raw)
    ? clampCanvasDenoisingStrength(raw)
    : DEFAULT_CANVAS_DENOISING_STRENGTH;
};
