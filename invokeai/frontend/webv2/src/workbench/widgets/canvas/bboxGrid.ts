/**
 * Maps a model base to the bbox snapping grid size (document px), mirroring the
 * legacy `getGridSize` rule: generation dimensions must land on a base-specific
 * multiple. React reads the active generate model's base and feeds the result
 * into `engine.viewport.setBboxGrid`; the engine itself stays model-agnostic.
 */

/** Default grid when no model is selected (or an unknown base). */
export const DEFAULT_MODEL_GRID = 8;

/**
 * The bbox grid size for a model base:
 * - `cogview4` → 32
 * - `flux` / `flux2` / `sd-3` / `qwen-image` / `z-image` → 16
 * - everything else (sd-1/sd-2/sdxl/anima/unknown) → 8
 */
export const gridSizeForModelBase = (base: string | null | undefined): number => {
  switch (base) {
    case 'cogview4':
      return 32;
    case 'flux':
    case 'flux2':
    case 'sd-3':
    case 'qwen-image':
    case 'z-image':
      return 16;
    default:
      return DEFAULT_MODEL_GRID;
  }
};
