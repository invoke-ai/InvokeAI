/**
 * Maps a model base to the bbox snapping grid size (document px), mirroring the
 * generation core's model policy: generation dimensions must land on a
 * base-specific multiple. React reads the active generate model's base and
 * feeds the result into `engine.viewport.setBboxGrid`; the engine itself stays
 * model-agnostic.
 */

import { getGenerationDimensions } from '@features/generation/settings';

/** Default grid when no model is selected (or an unknown base). */
export const DEFAULT_MODEL_GRID = getGenerationDimensions(undefined).grid;

/** The bbox grid for a model base, or the shared 8px fallback for unknown/empty bases. */
export const gridSizeForModelBase = (base: string | null | undefined): number =>
  base ? getGenerationDimensions({ base, type: 'main' }).grid : DEFAULT_MODEL_GRID;
