/** Backend grid policy for the engine; `resolveModelGrid` returns null while policy is unknown, so writers wait. */

import type { CanvasDocumentContractV3 } from '@workbench/canvas-engine/api';

import { getArchitectureCapabilitiesSnapshot, subscribeArchitectureCapabilities } from '@features/generation/runtime';
import { getDimensionGrid } from '@features/generation/settings';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { useCallback } from 'react';

/** Default grid when no model is selected. */
export const DEFAULT_MODEL_GRID = 8;

/** The architecture's grid, `DEFAULT_MODEL_GRID` without a model, or `null` until the backend answers. */
export const resolveModelGrid = (base: string | null | undefined, variant?: unknown): number | null =>
  base ? getDimensionGrid(base, variant) : DEFAULT_MODEL_GRID;

export const gridSizeForModelBase = (base: string | null | undefined, variant?: unknown): number =>
  resolveModelGrid(base, variant) ?? DEFAULT_MODEL_GRID;

/** Resolved inside the selector so table updates recompute it even when the persisted base is unchanged. */
export const useModelGridSize = (base: string | null | undefined, variant?: string | null): number =>
  useExternalStoreSelector(
    subscribeArchitectureCapabilities,
    getArchitectureCapabilitiesSnapshot,
    useCallback(() => gridSizeForModelBase(base, variant), [base, variant])
  );

type Bbox = CanvasDocumentContractV3['bbox'];

/** Resized around its centre, rounded to whole pixels; Generate's sync and the Frame tool share this anchor. */
export const resizeBboxAroundCentre = (bbox: Bbox, size: Pick<Bbox, 'height' | 'width'>): Bbox => ({
  height: size.height,
  width: size.width,
  // `+ 0` folds the -0 that rounding a small negative offset yields.
  x: Math.round(bbox.x + (bbox.width - size.width) / 2) + 0,
  y: Math.round(bbox.y + (bbox.height - size.height) / 2) + 0,
});

/** The frame's pixel count at `ratio`, snapped to `grid`, both sides scaled up if one falls below `minSide`. */
export const getBboxSizeForRatio = (
  area: number,
  ratio: number,
  grid: number,
  minSide: number
): Pick<Bbox, 'height' | 'width'> => {
  const exactWidth = Math.sqrt(area * ratio);
  const exactHeight = exactWidth / ratio;
  const scale = Math.max(1, minSide / Math.min(exactWidth, exactHeight));
  const snap = (value: number) => Math.max(minSide, Math.round((value * scale) / grid) * grid);

  return { height: snap(exactHeight), width: snap(exactWidth) };
};
