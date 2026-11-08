/**
 * Feed backend model grid policy into the model-agnostic engine. {@link resolveModelGrid} distinguishes
 * unavailable policy so persistent writers wait while readers may use defaults; {@link useModelGridSize} updates
 * readers when policy arrives.
 */

import type { CanvasDocumentContractV3 } from '@workbench/canvas-engine/api';

import { getArchitectureCapabilitiesSnapshot, subscribeArchitectureCapabilities } from '@features/generation/runtime';
import { getDimensionGrid } from '@features/generation/settings';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { useCallback } from 'react';

/** Default grid when no model is selected. */
export const DEFAULT_MODEL_GRID = 8;

/**
 * The grid the selected architecture declares, `DEFAULT_MODEL_GRID` when no model is selected, or
 * `null` while the backend has not answered for that architecture.
 */
export const resolveModelGrid = (base: string | null | undefined, variant?: unknown): number | null =>
  base ? getDimensionGrid(base, variant) : DEFAULT_MODEL_GRID;

export const gridSizeForModelBase = (base: string | null | undefined, variant?: unknown): number =>
  resolveModelGrid(base, variant) ?? DEFAULT_MODEL_GRID;

/**
 * Resolve grid inside the capability selector so table updates recompute it even when the persisted base stays
 * unchanged and compiler memoization would retain a fallback.
 */
export const useModelGridSize = (base: string | null | undefined, variant?: string | null): number =>
  useExternalStoreSelector(
    subscribeArchitectureCapabilities,
    getArchitectureCapabilitiesSnapshot,
    useCallback(() => gridSizeForModelBase(base, variant), [base, variant])
  );

type Bbox = CanvasDocumentContractV3['bbox'];

/**
 * The frame resized to `size` around its centre; the position rounds to whole pixels. Generate's size sync and the
 * Frame tool's size fields share it, so both resize around the same anchor.
 */
export const resizeBboxAroundCentre = (bbox: Bbox, size: Pick<Bbox, 'height' | 'width'>): Bbox => ({
  height: size.height,
  width: size.width,
  // `+ 0` folds the -0 that rounding a small negative offset yields.
  x: Math.round(bbox.x + (bbox.width - size.width) / 2) + 0,
  y: Math.round(bbox.y + (bbox.height - size.height) / 2) + 0,
});

/**
 * The frame size a ratio sets: the frame's pixel count at `ratio`, snapped to `grid`, with both sides scaled up together
 * when the smaller would fall below `minSide`. Unlike Generate's size the frame has no maximum, so a large frame keeps
 * its pixel count and only snapping moves the ratio.
 */
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
