/**
 * Pure zoom-menu helpers for the canvas HUD.
 *
 * The option set is derived from the engine's single source of truth for zoom
 * snap points (`ZOOM_SNAP_CANDIDATES`), so the HUD and the viewport's wheel
 * snapping always agree. No React, no side effects — unit-testable in node.
 */

import { ZOOM_SNAP_CANDIDATES } from '@workbench/canvas-engine/math/snapping';

/** Formats a zoom factor as a rounded whole-percent label (e.g. `1` → `"100%"`). */
export const formatZoomPercent = (zoom: number): string => `${Math.round(zoom * 100)}%`;

/** A single zoom-menu entry: the factor to apply and its display label. */
export interface ZoomMenuOption {
  value: number;
  label: string;
}

/** The selectable zoom levels for the HUD menu, largest first (matches snap points top-down). */
export const zoomMenuOptions = (): ZoomMenuOption[] =>
  [...ZOOM_SNAP_CANDIDATES].sort((a, b) => b - a).map((value) => ({ label: formatZoomPercent(value), value }));
