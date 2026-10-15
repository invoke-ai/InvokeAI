/** Derive HUD zoom choices from engine ZOOM_PRESETS so menu and wheel snapping agree. */

import { ZOOM_PRESETS } from '@workbench/canvas-engine/api';

/** Formats a zoom factor as a rounded whole-percent label (e.g. `1` → `"100%"`). */
export const formatZoomPercent = (zoom: number): string => `${Math.round(zoom * 100)}%`;

/** A single zoom-menu entry: the factor to apply and its display label. */
export interface ZoomMenuOption {
  value: number;
  label: string;
}

/** The selectable zoom levels for the HUD menu, largest first (matches snap points top-down). */
export const zoomMenuOptions = (): ZoomMenuOption[] =>
  [...ZOOM_PRESETS].sort((a, b) => b - a).map((value) => ({ label: formatZoomPercent(value), value }));
