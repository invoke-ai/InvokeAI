import type { FloatingWidgetState } from '@workbench/layoutContracts';
import type { WidgetInstanceId } from '@workbench/widgetContracts';

/**
 * Pure geometry and stacking helpers for floating widget windows. The reducer
 * owns the state; components clamp against the live viewport (the reducer
 * never reads window dimensions).
 */

export const FLOATING_DEFAULT_WIDTH_PX = 520;
export const FLOATING_DEFAULT_HEIGHT_PX = 440;
export const FLOATING_MIN_WIDTH_PX = 280;
export const FLOATING_MIN_HEIGHT_PX = 200;

/** Minimum sliver of a window that must stay reachable inside the viewport. */
const VIEWPORT_MARGIN_PX = 48;
const CASCADE_ORIGIN_PX = 96;
const CASCADE_STEP_PX = 32;
const CASCADE_WRAP = 8;

export interface FloatingGeometry {
  x: number;
  y: number;
  widthPx: number;
  heightPx: number;
}

export const nextStackOrder = (floatingWidgets: Record<WidgetInstanceId, FloatingWidgetState> | undefined): number =>
  Object.values(floatingWidgets ?? {}).reduce((max, state) => Math.max(max, state.stackOrder), 0) + 1;

/** Default placement for the Nth window: a classic cascading offset. */
export const cascadeDefaultGeometry = (existingCount: number): FloatingGeometry => {
  const step = (existingCount % CASCADE_WRAP) * CASCADE_STEP_PX;

  return {
    heightPx: FLOATING_DEFAULT_HEIGHT_PX,
    widthPx: FLOATING_DEFAULT_WIDTH_PX,
    x: CASCADE_ORIGIN_PX + step,
    y: CASCADE_ORIGIN_PX + step,
  };
};

export const clampSizeToMinimum = (geometry: FloatingGeometry): FloatingGeometry => ({
  ...geometry,
  heightPx: Math.max(FLOATING_MIN_HEIGHT_PX, geometry.heightPx),
  widthPx: Math.max(FLOATING_MIN_WIDTH_PX, geometry.widthPx),
});

/**
 * Keep at least a grabbable corner of the window inside the viewport so a
 * drag (or a shrunk browser window) can never strand it off-screen.
 */
export const clampWindowToViewport = (
  geometry: FloatingGeometry,
  viewport: { width: number; height: number }
): FloatingGeometry => ({
  ...geometry,
  x: Math.min(Math.max(geometry.x, VIEWPORT_MARGIN_PX - geometry.widthPx), viewport.width - VIEWPORT_MARGIN_PX),
  y: Math.min(Math.max(geometry.y, 0), Math.max(0, viewport.height - VIEWPORT_MARGIN_PX)),
});
