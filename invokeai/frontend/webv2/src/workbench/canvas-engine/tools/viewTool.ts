/**
 * The view tool: pan the canvas by dragging, zoom with the wheel. It is the
 * default tool and the one the engine temporarily swaps in while space is held.
 * All navigation flows through the viewport, so it never mutates the document
 * and never dispatches.
 *
 * Each engine builds its own instance via {@link createViewTool} so the private
 * drag state is per-engine (never module-global). Zero React, zero import-time
 * side effects.
 */

import type { PointerInput, PointerModifiers, Vec2 } from '@workbench/canvas-engine/types';

import type { Tool, ToolContext } from './tool';

/** Bit for the primary (usually left) mouse button in `PointerEvent.buttons`. */
const PRIMARY_BUTTON = 1;

/** Creates a fresh view tool with its own drag state. */
export const createViewTool = (): Tool => {
  let panning = false;
  let lastScreen: Vec2 | null = null;

  const begin = (input: PointerInput): void => {
    // Only the primary button pans; secondary/middle are handled by the engine.
    if ((input.buttons & PRIMARY_BUTTON) === 0) {
      return;
    }
    panning = true;
    lastScreen = input.screenPoint;
  };

  const move = (ctx: ToolContext, input: PointerInput): void => {
    if (!panning || !lastScreen) {
      return;
    }
    ctx.viewport.panBy({ x: input.screenPoint.x - lastScreen.x, y: input.screenPoint.y - lastScreen.y });
    lastScreen = input.screenPoint;
    ctx.invalidate({ view: true });
  };

  const end = (): void => {
    panning = false;
    lastScreen = null;
  };

  return {
    cursor: () => (panning ? 'grabbing' : 'grab'),
    id: 'view',
    onDeactivate: end,
    onPointerDown: (_ctx, input) => begin(input),
    onPointerMove: move,
    onPointerUp: end,
    onWheel: (ctx: ToolContext, deltaY: number, screenAnchor: Vec2, _modifiers: PointerModifiers) => {
      ctx.viewport.wheelZoom(deltaY, screenAnchor);
      ctx.invalidate({ view: true });
    },
  };
};
