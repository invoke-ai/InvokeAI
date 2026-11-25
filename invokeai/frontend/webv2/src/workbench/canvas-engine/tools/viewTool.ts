/** Per-engine view tool: pans and zooms the viewport without document dispatch; also the temporary Space tool. */

import type { PointerInput, Vec2 } from '@workbench/canvas-engine/types';

import type { Tool, ToolContext } from './tool';

/** Bit for the primary (usually left) mouse button in `PointerEvent.buttons`. */
const PRIMARY_BUTTON = 1;

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
  };
};
