/**
 * Wheel routing for the canvas.
 *
 * - Plain wheel → the active tool's `onWheel` if it defines one, else viewport
 *   zoom about the cursor (the default navigation behavior).
 * - Ctrl+wheel → when the active tool is brush/eraser, step its size; otherwise
 *   it is swallowed (reserved for the browser pinch-zoom gesture).
 *
 * The handler is a pure function of injected deps (viewport, the active tool, and
 * a `stepActiveBrushSize` callback the engine wires to the tool-options stores),
 * so it is driven directly in node tests. DOM is touched only through the passed
 * `WheelEvent`. Zero React, zero import-time side effects.
 */

import type { InvalidatePayload } from '@workbench/canvas-engine/render/scheduler';
import type { Tool, ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { PointerModifiers, Vec2 } from '@workbench/canvas-engine/types';
import type { Viewport } from '@workbench/canvas-engine/viewport';

/** Dependencies for {@link createWheelHandler}. */
export interface WheelHandlerDeps {
  viewport: Viewport;
  invalidate(payload: InvalidatePayload): void;
  getInputElement(): HTMLElement | null;
  getActiveTool(): Tool | undefined;
  getToolContext(): ToolContext;
  /** Steps the active brush/eraser diameter by one notch (`+1` grow, `-1` shrink). */
  stepActiveBrushSize(direction: 1 | -1): void;
  /**
   * Whether ctrl+wheel brush sizing is inverted. `false` (default): wheel-up
   * grows. `true`: wheel-up shrinks. A user preference read fresh per event.
   */
  getInvertBrushSizeScroll(): boolean;
}

const modifiersOf = (event: WheelEvent): PointerModifiers => ({
  alt: event.altKey,
  ctrl: event.ctrlKey,
  meta: event.metaKey,
  shift: event.shiftKey,
});

/** Creates the canvas wheel handler. See module docs for routing. */
export const createWheelHandler = (deps: WheelHandlerDeps): ((event: WheelEvent) => void) => {
  return (event: WheelEvent): void => {
    event.preventDefault();

    const rect = deps.getInputElement()?.getBoundingClientRect();
    const screenAnchor: Vec2 = { x: event.clientX - (rect?.left ?? 0), y: event.clientY - (rect?.top ?? 0) };

    const tool = deps.getActiveTool();

    // Ctrl+wheel: brush/eraser size step, else reserved (pinch-zoom) — swallowed.
    if (event.ctrlKey) {
      if (tool && (tool.id === 'brush' || tool.id === 'eraser')) {
        // Default: wheel-up (deltaY < 0) grows. The inversion preference flips it.
        const grow = event.deltaY < 0;
        const stepUp = grow !== deps.getInvertBrushSizeScroll();
        deps.stepActiveBrushSize(stepUp ? 1 : -1);
      }
      return;
    }

    if (tool?.onWheel) {
      tool.onWheel(deps.getToolContext(), event.deltaY, screenAnchor, modifiersOf(event));
      return;
    }
    deps.viewport.wheelZoom(event.deltaY, screenAnchor);
    deps.invalidate({ view: true });
  };
};
