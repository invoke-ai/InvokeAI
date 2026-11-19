/**
 * Plain wheel routes to the tool or cursor-centered zoom. Ctrl+wheel sizes brush/eraser and is otherwise swallowed.
 * Injected dependencies keep routing testable without DOM setup.
 */

import type { InvalidatePayload } from '@workbench/canvas-engine/render/scheduler';
import type { Tool, ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { PointerModifiers, Vec2 } from '@workbench/canvas-engine/types';
import type { Viewport } from '@workbench/canvas-engine/viewport';

export interface WheelHandlerDeps {
  viewport: Viewport;
  invalidate(payload: InvalidatePayload): void;
  getInputElement(): HTMLElement | null;
  getActiveTool(): Tool | undefined;
  getToolContext(): ToolContext;
  /** Steps the active brush/eraser diameter by one notch (`+1` grow, `-1` shrink). */
  stepActiveBrushSize(direction: 1 | -1): void;
  /** Inverts ctrl+wheel brush sizing (default false: wheel-up grows); a user preference read fresh per event. */
  getInvertBrushSizeScroll(): boolean;
}

const modifiersOf = (event: WheelEvent): PointerModifiers => ({
  alt: event.altKey,
  ctrl: event.ctrlKey,
  meta: event.metaKey,
  shift: event.shiftKey,
});

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
