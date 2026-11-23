/**
 * Mouse wheel: plain wheel zooms around the cursor, resting on snap points; ctrl+wheel sizes brush/eraser and is
 * otherwise swallowed. Trackpad: plain wheel pans on both axes for every tool; ctrl+wheel, how browsers deliver a
 * pinch, zooms continuously around the cursor. `scrollInput` picks one, or `auto` judges each event (`wheelDevice.ts`);
 * a two-axis event or a pinch proves a trackpad.
 */

import type { ScalarStore, ScrollInput } from '@workbench/canvas-engine/engineStores';
import type { ControlKeyState } from '@workbench/canvas-engine/input/pointerPipeline';
import type { WheelSample } from '@workbench/canvas-engine/input/wheelDevice';
import type { InvalidatePayload } from '@workbench/canvas-engine/render/scheduler';
import type { Tool } from '@workbench/canvas-engine/tools/tool';
import type { Vec2 } from '@workbench/canvas-engine/types';
import type { Viewport } from '@workbench/canvas-engine/viewport';

import { createWheelDeviceClassifier } from '@workbench/canvas-engine/input/wheelDevice';

export interface WheelHandlerDeps {
  viewport: Viewport;
  invalidate(payload: InvalidatePayload): void;
  getInputElement(): HTMLElement | null;
  getActiveTool(): Tool | undefined;
  /** Steps the active brush/eraser diameter by one notch (`+1` grow, `-1` shrink). */
  stepActiveBrushSize(direction: 1 | -1): void;
  /** Inverts ctrl+wheel brush sizing (default false: wheel-up grows); a user preference read fresh per event. */
  getInvertBrushSizeScroll(): boolean;
  getScrollInput(): ScrollInput;
  /** The physical Control key; a pinch's ctrl+wheel arrives with it `up` or, right after window focus, `unknown`. */
  getControlKeyState(): ControlKeyState;
  /** Set on trackpad proof; gates `auto` routing of smooth scrolling. */
  trackpadDetected: Pick<ScalarStore<boolean>, 'get' | 'set'>;
  /** A pointer gesture is under way: iPadOS also sends pinch gesture events for touches, mid-stroke. */
  isGestureActive(): boolean;
}

export interface WheelInputHandlers {
  onWheel(event: WheelEvent): void;
  /** Safari sends a trackpad pinch as these instead of ctrl+wheel; unhandled, it magnifies the page. */
  onPinchGestureStart(event: Event): void;
  onPinchGestureChange(event: Event): void;
}

/** Safari's non-standard `GestureEvent`, absent from lib.dom; `scale` is relative to the gesture's start. */
interface SafariGestureEvent extends Event {
  readonly scale: number;
  readonly clientX: number;
  readonly clientY: number;
}

/** Browser pinch steps are a few pixels each; matches the image map's pinch gain. */
const PINCH_ZOOM_GAIN = 0.01;
/** Bounds a ctrl+mouse-wheel notch (~100px, e^1 unbounded) while leaving pinch steps untouched. */
const MAX_PINCH_ZOOM_STEP = 1.25;

const pinchZoomFactor = (deltaY: number): number =>
  Math.min(MAX_PINCH_ZOOM_STEP, Math.max(1 / MAX_PINCH_ZOOM_STEP, Math.exp(-deltaY * PINCH_ZOOM_GAIN)));

export const createWheelInput = (deps: WheelHandlerDeps): WheelInputHandlers => {
  const classify = createWheelDeviceClassifier();
  // Null while the pinch belongs to a touch that is already drawing.
  let gestureStartZoom: number | null = null;

  const screenAnchor = (event: { clientX: number; clientY: number }): Vec2 => {
    const rect = deps.getInputElement()?.getBoundingClientRect();
    return { x: event.clientX - (rect?.left ?? 0), y: event.clientY - (rect?.top ?? 0) };
  };

  const routesAsTrackpad = (sample: WheelSample): boolean => {
    const scrollInput = deps.getScrollInput();
    if (scrollInput !== 'auto') {
      return scrollInput === 'trackpad';
    }
    // A sideways-only wheel has nothing to zoom by.
    if (sample.deltaY === 0 && sample.deltaX !== 0) {
      return true;
    }
    if (sample.kind === 'notched') {
      return false;
    }
    if (sample.twoAxis || sample.pinch) {
      deps.trackpadDetected.set(true);
    }
    // Until a trackpad shows itself, smooth scrolling may be a high-resolution mouse wheel, which zooms.
    return deps.trackpadDetected.get();
  };

  return {
    onPinchGestureChange: (event) => {
      event.preventDefault();
      if (gestureStartZoom === null) {
        return;
      }
      const gesture = event as SafariGestureEvent;
      deps.trackpadDetected.set(true);
      deps.viewport.zoomAtPoint(gestureStartZoom * gesture.scale, screenAnchor(gesture));
      deps.invalidate({ view: true });
    },
    onPinchGestureStart: (event) => {
      event.preventDefault();
      gestureStartZoom = deps.isGestureActive() ? null : deps.viewport.getZoom();
    },
    onWheel: (event) => {
      event.preventDefault();
      const sample = classify(event, deps.viewport.getViewportSize(), deps.getControlKeyState());
      const anchor = screenAnchor(event);

      if (routesAsTrackpad(sample)) {
        if (event.ctrlKey) {
          deps.viewport.zoomAtPoint(deps.viewport.getZoom() * pinchZoomFactor(sample.deltaY), anchor);
        } else {
          // Content follows the fingers, as native scrolling moves a page.
          deps.viewport.panBy({ x: -sample.deltaX, y: -sample.deltaY });
        }
        deps.invalidate({ view: true });
        return;
      }

      if (event.ctrlKey) {
        const tool = deps.getActiveTool();
        if ((tool?.id === 'brush' || tool?.id === 'eraser') && sample.deltaY !== 0) {
          const grow = sample.deltaY < 0;
          deps.stepActiveBrushSize(grow !== deps.getInvertBrushSizeScroll() ? 1 : -1);
        }
        return;
      }

      deps.viewport.wheelZoom(sample.deltaY, anchor);
      deps.invalidate({ view: true });
    },
  };
};
