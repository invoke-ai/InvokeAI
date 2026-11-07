/**
 * Default: plain wheel zooms around the cursor, resting on snap points; ctrl+wheel sizes brush/eraser and is
 * otherwise swallowed. Trackpad navigation (a user setting, since browsers cannot tell a trackpad from a mouse):
 * plain wheel pans on both axes for every tool, and ctrl+wheel, which is how browsers deliver a pinch, zooms
 * continuously around the cursor instead of sizing the brush. While the setting is off, a run of small ctrl+wheel steps
 * with the physical Control key known to be up can only be a pinch: the step completing the run is reported before
 * routing, so the frontend may turn the setting on for that event, which then zooms by the whole run, its earlier
 * steps included. Those earlier candidates are swallowed rather than sizing the brush, so a pinch too short to complete
 * a run does nothing while the setting is undecided. Once nobody listens (the setting was decided), candidates are plain
 * ctrl+wheel again. Injected dependencies keep routing testable without DOM setup.
 */

import type { ControlKeyState } from '@workbench/canvas-engine/input/pointerPipeline';
import type { InvalidatePayload } from '@workbench/canvas-engine/render/scheduler';
import type { Tool } from '@workbench/canvas-engine/tools/tool';
import type { Vec2 } from '@workbench/canvas-engine/types';
import type { Viewport } from '@workbench/canvas-engine/viewport';

/** Dependencies for {@link createWheelHandler}. */
export interface WheelHandlerDeps {
  viewport: Viewport;
  invalidate(payload: InvalidatePayload): void;
  getInputElement(): HTMLElement | null;
  getActiveTool(): Tool | undefined;
  /** Steps the active brush/eraser diameter by one notch (`+1` grow, `-1` shrink). */
  stepActiveBrushSize(direction: 1 | -1): void;
  /**
   * Whether ctrl+wheel brush sizing is inverted. `false` (default): wheel-up
   * grows. `true`: wheel-up shrinks. A user preference read fresh per event.
   */
  getInvertBrushSizeScroll(): boolean;
  /** Whether wheel pans and ctrl+wheel (pinch) zooms. A user preference read fresh per event. */
  getTrackpadNavigation(): boolean;
  /** The physical Control key; a pinch's ctrl+wheel arrives with it `up`. */
  getControlKeyState(): ControlKeyState;
  /**
   * Whether a detected pinch could still turn trackpad navigation on (someone listens for it). Only then are pinch
   * candidates detected and swallowed; otherwise ctrl+wheel keeps sizing the brush.
   */
  isPinchDetectionArmed(): boolean;
  /**
   * A pinch was detected while trackpad navigation is off. Runs before routing, so a listener that turns the setting
   * on has this same event handled as a pinch zoom.
   */
  reportTrackpadPinch(): void;
}

/** `WheelEvent.deltaMode` values, literal so the handler needs no DOM globals. */
const DOM_DELTA_LINE = 1;
const DOM_DELTA_PAGE = 2;

/** CSS pixels per wheel line (`deltaMode` 1, e.g. a Firefox mouse wheel). */
const WHEEL_LINE_PX = 16;

/**
 * Pinch zoom gain per wheel pixel. Browser pinch steps are a few pixels each, so 0.01 tracks the fingers closely
 * (matches the image map's pinch gain).
 */
const PINCH_ZOOM_GAIN = 0.01;

/**
 * Largest zoom factor one ctrl+wheel event may apply. Leaves pinch steps (under ~22px) untouched while a real
 * ctrl+mouse-wheel notch (~100px, which would be e^1) moves by a bounded step.
 */
const MAX_PINCH_ZOOM_STEP = 1.25;

/**
 * Pinch steps are a few pixels; a mouse notch is about 100-120 pixels, or 3 lines (48px). Only steps under this, in
 * pixels, count toward detecting a pinch, whatever `deltaMode` the browser reports them in.
 */
const PINCH_DELTA_LIMIT_PX = 40;

/** Pinch-shaped events needed within {@link PINCH_DETECTION_WINDOW_MS} before a pinch is reported. */
const PINCH_DETECTION_EVENTS = 3;
const PINCH_DETECTION_WINDOW_MS = 300;

/** Wheel deltas in CSS pixels: lines are a fixed height, pages the viewport size on that axis. */
const wheelDeltaPixels = (event: WheelEvent, viewport: Viewport): Vec2 => {
  if (event.deltaMode === DOM_DELTA_LINE) {
    return { x: event.deltaX * WHEEL_LINE_PX, y: event.deltaY * WHEEL_LINE_PX };
  }
  if (event.deltaMode === DOM_DELTA_PAGE) {
    const size = viewport.getViewportSize();
    return { x: event.deltaX * size.width, y: event.deltaY * size.height };
  }
  return { x: event.deltaX, y: event.deltaY };
};

/** The zoom factor of one ctrl+wheel step of `deltaY` pixels: wheel-up (negative) zooms in, bounded per step. */
const pinchZoomFactor = (deltaY: number): number =>
  Math.min(MAX_PINCH_ZOOM_STEP, Math.max(1 / MAX_PINCH_ZOOM_STEP, Math.exp(-deltaY * PINCH_ZOOM_GAIN)));

/** Creates the canvas wheel handler. See module docs for routing. */
export const createWheelHandler = (deps: WheelHandlerDeps): ((event: WheelEvent) => void) => {
  // Pinch-shaped events in the current run, when it began, and the zoom its swallowed steps would have applied (each
  // bounded like a live step, so the run zooms exactly as far as it would have with the setting on). Any other wheel
  // event ends the run.
  let pinchRun = 0;
  let pinchRunStart = 0;
  let pinchRunZoom = 1;

  const endPinchRun = (): void => {
    pinchRun = 0;
    pinchRunZoom = 1;
  };

  /**
   * A pinch candidate: a small ctrl+wheel step with the Control key known to be up. An unknown key state (a press may
   * have gone unseen) or a mouse notch, in pixels or lines, never is one.
   */
  const isPinchCandidate = (event: WheelEvent, delta: Vec2): boolean =>
    event.ctrlKey && deps.getControlKeyState() === 'up' && Math.abs(delta.y) < PINCH_DELTA_LIMIT_PX;

  return (event: WheelEvent): void => {
    event.preventDefault();

    const rect = deps.getInputElement()?.getBoundingClientRect();
    const screenAnchor: Vec2 = { x: event.clientX - (rect?.left ?? 0), y: event.clientY - (rect?.top ?? 0) };
    const delta = wheelDeltaPixels(event, deps.viewport);
    // The zoom of a just-detected run's earlier steps, which this step applies along with its own.
    let carriedZoom = 1;

    if (!deps.getTrackpadNavigation() && deps.isPinchDetectionArmed()) {
      if (!isPinchCandidate(event, delta)) {
        endPinchRun();
      } else {
        if (pinchRun === 0 || event.timeStamp - pinchRunStart > PINCH_DETECTION_WINDOW_MS) {
          endPinchRun();
          pinchRunStart = event.timeStamp;
        }
        pinchRun += 1;
        if (pinchRun < PINCH_DETECTION_EVENTS) {
          // Swallowed: almost surely a pinch, not a brush resize. Its zoom waits for the run to complete.
          pinchRunZoom *= pinchZoomFactor(delta.y);
          return;
        }
        carriedZoom = pinchRunZoom;
        endPinchRun();
        deps.reportTrackpadPinch();
        // A completed run that did not turn the setting on is swallowed too.
        if (!deps.getTrackpadNavigation()) {
          return;
        }
      }
    }

    if (deps.getTrackpadNavigation()) {
      if (event.ctrlKey) {
        deps.viewport.zoomAtPoint(deps.viewport.getZoom() * carriedZoom * pinchZoomFactor(delta.y), screenAnchor);
      } else {
        // Content follows the fingers, as native scrolling moves a page.
        deps.viewport.panBy({ x: -delta.x, y: -delta.y });
      }
      deps.invalidate({ view: true });
      return;
    }

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

    deps.viewport.wheelZoom(event.deltaY, screenAnchor);
    deps.invalidate({ view: true });
  };
};
