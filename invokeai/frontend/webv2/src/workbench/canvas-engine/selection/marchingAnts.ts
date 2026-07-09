/**
 * Marching ants: the animated dashed outline of a pixel selection.
 *
 * Two concerns live here:
 *
 * 1. {@link drawMarchingAnts} strokes the selection's committed path list onto
 *    the overlay surface as a two-tone (black/white) dashed outline. It draws in
 *    document space (paths are document-space `Path2D`s) via the shared `view`
 *    transform, scaling line width and dash by `1/scale` so the ants stay a
 *    constant pixel size at any zoom — the same screen-constant convention the
 *    rest of the overlay uses. The animated `lineDashOffset` (`phase`) makes the
 *    white dashes crawl.
 *
 * 2. {@link createAntsAnimator} drives the `phase` advance. It self-reschedules
 *    through an injected `requestFrame`/`cancelFrame` pair (the engine passes the
 *    global rAF, so it rides the same frame clock as the scheduler) and, throttled
 *    by an injected `now()` to ~a few fps (the legacy look, not 60fps), calls
 *    `onStep` — the engine advances the phase and requests an OVERLAY-ONLY
 *    invalidation there, so an ants tick never recomposites layers (Task-22 gate).
 *    It runs only while started (a selection exists AND the engine is attached)
 *    and cancels its pending frame on `stop`, so there are no timer leaks on
 *    detach/deselect/dispose.
 *
 * Zero React, zero import-time side effects.
 */

import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Mat2d } from '@workbench/canvas-engine/types';

import { getScale } from '@workbench/canvas-engine/math/mat2d';

/** Dash length in screen pixels; the offset advances by {@link ANTS_STEP_PX} per tick. */
export const ANTS_DASH_PX = 4;
/** Screen-pixel advance of the dash offset per animation step. */
export const ANTS_STEP_PX = 1;
/** Default step interval (ms) — ~5 fps, the legacy marching-ants cadence. */
export const ANTS_INTERVAL_MS = 200;

const ANTS_LINE_WIDTH_PX = 1;
const ANTS_DARK = '#000000';
const ANTS_LIGHT = '#ffffff';

/** What {@link drawMarchingAnts} needs: the committed outlines and the animation phase. */
export interface MarchingAntsRender {
  paths: readonly Path2D[];
  /** Animated dash offset in screen pixels. */
  phase: number;
}

/**
 * Strokes the selection outline(s) as animated two-tone marching ants, in
 * document space projected through `view`. A no-op with no paths.
 */
export const drawMarchingAnts = (ctx: RasterSurface['ctx'], view: Mat2d, render: MarchingAntsRender): void => {
  if (render.paths.length === 0) {
    return;
  }
  const scale = getScale(view) || 1;
  const dash = ANTS_DASH_PX / scale;
  const lineWidth = ANTS_LINE_WIDTH_PX / scale;
  const offset = render.phase / scale;

  ctx.save();
  ctx.setTransform(view.a, view.b, view.c, view.d, view.e, view.f);
  ctx.lineWidth = lineWidth;

  // Dark base run: a continuous dashed outline.
  ctx.setLineDash([dash, dash]);
  ctx.lineDashOffset = -offset;
  ctx.strokeStyle = ANTS_DARK;
  for (const path of render.paths) {
    ctx.stroke(path);
  }

  // Light run, offset by one dash so it fills the gaps and appears to crawl.
  ctx.lineDashOffset = -offset + dash;
  ctx.strokeStyle = ANTS_LIGHT;
  for (const path of render.paths) {
    ctx.stroke(path);
  }

  ctx.restore();
};

/** The animator handle the engine starts/stops as selection presence + attachment change. */
export interface AntsAnimator {
  /** Begins the animation loop (idempotent). */
  start(): void;
  /** Stops the loop and cancels any pending frame (idempotent). */
  stop(): void;
  /** True while the loop is running. */
  readonly isRunning: boolean;
}

/** Dependencies for {@link createAntsAnimator}; all injectable for deterministic tests. */
export interface AntsAnimatorDeps {
  requestFrame(callback: () => void): number;
  cancelFrame(handle: number): void;
  /** Monotonic clock (ms) used to throttle steps to {@link AntsAnimatorDeps.intervalMs}. */
  now(): number;
  /** Called on each throttled step (the engine advances the phase + invalidates the overlay). */
  onStep(): void;
  /** Step interval in ms; defaults to {@link ANTS_INTERVAL_MS}. */
  intervalMs?: number;
}

/**
 * Creates a self-rescheduling animation loop that fires `onStep` at most every
 * `intervalMs`. It reschedules on every frame while running (to keep polling the
 * clock) but only steps on the throttle boundary.
 */
export const createAntsAnimator = (deps: AntsAnimatorDeps): AntsAnimator => {
  const intervalMs = deps.intervalMs ?? ANTS_INTERVAL_MS;
  let running = false;
  let handle: number | null = null;
  let lastStep = 0;

  const frame = (): void => {
    handle = null;
    if (!running) {
      return;
    }
    const t = deps.now();
    if (t - lastStep >= intervalMs) {
      lastStep = t;
      deps.onStep();
    }
    // Reschedule while running (even between steps) so the clock keeps advancing.
    if (running) {
      handle = deps.requestFrame(frame);
    }
  };

  return {
    get isRunning() {
      return running;
    },
    start: () => {
      if (running) {
        return;
      }
      running = true;
      // Force the first frame to step immediately.
      lastStep = deps.now() - intervalMs;
      handle = deps.requestFrame(frame);
    },
    stop: () => {
      running = false;
      if (handle !== null) {
        deps.cancelFrame(handle);
        handle = null;
      }
    },
  };
};
