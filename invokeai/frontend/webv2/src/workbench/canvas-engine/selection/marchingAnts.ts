/**
 * Draws black/white selection paths with zoom-compensated widths and dashes. A throttled injected frame loop
 * advances phase through overlay-only invalidation; run only while selected and attached, and cancel pending
 * frames on stop.
 */

import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Mat2d } from '@workbench/canvas-engine/types';

import { getScale, identity, multiply } from '@workbench/canvas-engine/math/mat2d';

const IDENTITY = identity();

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
  /**
   * Optional document transform moves ants with live float pixels without rebuilding paths. Commit replaces
   * selection geometry and removes the override.
   */
  matrix?: Mat2d | null;
}

/**
 * `view` maps document to CSS pixels; `base` maps CSS to backing pixels, so dash and width stay CSS-constant at any
 * device-pixel ratio.
 */
export const drawMarchingAnts = (
  ctx: RasterSurface['ctx'],
  view: Mat2d,
  render: MarchingAntsRender,
  base: Mat2d = IDENTITY
): void => {
  if (render.paths.length === 0) {
    return;
  }
  const transform = multiply(base, render.matrix ? multiply(view, render.matrix) : view);
  // Line width and dash stay screen-constant, so they follow the VIEW scale
  // only — a float scaled up must not thicken its own outline.
  const scale = getScale(view) || 1;
  const dash = ANTS_DASH_PX / scale;
  const lineWidth = ANTS_LINE_WIDTH_PX / scale;
  const offset = render.phase / scale;

  ctx.save();
  ctx.setTransform(transform.a, transform.b, transform.c, transform.d, transform.e, transform.f);
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
  /** Every rAF, before the step check; a smooth-animation consumer invalidates from here. */
  onFrame?(): void;
  /** Called on each throttled step (the engine advances the phase + invalidates the overlay). */
  onStep(): void;
  /** Step interval in ms; defaults to {@link ANTS_INTERVAL_MS}. */
  intervalMs?: number;
}

/** Polls the injected frame clock while running, firing `onStep` only at `intervalMs` boundaries. */
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
    deps.onFrame?.();
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
