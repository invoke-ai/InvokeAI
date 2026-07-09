/**
 * A single-rAF render scheduler.
 *
 * Consumers call `invalidate(...)` many times per frame; the scheduler
 * coalesces those into one pending `RenderFlags` set and runs the injected
 * `render` callback at most once per animation frame. The rAF driver is
 * injectable so the scheduler can be driven deterministically in node tests.
 *
 * Zero React, zero import-time side effects; the only DOM touch point is the
 * default `requestFrame`/`cancelFrame`, which are resolved lazily at call
 * time (never at import) and are always overridden in tests.
 */

import type { RenderFlags } from '@workbench/canvas-engine/types';

/** The partial invalidation payload accepted by {@link RenderScheduler.invalidate}. */
export interface InvalidatePayload {
  /** The viewport transform (pan/zoom) changed. */
  view?: true;
  /** Ids of layers whose pixel content or transform changed. */
  layers?: string[];
  /** Interaction overlays (selection, cursors, guides) changed. */
  overlay?: true;
  /** Force a full repaint next frame. */
  all?: true;
}

/** Dependencies for {@link createRenderScheduler}; the rAF pair is injectable for tests. */
export interface RenderSchedulerDeps {
  /** Invoked once per scheduled frame with the coalesced flags. */
  render: (flags: RenderFlags) => void;
  /** Defaults to `globalThis.requestAnimationFrame`. */
  requestFrame?: (callback: FrameRequestCallback) => number;
  /** Defaults to `globalThis.cancelAnimationFrame`. */
  cancelFrame?: (handle: number) => void;
}

/** The imperative scheduler handle returned by {@link createRenderScheduler}. */
export interface RenderScheduler {
  /** Merge a partial invalidation into the pending flags and schedule a frame. */
  invalidate(payload: InvalidatePayload): void;
  /** Suspend frame scheduling (e.g. widget detach); invalidations still accumulate. */
  pause(): void;
  /** Resume scheduling; flushes any invalidations accumulated while paused. */
  resume(): void;
  /** True while paused. */
  readonly isPaused: boolean;
  /** Cancel any pending frame and stop accepting further work. */
  dispose(): void;
}

const createEmptyFlags = (): RenderFlags => ({
  all: false,
  layers: new Set<string>(),
  overlay: false,
  view: false,
});

const hasPending = (flags: RenderFlags): boolean => flags.all || flags.view || flags.overlay || flags.layers.size > 0;

const defaultRequestFrame = (callback: FrameRequestCallback): number => globalThis.requestAnimationFrame(callback);

const defaultCancelFrame = (handle: number): void => {
  globalThis.cancelAnimationFrame(handle);
};

/**
 * Creates a coalescing, single-frame render scheduler. See module docs for
 * the coalescing and pause/resume semantics.
 */
export const createRenderScheduler = (deps: RenderSchedulerDeps): RenderScheduler => {
  const requestFrame = deps.requestFrame ?? defaultRequestFrame;
  const cancelFrame = deps.cancelFrame ?? defaultCancelFrame;

  let pending = createEmptyFlags();
  let frameHandle: number | null = null;
  let paused = false;
  let disposed = false;

  const runFrame = (): void => {
    frameHandle = null;
    if (disposed) {
      return;
    }
    // Snapshot then reset before invoking render, so invalidations made from
    // within the render callback accumulate into a fresh frame rather than
    // being wiped by the post-render reset.
    const flags = pending;
    pending = createEmptyFlags();
    deps.render(flags);
  };

  const schedule = (): void => {
    if (disposed || paused || frameHandle !== null) {
      return;
    }
    if (!hasPending(pending)) {
      return;
    }
    frameHandle = requestFrame(runFrame);
  };

  const invalidate = (payload: InvalidatePayload): void => {
    if (disposed) {
      return;
    }
    if (payload.all) {
      pending.all = true;
    }
    if (payload.view) {
      pending.view = true;
    }
    if (payload.overlay) {
      pending.overlay = true;
    }
    if (payload.layers) {
      for (const id of payload.layers) {
        pending.layers.add(id);
      }
    }
    schedule();
  };

  const pause = (): void => {
    if (disposed || paused) {
      return;
    }
    paused = true;
    if (frameHandle !== null) {
      cancelFrame(frameHandle);
      frameHandle = null;
    }
  };

  const resume = (): void => {
    if (disposed || !paused) {
      return;
    }
    paused = false;
    schedule();
  };

  const dispose = (): void => {
    if (disposed) {
      return;
    }
    disposed = true;
    if (frameHandle !== null) {
      cancelFrame(frameHandle);
      frameHandle = null;
    }
  };

  return {
    dispose,
    invalidate,
    get isPaused() {
      return paused;
    },
    pause,
    resume,
  };
};
