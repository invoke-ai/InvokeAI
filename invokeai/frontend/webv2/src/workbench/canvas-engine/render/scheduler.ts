/**
 * Coalesces invalidations into one flag set and render per animation frame. Injected drivers support deterministic
 * tests; default browser functions resolve lazily.
 */

import type { FrameDamage, LayerDamage, RenderFlags } from '@workbench/canvas-engine/types';

import { FULL_DAMAGE, NO_DAMAGE } from '@workbench/canvas-engine/types';

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
  /**
   * Optional local damage is honored only for the sole named layer. View/all or unqualified layer invalidations
   * force full repaint; omission is safe.
   */
  damage?: LayerDamage;
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
  /**
   * Runs `task` at the start of the next frame, before that frame's flags are taken, so its invalidations render
   * in the same frame. Returns an idempotent cancel.
   */
  beforeNextFrame(task: () => void): () => void;
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
  damage: NO_DAMAGE,
  layers: new Set<string>(),
  overlay: false,
  view: false,
});

const addDamage = (pending: FrameDamage, damage: LayerDamage | null): FrameDamage => {
  if (!damage || pending.kind === 'full') {
    return FULL_DAMAGE;
  }
  return { kind: 'regions', regions: pending.kind === 'regions' ? [...pending.regions, damage] : [damage] };
};

const hasPending = (flags: RenderFlags): boolean => flags.all || flags.view || flags.overlay || flags.layers.size > 0;

const defaultRequestFrame = (callback: FrameRequestCallback): number => globalThis.requestAnimationFrame(callback);

const defaultCancelFrame = (handle: number): void => {
  globalThis.cancelAnimationFrame(handle);
};

export const createRenderScheduler = (deps: RenderSchedulerDeps): RenderScheduler => {
  const requestFrame = deps.requestFrame ?? defaultRequestFrame;
  const cancelFrame = deps.cancelFrame ?? defaultCancelFrame;

  let pending = createEmptyFlags();
  const frameTasks = new Set<() => void>();
  let frameHandle: number | null = null;
  let paused = false;
  let disposed = false;

  const runFrame = (): void => {
    if (disposed) {
      frameHandle = null;
      return;
    }
    // The handle stays set while tasks run, so their invalidations join this frame instead of requesting another.
    const tasks = [...frameTasks];
    frameTasks.clear();
    for (const task of tasks) {
      try {
        task();
      } catch (error) {
        // One failed task must not starve the frame or the tasks after it.
        globalThis.reportError?.(error);
      }
    }
    frameHandle = null;
    if (!paused && !disposed && hasPending(pending)) {
      // Snapshot then reset before invoking render, so invalidations made from
      // within the render callback accumulate into a fresh frame rather than
      // being wiped by the post-render reset.
      const flags = pending;
      pending = createEmptyFlags();
      deps.render(flags);
    }
    if (frameTasks.size > 0) {
      schedule();
    }
  };

  const schedule = (): void => {
    if (disposed || paused || frameHandle !== null) {
      return;
    }
    if (!hasPending(pending) && frameTasks.size === 0) {
      return;
    }
    try {
      frameHandle = requestFrame(runFrame);
    } catch {
      // Rendering is ancillary to the document mutation that requested it. A
      // faulty host scheduler must not make that already-applied mutation look
      // failed; retain the pending flags so a later invalidation can retry.
      frameHandle = null;
    }
  };

  const invalidate = (payload: InvalidatePayload): void => {
    if (disposed) {
      return;
    }
    // Partial repaint requires every invalidation to declare damage. Any unknown region widens to full, trading
    // performance for correctness.
    if (payload.all) {
      pending.all = true;
      pending.damage = FULL_DAMAGE;
    }
    if (payload.view) {
      // The whole viewport moves under a pan/zoom; no layer-local rect survives it.
      pending.view = true;
      pending.damage = FULL_DAMAGE;
    }
    if (payload.overlay) {
      // The overlay is its own canvas, redrawn whole every frame — it neither
      // needs nor invalidates composite damage.
      pending.overlay = true;
    }
    if (payload.layers) {
      for (const id of payload.layers) {
        pending.layers.add(id);
      }
      const damage =
        payload.damage && payload.layers.length === 1 && payload.layers[0] === payload.damage.layerId
          ? payload.damage
          : null;
      pending.damage = addDamage(pending.damage, damage);
    }
    schedule();
  };

  const beforeNextFrame = (task: () => void): (() => void) => {
    if (disposed) {
      return () => undefined;
    }
    // A wrapper gives repeated registrations of one function separate identities.
    const entry = (): void => task();
    frameTasks.add(entry);
    schedule();
    return () => {
      frameTasks.delete(entry);
    };
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
    frameTasks.clear();
    if (frameHandle !== null) {
      cancelFrame(frameHandle);
      frameHandle = null;
    }
  };

  return {
    beforeNextFrame,
    dispose,
    invalidate,
    get isPaused() {
      return paused;
    },
    pause,
    resume,
  };
};
