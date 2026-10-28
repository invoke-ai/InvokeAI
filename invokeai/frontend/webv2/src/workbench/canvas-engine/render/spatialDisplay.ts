/**
 * The asynchronous display of spatially adjusted pixels, shared by layers (`adjustedSurfaceCache.ts`) and groups
 * (`groupSurfaceCache.ts`). Each owner shows an interim until its stack's evaluation lands from the spatial worker;
 * the canvas-engine ARCHITECTURE "Spatial adjustments" section states the interim, install and scheduling rules.
 */

import type { CanvasAdjustmentsContract } from '@workbench/canvas-engine/contracts';
import type {
  SpatialRunRequest,
  SpatialRunResult,
} from '@workbench/canvas-engine/render/spatial/spatialAdjustmentService';
import type { Rect } from '@workbench/canvas-engine/types';

import { intersect, roundOut } from '@workbench/canvas-engine/math/rect';

import type { RasterSurface } from './raster';

import { type AdjustmentOrigin, adjustmentsKey, applyAdjustments, perPixelAdjustments } from './adjustments';
import { evaluateAdjustmentStack } from './exactAdjustments';

/** Why an owner's displayed spatial adjustments are not exact; absent means they are (or there are none). */
export type AdjustedSurfaceStatus = 'computing' | 'too-large' | 'over-budget' | 'error';

/**
 * How final displayed pixels are: `exact`; `pending` while a spatial job will still change them; `inexact` when none
 * can run until the owner's target changes (a failure, or a placeholder source).
 */
export type DisplayedPixelsState = 'exact' | 'pending' | 'inexact';

const STATE_RANK: Record<DisplayedPixelsState, number> = { exact: 0, inexact: 1, pending: 2 };

/** The less final of two states: pending outranks inexact, which outranks exact. */
export const worseDisplayedState = (a: DisplayedPixelsState, b: DisplayedPixelsState): DisplayedPixelsState =>
  STATE_RANK[a] >= STATE_RANK[b] ? a : b;

/** Content edits start a spatial job only after this much quiet, so a stroke does not queue one per tick. */
export const SPATIAL_CONTENT_DEBOUNCE_MS = 120;

export interface SpatialDisplayOptions {
  /** Evaluates spatial steps off the main thread (`SpatialAdjustmentService.run`). */
  readonly run: (request: SpatialRunRequest) => Promise<SpatialRunResult>;
  /** A spatial result changed the owner's displayed pixels without a source version change. */
  readonly onSettle?: (ownerId: string) => void;
  /** The owner's spatial status changed; null when exact or no longer spatial. */
  readonly onStatus?: (ownerId: string, status: AdjustedSurfaceStatus | null) => void;
}

/** What one display request asked for; a job snapshots it. */
export interface SpatialDisplayTarget {
  /** The owner's unadjusted pixels; a job reads all of them. */
  readonly source: RasterSurface;
  readonly version: number;
  /** `adjustmentsKey` of the stack. */
  readonly key: string;
  readonly adjustments: CanvasAdjustmentsContract;
  /** Where the source's (0,0) sits in the stack's reference space. */
  readonly origin: AdjustmentOrigin;
  /**
   * Why no job may start yet: a `placeholder` source with no real pixels (reported idle), or `inputs` whose own
   * spatial results are still computing (reported computing).
   */
  readonly hold: 'placeholder' | 'inputs' | null;
  /** The source's current version, or null once the owner no longer draws from `source`. */
  readonly liveVersion: () => number | null;
  /** Source-local damage since `version`, or null when unknown. */
  readonly damageSince: (version: number) => Rect | null;
  /** The display surface currently guarded for this target, or null once it was evicted or replaced. */
  readonly shown: () => RasterSurface | null;
}

interface TargetKey {
  readonly source: RasterSurface;
  readonly version: number;
  readonly key: string;
}

interface SpatialJob extends TargetKey {
  readonly width: number;
  readonly height: number;
  readonly seq: number;
  readonly controller: AbortController;
}

interface SpatialSlot {
  /** The latest display request. */
  target: SpatialDisplayTarget | null;
  /** Source version the surface's pixels reflect; null until this slot painted the surface. */
  shownVersion: number | null;
  /** Provenance of the installed spatial result; null while the surface is a per-pixel-only interim. */
  shown: { readonly seq: number; readonly version: number; readonly key: string } | null;
  /** At most one job per owner: a later target waits for it, so no readback is wasted while it runs. */
  job: SpatialJob | null;
  timer: ReturnType<typeof setTimeout> | null;
  /** `adjustmentsKey` of the per-pixel part a whole interim painted, while `shown` is null and no patch differs. */
  interimKey: string | null;
  /** The last terminal outcome, kept so an unchanged target is not retried every frame. */
  failure: SpatialFailure | null;
  status: AdjustedSurfaceStatus | null;
}

interface SpatialFailure extends TargetKey {
  readonly status: Exclude<AdjustedSurfaceStatus, 'computing'>;
}

/** Per-owner spatial display state returned by {@link createSpatialDisplay}. */
export interface SpatialDisplay {
  /**
   * Brings `surface` to the interim for `target`: a parameter change keeps the shown pixels, known damage is
   * re-patched with the per-pixel part only, and anything else (or `reusable` false) repaints the whole surface
   * per-pixel. Returns whether it repainted the whole surface.
   */
  paintInterim(ownerId: string, surface: RasterSurface, reusable: boolean, target: SpatialDisplayTarget): boolean;
  /** Records `target` as the owner's latest request and schedules a job toward it. */
  request(ownerId: string, target: SpatialDisplayTarget, rebuilt: boolean): void;
  /** Whether the owner shows the installed result for exactly `target`. */
  isExact(ownerId: string, target: TargetKey): boolean;
  /** How final the owner's displayed pixels are for `target`. */
  state(ownerId: string, target: TargetKey): DisplayedPixelsState;
  status(ownerId: string): AdjustedSurfaceStatus | null;
  /** Forgets an owner, abandoning its job and clearing its status; false when it had no state. */
  drop(ownerId: string): boolean;
  /**
   * Forgets every owner with a job running or scheduled, returning their ids so callers drop the interims those jobs
   * would have replaced. Exact, failed and waiting owners keep their state.
   */
  cancel(): string[];
  ownerIds(): string[];
}

const sameTarget = (a: TargetKey, b: TargetKey): boolean =>
  a.source === b.source && a.version === b.version && a.key === b.key;

/** Whether a failure still applies to `target`: too-large depends only on the content's alpha bounds, not the stack. */
const failureHolds = (failure: SpatialFailure, target: TargetKey): boolean =>
  failure.source === target.source &&
  failure.version === target.version &&
  (failure.status === 'too-large' || failure.key === target.key);

const interimKeyOf = (target: SpatialDisplayTarget): string => adjustmentsKey(perPixelAdjustments(target.adjustments));

/** Damage clamped to a `width` × `height` surface: `null` when unknown, `'none'` when it misses the surface. */
export const clampDamage = (damaged: Rect | null, width: number, height: number): Rect | 'none' | null => {
  if (!damaged) {
    return null;
  }
  const clamped = intersect(roundOut(damaged), { height, width, x: 0, y: 0 });
  return clamped && clamped.width > 0 && clamped.height > 0 ? clamped : 'none';
};

const resetContext = (surface: RasterSurface): RasterSurface['ctx'] => {
  const ctx = surface.ctx;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = 'source-over';
  return ctx;
};

/** Redraws `region` of `surface` from the same region of `source`, with `adjustments` applied at its place. */
export const paintAdjustedRegion = (
  surface: RasterSurface,
  source: RasterSurface,
  region: Rect,
  adjustments: CanvasAdjustmentsContract | undefined,
  origin: AdjustmentOrigin
): void => {
  const ctx = resetContext(surface);
  ctx.clearRect(region.x, region.y, region.width, region.height);
  ctx.drawImage(
    source.canvas,
    region.x,
    region.y,
    region.width,
    region.height,
    region.x,
    region.y,
    region.width,
    region.height
  );
  const imageData = ctx.getImageData(region.x, region.y, region.width, region.height);
  applyAdjustments(imageData, adjustments, { x: origin.x + region.x, y: origin.y + region.y });
  ctx.putImageData(imageData, region.x, region.y);
};

/**
 * Creates the state machine; `slotKeyPrefix` keeps owners of different kinds from superseding each other's queued
 * jobs in the shared service.
 */
export const createSpatialDisplay = (options: SpatialDisplayOptions, slotKeyPrefix: string): SpatialDisplay => {
  const slots = new Map<string, SpatialSlot>();
  let nextSeq = 0;

  const slotFor = (ownerId: string): SpatialSlot => {
    let slot = slots.get(ownerId);
    if (!slot) {
      slot = {
        failure: null,
        interimKey: null,
        job: null,
        shown: null,
        shownVersion: null,
        status: null,
        target: null,
        timer: null,
      };
      slots.set(ownerId, slot);
    }
    return slot;
  };

  /** Repaints all of `surface` with `target`'s per-pixel part only. */
  const paintWholeInterim = (slot: SpatialSlot, surface: RasterSurface, target: SpatialDisplayTarget): void => {
    const { height, width } = surface;
    paintAdjustedRegion(
      surface,
      target.source,
      { height, width, x: 0, y: 0 },
      perPixelAdjustments(target.adjustments),
      target.origin
    );
    slot.shownVersion = target.version;
    slot.shown = null;
    slot.interimKey = interimKeyOf(target);
  };

  /**
   * A target that cannot run shows its per-pixel part alone, as no result will replace what a parameter change kept.
   * Returns whether it repainted.
   */
  const showPerPixelOnly = (slot: SpatialSlot): boolean => {
    const target = slot.target!;
    if (slot.shown === null && slot.interimKey === interimKeyOf(target)) {
      return false;
    }
    const surface = target.shown();
    if (!surface || surface.width !== target.source.width || surface.height !== target.source.height) {
      return false;
    }
    paintWholeInterim(slot, surface, target);
    return true;
  };

  const setStatus = (ownerId: string, slot: SpatialSlot, status: AdjustedSurfaceStatus | null): void => {
    if (slot.status !== status) {
      slot.status = status;
      options.onStatus?.(ownerId, status);
    }
  };

  const clearTimer = (slot: SpatialSlot): void => {
    if (slot.timer !== null) {
      clearTimeout(slot.timer);
      slot.timer = null;
    }
  };

  const abandonJob = (slot: SpatialSlot): void => {
    slot.job?.controller.abort();
    slot.job = null;
  };

  const drop = (ownerId: string): boolean => {
    const slot = slots.get(ownerId);
    if (!slot) {
      return false;
    }
    slots.delete(ownerId);
    clearTimer(slot);
    abandonJob(slot);
    setStatus(ownerId, slot, null);
    return true;
  };

  const isExact = (slot: SpatialSlot): boolean =>
    slot.shown !== null &&
    slot.target !== null &&
    slot.shownVersion === slot.target.version &&
    slot.shown.version === slot.target.version &&
    slot.shown.key === slot.target.key;

  /**
   * Writes a result only when it is newer than the shown one, came from the same source surface and from a version
   * at or before the shown one whose later damage is known; that damage is then re-patched per-pixel.
   */
  const install = (slot: SpatialSlot, job: SpatialJob, pixels: Uint8ClampedArray): boolean => {
    const target = slot.target;
    const live = target?.liveVersion() ?? null;
    if (
      !target ||
      live === null ||
      target.source !== job.source ||
      slot.shownVersion === null ||
      job.version > slot.shownVersion ||
      (slot.shown !== null && job.seq <= slot.shown.seq)
    ) {
      return false;
    }
    const surface = target.shown();
    const { height, width } = job;
    if (
      !surface ||
      surface.width !== width ||
      surface.height !== height ||
      job.source.width !== width ||
      job.source.height !== height
    ) {
      return false;
    }
    const patch = live === job.version ? 'none' : clampDamage(target.damageSince(job.version), width, height);
    if (patch === null) {
      return false;
    }
    const ctx = resetContext(surface);
    const image = ctx.createImageData(width, height);
    image.data.set(pixels);
    ctx.putImageData(image, 0, 0);
    if (patch !== 'none') {
      paintAdjustedRegion(surface, target.source, patch, perPixelAdjustments(target.adjustments), target.origin);
    }
    slot.shown = { key: job.key, seq: job.seq, version: job.version };
    return true;
  };

  const finish = (ownerId: string, slot: SpatialSlot, job: SpatialJob, outcome: SpatialRunResult): void => {
    if (slots.get(ownerId) !== slot || slot.job !== job) {
      return;
    }
    slot.job = null;
    if (outcome.status === 'ok') {
      if (install(slot, job, outcome.pixels)) {
        options.onSettle?.(ownerId);
      } else if (slot.target && !slot.target.shown()) {
        // Evicted while computing: the next display request rebuilds and reschedules.
        setStatus(ownerId, slot, null);
        return;
      }
    } else if (outcome.status !== 'aborted' && outcome.status !== 'superseded') {
      slot.failure = { key: job.key, source: job.source, status: outcome.status, version: job.version };
      if (slot.target && failureHolds(slot.failure, slot.target) && showPerPixelOnly(slot)) {
        options.onSettle?.(ownerId);
      }
    }
    if (slot.timer === null) {
      schedule(ownerId, slot, true);
    }
  };

  const start = (ownerId: string, slot: SpatialSlot): void => {
    const target = slot.target!;
    // The source moved on without a display request: that request will reschedule.
    if (target.liveVersion() !== target.version) {
      setStatus(ownerId, slot, null);
      return;
    }
    const { height, width } = target.source;
    const job: SpatialJob = {
      controller: new AbortController(),
      height,
      key: target.key,
      seq: (nextSeq += 1),
      source: target.source,
      version: target.version,
      width,
    };
    slot.job = job;
    let outcome: Promise<SpatialRunResult>;
    try {
      const { data } = target.source.ctx.getImageData(0, 0, width, height);
      // Display jobs share one service slot per owner, so a newer target supersedes a queued older one.
      outcome = evaluateAdjustmentStack({ data, height, width }, target.adjustments, target.origin, {
        priority: 'display',
        run: options.run,
        signal: job.controller.signal,
        slotKey: `${slotKeyPrefix}${ownerId}`,
      });
    } catch {
      outcome = Promise.resolve({ status: 'error' });
    }
    void outcome.then(
      (result) => finish(ownerId, slot, job, result),
      () => finish(ownerId, slot, job, { status: 'error' })
    );
  };

  /** Brings the slot toward its target: idle when exact, else one job now or after the content debounce. */
  const schedule = (ownerId: string, slot: SpatialSlot, immediate: boolean): void => {
    const target = slot.target!;
    if (isExact(slot)) {
      clearTimer(slot);
      abandonJob(slot);
      slot.failure = null;
      setStatus(ownerId, slot, null);
      return;
    }
    if (target.hold) {
      // A placeholder is never worth a job, and inputs still computing would waste one; a later request starts it.
      clearTimer(slot);
      setStatus(ownerId, slot, target.hold === 'inputs' ? 'computing' : null);
      return;
    }
    if (slot.failure) {
      if (failureHolds(slot.failure, target)) {
        // No job will replace what this request shows, so it shows the per-pixel part now, as the caller draws it.
        clearTimer(slot);
        showPerPixelOnly(slot);
        setStatus(ownerId, slot, slot.failure.status);
        return;
      }
      slot.failure = null;
    }
    setStatus(ownerId, slot, 'computing');
    if (slot.job && sameTarget(slot.job, target)) {
      clearTimer(slot);
      return;
    }
    clearTimer(slot);
    if (immediate) {
      if (!slot.job) {
        start(ownerId, slot);
      }
      return;
    }
    slot.timer = setTimeout(() => {
      slot.timer = null;
      if (!slot.job && slots.get(ownerId) === slot) {
        schedule(ownerId, slot, true);
      }
    }, SPATIAL_CONTENT_DEBOUNCE_MS);
  };

  const isExactFor = (ownerId: string, target: TargetKey): boolean => {
    const slot = slots.get(ownerId);
    return slot?.target !== null && slot?.target !== undefined && sameTarget(slot.target, target) && isExact(slot);
  };

  return {
    cancel: () => {
      const cancelled = [...slots].filter(([, slot]) => slot.job || slot.timer !== null).map(([ownerId]) => ownerId);
      for (const ownerId of cancelled) {
        drop(ownerId);
      }
      return cancelled;
    },
    drop,
    isExact: isExactFor,
    ownerIds: () => [...slots.keys()],
    paintInterim: (ownerId, surface, reusable, target) => {
      const slot = slotFor(ownerId);
      const { height, width } = surface;
      const from = reusable ? slot.shownVersion : null;
      const damage =
        from === null ? null : from === target.version ? 'none' : clampDamage(target.damageSince(from), width, height);
      if (damage === null) {
        paintWholeInterim(slot, surface, target);
        return true;
      }
      slot.shownVersion = target.version;
      if (damage !== 'none') {
        paintAdjustedRegion(surface, target.source, damage, perPixelAdjustments(target.adjustments), target.origin);
        if (slot.interimKey !== interimKeyOf(target)) {
          slot.interimKey = null;
        }
      }
      return false;
    },
    request: (ownerId, target, rebuilt) => {
      const slot = slotFor(ownerId);
      const previous = slot.target;
      slot.target = target;
      if (!previous || rebuilt || !sameTarget(previous, target) || previous.hold !== target.hold) {
        // Content edits wait for quiet; a parameter change, first pixels, released inputs, a new slot or a rebuilt
        // surface start at once.
        const contentOnly =
          previous !== null &&
          previous.hold === null &&
          previous.key === target.key &&
          (previous.source !== target.source || previous.version !== target.version);
        schedule(ownerId, slot, !contentOnly);
      }
    },
    state: (ownerId, target) =>
      isExactFor(ownerId, target) ? 'exact' : slots.get(ownerId)?.status === 'computing' ? 'pending' : 'inexact',
    status: (ownerId) => slots.get(ownerId)?.status ?? null,
  };
};
