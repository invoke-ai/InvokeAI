/**
 * Share one engine per project through reference-counted leases. Delay disposal after the last release;
 * reacquisition cancels it so quick remounts reuse the warm engine.
 */

import type { CanvasEngine as PublicCanvasEngine } from '@workbench/canvas-engine/api';
import type { LiveCanvasEngines } from '@workbench/projects/projectAssets';

import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import {
  createCanvasEngine,
  type CanvasEngine,
  type CanvasEngineOptions,
} from '@workbench/canvas-operations/createCanvasEngine';

/** Engine creation dependencies, minus the project id the registry supplies. */
export type EngineDeps = Omit<CanvasEngineOptions, 'projectId'> & {
  /**
   * The mounted Workbench's held-media registry. It also identifies that Workbench: an engine is bound to the one
   * that created it, so a released engine is replaced rather than reused by the next Workbench.
   */
  liveEngines?: LiveCanvasEngines;
};

/** Default grace period before a released engine is disposed (30s). */
export const DEFAULT_GRACE_PERIOD_MS = 30_000;

/** Injectable timer seam (defaults to the global timers). */
export interface RegistryTimers {
  setTimeout(handler: () => void, ms: number): number;
  clearTimeout(handle: number): void;
}

/** The registry handle. */
export interface EngineRegistry {
  /** Immediately disposes every engine, including active and cooling leases. */
  disposeAll(): void;
  /** Returns the engine for `projectId`, creating it if needed, and adds a reference. */
  getOrCreateEngine(projectId: string, deps: EngineDeps): CanvasEngine;
  /** Returns the engine for `projectId` without changing its reference count. */
  getEngine(projectId: string): CanvasEngine | undefined;
  /** Drops a reference; schedules grace-period disposal when the last reference is released. */
  releaseEngine(projectId: string): void;
}

interface RegistryEntry {
  engine: CanvasEngine;
  liveEngines: LiveCanvasEngines | undefined;
  releaseLiveEngine: () => void;
  refCount: number;
  disposeHandle: number | null;
  generation: number;
  cooldown: Promise<'cooled' | 'dirty'> | null;
}

const defaultTimers: RegistryTimers = {
  clearTimeout: (handle) => globalThis.clearTimeout(handle),
  setTimeout: (handler, ms) => globalThis.setTimeout(handler, ms),
};

/** Creates an engine registry with an optional grace period and injectable timers. */
export const createEngineRegistry = (
  options: {
    gracePeriodMs?: number;
    timers?: RegistryTimers;
  } = {}
): EngineRegistry => {
  const gracePeriodMs = options.gracePeriodMs ?? DEFAULT_GRACE_PERIOD_MS;
  const timers = options.timers ?? defaultTimers;
  const entries = new Map<string, RegistryEntry>();

  const cancelDisposal = (entry: RegistryEntry): void => {
    if (entry.disposeHandle !== null) {
      timers.clearTimeout(entry.disposeHandle);
      entry.disposeHandle = null;
    }
  };

  const scheduleDisposal = (projectId: string, entry: RegistryEntry, generation: number): void => {
    entry.disposeHandle = timers.setTimeout(() => {
      entry.disposeHandle = null;
      void entry.cooldown?.then((result) => {
        if (entry.refCount !== 0 || entry.generation !== generation || entries.get(projectId) !== entry) {
          return;
        }
        if (result === 'dirty') {
          entry.cooldown = entry.engine.lifecycle.beginCooldown();
          scheduleDisposal(projectId, entry, generation);
          return;
        }
        entries.delete(projectId);
        entry.releaseLiveEngine();
        entry.engine.lifecycle.dispose();
      });
    }, gracePeriodMs);
  };

  return {
    disposeAll: () => {
      const ownedEntries = [...entries.values()];
      let firstError: unknown;

      entries.clear();
      for (const entry of ownedEntries) {
        cancelDisposal(entry);
        entry.releaseLiveEngine();
        try {
          entry.engine.lifecycle.dispose();
        } catch (error) {
          firstError ??= error;
        }
      }

      if (firstError !== undefined) {
        throw new Error(firstError instanceof Error ? firstError.message : 'A canvas engine failed to dispose.', {
          cause: firstError,
        });
      }
    },
    getEngine: (projectId) => entries.get(projectId)?.engine,
    getOrCreateEngine: (projectId, { liveEngines, ...deps }) => {
      const existing = entries.get(projectId);
      if (existing && existing.liveEngines !== liveEngines && existing.refCount > 0) {
        // Its edits and held media belong to the Workbench still using it; sharing it would cross that boundary.
        throw new Error(`The canvas engine for project ${projectId} is still in use by another Workbench.`);
      }
      if (existing && existing.liveEngines === liveEngines) {
        cancelDisposal(existing);
        existing.generation += 1;
        existing.refCount += 1;
        existing.engine.lifecycle.activate();
        existing.cooldown = null;
        return existing.engine;
      }
      if (existing) {
        // A released engine still writes through the Workbench that created it; a new Workbench gets its own.
        entries.delete(projectId);
        cancelDisposal(existing);
        existing.releaseLiveEngine();
        existing.engine.lifecycle.dispose();
      }
      const engine = createCanvasEngine({ projectId, ...deps });
      const releaseLiveEngine =
        liveEngines?.register(projectId, {
          // Document pushes cannot wait on an edit the user may never finish.
          flushPendingPixels: () => engine.lifecycle.flushPendingUploads({ waitForHeldPixels: false }),
          heldAssets: () => engine.history.getHeldAssetRefs(),
          subscribe: (listener) => engine.interaction.subscribe('historyEpoch', listener),
        }) ?? (() => undefined);
      entries.set(projectId, {
        cooldown: null,
        disposeHandle: null,
        engine,
        generation: 0,
        liveEngines,
        refCount: 1,
        releaseLiveEngine,
      });
      return engine;
    },
    releaseEngine: (projectId) => {
      const entry = entries.get(projectId);
      if (!entry) {
        return;
      }
      entry.refCount = Math.max(0, entry.refCount - 1);
      if (entry.refCount > 0 || entry.disposeHandle !== null) {
        return;
      }
      entry.generation += 1;
      const generation = entry.generation;
      entry.cooldown = entry.engine.lifecycle.beginCooldown();
      scheduleDisposal(projectId, entry, generation);
    },
  };
};

/** The process-wide default registry shared by all widget surfaces. */
const defaultRegistry = createEngineRegistry();

registerAccountOwnedResource({
  clear: defaultRegistry.disposeAll,
  name: 'canvas-engine-registry',
});

export const getOrCreateEngine = defaultRegistry.getOrCreateEngine;
export const releaseEngine = defaultRegistry.releaseEngine;

/** Non-owning public lookup. Engine construction and lease management stay inside Canvas composition. */
export const getCanvasEngine = (projectId: string): PublicCanvasEngine | undefined =>
  defaultRegistry.getEngine(projectId);
