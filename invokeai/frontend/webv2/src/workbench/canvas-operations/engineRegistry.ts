/**
 * The engine registry: one {@link CanvasEngine} per project, shared by the
 * canvas and layers widgets (and any other surface that needs the same live
 * document/pixels).
 *
 * Acquisition is reference-counted: `getOrCreateEngine` hands out (or creates)
 * the instance and bumps the count; `releaseEngine` drops it. When the count
 * reaches zero the engine is not disposed immediately — a grace-period timer
 * runs first, so a quick unmount/remount (route change, widget re-layout)
 * re-acquires the same warm instance instead of paying to rebuild it. A
 * re-acquire cancels the pending disposal. The timer is injectable so tests can
 * drive it deterministically.
 *
 * Zero React, zero import-time side effects.
 */

import type { CanvasEngine as PublicCanvasEngine } from '@workbench/canvas-engine/api';

import {
  createCanvasEngine,
  type CanvasEngine,
  type CanvasEngineOptions,
} from '@workbench/canvas-operations/createCanvasEngine';

/** Engine creation dependencies, minus the project id the registry supplies. */
export type EngineDeps = Omit<CanvasEngineOptions, 'projectId'>;

/** Default grace period before a released engine is disposed (30s). */
export const DEFAULT_GRACE_PERIOD_MS = 30_000;

/** Injectable timer seam (defaults to the global timers). */
export interface RegistryTimers {
  setTimeout(handler: () => void, ms: number): number;
  clearTimeout(handle: number): void;
}

/** The registry handle. */
export interface EngineRegistry {
  /** Returns the engine for `projectId`, creating it if needed, and adds a reference. */
  getOrCreateEngine(projectId: string, deps: EngineDeps): CanvasEngine;
  /** Returns the engine for `projectId` without changing its reference count. */
  getEngine(projectId: string): CanvasEngine | undefined;
  /** Drops a reference; schedules grace-period disposal when the last reference is released. */
  releaseEngine(projectId: string): void;
}

interface RegistryEntry {
  engine: CanvasEngine;
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
        entry.engine.lifecycle.dispose();
      });
    }, gracePeriodMs);
  };

  return {
    getEngine: (projectId) => entries.get(projectId)?.engine,
    getOrCreateEngine: (projectId, deps) => {
      const existing = entries.get(projectId);
      if (existing) {
        cancelDisposal(existing);
        existing.generation += 1;
        existing.refCount += 1;
        existing.engine.lifecycle.activate();
        existing.cooldown = null;
        return existing.engine;
      }
      const engine = createCanvasEngine({ projectId, ...deps });
      entries.set(projectId, { cooldown: null, disposeHandle: null, engine, generation: 0, refCount: 1 });
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

export const getOrCreateEngine = defaultRegistry.getOrCreateEngine;
export const releaseEngine = defaultRegistry.releaseEngine;

/** Non-owning public lookup. Engine construction and lease management stay inside Canvas composition. */
export const getCanvasEngine = (projectId: string): PublicCanvasEngine | undefined =>
  defaultRegistry.getEngine(projectId);
