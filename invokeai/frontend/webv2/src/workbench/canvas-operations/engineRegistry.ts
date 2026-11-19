/** Reference-counted engine leases with delayed disposal, so quick remounts reuse the warm engine. */

import type {
  CanvasDocumentContractV3,
  CanvasEngine as PublicCanvasEngine,
  ToolId,
} from '@workbench/canvas-engine/api';
import type { LiveCanvasEngines } from '@workbench/projects/projectAssets';

import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { getDocumentLeaves } from '@workbench/canvas-engine/api';
import {
  createCanvasEngine,
  type CanvasEngine,
  type CanvasEngineOptions,
} from '@workbench/canvas-operations/createCanvasEngine';

export type EngineDeps = Omit<CanvasEngineOptions, 'droppedHistorySteps' | 'projectId'> & {
  /** Identifies the Workbench: a released engine is replaced rather than reused by the next one. */
  liveEngines?: LiveCanvasEngines;
};

export const DEFAULT_GRACE_PERIOD_MS = 30_000;

const MAX_CARRIES = 32;

interface EngineCarry {
  activeTool: ToolId;
  bookmarkedLayerId: string | null;
  droppedHistorySteps: number;
}

export const isBlankCanvasDocument = (document: CanvasDocumentContractV3 | null): boolean =>
  getDocumentLeaves(document).every((layer) => layer.type === 'inpaint_mask' && layer.mask.bitmap === null);

/** A selected empty mask opens on View, so a first stroke never paints the mask by surprise. */
const initialTool = (carry: EngineCarry | undefined, document: CanvasDocumentContractV3 | null): ToolId =>
  carry?.activeTool ?? (isBlankCanvasDocument(document) && !document?.selectedLayerId ? 'brush' : 'view');

export interface RegistryTimers {
  setTimeout(handler: () => void, ms: number): number;
  clearTimeout(handle: number): void;
}

export interface EngineRegistry {
  disposeAll(): void;
  getOrCreateEngine(projectId: string, deps: EngineDeps): CanvasEngine;
  getEngine(projectId: string): CanvasEngine | undefined;
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

export const createEngineRegistry = (
  options: {
    gracePeriodMs?: number;
    timers?: RegistryTimers;
  } = {}
): EngineRegistry => {
  const gracePeriodMs = options.gracePeriodMs ?? DEFAULT_GRACE_PERIOD_MS;
  const timers = options.timers ?? defaultTimers;
  const entries = new Map<string, RegistryEntry>();
  const carries = new Map<string, EngineCarry>();

  const keepCarry = (projectId: string, engine: CanvasEngine): void => {
    const { dropped = 0, evicted, future, past } = engine.history.getEntries();
    const activeTool = engine.interaction.get('activeTool');
    carries.delete(projectId);
    carries.set(projectId, {
      // Select Object's operation ended with the engine; Transform would open an edit session nobody asked for.
      activeTool: activeTool === 'sam' || activeTool === 'transform' ? 'view' : activeTool,
      bookmarkedLayerId: engine.interaction.get('layerQuickSwitch').bookmarkedLayerId,
      droppedHistorySteps: past.length + future.length + evicted + dropped,
    });
    if (carries.size > MAX_CARRIES) {
      carries.delete(carries.keys().next().value!);
    }
  };

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
        keepCarry(projectId, entry.engine);
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
      carries.clear();
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
        keepCarry(projectId, existing.engine);
        existing.releaseLiveEngine();
        existing.engine.lifecycle.dispose();
      }
      const carry = carries.get(projectId);
      carries.delete(projectId);
      const engine = createCanvasEngine({ projectId, ...deps, droppedHistorySteps: carry?.droppedHistorySteps ?? 0 });
      engine.tools.setTool(initialTool(carry, engine.document.getDocument()));
      if (carry?.bookmarkedLayerId) {
        engine.interaction.set('layerQuickSwitch', {
          bookmarkedLayerId: carry.bookmarkedLayerId,
          previousLayerId: null,
        });
      }
      const releaseLiveEngine =
        liveEngines?.register(projectId, {
          commitFrame: (from, to, label) =>
            engine.layers.commitStructural(
              label,
              { bbox: to, type: 'setCanvasBbox' },
              { bbox: from, type: 'setCanvasBbox' }
            ).status === 'committed',
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

const defaultRegistry = createEngineRegistry();

registerAccountOwnedResource({
  clear: defaultRegistry.disposeAll,
  name: 'canvas-engine-registry',
});

export const getOrCreateEngine = defaultRegistry.getOrCreateEngine;
export const releaseEngine = defaultRegistry.releaseEngine;

/** Queries the default registry without constructing an engine or acquiring a lease. */
export const getCanvasEngine = (projectId: string): PublicCanvasEngine | undefined =>
  defaultRegistry.getEngine(projectId);
