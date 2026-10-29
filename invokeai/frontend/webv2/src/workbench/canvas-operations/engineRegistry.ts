/**
 * Share one engine per project through reference-counted leases. Delay disposal after the last release;
 * reacquisition cancels it so quick remounts reuse the warm engine. A disposed engine leaves its project's next engine
 * the tool the person was on, their quick-switch bookmark and how many undo steps went with it, for this session.
 */

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

/** Engine creation dependencies, minus what the registry supplies: the project id and its previous engine's carry. */
export type EngineDeps = Omit<CanvasEngineOptions, 'droppedHistorySteps' | 'projectId'> & {
  /**
   * The mounted Workbench's held-media registry. It also identifies that Workbench: an engine is bound to the one
   * that created it, so a released engine is replaced rather than reused by the next Workbench.
   */
  liveEngines?: LiveCanvasEngines;
};

/** Default grace period before a released engine is disposed (30s). */
export const DEFAULT_GRACE_PERIOD_MS = 30_000;

/** Projects whose disposed engine's carry is kept; the oldest goes first. */
const MAX_CARRIES = 32;

/** What a disposed engine hands its project's next engine within the session. */
interface EngineCarry {
  activeTool: ToolId;
  bookmarkedLayerId: string | null;
  /** Undo and redo steps the disposal dropped, with those its history had already let go; History says so. */
  droppedHistorySteps: number;
}

/** Whether the canvas holds nothing yet: no raster, control or regional layer, and no mask with coverage. */
export const isBlankCanvasDocument = (document: CanvasDocumentContractV3 | null): boolean =>
  getDocumentLeaves(document).every((layer) => layer.type === 'inpaint_mask' && layer.mask.bitmap === null);

/**
 * The tool a fresh engine starts on: the one its project's last engine was on, else Brush on a blank canvas with
 * nothing selected (a new project's), whose first stroke paints a new layer. A selected empty mask, as older
 * projects have, opens on View, so a first stroke never paints the mask by surprise.
 */
const initialTool = (carry: EngineCarry | undefined, document: CanvasDocumentContractV3 | null): ToolId =>
  carry?.activeTool ?? (isBlankCanvasDocument(document) && !document?.selectedLayerId ? 'brush' : 'view');

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
  const carries = new Map<string, EngineCarry>();

  const keepCarry = (projectId: string, engine: CanvasEngine): void => {
    const { dropped = 0, evicted, future, past } = engine.history.getEntries();
    const activeTool = engine.interaction.get('activeTool');
    carries.delete(projectId);
    carries.set(projectId, {
      // Select Object's tool belongs to its operation, which ends with the engine.
      activeTool: activeTool === 'sam' ? 'view' : activeTool,
      bookmarkedLayerId: engine.interaction.get('layerQuickSwitch').bookmarkedLayerId,
      // Steps trimmed at cooldown or evicted before count too: History's start point says how many went.
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
        // A bookmark naming a layer that is gone is ignored by its readers.
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
