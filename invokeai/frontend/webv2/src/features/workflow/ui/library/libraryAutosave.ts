/**
 * Figma-style library sync: a project graph bound to a library workflow saves
 * itself back to the library after edits settle. Content-addressed dedupe (the
 * serialized JSON string) prevents echo saves; a failed save parks in 'error'
 * and is retried by the next graph change or an explicit flush(). No automatic
 * retry loop — persistent failures must not spin (cf. bitmapStore anti-spin).
 */

export type LibrarySyncStatus = 'idle' | 'dirty' | 'saving' | 'saved' | 'error';

export interface LibraryAutosaverDeps {
  read(): { libraryWorkflowId: string | undefined; serialized: Record<string, unknown> };
  save(workflowId: string, serialized: Record<string, unknown>): Promise<void>;
  onStatus(status: LibrarySyncStatus): void;
  /** Idle window before a save (default 2000ms). */
  debounceMs?: number;
  timers?: {
    setTimeout(fn: () => void, ms: number): number;
    clearTimeout(handle: number): void;
  };
}

export const DEFAULT_LIBRARY_AUTOSAVE_DEBOUNCE_MS = 2000;

export const createLibraryAutosaver = (deps: LibraryAutosaverDeps) => {
  const debounceMs = deps.debounceMs ?? DEFAULT_LIBRARY_AUTOSAVE_DEBOUNCE_MS;
  const timers = deps.timers ?? {
    clearTimeout: (handle: number) => globalThis.clearTimeout(handle),
    setTimeout: (fn: () => void, ms: number) => globalThis.setTimeout(fn, ms),
  };

  let timerHandle: number | null = null;
  let inFlight: Promise<void> | null = null;
  let lastSavedJson: string | null = null;
  let disposed = false;

  const clearTimer = (): void => {
    if (timerHandle !== null) {
      timers.clearTimeout(timerHandle);
      timerHandle = null;
    }
  };

  const runSave = (): Promise<void> => {
    if (inFlight) {
      // A save is running; chain another pass after it so content edited
      // mid-save is picked up. The rerun re-reads and dedupes, so it no-ops
      // when nothing newer landed — the chain always terminates.
      return inFlight.then(() => runSave());
    }

    const { libraryWorkflowId, serialized } = deps.read();

    if (disposed || !libraryWorkflowId) {
      return Promise.resolve();
    }

    const json = JSON.stringify(serialized);

    if (json === lastSavedJson) {
      deps.onStatus('saved');
      return Promise.resolve();
    }

    deps.onStatus('saving');
    inFlight = deps
      .save(libraryWorkflowId, serialized)
      .then(() => {
        lastSavedJson = json;
        if (!disposed) {
          deps.onStatus('saved');
        }
      })
      .catch(() => {
        if (!disposed) {
          deps.onStatus('error');
        }
      })
      .finally(() => {
        inFlight = null;
      });

    return inFlight;
  };

  return {
    dispose: (): void => {
      disposed = true;
      clearTimer();
    },
    flush: (): Promise<void> => {
      clearTimer();
      return runSave();
    },
    /** Marks the last save as matching `serialized` — call after load/bind so the loaded state is not re-saved. */
    markSynced: (serialized: Record<string, unknown>): void => {
      lastSavedJson = JSON.stringify(serialized);
    },
    notifyGraphChanged: (): void => {
      if (disposed || !deps.read().libraryWorkflowId) {
        return;
      }
      deps.onStatus('dirty');
      clearTimer();
      timerHandle = timers.setTimeout(() => {
        timerHandle = null;
        void runSave();
      }, debounceMs);
    },
  };
};
