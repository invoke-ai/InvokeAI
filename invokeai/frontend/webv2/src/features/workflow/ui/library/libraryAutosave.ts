/**
 * Figma-style library sync: a project graph bound to a library workflow saves
 * itself back to the library after edits settle. Content-addressed dedupe (the
 * serialized JSON string) prevents echo saves; a failed save parks in 'error'
 * and is retried by the next graph change or an explicit flush(). No automatic
 * retry loop — persistent failures must not spin (cf. bitmapStore anti-spin).
 *
 * A save's acknowledgement only reports 'saved' if no edit arrived while it was
 * in flight; otherwise the status stays 'dirty' and the queued write reports
 * for itself. The status is a promise about the person's current work, not a
 * receipt for whichever bytes happened to be in the last request.
 *
 * dispose() flushes rather than drops: a pending debounced edit is an unsaved
 * change, so unmount collapses it into one final save (status callbacks stay
 * muted). Known limitation: a hard tab close inside the debounce window can
 * still lose the final write — there is no pagehide/keepalive path. Fixing
 * that needs a transport-level keepalive decision and is out of scope here.
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
  /**
   * Bumped by every graph change. A save captures it after serializing, so a
   * captured value that no longer matches means the person edited again while
   * the write was in flight and the answer we just got is about older content.
   *
   * Only the success path consults it, because the two ways of being wrong are
   * not symmetric: claiming 'saved' over unwritten work invites someone to
   * close the tab, while claiming 'error' or 'dirty' over work that did land
   * costs one redundant save. Failures still report as failures.
   */
  let editGeneration = 0;

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

    if (!libraryWorkflowId) {
      return Promise.resolve();
    }

    const json = JSON.stringify(serialized);

    if (json === lastSavedJson) {
      if (!disposed) {
        deps.onStatus('saved');
      }
      return Promise.resolve();
    }

    if (!disposed) {
      deps.onStatus('saving');
    }

    const generationAtCapture = editGeneration;

    inFlight = deps
      .save(libraryWorkflowId, serialized)
      .then(() => {
        // `json` did reach the server, so it is still the dedupe baseline even
        // when newer edits exist — the next pass compares against it and saves
        // only the difference.
        lastSavedJson = json;
        if (!disposed) {
          deps.onStatus(editGeneration === generationAtCapture ? 'saved' : 'dirty');
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
      if (disposed) {
        return;
      }
      disposed = true;
      // A pending debounce is an unsaved edit; losing it on unmount is data
      // loss, so the timer collapses into one immediate final save. Status
      // callbacks stay muted (disposed) — only the write itself survives.
      const hasPendingEdit = timerHandle !== null;
      clearTimer();
      if (hasPendingEdit) {
        void runSave();
      }
    },
    flush: (): Promise<void> => {
      if (disposed) {
        return Promise.resolve();
      }
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
      editGeneration += 1;
      deps.onStatus('dirty');
      clearTimer();
      timerHandle = timers.setTimeout(() => {
        timerHandle = null;
        void runSave();
      }, debounceMs);
    },
  };
};
