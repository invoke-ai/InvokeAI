import type { QueueWorkflowRunCompletedEvent, QueueWorkflowRunSink } from '@features/queue/contracts';
import type { AccountScope } from '@platform/state/accountLifecycle';

import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';

import { setLibraryWorkflowThumbnail, touchLibraryWorkflowLastRunAt } from './api';
import { getLibraryWorkflowCached, invalidateWorkflowLibraryCache } from './libraryCache';

/**
 * Turns a completed workflow run into the library record's cover image and
 * last-run stamp. The Queue reports the run; this module owns everything about
 * the library, and every step is best-effort — a capture that fails leaves the
 * record exactly as it was and never surfaces to the user, because the run it
 * decorates already succeeded.
 */

export interface RunCaptureDeps {
  /** Downloads the gallery thumbnail bytes for one result image. */
  fetchThumbnailBlob(imageName: string, signal: AbortSignal): Promise<Blob>;
  getWorkflow(workflowId: string, signal?: AbortSignal): Promise<Record<string, unknown>>;
  invalidateCache(): void;
  setThumbnail(workflowId: string, image: Blob, signal?: AbortSignal): Promise<void>;
  touchLastRunAt(workflowId: string, signal?: AbortSignal): Promise<void>;
}

/**
 * Gallery owns image URLs, but importing its public surface statically would
 * drag the gallery UI (drag-and-drop included) into every bundle that composes
 * the queue runtime. The URL is needed once per completed run, so it is loaded
 * on demand instead.
 *
 * Plain `fetch` rather than the authenticated API helper: image media routes are
 * served against the path-scoped session cookie — the same credentials the
 * gallery's own `<img>` tags use to render this exact URL.
 */
const fetchThumbnailBlob = async (imageName: string, signal: AbortSignal): Promise<Blob> => {
  const { galleryImageUrls } = await import('@features/gallery/utility');
  const response = await fetch(galleryImageUrls.thumbnail(imageName), { credentials: 'same-origin', signal });

  if (!response.ok) {
    throw new Error(`Thumbnail request for ${imageName} failed with status ${response.status}.`);
  }

  return response.blob();
};

const PRODUCTION_DEPS: RunCaptureDeps = {
  fetchThumbnailBlob,
  getWorkflow: getLibraryWorkflowCached,
  invalidateCache: invalidateWorkflowLibraryCache,
  setThumbnail: setLibraryWorkflowThumbnail,
  touchLastRunAt: touchLibraryWorkflowLastRunAt,
};

/**
 * Bundled defaults ship in `meta.category: 'default'` and are read-only for the
 * account, so a run of one must not write a thumbnail or a last-run stamp.
 */
const isUserWorkflow = (workflow: Record<string, unknown>): boolean => {
  const meta = workflow.meta;

  return typeof meta === 'object' && meta !== null && (meta as { category?: unknown }).category === 'user';
};

interface PendingCapture {
  event: QueueWorkflowRunCompletedEvent;
  /** The scope the run settled under; a later account owns none of its writes. */
  owner: AccountScope;
}

export const createWorkflowRunCaptureSink = (overrides?: Partial<RunCaptureDeps>): QueueWorkflowRunSink => {
  const deps: RunCaptureDeps = { ...PRODUCTION_DEPS, ...overrides };
  // One drain chain per library record. Two runs of the SAME workflow must not
  // race their thumbnail uploads (the loser would win the record), while runs of
  // different workflows are independent and proceed in parallel.
  const draining = new Map<string, Promise<void>>();
  // At most one queued capture per record: while one is uploading, a newer run
  // supersedes any other run still waiting, because only the newest output
  // should end up as the cover.
  const pending = new Map<string, PendingCapture>();

  const capture = async ({ event, owner }: PendingCapture): Promise<void> => {
    const { imageNames, libraryWorkflowId } = event;
    const imageName = imageNames[imageNames.length - 1];

    if (!imageName || !isAccountScopeCurrent(owner)) {
      return;
    }

    const workflow = await deps.getWorkflow(libraryWorkflowId, owner.signal);

    if (!isAccountScopeCurrent(owner) || !isUserWorkflow(workflow)) {
      return;
    }

    const blob = await deps.fetchThumbnailBlob(imageName, owner.signal);

    if (!isAccountScopeCurrent(owner)) {
      return;
    }

    await deps.setThumbnail(libraryWorkflowId, blob, owner.signal);

    if (!isAccountScopeCurrent(owner)) {
      return;
    }

    // Ordered after the upload so a library row never advertises a fresh run
    // against a stale cover.
    await deps.touchLastRunAt(libraryWorkflowId, owner.signal);

    if (isAccountScopeCurrent(owner)) {
      deps.invalidateCache();
    }
  };

  const drain = async (libraryWorkflowId: string): Promise<void> => {
    for (let next = pending.get(libraryWorkflowId); next !== undefined; next = pending.get(libraryWorkflowId)) {
      pending.delete(libraryWorkflowId);

      try {
        await capture(next);
      } catch {
        // Capture is decoration around an already-successful run: one failed
        // record must not stop the next run from trying again.
      }
    }

    draining.delete(libraryWorkflowId);
  };

  return {
    onWorkflowRunCompleted: (event) => {
      if (event.imageNames.length === 0) {
        return;
      }

      pending.set(event.libraryWorkflowId, { event, owner: captureAccountScope() });

      if (!draining.has(event.libraryWorkflowId)) {
        draining.set(event.libraryWorkflowId, drain(event.libraryWorkflowId));
      }
    },
  };
};
