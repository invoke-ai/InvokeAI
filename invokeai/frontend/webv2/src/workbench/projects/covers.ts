import {
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
  type AccountScope,
} from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';
import { createSingleFlight } from '@platform/state/singleFlight';
import { absolutizeApiUrl } from '@platform/transport/http';

import { getClientStateValue, setClientStateValue } from './api';

/**
 * Which image stands for each project, as an index in the per-user client-state KV.
 *
 * The document is the source of truth, but the library grid lists projects from `GET /projects`,
 * which returns summaries with no document. So the answer is written down once, on save or import.
 * Being an index, it is allowed to be stale: a project with no entry shows the folder glyph.
 *
 * ### The whole index is one value, so a write must not precede a read
 *
 * The KV holds one blob and `setClientStateValue` replaces it outright, so writing from a store
 * that has not loaded would delete every entry but the one being written. Not hypothetical:
 * autosave reaches {@link recordProjectCover} from the editor, which loads the index only when the
 * switcher or Open dialog opens. Records made before the load wait in {@link pendingCoverNames}.
 *
 * A read that *fails* is not a read — merging onto a blank answer is the same destructive write.
 */

export const PROJECT_COVERS_KEY = 'webv2:project-covers';

/** Cap on tracked projects, so a long-lived account cannot grow the blob without bound. */
const MAX_TRACKED_COVERS = 500;

interface ProjectCoversSnapshot {
  /** Project id to the server image name of its cover. */
  coverImageNames: Record<string, string>;
  isDirty: boolean;
  isLoaded: boolean;
}

interface CoverMutationQueue {
  latestRevision: number;
  owner: AccountScope;
  tail: Promise<void>;
}

const EMPTY: ProjectCoversSnapshot = { coverImageNames: {}, isDirty: false, isLoaded: false };
const store = createExternalStore<ProjectCoversSnapshot>(EMPTY);
let mutationQueue: CoverMutationQueue | null = null;

/** Covers recorded before the index loaded, newest per project. Drained by {@link loadProjectCovers}. */
const pendingCoverNames = new Map<string, string | null>();

registerAccountOwnedResource({
  clear: () => {
    pendingCoverNames.clear();
    mutationQueue = null;
    store.setSnapshot(EMPTY);
  },
  name: 'project-covers',
});

export const parseProjectCovers = (raw: string | null): Record<string, string> => {
  if (!raw) {
    return {};
  }

  try {
    const parsed: unknown = JSON.parse(raw);

    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
      return {};
    }

    return Object.fromEntries(
      Object.entries(parsed).filter(
        (entry): entry is [string, string] => typeof entry[1] === 'string' && entry[1] !== ''
      )
    );
  } catch {
    return {};
  }
};

const enqueuePersist = (coverImageNames: Record<string, string>, owner: AccountScope): void => {
  if (mutationQueue === null || mutationQueue.owner !== owner) {
    mutationQueue = { latestRevision: 0, owner, tail: Promise.resolve() };
  }

  const queue = mutationQueue;
  const revision = ++queue.latestRevision;
  const value = JSON.stringify(coverImageNames);

  queue.tail = queue.tail.then(async () => {
    if (mutationQueue !== queue || !isAccountScopeCurrent(owner)) {
      return;
    }

    try {
      await setClientStateValue(PROJECT_COVERS_KEY, value, owner.signal);
    } catch {
      // Losing the write costs a glyph until the next save. Keeping the snapshot dirty is what
      // lets that save retry the complete blob.
      return;
    }

    if (mutationQueue !== queue || !isAccountScopeCurrent(owner) || queue.latestRevision !== revision) {
      return;
    }

    store.setSnapshot({ ...store.getSnapshot(), isDirty: false });
  });
};

/**
 * Trim to the cap, oldest first. Insertion order is recency order because
 * {@link recordProjectCover} re-inserts what it touches, so the entry being written is never evicted.
 */
const boundCovers = (coverImageNames: Record<string, string>): Record<string, string> => {
  const entries = Object.entries(coverImageNames);

  return entries.length > MAX_TRACKED_COVERS ? Object.fromEntries(entries.slice(-MAX_TRACKED_COVERS)) : coverImageNames;
};

const loadFlight = createSingleFlight<void>();

/** Fetch the index once per account scope; concurrent calls share the request. */
export const loadProjectCovers = (): Promise<void> => {
  const owner = captureAccountScope();

  if (store.getSnapshot().isLoaded) {
    return Promise.resolve();
  }

  return loadFlight.run(`project-covers:${owner.epoch}`, async () => {
    let raw: string | null = null;

    try {
      raw = await getClientStateValue(PROJECT_COVERS_KEY, owner.signal);
    } catch {
      // Not "no covers yet": a failed read leaves the store unloaded so nothing writes over an
      // index we could not see. Pending records wait for the next refresh.
      return;
    }

    if (!isAccountScopeCurrent(owner)) {
      return;
    }

    const coverImageNames = parseProjectCovers(raw);
    let hasPendingChange = false;

    for (const [projectId, coverImageName] of pendingCoverNames) {
      if ((coverImageNames[projectId] ?? null) === coverImageName) {
        continue;
      }

      hasPendingChange = true;
      delete coverImageNames[projectId];

      if (coverImageName !== null) {
        coverImageNames[projectId] = coverImageName;
      }
    }

    pendingCoverNames.clear();

    const bounded = boundCovers(coverImageNames);

    store.setSnapshot({ coverImageNames: bounded, isDirty: hasPendingChange, isLoaded: true });

    if (hasPendingChange) {
      enqueuePersist(bounded, owner);
    }
  });
};

export const getProjectCoverImageName = (projectId: string): string | undefined =>
  store.getSnapshot().coverImageNames[projectId];

export const subscribeProjectCovers = store.subscribe;

/** Record (or clear) a project's cover. A no-op when nothing changed — the autosave case. */
export const recordProjectCover = (
  projectId: string,
  coverImageName: string | null,
  owner: AccountScope = captureAccountScope()
): void => {
  if (!isAccountScopeCurrent(owner)) {
    return;
  }

  const { coverImageNames, isDirty, isLoaded } = store.getSnapshot();

  if (isLoaded && (coverImageNames[projectId] ?? null) === coverImageName && !isDirty) {
    return;
  }

  const next = { ...coverImageNames };

  // Deleted before it is set, so the project moves to the end and survives the insertion-order
  // eviction below.
  delete next[projectId];

  if (coverImageName !== null) {
    next[projectId] = coverImageName;
  }

  const bounded = boundCovers(next);

  store.setSnapshot({ coverImageNames: bounded, isDirty: true, isLoaded });

  if (!isLoaded) {
    // The grid can show this immediately, but the KV cannot receive it until we
    // know what else is in there. See the module docblock.
    pendingCoverNames.set(projectId, coverImageName);
    void loadProjectCovers();

    return;
  }

  enqueuePersist(bounded, owner);
};

/** Drop a deleted project's entry so the blob does not accumulate dead ids. */
export const forgetProjectCover = (projectId: string, owner: AccountScope = captureAccountScope()): void => {
  recordProjectCover(projectId, null, owner);
};

/** Built here rather than imported from `@features/gallery`, which is private to that feature. */
export const getProjectCoverUrl = (coverImageName: string): string =>
  absolutizeApiUrl(`/api/v1/images/i/${encodeURIComponent(coverImageName)}/thumbnail`);
