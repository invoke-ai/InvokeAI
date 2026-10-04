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
 * The cover index is nonauthoritative. Load it before replacing the whole blob; failed reads must not unlock
 * writes.
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
      // Retain dirty cover state after a failed write for retry.
      return;
    }

    if (mutationQueue !== queue || !isAccountScopeCurrent(owner) || queue.latestRevision !== revision) {
      return;
    }

    store.setSnapshot({ ...store.getSnapshot(), isDirty: false });
  });
};

/** Insertion order records recency because updates reinsert entries. */
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
      // Failed reads leave the index unloaded so pending covers cannot overwrite unseen entries.
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

  // Reinsert updated covers at the newest eviction position.
  delete next[projectId];

  if (coverImageName !== null) {
    next[projectId] = coverImageName;
  }

  const bounded = boundCovers(next);

  store.setSnapshot({ coverImageNames: bounded, isDirty: true, isLoaded });

  if (!isLoaded) {
    // Display local covers immediately; durable writes must await index loading.
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
