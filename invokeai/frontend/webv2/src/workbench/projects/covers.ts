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
 * Which image stands for each project, as an index in the per-user client-state
 * KV — the same place the editor session lives (`session.ts`).
 *
 * The project document is the source of truth: `selectCoverImageName` derives a
 * cover from the newest result or the top-most canvas layer, and an `.invk`
 * carries those bytes as its `cover` entry. But the library grid lists projects
 * from `GET /projects`, which returns summaries with no document — deriving a
 * cover there would mean fetching every project's document to render a page of
 * thumbnails. So the answer is written down once, when a project is saved or
 * imported, and read back as one small blob.
 *
 * Being an index and not a truth, it is allowed to be stale or incomplete. A
 * project saved by a build without this has no entry and shows the folder glyph,
 * which is also what a project that has produced nothing shows — the absence is
 * already a state `ProjectCover` handles.
 *
 * Writes are skipped when the value has not changed, which is almost always: a
 * cover changes when a generation lands, not on every autosave.
 *
 * ### The whole index is one value, so a write must not precede a read
 *
 * The KV holds one blob and `setClientStateValue` replaces it outright. Writing
 * from a store that has not loaded would therefore not add an entry — it would
 * delete every entry but the one being written.
 *
 * That ordering is not hypothetical. Autosave reaches {@link recordProjectCover}
 * from the editor, which loads the index only when the project switcher or the
 * Open dialog is opened; someone who reloads straight into a project and
 * generates has never loaded it. So a record made before the load is held in
 * {@link pendingCoverNames}, applied on top of the server's answer once it
 * arrives, and persisted then. A read that *fails* is not a read: the store
 * stays unloaded and the pending records wait, because merging onto a blank
 * answer is the same destructive write by another route.
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

/**
 * Covers recorded before the index loaded, newest write per project. Drained by
 * {@link loadProjectCovers} — see the module docblock for why they cannot be
 * persisted where they are made.
 */
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
      // A cover is a thumbnail on a card. Losing the write costs a glyph until
      // the next save, which is not worth surfacing to anyone. Keeping the
      // snapshot dirty lets the next matching save retry the complete blob.
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
 * {@link recordProjectCover} re-inserts the project it touches, so the entry
 * being written can never be the one evicted.
 */
const boundCovers = (coverImageNames: Record<string, string>): Record<string, string> => {
  const entries = Object.entries(coverImageNames);

  return entries.length > MAX_TRACKED_COVERS ? Object.fromEntries(entries.slice(-MAX_TRACKED_COVERS)) : coverImageNames;
};

const loadFlight = createSingleFlight<void>();

/** Fetch the index once per account scope; concurrent calls share the request. */
export const loadProjectCovers = (): Promise<void> => {
  const owner = captureAccountScope();

  return loadFlight.run(`project-covers:${owner.epoch}`, async () => {
    let raw: string | null = null;

    try {
      raw = await getClientStateValue(PROJECT_COVERS_KEY, owner.signal);
    } catch {
      // Not "no covers yet": a failed read leaves the store unloaded so that
      // nothing writes over an index we were unable to see. The next refresh
      // tries again, and any pending records wait for it.
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

/**
 * Record (or clear) a project's cover. A no-op when nothing changed, which is
 * the common case on autosave.
 */
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

  // Deleted before it is set, so an existing project moves to the end: the cap
  // below evicts in insertion order, and the entry being written now is the one
  // that must survive it.
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

/**
 * Thumbnail URL for a cover image name. Built here rather than imported from
 * `@features/gallery/core/imagePaths`, which is private to that feature; the
 * palette's provider does the same for the same reason.
 */
export const getProjectCoverUrl = (coverImageName: string): string =>
  absolutizeApiUrl(`/api/v1/images/i/${encodeURIComponent(coverImageName)}/thumbnail`);
