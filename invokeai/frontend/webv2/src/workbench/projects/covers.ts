import {
  assertAccountScopeCurrent,
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
 */

export const PROJECT_COVERS_KEY = 'webv2:project-covers';

/** Cap on tracked projects, so a long-lived account cannot grow the blob without bound. */
const MAX_TRACKED_COVERS = 500;

interface ProjectCoversSnapshot {
  /** Project id to the server image name of its cover. */
  coverImageNames: Record<string, string>;
  isLoaded: boolean;
}

const EMPTY: ProjectCoversSnapshot = { coverImageNames: {}, isLoaded: false };
const store = createExternalStore<ProjectCoversSnapshot>(EMPTY);

registerAccountOwnedResource({
  clear: () => {
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

const persist = (coverImageNames: Record<string, string>, owner: AccountScope): void => {
  void setClientStateValue(PROJECT_COVERS_KEY, JSON.stringify(coverImageNames), owner.signal).catch(() => {
    // A cover is a thumbnail on a card. Losing the write costs a glyph until
    // the next save, which is not worth surfacing to anyone.
  });
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
      // Treated as "no covers yet" — see the module docblock.
    }

    if (!isAccountScopeCurrent(owner)) {
      return;
    }

    store.setSnapshot({ coverImageNames: parseProjectCovers(raw), isLoaded: true });
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

  const { coverImageNames } = store.getSnapshot();

  if ((coverImageNames[projectId] ?? null) === coverImageName) {
    return;
  }

  const next = { ...coverImageNames };

  if (coverImageName === null) {
    delete next[projectId];
  } else {
    next[projectId] = coverImageName;
  }

  const entries = Object.entries(next);
  const bounded = entries.length > MAX_TRACKED_COVERS ? Object.fromEntries(entries.slice(-MAX_TRACKED_COVERS)) : next;

  store.setSnapshot({ coverImageNames: bounded, isLoaded: true });
  persist(bounded, owner);
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

/** Load the index and report whether it changed anything, for the library's refresh pass. */
export const ensureProjectCoversLoaded = async (owner: AccountScope): Promise<void> => {
  await loadProjectCovers();
  assertAccountScopeCurrent(owner);
};
