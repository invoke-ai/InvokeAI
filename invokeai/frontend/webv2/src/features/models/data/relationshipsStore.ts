import {
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';
import { areArraysEqual } from '@platform/state/selectors';

import { addModelRelationship, getRelatedModelKeys, removeModelRelationship } from './api';

/**
 * Shared cache of bidirectional "related model" links, keyed by model key.
 * One store keeps the detail pane and every open picker consistent: link and
 * unlink patch both cached directions instead of refetching.
 *
 * The backend's batch endpoint (`POST /model_relationships/batch`) is
 * deliberately unused — it returns a flat de-duplicated union across the input
 * keys, which cannot populate a per-key cache. Per-key fetches through
 * `ensureRelatedModelKeysLoaded` stay attributable and dedupe on the inflight
 * map.
 */

export interface ModelRelationshipsSnapshot {
  /** Related keys per model key; a missing entry means "not fetched yet". */
  relatedKeysByModelKey: Readonly<Record<string, readonly string[]>>;
}

const EMPTY_RELATIONSHIPS_SNAPSHOT: ModelRelationshipsSnapshot = { relatedKeysByModelKey: {} };
const store = createExternalStore<ModelRelationshipsSnapshot>(EMPTY_RELATIONSHIPS_SNAPSHOT);

/**
 * A fetch may only write its result while it is still the promise in this map.
 * Evicting a key therefore both invalidates its inflight GET and unblocks the
 * dedupe so a follow-up fetch genuinely starts.
 */
const inflightByKey = new Map<string, Promise<void>>();

/** Keys whose last settled fetch rejected; `ensure` revalidates these. */
const failedFetchKeys = new Set<string>();

registerAccountOwnedResource({
  clear: () => {
    inflightByKey.clear();
    failedFetchKeys.clear();
    store.setSnapshot(EMPTY_RELATIONSHIPS_SNAPSHOT);
  },
  name: 'model-relationships',
});

const setEntry = (modelKey: string, keys: readonly string[]): void => {
  store.patchSnapshot({
    relatedKeysByModelKey: { ...store.getSnapshot().relatedKeysByModelKey, [modelKey]: keys },
  });
};

const fetchRelatedKeys = (modelKey: string): Promise<void> => {
  const existing = inflightByKey.get(modelKey);

  if (existing) {
    return existing;
  }

  const owner = captureAccountScope();
  const fetch = getRelatedModelKeys(modelKey, owner.signal)
    .then((keys) => {
      if (isAccountScopeCurrent(owner) && inflightByKey.get(modelKey) === fetch) {
        failedFetchKeys.delete(modelKey);
        setEntry(modelKey, keys);
      }
    })
    .catch((error: unknown) => {
      // Leave an empty entry so consumers stop showing a spinner, but still
      // reject so callers that care (the detail pane) can surface the error.
      // A superseded fetch writes nothing and does not mark the key failed.
      if (isAccountScopeCurrent(owner) && inflightByKey.get(modelKey) === fetch) {
        failedFetchKeys.add(modelKey);

        if (store.getSnapshot().relatedKeysByModelKey[modelKey] === undefined) {
          setEntry(modelKey, []);
        }
      }
      throw error;
    })
    .finally(() => {
      if (inflightByKey.get(modelKey) === fetch) {
        inflightByKey.delete(modelKey);
      }
    });

  inflightByKey.set(modelKey, fetch);
  return fetch;
};

/**
 * Fetch once per key; concurrent callers share one request. Cache-first,
 * except that a key whose last fetch failed is revalidated on the next call.
 */
export const ensureRelatedModelKeysLoaded = (modelKey: string): Promise<void> => {
  if (store.getSnapshot().relatedKeysByModelKey[modelKey] !== undefined && !failedFetchKeys.has(modelKey)) {
    return inflightByKey.get(modelKey) ?? Promise.resolve();
  }

  return fetchRelatedKeys(modelKey);
};

/** Stale-while-revalidate: any cached entry keeps rendering while it refetches. */
export const refreshRelatedModelKeys = (modelKey: string): Promise<void> => fetchRelatedKeys(modelKey);

type EntryPatch = (entry: readonly string[], otherKey: string) => readonly string[];

/** Links are bidirectional: apply the patch to both cached directions. */
const patchBothDirections = (modelKey: string, otherKey: string, patch: EntryPatch): void => {
  const next = { ...store.getSnapshot().relatedKeysByModelKey };
  let changed = false;

  for (const [key, other] of [
    [modelKey, otherKey],
    [otherKey, modelKey],
  ] as const) {
    const entry = next[key];

    if (entry !== undefined) {
      next[key] = patch(entry, other);
      changed = true;
    }
  }

  if (changed) {
    store.patchSnapshot({ relatedKeysByModelKey: next });
  }
};

/**
 * Apply a successful mutation to the cache: patch the entries that exist, and
 * for any key with a GET inflight, evict it (a request issued before the
 * mutation is stale) and refetch so the entry converges on server truth. This
 * also covers a link made while the entry's first fetch is still inflight —
 * the refetch was issued after the POST, so it includes the new link.
 */
const commitMutation = (modelKey: string, otherKey: string, patch: EntryPatch): void => {
  patchBothDirections(modelKey, otherKey, patch);

  for (const key of [modelKey, otherKey]) {
    if (inflightByKey.delete(key)) {
      void fetchRelatedKeys(key).catch(() => {});
    }
  }
};

export const linkModels = async (modelKey: string, otherKey: string): Promise<void> => {
  const owner = captureAccountScope();

  await addModelRelationship(modelKey, otherKey, owner.signal);

  if (isAccountScopeCurrent(owner)) {
    commitMutation(modelKey, otherKey, (entry, other) => (entry.includes(other) ? entry : [...entry, other]));
  }
};

export const unlinkModels = async (modelKey: string, otherKey: string): Promise<void> => {
  const owner = captureAccountScope();

  await removeModelRelationship(modelKey, otherKey, owner.signal);

  if (isAccountScopeCurrent(owner)) {
    commitMutation(modelKey, otherKey, (entry, other) => entry.filter((key) => key !== other));
  }
};

/** Drop deleted models' entries and scrub their keys from the remaining ones. */
export const removeModelsFromRelationships = (keys: readonly string[]): void => {
  for (const key of keys) {
    // A late GET for a deleted model must not resurrect its entry or retry.
    inflightByKey.delete(key);
    failedFetchKeys.delete(key);
  }

  const removed = new Set(keys);
  const next: Record<string, readonly string[]> = {};

  for (const [key, entry] of Object.entries(store.getSnapshot().relatedKeysByModelKey)) {
    if (!removed.has(key)) {
      next[key] = entry.some((related) => removed.has(related))
        ? entry.filter((related) => !removed.has(related))
        : entry;
    }
  }

  store.patchSnapshot({ relatedKeysByModelKey: next });
};

export const getRelationshipsSnapshot = (): ModelRelationshipsSnapshot => store.getSnapshot();

/** Related keys for one model; `null` means not fetched yet (or no model). */
export const useRelatedModelKeys = (modelKey: string | null): readonly string[] | null =>
  store.useSelector(
    (snapshot) => (modelKey === null ? null : (snapshot.relatedKeysByModelKey[modelKey] ?? null)),
    (left, right) => (left === null || right === null ? left === right : areArraysEqual(left, right))
  );

export const setRelationshipsSnapshotForTests = (next: Partial<ModelRelationshipsSnapshot>): void => {
  store.patchSnapshot(next);
};
