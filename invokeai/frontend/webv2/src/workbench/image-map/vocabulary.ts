/**
 * The supplementary cluster-labeling vocabulary: a server-wide term list that
 * admins maintain and the backend merges with its bundled vocabulary when it
 * builds the label embeddings.
 *
 * Reads are open to every user; the PUT is admin-only (the router's
 * `AdminUserOrDefault`). The list is replaced whole on save — the editor
 * always holds the full list, and replace semantics keep the API idempotent.
 */

import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { apiFetchJson } from '@platform/transport/http';
import { queryOptions } from '@tanstack/react-query';

import { refetchClusterLabels } from './imageMapStore';

/** Mirrors the backend's VocabBuildState literal. */
export type ImageMapVocabState = 'unavailable' | 'idle' | 'building' | 'ready' | 'error';

export interface ImageMapVocab {
  /** The stored terms, normalized and sorted alphabetically by the server. */
  terms: string[];
  /** The label-embedding build's state; 'building' after a save until the worker rebuilds. */
  state: ImageMapVocabState;
  /** Why the last embedding build failed; only set when state is 'error'. */
  error: string | null;
  maxTerms: number;
  maxTermLength: number;
}

interface BackendImageMapVocabResponse {
  terms: string[];
  state: ImageMapVocabState;
  error?: string | null;
  max_terms: number;
  max_term_length: number;
}

const mapVocab = (body: BackendImageMapVocabResponse): ImageMapVocab => ({
  error: body.error ?? null,
  maxTerms: body.max_terms,
  maxTermLength: body.max_term_length,
  state: body.state,
  terms: body.terms,
});

export const imageMapVocabKeys = {
  all: ['image-map', 'vocab'] as const,
};

export const imageMapVocabQueryOptions = () =>
  (() => {
    const owner = captureAccountScope();

    return queryOptions({
      queryFn: async ({ signal }): Promise<ImageMapVocab> => {
        const requestSignal = AbortSignal.any([signal, owner.signal]);
        const body = await apiFetchJson<BackendImageMapVocabResponse>('/api/v1/image_map/vocab', {
          signal: requestSignal,
        });

        assertAccountScopeCurrent(owner);
        return mapVocab(body);
      },
      queryKey: imageMapVocabKeys.all,
      staleTime: 5_000,
    });
  })();

/**
 * Replace the whole term list. The server normalizes (lowercase, collapsed
 * whitespace), dedupes, and rejects over-limit input with a 422 whose detail
 * names the offending term; the response carries the stored list.
 *
 * When the response reports a background embedding rebuild, a module-level
 * watcher is started so any open map's labels are re-fetched when it lands —
 * a vocabulary edit changes what the labels say without moving a single
 * point, so nothing in the map's own flow would notice, and the watcher must
 * outlive the settings dialog the save was made from.
 */
export const updateImageMapVocab = async (terms: string[]): Promise<ImageMapVocab> => {
  const body = await apiFetchJson<BackendImageMapVocabResponse>('/api/v1/image_map/vocab', {
    body: JSON.stringify({ terms }),
    method: 'PUT',
  });
  const vocab = mapVocab(body);

  if (vocab.state === 'building') {
    watchRebuild();
  }

  return vocab;
};

// The rebuild usually lands in seconds, but the index worker parks during
// generations, so it can also take a long while: back the poll off rather
// than holding a 2s cadence indefinitely.
const REBUILD_POLL_MIN_MS = 2_000;
const REBUILD_POLL_MAX_MS = 30_000;

let rebuildWatchActive = false;

const delay = (ms: number): Promise<void> =>
  new Promise((resolve) => {
    setTimeout(resolve, ms);
  });

const watchRebuild = (): void => {
  // One watcher covers every save it overlaps: it acts on the final state,
  // not on which save produced it.
  if (rebuildWatchActive) {
    return;
  }

  rebuildWatchActive = true;
  const owner = captureAccountScope();

  void (async () => {
    try {
      let interval = REBUILD_POLL_MIN_MS;

      for (;;) {
        await delay(interval);
        interval = Math.min(interval * 2, REBUILD_POLL_MAX_MS);

        if (!isAccountScopeCurrent(owner)) {
          return;
        }

        const body = await apiFetchJson<BackendImageMapVocabResponse>('/api/v1/image_map/vocab');

        if (!isAccountScopeCurrent(owner)) {
          return;
        }

        if (body.state === 'building') {
          continue;
        }

        if (body.state === 'ready') {
          refetchClusterLabels();
        }

        // 'error', 'unavailable', and 'idle' all mean no new labels are
        // coming from this rebuild; the settings UI reports those itself.
        return;
      }
    } catch {
      // Label refresh is best-effort decoration; the next points refresh
      // fetches labels anyway.
    } finally {
      rebuildWatchActive = false;
    }
  })();
};
