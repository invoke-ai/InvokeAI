/**
 * Admin-maintained server-wide supplementary terms augment bundled label vocabulary. All users can read;
 * admin-only saves replace the full list idempotently.
 */

import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { apiFetchJson } from '@platform/transport/http';
import { queryOptions } from '@tanstack/react-query';

import { clearImageLabels } from './imageLabelCache';
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
 * Replace all terms; server normalizes/dedupes and returns stored values or term-specific 422 limits. Background
 * rebuild watching outlives the settings dialog and refreshes map labels even when points do not move.
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

// Back off rebuild polling because generation pauses can delay completion substantially.
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
          // Vocabulary rebuild also invalidates cached per-item hover tags.
          clearImageLabels();
          refetchClusterLabels();
        }

        // Error, unavailable and idle end this rebuild's label wait; settings owns their feedback.
        return;
      }
    } catch {
      // Label refresh is best-effort; the next point refresh fetches labels again.
    } finally {
      rebuildWatchActive = false;
    }
  })();
};
