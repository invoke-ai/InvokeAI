/**
 * The supplementary cluster-labeling vocabulary: a server-wide term list that
 * admins maintain and the backend merges with its bundled vocabulary when it
 * builds the label embeddings.
 *
 * Reads are open to every user; the PUT is admin-only (the router's
 * `AdminUserOrDefault`). The list is replaced whole on save — the editor
 * always holds the full list, and replace semantics keep the API idempotent.
 */

import { assertAccountScopeCurrent, captureAccountScope } from '@platform/state/accountLifecycle';
import { apiFetchJson } from '@platform/transport/http';
import { queryOptions } from '@tanstack/react-query';

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
 */
export const updateImageMapVocab = async (terms: string[]): Promise<ImageMapVocab> => {
  const body = await apiFetchJson<BackendImageMapVocabResponse>('/api/v1/image_map/vocab', {
    body: JSON.stringify({ terms }),
    method: 'PUT',
  });

  return mapVocab(body);
};
