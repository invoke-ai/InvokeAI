/**
 * modelRelationships.ts
 *
 * RTK Query API slice for managing model-to-model relationships.
 *
 * Endpoints provided:
 * - Fetch related models for a single model
 * - Add a relationship between two models
 * - Remove a relationship between two models
 * - Fetch related models for multiple models in batch
 *
 * Provides and invalidates cache tags for seamless UI updates after add/remove operations.
 */

import { api } from '..';

const REL_TAG = 'ModelRelationships'; // Needed for UI updates on relationship changes.

const modelRelationshipsApi = api.injectEndpoints({
  endpoints: (build) => ({
    getRelatedModelIds: build.query<string[], string>({
      query: (model_key) => `/api/v1/model_relationships/i/${model_key}`,
      providesTags: (result, error, model_key) => [{ type: REL_TAG, id: model_key }],
    }),

    addModelRelationship: build.mutation<void, { model_key_1: string; model_key_2: string }>({
      query: (payload) => ({
        url: `/api/v1/model_relationships/`,
        method: 'POST',
        body: payload,
      }),
      invalidatesTags: (result, error, { model_key_1, model_key_2 }) => [
        { type: REL_TAG, id: model_key_1 },
        { type: REL_TAG, id: model_key_2 },
      ],
    }),

    removeModelRelationship: build.mutation<void, { model_key_1: string; model_key_2: string }>({
      query: (payload) => ({
        url: `/api/v1/model_relationships/`,
        method: 'DELETE',
        body: payload,
      }),
      invalidatesTags: (result, error, { model_key_1, model_key_2 }) => [
        { type: REL_TAG, id: model_key_1 },
        { type: REL_TAG, id: model_key_2 },
      ],
    }),

    getRelatedModelIdsBatch: build.query<string[], string[]>({
      query: (model_keys) => ({
        url: `/api/v1/model_relationships/batch`,
        method: 'POST',
        body: { model_keys },
      }),
      providesTags: (result, error, model_keys) => model_keys.map((key) => ({ type: 'ModelRelationships', id: key })),
    }),
  }),
  overrideExisting: false,
});

export const {
  useGetRelatedModelIdsQuery,
  useAddModelRelationshipMutation,
  useRemoveModelRelationshipMutation,
  useGetRelatedModelIdsBatchQuery,
} = modelRelationshipsApi;
