import type { S } from 'services/api/types';

import { api, buildV1Url, LIST_TAG } from '..';

// NOTE: schema.ts is regenerated separately. Until that lands, we manually augment the
// generated DTOs with the multi-user fields the backend already returns/accepts. After a
// `pnpm typegen` run that includes migration 32's schema, these intersections become no-ops
// and can be removed.
export type SystemPromptRecordDTO = S['SystemPromptRecordDTO'] & {
  user_id: string;
  is_public: boolean;
};
type SystemPromptWithoutId = S['SystemPromptWithoutId'];
type SystemPromptChanges = S['SystemPromptChanges'] & {
  is_public?: boolean | null;
};

const buildSystemPromptsUrl = (path: string = '') => buildV1Url(`system_prompts/${path}`);

export const systemPromptsApi = api.injectEndpoints({
  endpoints: (build) => ({
    listSystemPrompts: build.query<SystemPromptRecordDTO[], void>({
      query: () => ({ url: buildSystemPromptsUrl() }),
      providesTags: ['FetchOnReconnect', { type: 'SystemPrompt', id: LIST_TAG }],
    }),
    createSystemPrompt: build.mutation<SystemPromptRecordDTO, SystemPromptWithoutId>({
      query: (body) => ({
        url: buildSystemPromptsUrl(),
        method: 'POST',
        body,
      }),
      invalidatesTags: [{ type: 'SystemPrompt', id: LIST_TAG }],
    }),
    updateSystemPrompt: build.mutation<SystemPromptRecordDTO, { id: string; changes: SystemPromptChanges }>({
      query: ({ id, changes }) => ({
        url: buildSystemPromptsUrl(`i/${id}`),
        method: 'PATCH',
        body: changes,
      }),
      invalidatesTags: (_result, _error, { id }) => [
        { type: 'SystemPrompt', id: LIST_TAG },
        { type: 'SystemPrompt', id },
      ],
    }),
    deleteSystemPrompt: build.mutation<void, string>({
      query: (id) => ({
        url: buildSystemPromptsUrl(`i/${id}`),
        method: 'DELETE',
      }),
      invalidatesTags: (_result, _error, id) => [
        { type: 'SystemPrompt', id: LIST_TAG },
        { type: 'SystemPrompt', id },
      ],
    }),
  }),
});

export const {
  useListSystemPromptsQuery,
  useCreateSystemPromptMutation,
  useUpdateSystemPromptMutation,
  useDeleteSystemPromptMutation,
} = systemPromptsApi;
