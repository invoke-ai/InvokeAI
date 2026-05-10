import type { paths } from 'services/api/schema';
import type { S } from 'services/api/types';

import { api, buildV1Url, LIST_TAG } from '..';

export type SystemPromptRecordDTO = S['SystemPromptRecordDTO'];
type SystemPromptWithoutId = S['SystemPromptWithoutId'];
type SystemPromptChanges = S['SystemPromptChanges'];

const buildSystemPromptsUrl = (path: string = '') => buildV1Url(`system_prompts/${path}`);

export const systemPromptsApi = api.injectEndpoints({
  endpoints: (build) => ({
    listSystemPrompts: build.query<
      paths['/api/v1/system_prompts/']['get']['responses']['200']['content']['application/json'],
      void
    >({
      query: () => ({ url: buildSystemPromptsUrl() }),
      providesTags: ['FetchOnReconnect', { type: 'SystemPrompt', id: LIST_TAG }],
    }),
    createSystemPrompt: build.mutation<
      paths['/api/v1/system_prompts/']['post']['responses']['200']['content']['application/json'],
      SystemPromptWithoutId
    >({
      query: (body) => ({
        url: buildSystemPromptsUrl(),
        method: 'POST',
        body,
      }),
      invalidatesTags: [{ type: 'SystemPrompt', id: LIST_TAG }],
    }),
    updateSystemPrompt: build.mutation<
      paths['/api/v1/system_prompts/i/{system_prompt_id}']['patch']['responses']['200']['content']['application/json'],
      { id: string; changes: SystemPromptChanges }
    >({
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
