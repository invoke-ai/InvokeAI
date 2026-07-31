import type { QueryClient } from '@tanstack/react-query';

import { assertAccountScopeCurrent, captureAccountScope } from '@platform/state/accountLifecycle';
import { apiFetch, apiFetchJson } from '@platform/transport/http';
import { queryOptions } from '@tanstack/react-query';

const SYSTEM_PROMPTS_BASE = '/api/v1/system_prompts';

interface SystemPromptDTO {
  id: string;
  name: string;
  content: string;
  user_id: string;
  is_public: boolean;
}

export interface SystemPromptRecord {
  id: string;
  name: string;
  content: string;
  userId: string;
  isPublic: boolean;
}

export interface SystemPromptDraft {
  name: string;
  content: string;
}

export const systemPromptKeys = {
  all: ['generation', 'systemPrompts'] as const,
  list: () => [...systemPromptKeys.all, 'list'] as const,
};

const mapSystemPrompt = (dto: SystemPromptDTO): SystemPromptRecord => ({
  content: dto.content,
  id: dto.id,
  isPublic: dto.is_public,
  name: dto.name,
  userId: dto.user_id,
});

export const systemPromptsQueryOptions = () =>
  (() => {
    const owner = captureAccountScope();

    return queryOptions({
      queryFn: async ({ signal }): Promise<SystemPromptRecord[]> => {
        const prompts = await apiFetchJson<SystemPromptDTO[]>(`${SYSTEM_PROMPTS_BASE}/`, {
          signal: AbortSignal.any([signal, owner.signal]),
        });

        assertAccountScopeCurrent(owner);
        return prompts.map(mapSystemPrompt);
      },
      queryKey: systemPromptKeys.list(),
      staleTime: 30_000,
    });
  })();

export const createSystemPrompt = async (draft: SystemPromptDraft): Promise<SystemPromptRecord> =>
  mapSystemPrompt(
    await apiFetchJson<SystemPromptDTO>(`${SYSTEM_PROMPTS_BASE}/`, {
      body: JSON.stringify({ content: draft.content, name: draft.name }),
      method: 'POST',
    })
  );

export const updateSystemPrompt = async (id: string, draft: SystemPromptDraft): Promise<SystemPromptRecord> =>
  mapSystemPrompt(
    await apiFetchJson<SystemPromptDTO>(`${SYSTEM_PROMPTS_BASE}/i/${encodeURIComponent(id)}`, {
      body: JSON.stringify({ content: draft.content, name: draft.name }),
      method: 'PATCH',
    })
  );

export const deleteSystemPrompt = async (id: string): Promise<void> => {
  await apiFetch(`${SYSTEM_PROMPTS_BASE}/i/${encodeURIComponent(id)}`, { method: 'DELETE' });
};

export const invalidateSystemPrompts = async (queryClient: QueryClient): Promise<void> => {
  await queryClient.invalidateQueries({ exact: true, queryKey: systemPromptKeys.list() });
};
