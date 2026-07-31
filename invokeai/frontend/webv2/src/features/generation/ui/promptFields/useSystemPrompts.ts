import type { SystemPromptDraft, SystemPromptRecord } from '@features/generation/data/systemPrompts';

import {
  classifySystemPrompts,
  isOwnedSystemPrompt,
  requireOwnedSystemPrompt,
} from '@features/generation/core/systemPrompts';
import {
  createSystemPrompt,
  deleteSystemPrompt,
  invalidateSystemPrompts,
  systemPromptsQueryOptions,
  updateSystemPrompt,
} from '@features/generation/data/systemPrompts';
import { useGenerationUi } from '@features/generation/ui/GenerationUiContext';
import { assertAccountScopeCurrent, captureAccountScope } from '@platform/state/accountLifecycle';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useCallback, useMemo } from 'react';

export interface SystemPromptCatalog {
  personalPrompts: SystemPromptRecord[];
  sharedPrompts: SystemPromptRecord[];
  prompts: SystemPromptRecord[];
  isLoading: boolean;
  isLoaded: boolean;
  canEdit: (prompt: SystemPromptRecord) => boolean;
  create: (draft: SystemPromptDraft) => Promise<SystemPromptRecord>;
  update: (prompt: SystemPromptRecord, draft: SystemPromptDraft) => Promise<SystemPromptRecord>;
  remove: (prompt: SystemPromptRecord) => Promise<void>;
}

export const useSystemPrompts = ({ isEnabled = true }: { isEnabled?: boolean } = {}): SystemPromptCatalog => {
  const queryClient = useQueryClient();
  const { account } = useGenerationUi();
  const query = useQuery({ ...systemPromptsQueryOptions(), enabled: isEnabled });
  const classified = useMemo(() => classifySystemPrompts(query.data ?? [], account), [account, query.data]);

  const runAndInvalidate = useCallback(
    async <T>(run: () => Promise<T>): Promise<T> => {
      const owner = captureAccountScope();
      const result = await run();

      assertAccountScopeCurrent(owner);
      await invalidateSystemPrompts(queryClient);
      return result;
    },
    [queryClient]
  );

  const create = useCallback(
    (draft: SystemPromptDraft) => runAndInvalidate(() => createSystemPrompt(draft)),
    [runAndInvalidate]
  );

  const update = useCallback(
    (prompt: SystemPromptRecord, draft: SystemPromptDraft) => {
      // The backend rejects this too; failing here keeps a shared prompt's disabled edit
      // affordance from turning into a round-trip 403 if it is ever mis-rendered.
      requireOwnedSystemPrompt(prompt, account);
      return runAndInvalidate(() => updateSystemPrompt(prompt.id, draft));
    },
    [account, runAndInvalidate]
  );

  const remove = useCallback(
    (prompt: SystemPromptRecord) => {
      requireOwnedSystemPrompt(prompt, account);
      return runAndInvalidate(() => deleteSystemPrompt(prompt.id));
    },
    [account, runAndInvalidate]
  );

  const canEdit = useCallback((prompt: SystemPromptRecord) => isOwnedSystemPrompt(prompt, account), [account]);

  return {
    canEdit,
    create,
    isLoaded: query.isSuccess,
    isLoading: query.isPending,
    personalPrompts: classified.personalPrompts,
    prompts: classified.prompts,
    remove,
    sharedPrompts: classified.sharedPrompts,
    update,
  };
};
