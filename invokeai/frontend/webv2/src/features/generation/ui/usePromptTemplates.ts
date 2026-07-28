import type { PromptTemplateDraft, PromptTemplateRecord } from '@features/generation/data/promptTemplates';

import {
  createPromptTemplate,
  deletePromptTemplate,
  exportPromptTemplates,
  importPromptTemplates,
  invalidatePromptTemplates,
  promptTemplatesQueryOptions,
  updatePromptTemplate,
} from '@features/generation/data/promptTemplates';
import { assertAccountScopeCurrent, captureAccountScope } from '@platform/state/accountLifecycle';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useCallback, useMemo } from 'react';

export interface PromptTemplateCatalog {
  /** The user's own templates, name-ascending as the backend orders them. */
  userTemplates: PromptTemplateRecord[];
  /** Templates shipped with the backend; applicable but not editable. */
  defaultTemplates: PromptTemplateRecord[];
  /** Every visible template, for resolving a stored snapshot. */
  templates: PromptTemplateRecord[];
  isLoading: boolean;
  /** Resolves with the saved record so a caller can refresh an applied snapshot. */
  create: (draft: PromptTemplateDraft) => Promise<PromptTemplateRecord>;
  update: (id: string, draft: PromptTemplateDraft) => Promise<PromptTemplateRecord>;
  remove: (id: string) => Promise<void>;
  importFile: (file: File) => Promise<void>;
  exportCsv: () => Promise<Blob>;
}

/**
 * The shared prompt template catalog. Each mutation captures the identity that
 * started it, so a sign-out mid-flight does not invalidate the next account's
 * cache on the way back — the same rule `useWildcards` follows.
 */
export const usePromptTemplates = (): PromptTemplateCatalog => {
  const queryClient = useQueryClient();
  const query = useQuery(promptTemplatesQueryOptions());
  const templates = useMemo(() => query.data ?? [], [query.data]);
  const defaultTemplates = useMemo(() => templates.filter((template) => template.isDefault), [templates]);
  const userTemplates = useMemo(() => templates.filter((template) => !template.isDefault), [templates]);

  const runAndInvalidate = useCallback(
    async <T>(run: () => Promise<T>): Promise<T> => {
      const owner = captureAccountScope();
      const result = await run();

      assertAccountScopeCurrent(owner);
      await invalidatePromptTemplates(queryClient);
      return result;
    },
    [queryClient]
  );

  const create = useCallback(
    (draft: PromptTemplateDraft) => runAndInvalidate(() => createPromptTemplate(draft)),
    [runAndInvalidate]
  );

  const update = useCallback(
    (id: string, draft: PromptTemplateDraft) => runAndInvalidate(() => updatePromptTemplate(id, draft)),
    [runAndInvalidate]
  );

  const remove = useCallback((id: string) => runAndInvalidate(() => deletePromptTemplate(id)), [runAndInvalidate]);

  const importFile = useCallback(
    (file: File) => runAndInvalidate(() => importPromptTemplates(file)),
    [runAndInvalidate]
  );

  return {
    create,
    defaultTemplates,
    exportCsv: exportPromptTemplates,
    importFile,
    isLoading: query.isPending,
    remove,
    templates,
    update,
    userTemplates,
  };
};
