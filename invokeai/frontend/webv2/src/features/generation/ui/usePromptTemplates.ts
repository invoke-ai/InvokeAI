import type { PromptTemplateDraft, PromptTemplateRecord } from '@features/generation/data/promptTemplates';

import {
  createPromptTemplate,
  deletePromptTemplate,
  exportPromptTemplates,
  fetchPromptTemplateImage,
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
  /**
   * The preview image behind a record's `imageUrl`, or null.
   *
   * On the port rather than imported straight from `data/` by the one component
   * that needs it: everything else here is injectable, and a single direct
   * import made the editor's tests mock a module to stand in for a dependency
   * they were otherwise handed.
   */
  fetchImage: (imageUrl: string) => Promise<Blob | null>;
}

/**
 * The shared prompt template catalog. Each mutation captures the identity that
 * started it, so a sign-out mid-flight does not invalidate the next account's
 * cache on the way back — the same rule `useWildcards` follows.
 */
export const usePromptTemplates = ({ isEnabled = true }: { isEnabled?: boolean } = {}): PromptTemplateCatalog => {
  const queryClient = useQueryClient();
  // Off by default for the widget, which only wants to re-read an applied
  // template and has nothing to re-read when none is applied. The picker asks
  // for it in earnest, and shares this cache when it does.
  const query = useQuery({ ...promptTemplatesQueryOptions(), enabled: isEnabled });
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
    fetchImage: fetchPromptTemplateImage,
    importFile,
    isLoading: query.isPending,
    remove,
    templates,
    update,
    userTemplates,
  };
};
