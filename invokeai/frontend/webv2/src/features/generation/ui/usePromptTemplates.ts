import type { PromptTemplateSnapshot } from '@features/generation/core/promptTemplates';
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
  /**
   * The catalog has been read at least once, successfully. Distinct from
   * `!isLoading`, which is also true of a fetch that failed and of one that
   * never ran — telling a deleted template apart from an unread catalog needs
   * the difference.
   */
  isLoaded: boolean;
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
    isLoaded: query.isSuccess,
    isLoading: query.isPending,
    remove,
    templates,
    update,
    userTemplates,
  };
};

/**
 * Whether the applied template has been deleted out from under this tab.
 *
 * The snapshot keeps applying either way — a queue item has to explain itself
 * after its template is gone, and a catalog that is empty because it is still
 * loading or because the fetch failed must not silently change what generates.
 * That policy is right and stays; this is only what makes it visible.
 */
export const isPromptTemplateMissing = (
  catalog: PromptTemplateCatalog,
  active: PromptTemplateSnapshot | null
): boolean => active !== null && catalog.isLoaded && !catalog.templates.some((template) => template.id === active.id);
