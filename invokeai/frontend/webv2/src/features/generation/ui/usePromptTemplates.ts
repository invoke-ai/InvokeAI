import type { PromptTemplateSnapshot } from '@features/generation/core/promptTemplates';
import type {
  PromptTemplateCreateDraft,
  PromptTemplateRecord,
  PromptTemplateUpdateDraft,
} from '@features/generation/data/promptTemplates';

import { classifyPromptTemplates, requireOwnedPromptTemplate } from '@features/generation/core/promptTemplateOwnership';
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

import { useGenerationUi } from './GenerationUiContext';

export interface PromptTemplateCatalog {
  /** The user's own templates, name-ascending as the backend orders them. */
  personalTemplates: PromptTemplateRecord[];
  /** Public templates owned by somebody else; applicable but read-only. */
  sharedTemplates: PromptTemplateRecord[];
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
  create: (draft: PromptTemplateCreateDraft) => Promise<PromptTemplateRecord>;
  update: (template: PromptTemplateRecord, draft: PromptTemplateUpdateDraft) => Promise<PromptTemplateRecord>;
  remove: (template: PromptTemplateRecord) => Promise<void>;
  importFile: (file: File) => Promise<void>;
  exportCsv: () => Promise<Blob>;
}

/**
 * The shared prompt template catalog. Each mutation captures the identity that
 * started it, so a sign-out mid-flight does not invalidate the next account's
 * cache on the way back — the same rule `useWildcards` follows.
 */
export const usePromptTemplates = ({ isEnabled = true }: { isEnabled?: boolean } = {}): PromptTemplateCatalog => {
  const queryClient = useQueryClient();
  const { account } = useGenerationUi();
  // Off by default for the widget, which only wants to re-read an applied
  // template and has nothing to re-read when none is applied. The picker asks
  // for it in earnest, and shares this cache when it does.
  const query = useQuery({ ...promptTemplatesQueryOptions(), enabled: isEnabled });
  const classified = useMemo(() => classifyPromptTemplates(query.data ?? [], account), [account, query.data]);

  const runAndInvalidate = useCallback(
    async <T>(run: () => Promise<T>, imageId?: string): Promise<T> => {
      const owner = captureAccountScope();
      const result = await run();

      assertAccountScopeCurrent(owner);
      await invalidatePromptTemplates(queryClient, imageId);
      return result;
    },
    [queryClient]
  );

  const create = useCallback(
    async (draft: PromptTemplateCreateDraft) => {
      return await runAndInvalidate(() => createPromptTemplate(draft));
    },
    [runAndInvalidate]
  );

  const update = useCallback(
    (template: PromptTemplateRecord, draft: PromptTemplateUpdateDraft) => {
      requireOwnedPromptTemplate(template, account);
      return runAndInvalidate(() => updatePromptTemplate(template.id, draft), template.id);
    },
    [account, runAndInvalidate]
  );

  const remove = useCallback(
    (template: PromptTemplateRecord) => {
      requireOwnedPromptTemplate(template, account);
      return runAndInvalidate(() => deletePromptTemplate(template.id), template.id);
    },
    [account, runAndInvalidate]
  );

  const importFile = useCallback(
    (file: File) => runAndInvalidate(() => importPromptTemplates(file)),
    [runAndInvalidate]
  );

  return {
    create,
    defaultTemplates: classified.defaultTemplates,
    exportCsv: exportPromptTemplates,
    importFile,
    isLoaded: query.isSuccess,
    isLoading: query.isPending,
    remove,
    personalTemplates: classified.personalTemplates,
    sharedTemplates: classified.sharedTemplates,
    templates: classified.templates,
    update,
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
