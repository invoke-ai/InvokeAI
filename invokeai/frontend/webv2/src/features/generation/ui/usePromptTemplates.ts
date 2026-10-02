import type { PromptTemplateSnapshot } from '@features/generation/core/promptTemplates';
import type {
  PromptTemplateCreateDraft,
  PromptTemplateRecord,
  PromptTemplateUpdateDraft,
} from '@features/generation/data/promptTemplates';
import type { AccountScope } from '@platform/state/accountLifecycle';

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
  personalTemplates: PromptTemplateRecord[];
  sharedTemplates: PromptTemplateRecord[];
  defaultTemplates: PromptTemplateRecord[];
  templates: PromptTemplateRecord[];
  isLoading: boolean;
  isLoaded: boolean;
  create: (draft: PromptTemplateCreateDraft) => Promise<PromptTemplateRecord>;
  update: (template: PromptTemplateRecord, draft: PromptTemplateUpdateDraft) => Promise<PromptTemplateRecord>;
  remove: (template: PromptTemplateRecord) => Promise<void>;
  importFile: (file: File, owner: AccountScope) => Promise<void>;
  exportCsv: (owner: AccountScope) => Promise<Blob>;
}

export const usePromptTemplates = ({ isEnabled = true }: { isEnabled?: boolean } = {}): PromptTemplateCatalog => {
  const queryClient = useQueryClient();
  const { account } = useGenerationUi();
  const query = useQuery({ ...promptTemplatesQueryOptions(), enabled: isEnabled });
  const classified = useMemo(() => classifyPromptTemplates(query.data ?? [], account), [account, query.data]);

  const runAndInvalidate = useCallback(
    async <T>(run: () => Promise<T>, imageId?: string, owner: AccountScope = captureAccountScope()): Promise<T> => {
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
    (file: File, owner: AccountScope) => runAndInvalidate(() => importPromptTemplates(file, owner), undefined, owner),
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

export const isPromptTemplateMissing = (
  catalog: PromptTemplateCatalog,
  active: PromptTemplateSnapshot | null
): boolean => active !== null && catalog.isLoaded && !catalog.templates.some((template) => template.id === active.id);
