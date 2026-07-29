import type { GenerationModelCatalogItem, GenerateWidgetValues, MainModelConfig } from '@features/generation/contracts';
import type { PromptTemplateRecord } from '@features/generation/data/promptTemplates';
import type * as GenerationQueries from '@features/generation/queries';

import { getDefaultGenerateSettings } from '@features/generation/settings';
import { QueryClient } from '@tanstack/react-query';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const templateQuery = vi.hoisted(() => ({
  queryFn: vi.fn<() => Promise<PromptTemplateRecord[]>>(),
}));

vi.mock('@features/generation/queries', async (importOriginal) => {
  const actual = await importOriginal<typeof GenerationQueries>();

  return {
    ...actual,
    promptTemplatesQueryOptions: () => ({
      queryFn: templateQuery.queryFn,
      queryKey: ['generation', 'promptTemplates', 'list'],
      staleTime: 0,
    }),
  };
});

import {
  createGenerateWidgetSyncRuntime,
  type GenerateWidgetSyncProjectSnapshot,
  type GenerateWidgetSyncRuntimeDeps,
} from './runtime';

const createModel = (key: string, overrides: Partial<MainModelConfig> = {}): MainModelConfig => ({
  base: 'sdxl',
  key,
  name: key,
  type: 'main',
  ...overrides,
});

const createValues = (model: MainModelConfig, overrides: Partial<GenerateWidgetValues> = {}): GenerateWidgetValues => ({
  ...getDefaultGenerateSettings(model),
  model,
  seed: 1,
  shouldRandomizeSeed: false,
  ...overrides,
});

const createReadableStore = <Snapshot>(initialSnapshot: Snapshot) => {
  let snapshot = initialSnapshot;
  const listeners = new Set<() => void>();

  return {
    getListenerCount: () => listeners.size,
    getSnapshot: () => snapshot,
    setSnapshot: (next: Snapshot) => {
      snapshot = next;
      for (const listener of listeners) {
        listener();
      }
    },
    subscribe: (listener: () => void) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
};

const setup = ({
  models: initialModels,
  project: initialProject,
}: {
  models: readonly GenerationModelCatalogItem[];
  project: GenerateWidgetSyncProjectSnapshot;
}) => {
  const models = createReadableStore(initialModels);
  const project = createReadableStore(initialProject);
  const patches: Array<{
    origin: 'system';
    projectId: string;
    values: Partial<GenerateWidgetValues>;
  }> = [];
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const deps: GenerateWidgetSyncRuntimeDeps = {
    models,
    patchValues: (values, projectId, origin) => {
      patches.push({ origin, projectId, values });
      const current = project.getSnapshot();

      if (current.id === projectId) {
        project.setSnapshot({ ...current, values: { ...current.values, ...values } });
      }
    },
    project,
    queryClient,
  };
  const runtime = createGenerateWidgetSyncRuntime(deps);

  return { models, patches, project, queryClient, runtime };
};

beforeEach(() => {
  templateQuery.queryFn.mockReset();
  templateQuery.queryFn.mockResolvedValue([]);
});

describe('createGenerateWidgetSyncRuntime', () => {
  it('reconciles the newly active project after a project switch', () => {
    const model = createModel('model');
    const initialValues = createValues(model);
    const { patches, project, runtime } = setup({
      models: [model],
      project: { id: 'project-1', values: initialValues },
    });

    project.setSnapshot({ id: 'project-2', values: {} });

    expect(patches).toHaveLength(1);
    expect(patches[0]).toMatchObject({ origin: 'system', projectId: 'project-2' });
    runtime.dispose();
  });

  it('reconciles an applied template after the observer publishes a successful result', async () => {
    const model = createModel('model');
    const staleTemplate = {
      id: 'template-1',
      name: 'Stale name',
      negativePrompt: '',
      positivePrompt: 'stale {prompt}',
    };
    const currentTemplate: PromptTemplateRecord = {
      ...staleTemplate,
      hasImage: false,
      isDefault: false,
      isPublic: false,
      name: 'Current name',
      positivePrompt: 'current {prompt}',
      userId: 'user-1',
    };
    let resolveQuery: (templates: PromptTemplateRecord[]) => void = () => undefined;
    templateQuery.queryFn.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveQuery = resolve;
        })
    );
    const { patches, runtime } = setup({
      models: [model],
      project: {
        id: 'project-1',
        values: createValues(model, { promptTemplate: staleTemplate, promptTemplateViewMode: true }),
      },
    });

    expect(patches).toHaveLength(0);
    resolveQuery([currentTemplate]);

    await vi.waitFor(() => {
      expect(patches).toHaveLength(1);
    });
    expect(patches[0]?.values.promptTemplate).toEqual({
      id: currentTemplate.id,
      name: currentTemplate.name,
      negativePrompt: currentTemplate.negativePrompt,
      positivePrompt: currentTemplate.positivePrompt,
    });

    runtime.dispose();
  });

  it('queries only while the active project has an applied template', async () => {
    const model = createModel('model');
    const values = createValues(model);
    const { project, queryClient, runtime } = setup({
      models: [model],
      project: { id: 'project-1', values },
    });

    expect(templateQuery.queryFn).not.toHaveBeenCalled();

    project.setSnapshot({
      id: 'project-1',
      values: {
        ...values,
        promptTemplate: {
          id: 'template-1',
          name: 'Template',
          negativePrompt: '',
          positivePrompt: '{prompt}',
        },
      },
    });

    await vi.waitFor(() => {
      expect(templateQuery.queryFn).toHaveBeenCalledTimes(1);
    });

    project.setSnapshot({ id: 'project-1', values });
    await queryClient.invalidateQueries({ queryKey: ['generation', 'promptTemplates', 'list'] });
    expect(templateQuery.queryFn).toHaveBeenCalledTimes(1);

    runtime.dispose();
  });

  it('unsubscribes from both stores and the template observer on disposal', async () => {
    const model = createModel('model');
    const template = {
      id: 'template-1',
      name: 'Template',
      negativePrompt: '',
      positivePrompt: '{prompt}',
    };
    let resolveQuery: (templates: PromptTemplateRecord[]) => void = () => undefined;
    templateQuery.queryFn.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveQuery = resolve;
        })
    );
    const { models, patches, project, runtime } = setup({
      models: [model],
      project: {
        id: 'project-1',
        values: createValues(model, { promptTemplate: template }),
      },
    });

    expect(project.getListenerCount()).toBe(1);
    expect(models.getListenerCount()).toBe(1);

    runtime.dispose();
    expect(project.getListenerCount()).toBe(0);
    expect(models.getListenerCount()).toBe(0);

    project.setSnapshot({ id: 'project-2', values: {} });
    models.setSnapshot([createModel('renamed', { name: 'Renamed' })]);
    resolveQuery([{ ...template, hasImage: false, isDefault: false, isPublic: false, userId: 'user-1' }]);
    await Promise.resolve();

    expect(patches).toHaveLength(0);
  });
});
