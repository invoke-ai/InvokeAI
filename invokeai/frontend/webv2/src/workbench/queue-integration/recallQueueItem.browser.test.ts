import type { GenerateWidgetValues, MainModelConfig } from '@features/generation/contracts';
import type * as ModelsModule from '@features/models';
import type { QueueItemReadModel } from '@features/queue/contracts';
import type { Project } from '@workbench/projectContracts';
import type { TFunction } from 'i18next';

import { getDefaultGenerateSettings } from '@features/generation/settings';
import { seedArchitectureCapabilities } from '@features/models/core/architectureCapabilities.testing';
import { buildQueueItemOrigin } from '@features/queue/contracts';
import { accountLifecycle, captureAccountScope } from '@platform/state/accountLifecycle';
import { QueryClient } from '@tanstack/react-query';
import { getResolvedProjectWidgetValues } from '@workbench/widgetState';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { afterEach, expect, it, vi } from 'vitest';

import { recallQueueItem } from './recallQueueItem';

const models = vi.hoisted(() => ({ release: () => {} }));

// The catalog read is where the key's recall waits; the test decides when it settles.
vi.mock('@features/models', async (importOriginal) => ({
  ...(await importOriginal<typeof ModelsModule>()),
  ensureModelsLoaded: () =>
    new Promise<void>((resolve) => {
      models.release = resolve;
    }),
}));

seedArchitectureCapabilities();

afterEach(() => accountLifecycle.invalidate());

it.each(['its module loaded', 'it read the catalog'] as const)(
  'drops a key recall whose project stopped being active while %s',
  async (moment) => {
    accountLifecycle.activate('queue-recall-project-switch');
    const main = { base: 'sdxl', key: 'main', name: 'SDXL', type: 'main' } as MainModelConfig;
    const state = createInitialWorkbenchState();
    const project = state.projects[0]!;
    project.queue = {
      items: [
        {
          backendItemIds: [1],
          cancellable: false,
          id: 'run-1',
          snapshot: {
            recall: {
              generateValues: { ...getDefaultGenerateSettings(main), model: main, positivePrompt: 'Recalled' },
            },
            sourceId: 'generate',
          },
          status: 'completed',
        } as unknown as Project['queue']['items'][number],
      ],
    };
    const store = createWorkbenchStore(state);
    const notify = { info: vi.fn(), success: vi.fn() };
    const openWidget = vi.fn();
    const item = {
      batchId: 'batch-1',
      createdAt: '2026-01-01T00:00:00Z',
      id: 1,
      origin: buildQueueItemOrigin('run-1', project.id),
      resultImageNames: [],
      sessionId: 'session-1',
      status: 'completed',
      updatedAt: '2026-01-01T00:00:00Z',
    } satisfies QueueItemReadModel;

    // Captured at the press, as the queue's adapter does before it loads this module.
    const owner = captureAccountScope();
    const recall = () =>
      recallQueueItem({
        context: {
          commands: store.commands,
          notify,
          openWidget,
          queries: store.queries,
          t: ((key: string) => key) as TFunction,
        },
        item,
        kind: 'all',
        owner,
        projectId: project.id,
        queryClient: new QueryClient(),
      });
    let pending: Promise<void>;
    if (moment === 'its module loaded') {
      store.commands.projects.create();
      pending = recall();
    } else {
      pending = recall();
      store.commands.projects.create();
    }
    const switchedTo = store.queries.getSnapshot().activeProject.id;
    models.release();
    await pending;

    expect(switchedTo).not.toBe(project.id);
    for (const id of [project.id, switchedTo]) {
      const values = getResolvedProjectWidgetValues(
        store.queries.getProject(id)!,
        'generate',
        store.queries.getSnapshot().scopedDefaults
      ) as Partial<GenerateWidgetValues>;
      expect(values.positivePrompt).not.toBe('Recalled');
    }
    expect(notify.success).not.toHaveBeenCalled();
    expect(notify.info).not.toHaveBeenCalled();
    expect(openWidget).not.toHaveBeenCalled();
  }
);
