import type { GenerationUiAdapter } from '@features/generation/react';
import type { QueueServerItemDTO } from '@features/queue/data/serverTypes';
import type { Project } from '@workbench/projectContracts';
import type { ReactNode } from 'react';

import { getQueueReadModelOptions } from '@features/queue';
import { buildProjectQueueItemOriginPrefix } from '@features/queue/contracts';
import { createQueueItemDTO, createQueueServer } from '@features/queue/data/queueServer.testing';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { shallowEqual, useExternalStoreSelector } from '@platform/state/selectors';
import { QueryClient, QueryClientProvider, QueryObserver } from '@tanstack/react-query';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { GenerationUiAdapterProvider } from './GenerationUiAdapter';

type Insights = ReturnType<GenerationUiAdapter['queueInsights']['getSnapshot']>;

let store: ReturnType<typeof createWorkbenchStore>;
let adapter: GenerationUiAdapter;
let server: ReturnType<typeof createQueueServer>;
let backendItems: QueueServerItemDTO[];
let queryClient: QueryClient;
let host: HTMLDivElement;
let root: Root;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

vi.mock('@features/generation/react', () => ({
  GenerationUiProvider: ({ adapter: next, children }: { adapter: GenerationUiAdapter; children: ReactNode }) => {
    adapter = next;
    return children;
  },
}));
vi.mock('@workbench/image-actions/useFindGalleryItem', () => ({ useFindGalleryItem: () => () => undefined }));
vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectSelector: (selector: (project: Project) => unknown) =>
    useExternalStoreSelector(
      store.subscribe,
      store.getSnapshot,
      (snapshot) => selector(snapshot.activeProject),
      shallowEqual
    ),
  useOptionalWorkbenchCommands: () => store.commands,
  useWorkbenchCommands: () => store.commands,
  useWorkbenchInternalStore: () => store,
}));

const selectSeeds = (insights: Insights) => insights.seedHistory;
const selectSecondsPerRun = (insights: Insights) => insights.secondsPerRun;

/** Read like Generate's seed field and run-time estimate: each subscribes to its own slice. */
const SeedsConsumer = () => {
  const seeds = useExternalStoreSelector(
    adapter.queueInsights.subscribe,
    adapter.queueInsights.getSnapshot,
    selectSeeds
  );
  return <output data-testid="seeds">{seeds.map((item) => item.seed).join(',')}</output>;
};
const DurationConsumer = () => {
  const seconds = useExternalStoreSelector(
    adapter.queueInsights.subscribe,
    adapter.queueInsights.getSnapshot,
    selectSecondsPerRun
  );
  return <output data-testid="duration">{String(seconds)}</output>;
};

const rendered = (testId: string) => host.querySelector(`[data-testid="${testId}"]`)?.textContent;

const GENERATE_ID = 51;
const LOCAL_RUN_ID = 'local-generate-run';

const completedRun = (itemId: number, seed: number): QueueServerItemDTO =>
  createQueueItemDTO(itemId, {
    completed_at: '2026-01-01T00:00:12',
    field_values: [{ field_name: 'value', node_path: 'seed', value: seed }],
    started_at: '2026-01-01T00:00:10',
  });

const setGenerateSeed = (seed: number) => {
  backendItems[GENERATE_ID - 1] = completedRun(GENERATE_ID, seed);
};

const itemIdReads = () => server.requests.filter((request) => request.startsWith('GET item_ids'));

const render = async (children: ReactNode) => {
  await act(async () => {
    root.render(
      <QueryClientProvider client={queryClient}>
        <GenerationUiAdapterProvider>{children}</GenerationUiAdapterProvider>
      </QueryClientProvider>
    );
    await Promise.resolve();
  });
};

const settleQueries = () => act(() => vi.waitFor(() => expect(queryClient.isFetching()).toBe(0)));
const invalidateQueue = async () => {
  await act(() => queryClient.invalidateQueries({ queryKey: ['queue'] }));
  await settleQueries();
};

const projectScopeQuery = (projectId: string) =>
  `GET item_ids?${new URLSearchParams({
    limit: '50',
    order_dir: 'DESC',
    origin_prefix: buildProjectQueueItemOriginPrefix(projectId),
  }).toString()}`;

beforeEach(() => {
  accountLifecycle.activate('generation-queue-insights');
  const state = createInitialWorkbenchState();
  const project = state.projects[0]!;
  // The newest item is this project's Generate run; the 50 before it came from elsewhere.
  backendItems = Array.from({ length: GENERATE_ID }, (_, index) => completedRun(index + 1, 900 + index));
  setGenerateSeed(1234);
  project.queue = {
    items: [
      {
        backendItemIds: [GENERATE_ID],
        cancellable: false,
        id: LOCAL_RUN_ID,
        snapshot: { sourceId: 'generate' } as Project['queue']['items'][number]['snapshot'],
        status: 'completed',
      },
    ],
  };
  store = createWorkbenchStore(state);
  server = createQueueServer(backendItems);
  queryClient = new QueryClient({ defaultOptions: { queries: { refetchOnWindowFocus: false, retry: false } } });
  vi.stubGlobal('fetch', server.fetch);
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
  queryClient.clear();
  vi.unstubAllGlobals();
  accountLifecycle.invalidate();
});

describe('Generation adapter queue insights', () => {
  it('does not read or refetch the queue while no Generate control shows insights', async () => {
    await render(null);
    await invalidateQueue();

    expect(server.requests).toEqual([]);
  });

  it("reads the project's recent window while a control is mounted", async () => {
    await render(<SeedsConsumer />);
    await settleQueries();

    // Only the run this project submitted from Generate contributes, with its executed seed.
    expect(rendered('seeds')).toBe('1234');
    expect(itemIdReads()).toEqual([projectScopeQuery(store.getSnapshot().activeProject.id)]);
  });

  it('keeps refetching for the remaining control and stops once the last one unmounts', async () => {
    await render(
      <>
        <SeedsConsumer />
        <DurationConsumer />
      </>
    );
    await settleQueries();
    expect([rendered('seeds'), rendered('duration')]).toEqual(['1234', '2']);

    await render(<SeedsConsumer />);
    setGenerateSeed(5678);
    await invalidateQueue();

    expect(rendered('seeds')).toBe('5678');
    expect(itemIdReads()).toHaveLength(2);

    await render(null);
    await invalidateQueue();

    expect(itemIdReads()).toHaveLength(2);
  });

  it('re-targets the read to the new project on a project switch and releases the old one', async () => {
    const firstProjectId = store.getSnapshot().activeProject.id;
    await render(<SeedsConsumer />);
    await settleQueries();

    let secondProjectId = '';
    await act(() => {
      secondProjectId = store.commands.projects.create().id;
    });
    await settleQueries();
    server.requests.length = 0;
    await invalidateQueue();

    // The new project has no local Generate runs, so nothing in the shared window is its own.
    expect(rendered('seeds')).toBe('');
    expect(itemIdReads()).toEqual([projectScopeQuery(secondProjectId)]);
    expect(secondProjectId).not.toBe(firstProjectId);
  });

  it('updates when the local queue learns its backend ids, without another queue read', async () => {
    await act(() => {
      store.commands.queue.markBackendSubmitted({
        backendItemIds: [],
        projectId: store.getSnapshot().activeProject.id,
        queueItemId: LOCAL_RUN_ID,
      });
    });
    await render(<SeedsConsumer />);
    await settleQueries();
    expect(rendered('seeds')).toBe('');
    const requestCount = server.requests.length;

    await act(() => {
      store.commands.queue.markBackendSubmitted({
        backendItemIds: [GENERATE_ID],
        projectId: store.getSnapshot().activeProject.id,
        queueItemId: LOCAL_RUN_ID,
      });
    });

    expect(rendered('seeds')).toBe('1234');
    expect(server.requests).toHaveLength(requestCount);
  });

  it('updates every control even when the new data is read before its observer notifies', async () => {
    const { queryKey } = getQueueReadModelOptions({
      originPrefix: buildProjectQueueItemOriginPrefix(store.getSnapshot().activeProject.id),
    });
    // Another observer of the same query is notified first and reads the insights store in its callback, as a
    // render driven by that observer would, before the store's own observer delivers the change.
    const stopEarlierReader = new QueryObserver(queryClient, { enabled: false, queryKey }).subscribe(() => {
      adapter.queueInsights.getSnapshot();
    });
    await render(
      <>
        <SeedsConsumer />
        <DurationConsumer />
      </>
    );
    await settleQueries();
    expect(rendered('seeds')).toBe('1234');

    await act(() => {
      queryClient.setQueryData(queryKey, (model) =>
        model
          ? {
              ...model,
              items: model.items.map((item) =>
                item.id === GENERATE_ID
                  ? { ...item, fieldValues: [{ fieldName: 'value', nodePath: 'seed', value: 4321 }] }
                  : item
              ),
            }
          : model
      );
    });
    stopEarlierReader();

    expect(rendered('seeds')).toBe('4321');
  });

  it('keeps the seed history identical across a status-only refetch', async () => {
    await render(<SeedsConsumer />);
    await settleQueries();
    const before = adapter.queueInsights.getSnapshot();

    // An item outside the 50-item window fails: counts change, the window's items do not.
    backendItems[0] = { ...backendItems[0]!, status: 'failed' };
    await invalidateQueue();

    expect(itemIdReads()).toHaveLength(2);
    expect(adapter.queueInsights.getSnapshot().seedHistory).toBe(before.seedHistory);
  });
});
