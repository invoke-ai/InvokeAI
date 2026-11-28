/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type * as QueueDataStore from '@features/queue/ui/queueDataStore';

import { ChakraProvider } from '@chakra-ui/react';
import { activeProgressTargetStore } from '@features/queue/data/activeProgressTargetStore';
import { itemProgressStore } from '@features/queue/data/itemProgressStore';
import { progressImageStore } from '@features/queue/data/progressImageStore';
import { queueItemProgressStore } from '@features/queue/data/progressStore';
import { CANCEL_CURRENT_SETTLE_MS } from '@features/queue/ui/cancelCurrent';
import { clearQueueConfirmation } from '@features/queue/ui/queueConfirmationStore';
import { QueueDataRuntime } from '@features/queue/ui/QueueDataRuntime';
import { createTestQueueUiAdapter } from '@features/queue/ui/queueUiAdapter.testing';
import { QueueUiProvider, type QueueLocalRun } from '@features/queue/ui/QueueUiContext';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

const state = vi.hoisted(() => ({
  isConnected: true,
  isPaused: false,
  projectScoped: false,
  wholeQueueInProgress: 0,
  localItems: [] as unknown[],
  modelLoads: [] as Array<{ label: string }>,
  localRuns: [] as QueueLocalRun[],
  readFails: false,
  serverItems: null as null | Array<{ id: number; status: string }>,
}));
const commands = vi.hoisted(() => ({
  cancelItem: vi.fn((id: number) => {
    const item = state.serverItems?.find((candidate) => candidate.id === id);
    if (item && ['pending', 'in_progress'].includes(item.status)) {
      item.status = 'canceled';
    }
    return Promise.resolve({ ...item, id, status: item?.status ?? 'canceled' });
  }),
  cancelScopedItems: vi.fn(() => Promise.resolve()),
}));
const notify = vi.hoisted(() => ({ error: vi.fn(), info: vi.fn(), success: vi.fn() }));

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: Record<string, unknown>) => (values ? `${key} ${JSON.stringify(values)}` : key),
  }),
}));
vi.mock('@features/models', () => ({ useModelLoads: () => state.modelLoads }));
vi.mock('@features/queue/publicApi', () => ({
  createProductionQueueRealtimeRuntime: () => ({ dispose: vi.fn(), start: vi.fn() }),
  getQueueItemDetailOptions: (itemId: number) => ({ queryFn: () => null, queryKey: ['queue', 'item-detail', itemId] }),
  getQueueReadModelOptions: () => ({
    queryFn: () => serverReading(),
    queryKey: ['queue', 'read-model', 'all'],
    staleTime: Number.POSITIVE_INFINITY,
  }),
  queueCommands: commands,
}));
vi.mock('@features/queue/data/generationDevicesStore', () => ({ refreshGenerationDevices: vi.fn() }));
vi.mock('@features/queue/ui/queueDataStore', async (importOriginal) => ({
  ...(await importOriginal<typeof QueueDataStore>()),
  refreshQueue: () => Promise.resolve(),
}));
vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectSelector: (select: (project: unknown) => unknown) => select({ queue: { items: state.localItems } }),
  useWorkbenchSelector: (select: (snapshot: unknown) => unknown) =>
    select({ backendConnection: { status: state.isConnected ? 'connected' : 'disconnected' } }),
}));
vi.mock('@workbench/useOpenWorkbenchWidget', () => ({ useOpenWorkbenchWidget: () => vi.fn() }));

const { DocumentTitleProgress } = await import('@workbench/shell/DocumentTitleProgress');
const { QueueCluster } = await import('./QueueCluster');

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let client: QueryClient;
let host: HTMLDivElement;
let root: Root;

const running = {
  batchId: 'batch-1',
  createdAt: '2026-06-27T00:00:00Z',
  id: 7,
  resultImageNames: [],
  sessionId: 'session-7',
  status: 'in_progress' as const,
  updatedAt: '2026-06-27T00:00:00Z',
};

const serverItem = (id: number, status: 'canceled' | 'completed' | 'in_progress' | 'pending') => ({
  ...running,
  id,
  sessionId: `session-${id}`,
  status,
});

const render = async ({
  current = running as ReturnType<typeof serverItem> | null,
  items = [running, serverItem(8, 'pending'), serverItem(9, 'pending')],
  showDocumentTitle = false,
}: {
  current?: ReturnType<typeof serverItem> | null;
  items?: ReturnType<typeof serverItem>[];
  showDocumentTitle?: boolean;
} = {}) => {
  client = new QueryClient();
  client.setQueryData(['queue', 'read-model', 'all'], readModel({ current, items }));
  if (state.projectScoped) {
    const whole = readModel({ current: null, items: [] }).status;
    client.setQueryData(['queue', 'status', 'all'], {
      ...whole,
      queue: { ...whole.queue, inProgress: state.wholeQueueInProgress },
    });
  }
  state.serverItems ??= structuredClone(items);

  await act(() =>
    root.render(
      <ChakraProvider value={system}>
        <QueryClientProvider client={client}>
          <QueueUiProvider
            adapter={createTestQueueUiAdapter({
              activeProjectId: 'project-1',
              canManageItem: (item) => item.userId !== 'someone-else',
              queueJobsScope: state.projectScoped ? 'active-project' : 'all',
              isConnected: state.isConnected,
              localRuns: state.localRuns,
              notify,
            })}
          >
            <QueueCluster />
            {showDocumentTitle ? <DocumentTitleProgress /> : null}
            <QueueDataRuntime />
          </QueueUiProvider>
        </QueryClientProvider>
      </ChakraProvider>
    )
  );
};

function serverReading() {
  if (state.readFails) {
    throw new Error('Server unavailable');
  }
  const items = (state.serverItems ?? []) as ReturnType<typeof serverItem>[];
  return readModel({ current: items.find((item) => item.status === 'in_progress') ?? null, items });
}

const readModel = ({
  current,
  items,
}: {
  current: ReturnType<typeof serverItem> | null;
  items: ReturnType<typeof serverItem>[];
}) => {
  const open = items.filter((item) => item.status === 'pending' || item.status === 'in_progress');

  return {
    current,
    items,
    next: open.filter((item) => item.status === 'pending').sort((left, right) => left.id - right.id)[0] ?? null,
    scope: {},
    status: {
      processor: { isProcessing: true, isStarted: !state.isPaused },
      queue: {
        canceled: 0,
        completed: 0,
        failed: 0,
        inProgress: open.filter((item) => item.status === 'in_progress').length,
        pending: open.filter((item) => item.status === 'pending').length,
        queueId: 'default',
        total: items.length,
        waiting: 0,
      },
    },
  };
};

const cancelButton = () =>
  host.querySelector<HTMLButtonElement>(
    'button[aria-label="widgets.queue.cancelCurrentItem"], button[aria-label="widgets.queue.cancelNextItem"]'
  )!;
const confirmation = () => document.querySelector('[role="alertdialog"]');
const queueButton = () => host.querySelector<HTMLButtonElement>('button[aria-label^="topbar.queue.remaining"]')!;
const announcement = () => host.querySelector('[aria-live="polite"]')?.textContent;

const projectRun = (status: 'completed' | 'pending' | 'running', backendItemIds = [7, 8, 9]) => ({
  backendItemIds,
  cancellable: true,
  id: 'run-1',
  snapshot: {
    destination: 'gallery',
    graph: { id: 'graph', label: 'Graph' },
    presentation: { positivePrompt: 'a fjord at dawn' },
    sourceId: 'generate',
    submittedAt: '2026-06-27T00:00:00Z',
  },
  status,
});

beforeEach(() => {
  state.isConnected = true;
  state.localItems = [];
  state.modelLoads = [];
  state.localRuns = [];
  state.isPaused = false;
  state.projectScoped = false;
  state.wholeQueueInProgress = 0;
  state.readFails = false;
  state.serverItems = null;
  notify.error.mockClear();
  notify.info.mockClear();
  notify.success.mockClear();
  commands.cancelItem.mockClear();
  commands.cancelScopedItems.mockClear();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  clearQueueConfirmation();
  await act(() => root.unmount());
  host.remove();
  accountLifecycle.invalidate();
  accountLifecycle.activate('queue-cluster-test');
});

describe('QueueCluster', () => {
  it('cancels the running item on a plain click without asking', async () => {
    await render();

    await act(() => userEvent.click(cancelButton()));

    await vi.waitFor(() => expect(commands.cancelItem).toHaveBeenCalledWith(7));
    expect(confirmation()).toBeNull();
    await vi.waitFor(() =>
      expect(notify.success).toHaveBeenCalledExactlyOnceWith('widgets.queue.cancellationRequested', expect.any(String))
    );
  });

  it('names and stops the next waiting item between items, while nothing runs', async () => {
    await render({
      current: null,
      items: [serverItem(7, 'completed'), serverItem(8, 'pending'), serverItem(9, 'pending')],
    });

    expect(cancelButton().disabled).toBe(false);
    expect(cancelButton().getAttribute('aria-label')).toBe('widgets.queue.cancelNextItem');
    await act(() => userEvent.click(cancelButton()));

    await vi.waitFor(() => expect(commands.cancelItem).toHaveBeenCalledExactlyOnceWith(8));
    await vi.waitFor(() =>
      expect(notify.success).toHaveBeenCalledExactlyOnceWith('widgets.queue.cancellationRequested', expect.any(String))
    );
  });

  it('stops one item for a double click, and the next one on a later click', async () => {
    let answer!: () => void;
    commands.cancelItem.mockImplementationOnce((id: number) => {
      const item = state.serverItems!.find((candidate) => candidate.id === id)!;
      item.status = 'canceled';
      return new Promise((resolve) => {
        answer = () => resolve({ ...item, id, status: 'canceled' });
      });
    });
    await render();

    await act(() => userEvent.dblClick(cancelButton()));
    await act(() => answer());
    await vi.waitFor(() => expect(notify.success).toHaveBeenCalledOnce());
    expect(commands.cancelItem.mock.calls).toEqual([[7]]);

    await vi.waitFor(() => expect(cancelButton().getAttribute('aria-label')).toBe('widgets.queue.cancelNextItem'));
    await act(
      () =>
        new Promise<void>((resolve) => {
          setTimeout(resolve, CANCEL_CURRENT_SETTLE_MS + 100);
        })
    );
    await act(() => userEvent.click(cancelButton()));
    await vi.waitFor(() => expect(commands.cancelItem.mock.calls).toEqual([[7], [8]]));
  });

  it("in a project view, stops nothing waiting while another project's item runs: the × is disabled", async () => {
    state.projectScoped = true;
    state.wholeQueueInProgress = 1;
    await render({ current: null, items: [serverItem(7, 'completed'), serverItem(8, 'pending')] });

    expect(cancelButton().disabled).toBe(true);
  });

  it("stops nothing waiting while someone else's item runs: the × is disabled", async () => {
    const theirs = { ...serverItem(20, 'in_progress'), userId: 'someone-else' };
    await render({ current: theirs, items: [serverItem(21, 'pending'), theirs] });

    expect(cancelButton().disabled).toBe(true);
  });

  it('shows the item running again and says the cancel failed, so the next click tries it again', async () => {
    commands.cancelItem.mockImplementationOnce(() => Promise.reject(new Error('Server unavailable')));
    await render();

    await act(() => userEvent.click(cancelButton()));

    await vi.waitFor(() =>
      expect(notify.error).toHaveBeenCalledExactlyOnceWith(
        'widgets.queue.failedToCancelCurrentItem',
        'Server unavailable'
      )
    );
    expect(notify.success).not.toHaveBeenCalled();
    expect(cancelButton().getAttribute('aria-label')).toBe('widgets.queue.cancelCurrentItem');
    await act(() => userEvent.click(cancelButton()));
    await vi.waitFor(() => expect(commands.cancelItem.mock.calls).toEqual([[7], [7]]));
  });

  it('stops what runs now when the item it read as running had already finished', async () => {
    state.serverItems = [serverItem(7, 'completed'), serverItem(8, 'in_progress'), serverItem(9, 'pending')];
    await render();

    await act(() => userEvent.click(cancelButton()));

    await vi.waitFor(() => expect(commands.cancelItem.mock.calls).toEqual([[7], [8]]));
    await vi.waitFor(() => expect(notify.success).toHaveBeenCalledOnce());
    expect(notify.info).not.toHaveBeenCalled();
  });

  it('asks before cancelling everything on shift-click, and cancels nothing until confirmed', async () => {
    await render();

    await act(() => userEvent.click(cancelButton(), { modifiers: ['Shift'] } as never));

    await expect.poll(() => confirmation()?.getAttribute('data-state')).toBe('open');
    expect(commands.cancelScopedItems).not.toHaveBeenCalled();
    expect(commands.cancelItem).not.toHaveBeenCalled();
  });

  it("counts the project's run once though the server lists its items too, and shows its image progress", async () => {
    state.localItems = [projectRun('running')];
    await render();

    expect(queueButton().textContent).toBe('3');
    expect(announcement()).toBe('topbar.queue.imageProgress {"current":1,"total":3}');
  });

  it('preserves ordinary progress reporting for sequential multi-image batches', async () => {
    state.localItems = [projectRun('running', [7, 8, 9, 10])];
    const first = { queueItemId: 'run-1', itemIndex: 1 };
    const second = { queueItemId: 'run-1', itemIndex: 2 };
    activeProgressTargetStore.set(first);
    itemProgressStore.set(7, { message: 'First image', percentage: 0.8 });
    queueItemProgressStore.set('run-1', {
      activeItemIndex: 1,
      completedItemCount: 0,
      message: 'First image',
      percentage: 0.8,
      totalItemCount: 4,
    });

    await render({
      items: [running, serverItem(8, 'pending'), serverItem(9, 'pending'), serverItem(10, 'pending')],
    });

    expect(announcement()).toBe('topbar.queue.imageProgressPercent {"current":1,"percent":80,"total":4}');

    // The first result is still hydrating. Its backend progress
    // entry remains, but the coordinator has marked its slot settling.
    await act(() => {
      activeProgressTargetStore.settle(first);
      activeProgressTargetStore.set(second);
      itemProgressStore.set(8, { message: 'Denoising', percentage: 0.3 });
      queueItemProgressStore.set('run-1', {
        activeItemIndex: 2,
        completedItemCount: 1,
        message: 'Denoising',
        percentage: 0.3,
        totalItemCount: 4,
      });
    });

    // Both backend entries still exist, but only one slot is running.
    expect(announcement()).toBe('topbar.queue.imageProgressPercent {"current":2,"percent":30,"total":4}');

    await act(() => {
      state.localItems = [{ ...projectRun('running', [7, 8, 9, 10]), completedBackendItemIds: [7] }];
      itemProgressStore.clear(7);
      activeProgressTargetStore.clear(first);
    });

    expect(announcement()).toBe('topbar.queue.imageProgressPercent {"current":2,"percent":30,"total":4}');

    await act(() => userEvent.hover(queueButton()));
    await vi.waitFor(() =>
      expect(document.body.textContent).toContain('topbar.queue.generatingImage {"current":2,"total":4}')
    );
    expect(document.querySelector('[data-queue-backend-progress]')).toBeNull();
  });

  it.each([1, 2])('retains the live tab percentage for sequential image %i of four', async (index) => {
    state.localItems = [{ ...projectRun('running', [7, 8, 9, 10]), completedBackendItemIds: index === 2 ? [7] : [] }];
    queueItemProgressStore.set('run-1', {
      activeItemIndex: index,
      completedItemCount: index - 1,
      message: 'Denoising',
      percentage: 0.6,
      totalItemCount: 4,
    });
    const current = serverItem(6 + index, 'in_progress');
    await render({
      current,
      items: [
        current,
        ...[7, 8, 9, 10]
          .filter((id) => id !== current.id)
          .map((id) => serverItem(id, id < current.id ? 'completed' : 'pending')),
      ],
      showDocumentTitle: true,
    });

    expect(document.title).toBe(`60% · ${index}/4 · Invoke`);
  });

  it('uses settled-slot progress in the tab after concurrency is observed', async () => {
    state.localItems = [{ ...projectRun('running', [7, 8, 9, 10]), completedBackendItemIds: [7] }];
    activeProgressTargetStore.observeConcurrency('run-1');
    queueItemProgressStore.set('run-1', {
      activeItemIndex: 3,
      completedItemCount: 1,
      message: 'Denoising',
      percentage: 0.6,
      totalItemCount: 4,
    });
    await render({
      current: serverItem(8, 'in_progress'),
      items: [
        serverItem(7, 'completed'),
        serverItem(8, 'in_progress'),
        serverItem(9, 'in_progress'),
        serverItem(10, 'pending'),
      ],
      showDocumentTitle: true,
    });

    expect(document.title).toBe('25% · 2/4 · Invoke');
  });

  it.each([false, true])(
    'keeps title progress in the correct scope with additional queued work (concurrent: %s)',
    async (concurrent) => {
      state.localItems = [
        { ...projectRun('running', [7, 8, 9, 10]), completedBackendItemIds: [7] },
        { ...projectRun('pending', [11, 12, 13, 14]), id: 'run-2' },
      ];
      if (concurrent) {
        activeProgressTargetStore.observeConcurrency('run-1');
      }
      queueItemProgressStore.set('run-1', {
        activeItemIndex: 2,
        completedItemCount: 1,
        message: 'Denoising',
        percentage: 0.6,
        totalItemCount: 4,
      });
      await render({
        current: serverItem(8, 'in_progress'),
        items: [
          serverItem(7, 'completed'),
          serverItem(8, 'in_progress'),
          serverItem(9, concurrent ? 'in_progress' : 'pending'),
          ...[10, 11, 12, 13, 14].map((id) => serverItem(id, 'pending')),
        ],
        showDocumentTitle: true,
      });
      expect(document.title).toBe(concurrent ? '25% · 2/4 · Invoke' : '60% · 2/8 · Invoke');
      expect(queueButton().textContent).toBe('7');
    }
  );

  it('retains concurrent progress after the first iteration finishes in order', async () => {
    state.localItems = [projectRun('running', [7, 8])];
    activeProgressTargetStore.set({ queueItemId: 'run-1', itemIndex: 1 });
    activeProgressTargetStore.set({ queueItemId: 'run-1', itemIndex: 2 });
    itemProgressStore.set(7, { message: 'First iteration', percentage: 0.8 });
    progressImageStore.setProgress(
      { message: 'Second iteration', percentage: 0.4 },
      { queueItemId: 'run-1', itemIndex: 2 }
    );

    activeProgressTargetStore.observeConcurrency('run-1');
    await render({ items: [running, serverItem(8, 'in_progress')] });

    expect(announcement()).toBe('topbar.queue.imagesFinished {"current":0,"total":2}');

    await act(() => {
      state.localItems = [{ ...projectRun('running', [7, 8]), completedBackendItemIds: [7] }];
      activeProgressTargetStore.settle({ queueItemId: 'run-1', itemIndex: 1 });
      itemProgressStore.clear(7);
    });

    await vi.waitFor(() => expect(announcement()).toBe('topbar.queue.imagesFinished {"current":1,"total":2}'));

    await act(() => userEvent.hover(queueButton()));
    await vi.waitFor(() =>
      expect(document.querySelector('[data-queue-backend-progress="8"]')?.textContent).toContain('Second iteration')
    );
    expect(document.querySelector('[data-queue-backend-progress="7"]')).toBeNull();
  });

  it('retains observed concurrency across project switches and gaps between iterations', async () => {
    const first = { queueItemId: 'run-1', itemIndex: 1 };
    const second = { queueItemId: 'run-1', itemIndex: 2 };
    const third = { queueItemId: 'run-1', itemIndex: 3 };
    state.localItems = [projectRun('running')];
    activeProgressTargetStore.set(first);
    activeProgressTargetStore.set(second);

    activeProgressTargetStore.observeConcurrency('run-1');
    await render({ items: [running, serverItem(8, 'in_progress'), serverItem(9, 'pending')] });

    expect(announcement()).toBe('topbar.queue.imagesFinished {"current":0,"total":3}');

    const originalItems = [{ ...projectRun('running'), completedBackendItemIds: [7] }];

    await act(() => {
      state.localItems = originalItems;
      activeProgressTargetStore.clear(first);
    });

    await vi.waitFor(() => expect(announcement()).toBe('topbar.queue.imagesFinished {"current":1,"total":3}'));

    // View another project, then return to the original run.
    state.localItems = [];
    await render({ current: null, items: [] });

    state.localItems = originalItems;
    await render({
      current: serverItem(8, 'in_progress'),
      items: [serverItem(7, 'completed'), serverItem(8, 'in_progress'), serverItem(9, 'pending')],
    });

    expect(announcement()).toBe('topbar.queue.imagesFinished {"current":1,"total":3}');

    // There is now a gap with no running slots.
    await act(() => {
      state.localItems = [{ ...projectRun('running'), completedBackendItemIds: [7, 8] }];
      activeProgressTargetStore.settle(second);
    });

    await vi.waitFor(() => expect(announcement()).toBe('topbar.queue.imagesFinished {"current":2,"total":3}'));

    await act(() => activeProgressTargetStore.set(third));

    expect(announcement()).toBe('topbar.queue.imagesFinished {"current":2,"total":3}');
  });

  it('shows waiting progress between iterations of a concurrent run', async () => {
    state.localItems = [{ ...projectRun('running'), completedBackendItemIds: [7, 8] }];
    activeProgressTargetStore.observeConcurrency('run-1');
    await render({
      current: null,
      items: [serverItem(7, 'completed'), serverItem(8, 'completed'), serverItem(9, 'pending')],
    });

    await act(() => userEvent.hover(queueButton()));
    const tooltip = () => document.querySelector('[role="tooltip"]');
    await vi.waitFor(() =>
      expect(tooltip()?.textContent).toContain('topbar.queue.imagesFinished {"current":2,"total":3}')
    );
    expect(tooltip()?.textContent).toContain('topbar.queue.waitingForProgress');
    expect(tooltip()?.querySelector('[role="progressbar"]')).not.toBeNull();
  });

  it('shows a single model loading label alongside concurrent iteration progress', async () => {
    state.localItems = [projectRun('running')];
    state.modelLoads = [{ label: 'model-a' }];
    activeProgressTargetStore.set({ queueItemId: 'run-1', itemIndex: 1 });
    activeProgressTargetStore.set({ queueItemId: 'run-1', itemIndex: 2 });
    activeProgressTargetStore.observeConcurrency('run-1');
    await render({ items: [running, serverItem(8, 'in_progress'), serverItem(9, 'pending')] });

    await act(() => userEvent.hover(queueButton()));
    const tooltip = () => document.querySelector('[role="tooltip"]');
    await vi.waitFor(() => expect(tooltip()?.querySelector('[data-queue-backend-progress="7"]')).toBeTruthy());
    expect(tooltip()?.textContent).toContain('widgets.queue.loadingModel {"model":"model-a"}');
    expect(tooltip()?.querySelector('[data-queue-backend-progress="8"]')).toBeTruthy();
  });

  it('shows separate progress rows for concurrent iterations of one Invoke', async () => {
    state.localItems = [projectRun('running')];
    activeProgressTargetStore.set({ queueItemId: 'run-1', itemIndex: 1 });
    activeProgressTargetStore.set({ queueItemId: 'run-1', itemIndex: 2 });
    progressImageStore.setProgress(
      { message: 'Denoising first', percentage: 0.25 },
      { queueItemId: 'run-1', itemIndex: 1 }
    );
    progressImageStore.setProgress(
      { message: 'Denoising second', percentage: 0.75 },
      { queueItemId: 'run-1', itemIndex: 2 }
    );
    activeProgressTargetStore.observeConcurrency('run-1');
    await render({ items: [running, serverItem(8, 'in_progress'), serverItem(9, 'pending')] });

    await act(() => userEvent.hover(queueButton()));
    await vi.waitFor(() => expect(document.querySelector('[data-queue-backend-progress="7"]')).not.toBeNull());
    const first = () => document.querySelector('[data-queue-backend-progress="7"]');
    const second = () => document.querySelector('[data-queue-backend-progress="8"]');
    expect(first()?.textContent).toContain('Denoising first');
    expect(second()?.textContent).toContain('Denoising second');
    expect(document.querySelector('[role="tooltip"]')?.textContent).toContain(
      'topbar.queue.imagesFinished {"current":0,"total":3}'
    );

    await act(() =>
      progressImageStore.setProgress(
        { message: 'Second at 90%', percentage: 0.9 },
        { queueItemId: 'run-1', itemIndex: 2 }
      )
    );
    expect(first()?.textContent).toContain('Denoising first');
    await vi.waitFor(() => expect(second()?.textContent).toContain('Second at 90%'));
    expect(announcement()).toBe('topbar.queue.imagesFinished {"current":0,"total":3}');

    // The backend store can retain a previous leg after the coordinator clears the slot.
    await act(() => {
      itemProgressStore.set(8, { message: 'Stale backend progress', percentage: 0.9 });
      progressImageStore.setProgress(null, { queueItemId: 'run-1', itemIndex: 2 });
    });
    await vi.waitFor(() => expect(second()?.textContent).toContain('topbar.queue.waitingForProgress'));
    expect(second()?.textContent).not.toContain('Stale backend progress');
  });

  it('shows called workflow progress on the parent iteration even when only the child has backend progress', async () => {
    state.localItems = [projectRun('running')];
    const target = { queueItemId: 'run-1', itemIndex: 1 };
    // Reproduce the stores after a child event (backend ID 70, root ID 7):
    // realtime owns the child ID, while the coordinator owns the parent's local slot.
    itemProgressStore.set(70, { message: 'Child sampling', percentage: 0.5 });
    activeProgressTargetStore.set(target);
    progressImageStore.setProgress({ message: 'Child sampling', percentage: 0.5 }, target);
    activeProgressTargetStore.set({ queueItemId: 'run-1', itemIndex: 2 });
    progressImageStore.setProgress(
      { message: 'Other iteration', percentage: 0.25 },
      { queueItemId: 'run-1', itemIndex: 2 }
    );
    activeProgressTargetStore.observeConcurrency('run-1');
    await render({ items: [running, serverItem(8, 'in_progress'), serverItem(9, 'pending')] });

    await act(() => userEvent.hover(queueButton()));
    const parentRow = () => document.querySelector('[data-queue-backend-progress="7"]');
    const otherRow = () => document.querySelector('[data-queue-backend-progress="8"]');
    await vi.waitFor(() => expect(parentRow()?.textContent).toContain('Child sampling'));
    expect(parentRow()?.querySelector('[role="progressbar"]')?.getAttribute('aria-valuenow')).toBe('0.5');
    expect(otherRow()?.textContent).toContain('Other iteration');
    expect(document.querySelector('[data-queue-backend-progress="70"]')).toBeNull();

    // A stale parent reading must not override the coordinator's latest child progress.
    await act(() => {
      itemProgressStore.set(7, { message: 'Old parent progress', percentage: 0.1 });
      itemProgressStore.set(70, { message: 'Child at 80%', percentage: 0.8 });
      progressImageStore.setProgress({ message: 'Child at 80%', percentage: 0.8 }, target);
    });
    await vi.waitFor(() => expect(parentRow()?.textContent).toContain('Child at 80%'));
    expect(parentRow()?.querySelector('[role="progressbar"]')?.getAttribute('aria-valuenow')).toBe('0.8');
    expect(otherRow()?.textContent).toContain('Other iteration');
  });

  it('keeps the remaining GPU iteration and its progress after the other iteration finishes first', async () => {
    activeProgressTargetStore.set({ queueItemId: 'run-1', itemIndex: 1 });
    activeProgressTargetStore.set({ queueItemId: 'run-1', itemIndex: 2 });
    activeProgressTargetStore.settle({ queueItemId: 'run-1', itemIndex: 2 });
    state.localItems = [{ ...projectRun('running', [7, 8]), completedBackendItemIds: [8] }];
    progressImageStore.setProgress(
      { message: 'First iteration at 40%', percentage: 0.4 },
      { queueItemId: 'run-1', itemIndex: 1 }
    );
    itemProgressStore.set(8, { message: 'Second iteration finished', percentage: 1 });
    itemProgressStore.clear(8);
    activeProgressTargetStore.observeConcurrency('run-1');
    await render({ items: [running, serverItem(8, 'completed')] });

    await act(() => userEvent.hover(queueButton()));
    await vi.waitFor(() => expect(document.querySelector('[data-queue-backend-progress="7"]')).not.toBeNull());
    expect(document.querySelector('[data-queue-backend-progress="7"]')?.textContent).toContain(
      'First iteration at 40%'
    );
    expect(document.querySelector('[data-queue-backend-progress="8"]')).toBeNull();
    expect(document.body.textContent).toContain('topbar.queue.batchImage {"current":1,"total":2}');
    expect(announcement()).toBe('topbar.queue.imagesFinished {"current":1,"total":2}');
  });

  it('announces the active batch total rather than all images from separate batches', async () => {
    state.localItems = [
      { ...projectRun('running', [7, 8, 9]), completedBackendItemIds: [7] },
      { ...projectRun('running', [10, 11, 12]), id: 'run-2', completedBackendItemIds: [10, 11] },
    ];
    activeProgressTargetStore.set({ queueItemId: 'run-1', itemIndex: 2 });
    activeProgressTargetStore.set({ queueItemId: 'run-1', itemIndex: 3 });
    activeProgressTargetStore.set({ queueItemId: 'run-2', itemIndex: 3 });
    progressImageStore.setProgress(
      { message: 'First batch iteration 2', percentage: 0.4 },
      { queueItemId: 'run-1', itemIndex: 2 }
    );
    progressImageStore.setProgress(
      { message: 'First batch iteration 3', percentage: 0.6 },
      { queueItemId: 'run-1', itemIndex: 3 }
    );
    progressImageStore.setProgress(
      { message: 'Second batch iteration 3', percentage: 0.8 },
      { queueItemId: 'run-2', itemIndex: 3 }
    );
    activeProgressTargetStore.observeConcurrency('run-1');
    await render({
      items: [
        serverItem(7, 'completed'),
        serverItem(8, 'in_progress'),
        serverItem(9, 'in_progress'),
        serverItem(10, 'completed'),
        serverItem(11, 'completed'),
        serverItem(12, 'in_progress'),
      ],
    });

    expect(queueButton().textContent).toBe('3');
    expect(announcement()).toBe('topbar.queue.imagesFinished {"current":1,"total":3}');
    await act(() => userEvent.hover(queueButton()));
    await vi.waitFor(() => expect(document.querySelector('[data-queue-backend-progress="8"]')).not.toBeNull());
    expect(document.querySelector('[role="tooltip"]')?.textContent).toContain(
      'topbar.queue.imagesFinished {"current":1,"total":3}'
    );
  });

  it("shows the project's runs waiting on a lost connection instead of an idle queue", async () => {
    state.isConnected = false;
    state.localItems = [projectRun('pending', [])];
    await render({ current: null, items: [] });

    const offlineButton = host.querySelector<HTMLButtonElement>('button[aria-label^="topbar.queue.remainingOffline"]');

    expect(offlineButton).not.toBeNull();
    expect(offlineButton!.getAttribute('aria-label')).toContain('"count":1');
    expect(offlineButton!.querySelector('svg.lucide-cloud-off')).not.toBeNull();
    expect(announcement()).toBe('topbar.queue.offlineWaiting {"count":1}');
  });
});
