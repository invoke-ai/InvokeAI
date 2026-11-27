/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type * as QueueDataStore from '@features/queue/ui/queueDataStore';

import { ChakraProvider } from '@chakra-ui/react';
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
vi.mock('@features/models', () => ({ useModelLoads: () => [] }));
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
}: { current?: ReturnType<typeof serverItem> | null; items?: ReturnType<typeof serverItem>[] } = {}) => {
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
