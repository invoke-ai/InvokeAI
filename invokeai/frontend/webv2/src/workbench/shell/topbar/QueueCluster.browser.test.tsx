/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type * as QueueDataStore from '@features/queue/ui/queueDataStore';

import { ChakraProvider } from '@chakra-ui/react';
import { CANCEL_CURRENT_SETTLE_MS } from '@features/queue/ui/cancelCurrent';
import { clearQueueConfirmation } from '@features/queue/ui/queueConfirmationStore';
import { QueueDataRuntime } from '@features/queue/ui/QueueDataRuntime';
import { createTestLocalRun, createTestQueueUiAdapter } from '@features/queue/ui/queueUiAdapter.testing';
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
  /** The queue processor is paused. */
  isPaused: false,
  /** "Show jobs from" the active project; with it, how many items run anywhere in the queue. */
  projectScoped: false,
  wholeQueueInProgress: 0,
  localItems: [] as unknown[],
  /** This tab's runs the backend has not accepted, which only cancel-all reaches. */
  localRuns: [] as QueueLocalRun[],
  /**
   * The server's items now. The seeded reading stays cached until a read asks again; null keeps the server as the test
   * seeded it.
   */
  serverItems: null as null | Array<{ id: number; status: string }>,
}));
const commands = vi.hoisted(() => ({
  // Cancels an unfinished item; returns a finished one unchanged, as the server does.
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
  // The seeded read model stays put: nothing refetches it over the test's data unless an action asks afresh.
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

/** The server's reading of the scope: item 7 running and items 8 and 9 waiting, unless a test says otherwise. */
const render = async ({
  current = running as ReturnType<typeof serverItem> | null,
  items = [running, serverItem(8, 'pending'), serverItem(9, 'pending')],
}: { current?: ReturnType<typeof serverItem> | null; items?: ReturnType<typeof serverItem>[] } = {}) => {
  client = new QueryClient();
  client.setQueryData(['queue', 'read-model', 'all'], readModel({ current, items }));
  if (state.projectScoped) {
    // The whole queue's status read the Launchpad shares, fresh, so nothing reads it again here.
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

/** What a read of the server finds now. */
function serverReading() {
  const items = (state.serverItems ?? []) as ReturnType<typeof serverItem>[];
  return readModel({ current: items.find((item) => item.status === 'in_progress') ?? null, items });
}

/** A server reading of the scope; the next item is the oldest one waiting, as the server picks it. */
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

/** The ×: named for the running item, or between items for the next one, unless Shift arms cancel-all. */
const cancelButton = () =>
  host.querySelector<HTMLButtonElement>(
    'button[aria-label="widgets.queue.cancelCurrentItem"], button[aria-label="widgets.queue.cancelNextItem"]'
  )!;
const confirmation = () => document.querySelector('[role="alertdialog"]');
const queueButton = () => host.querySelector<HTMLButtonElement>('button[aria-label^="topbar.queue.remaining"]')!;
const announcement = () => host.querySelector('[aria-live="polite"]')?.textContent;

/** One of this project's runs: three images, the backend items 7 to 9, the first of them generating. */
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
  // A cancel that stopped an item keeps the command busy for its settle window; a new account ends it, so the next
  // test starts free.
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

  it('keeps the cancel button in place, disabled, while nothing is running or waiting', async () => {
    await render({ current: null, items: [serverItem(6, 'completed')] });

    expect(cancelButton().disabled).toBe(true);
  });

  it('names and stops the next waiting item between items, while nothing runs', async () => {
    // Item 7 finished and item 8 has not started yet.
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

    // Once the reconciling read has landed and the settle window is over, a deliberate click stops the next item.
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

  it('stops nothing waiting while the processor is paused: the × is disabled', async () => {
    state.isPaused = true;
    await render({ current: null, items: [serverItem(7, 'completed'), serverItem(8, 'pending')] });

    expect(cancelButton().disabled).toBe(true);
  });

  it("in a project view, stops nothing waiting while another project's item runs: the × is disabled", async () => {
    state.projectScoped = true;
    state.wholeQueueInProgress = 1;
    // The project's own reading: nothing of its own runs, and its next item waits.
    await render({ current: null, items: [serverItem(7, 'completed'), serverItem(8, 'pending')] });

    expect(cancelButton().disabled).toBe(true);
  });

  it('in a project view, offers the next item while nothing runs anywhere', async () => {
    state.projectScoped = true;
    await render({ current: null, items: [serverItem(7, 'completed'), serverItem(8, 'pending')] });

    expect(cancelButton().disabled).toBe(false);
    expect(cancelButton().getAttribute('aria-label')).toBe('widgets.queue.cancelNextItem');
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
    // The server has moved on: item 7 completed and item 8 runs.
    state.serverItems = [serverItem(7, 'completed'), serverItem(8, 'in_progress'), serverItem(9, 'pending')];
    await render();

    await act(() => userEvent.click(cancelButton()));

    await vi.waitFor(() => expect(commands.cancelItem.mock.calls).toEqual([[7], [8]]));
    await vi.waitFor(() => expect(notify.success).toHaveBeenCalledOnce());
    expect(notify.info).not.toHaveBeenCalled();
  });

  it('says there is nothing to cancel, and claims no cancel, when everything had already finished', async () => {
    state.serverItems = [serverItem(7, 'completed')];
    await render({ current: running, items: [running] });

    await act(() => userEvent.click(cancelButton()));

    await vi.waitFor(() =>
      expect(notify.info).toHaveBeenCalledExactlyOnceWith('widgets.queue.noCurrentItem', expect.any(String))
    );
    expect(commands.cancelItem.mock.calls).toEqual([[7]]);
    expect(notify.success).not.toHaveBeenCalled();
  });

  it('does not arm cancel-all on the disabled button, though Shift is held over it with work waiting', async () => {
    // Offline, nothing on the server is in reach; this tab's unaccepted run still is, for cancel-all alone.
    state.isConnected = false;
    state.localItems = [projectRun('pending', [])];
    state.localRuns = [createTestLocalRun()];
    await render({ current: null, items: [] });
    const button = cancelButton();

    await act(() => userEvent.hover(button));
    await act(() => userEvent.keyboard('{Shift>}'));
    const labelWithShift = button.getAttribute('aria-label');
    const disabledWithShift = button.disabled;
    await act(() => userEvent.keyboard('{/Shift}'));

    expect(labelWithShift).toBe('widgets.queue.cancelCurrentItem');
    expect(disabledWithShift).toBe(true);
  });

  it('asks before cancelling everything on shift-click, and cancels nothing until confirmed', async () => {
    await render();

    await act(() => userEvent.click(cancelButton(), { modifiers: ['Shift'] } as never));

    await expect.poll(() => confirmation()?.getAttribute('data-state')).toBe('open');
    expect(commands.cancelScopedItems).not.toHaveBeenCalled();
    expect(commands.cancelItem).not.toHaveBeenCalled();
  });

  it('asks the same way from the keyboard with Shift+Enter', async () => {
    await render();

    cancelButton().focus();
    await act(() => userEvent.keyboard('{Shift>}{Enter}{/Shift}'));

    await expect.poll(() => confirmation()?.getAttribute('data-state')).toBe('open');
    expect(commands.cancelItem).not.toHaveBeenCalled();
  });

  it("counts the project's run once though the server lists its items too, and shows its image progress", async () => {
    state.localItems = [projectRun('running')];
    await render();

    expect(queueButton().textContent).toBe('3');
    expect(announcement()).toBe('topbar.queue.imageProgress {"current":1,"total":3}');
  });

  it('adds work queued beyond the project, and says where it comes from', async () => {
    state.localItems = [projectRun('running')];
    await render({
      items: [
        running,
        serverItem(8, 'pending'),
        serverItem(9, 'pending'),
        serverItem(20, 'pending'),
        serverItem(21, 'pending'),
      ],
    });

    expect(queueButton().textContent).toBe('5');
    // The live region keeps to the project's own progress.
    expect(announcement()).toBe('topbar.queue.imageProgress {"current":1,"total":3}');

    await act(() => userEvent.hover(queueButton()));
    await expect
      .poll(() => document.querySelector('[role="tooltip"]')?.textContent ?? '', { timeout: 3_000 })
      .toContain('topbar.queue.alsoQueuedElsewhere {"count":2}');
  });

  it('counts a run larger than the read window once, and nothing as elsewhere', async () => {
    // 80 images: the read model's 50-item window shows only the newest of them.
    const backendItemIds = Array.from({ length: 80 }, (_, index) => 7 + index);
    state.localItems = [projectRun('running', backendItemIds)];
    const window = backendItemIds.slice(-50).map((id) => serverItem(id, id === 7 ? 'in_progress' : 'pending'));
    await render({ items: window });
    // The server counts all 80 open items, not just the window it lists.
    await act(() =>
      client.setQueryData(['queue', 'read-model', 'all'], (model: { status: { queue: object } }) => ({
        ...model,
        status: { ...model.status, queue: { ...model.status.queue, inProgress: 1, pending: 79 } },
      }))
    );

    expect(queueButton().textContent).toBe('80');

    await act(() => userEvent.hover(queueButton()));
    await expect
      .poll(() => document.querySelector('[role="tooltip"]')?.textContent ?? '', { timeout: 3_000 })
      .toContain('topbar.queue.generatingImage');
    expect(document.querySelector('[role="tooltip"]')?.textContent).not.toContain('Elsewhere');
  });

  it('counts work from other projects alone, without naming its source in the live region', async () => {
    await render();

    expect(queueButton().textContent).toBe('3');
    expect(announcement()).toBe('widgets.queueStatus.queued {"count":3}');
  });

  it("reads idle once the project's run finished, though the server has not caught up yet", async () => {
    state.localItems = [projectRun('completed')];
    await render();

    expect(queueButton().textContent).toBe('0');
    expect(announcement()).toBe('topbar.queue.idle');
  });

  it("shows the project's runs waiting on a lost connection instead of an idle queue", async () => {
    state.isConnected = false;
    state.localItems = [projectRun('pending', [])];
    // Nothing reached the server: the run waits locally for the connection.
    await render({ current: null, items: [] });

    const offlineButton = host.querySelector<HTMLButtonElement>('button[aria-label^="topbar.queue.remainingOffline"]');

    expect(offlineButton).not.toBeNull();
    expect(offlineButton!.getAttribute('aria-label')).toContain('"count":1');
    expect(offlineButton!.querySelector('svg.lucide-cloud-off')).not.toBeNull();
    expect(announcement()).toBe('topbar.queue.offlineWaiting {"count":1}');
  });
});
