/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type * as QueueDataStore from '@features/queue/ui/queueDataStore';

import { ChakraProvider } from '@chakra-ui/react';
import { clearQueueConfirmation } from '@features/queue/ui/queueConfirmationStore';
import { QueueDataRuntime } from '@features/queue/ui/QueueDataRuntime';
import { createTestQueueUiAdapter } from '@features/queue/ui/queueUiAdapter.testing';
import { QueueUiProvider } from '@features/queue/ui/QueueUiContext';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

const state = vi.hoisted(() => ({ isConnected: true, localItems: [] as unknown[] }));
const commands = vi.hoisted(() => ({
  cancelItem: vi.fn(() => Promise.resolve()),
  cancelScopedItems: vi.fn(() => Promise.resolve()),
}));

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: Record<string, unknown>) => (values ? `${key} ${JSON.stringify(values)}` : key),
  }),
}));
vi.mock('@features/models', () => ({ useModelLoads: () => [] }));
vi.mock('@features/queue/publicApi', () => ({
  createProductionQueueRealtimeRuntime: () => ({ dispose: vi.fn(), start: vi.fn() }),
  getQueueItemDetailOptions: (itemId: number) => ({ queryFn: () => null, queryKey: ['queue', 'item-detail', itemId] }),
  // The seeded read model stays put: nothing refetches it over the test's data.
  getQueueReadModelOptions: () => ({ queryFn: () => null, queryKey: ['queue'], staleTime: Number.POSITIVE_INFINITY }),
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

const serverItem = (id: number, status: 'in_progress' | 'pending') => ({
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
  const open = items.filter((item) => item.status === 'pending' || item.status === 'in_progress');
  client.setQueryData(['queue'], {
    current,
    items,
    next: null,
    scope: {},
    status: {
      processor: { isProcessing: true, isStarted: true },
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
  });

  await act(() =>
    root.render(
      <ChakraProvider value={system}>
        <QueryClientProvider client={client}>
          <QueueUiProvider adapter={createTestQueueUiAdapter({ isConnected: state.isConnected })}>
            <QueueCluster />
            <QueueDataRuntime />
          </QueueUiProvider>
        </QueryClientProvider>
      </ChakraProvider>
    )
  );
};

const cancelButton = () =>
  host.querySelector<HTMLButtonElement>('button[aria-label="widgets.queue.cancelCurrentItem"]')!;
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
});

describe('QueueCluster', () => {
  it('cancels the running item on a plain click without asking', async () => {
    await render();

    await act(() => userEvent.click(cancelButton()));

    await vi.waitFor(() => expect(commands.cancelItem).toHaveBeenCalledWith(7));
    expect(confirmation()).toBeNull();
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
      client.setQueryData(['queue'], (model: { status: { queue: object } }) => ({
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
