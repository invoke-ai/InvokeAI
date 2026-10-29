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

const state = vi.hoisted(() => ({ isConnected: true }));
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
  useWorkbenchSelector: (select: (snapshot: unknown) => unknown) =>
    select({ backendConnection: { status: state.isConnected ? 'connected' : 'disconnected' } }),
}));
vi.mock('@workbench/useOpenWorkbenchWidget', () => ({ useOpenWorkbenchWidget: () => vi.fn() }));
vi.mock('@workbench/queue-integration/useActiveQueueProgress', () => ({
  useActiveQueueProgress: () => ({
    progress: null,
    queueItems: [],
    summary: { current: 1, remaining: 3, runningQueueItemId: null, total: 3 },
  }),
}));

const { QueueCluster } = await import('./QueueCluster');

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement;
let root: Root;

const running = {
  batchId: 'batch-1',
  createdAt: '2026-06-27T00:00:00Z',
  id: 7,
  resultImageNames: [],
  sessionId: 'session-7',
  status: 'in_progress',
  updatedAt: '2026-06-27T00:00:00Z',
};

const render = async () => {
  const client = new QueryClient();
  client.setQueryData(['queue'], {
    current: running,
    items: [running],
    next: null,
    scope: {},
    status: {
      processor: { isProcessing: true, isStarted: true },
      queue: {
        canceled: 0,
        completed: 0,
        failed: 0,
        inProgress: 1,
        pending: 2,
        queueId: 'default',
        total: 3,
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

beforeEach(() => {
  state.isConnected = true;
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

  it('shows open work waiting on a lost connection instead of an idle queue', async () => {
    state.isConnected = false;
    await render();

    const queueButton = host.querySelector<HTMLButtonElement>('button[aria-label^="topbar.queue.remainingOffline"]');

    expect(queueButton).not.toBeNull();
    expect(queueButton!.getAttribute('aria-label')).toContain('"count":3');
    expect(queueButton!.querySelector('svg.lucide-cloud-off')).not.toBeNull();
  });
});
