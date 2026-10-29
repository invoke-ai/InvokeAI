import { ChakraProvider } from '@chakra-ui/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { QueueUiAdapter } from './QueueUiContext';

const commands = vi.hoisted(() => ({ cancelItem: vi.fn(() => Promise.resolve()) }));

vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));
vi.mock('@features/queue/publicApi', () => ({
  getQueueItemDetailOptions: (itemId: number) => ({ queryFn: () => null, queryKey: ['queue', 'item-detail', itemId] }),
  getQueueReadModelOptions: () => ({ queryFn: () => null, queryKey: ['queue'], staleTime: Number.POSITIVE_INFINITY }),
  queueCommands: commands,
}));

import { QueueHeaderActions } from './QueueHeaderActions';
import { createTestQueueUiAdapter } from './queueUiAdapter.testing';
import { QueueUiProvider } from './QueueUiContext';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement;
let root: Root;

const running = {
  batchId: 'b',
  createdAt: '',
  id: 7,
  resultImageNames: [],
  sessionId: 's',
  status: 'in_progress',
  updatedAt: '',
};

const render = async (adapter: QueueUiAdapter) => {
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
        pending: 0,
        queueId: 'default',
        total: 1,
        waiting: 0,
      },
    },
  });

  await act(() =>
    root.render(
      <ChakraProvider value={system}>
        <QueryClientProvider client={client}>
          <QueueUiProvider adapter={adapter}>
            <QueueHeaderActions />
          </QueueUiProvider>
        </QueryClientProvider>
      </ChakraProvider>
    )
  );

  return host.querySelector<HTMLButtonElement>('button')!;
};

beforeEach(() => {
  commands.cancelItem.mockClear();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

describe('QueueHeaderActions', () => {
  it('cancels the running item at once', async () => {
    const cancel = await render(createTestQueueUiAdapter());

    await act(() => userEvent.click(cancel));

    await vi.waitFor(() => expect(commands.cancelItem).toHaveBeenCalledWith(7));
  });

  it("offers no cancel for another user's running item, nor while disconnected", async () => {
    expect((await render(createTestQueueUiAdapter({ canManageItem: () => false }))).disabled).toBe(true);
    expect((await render(createTestQueueUiAdapter({ isConnected: false }))).disabled).toBe(true);
  });
});
