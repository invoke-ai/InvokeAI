import { ChakraProvider } from '@chakra-ui/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act, StrictMode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));
vi.mock('@features/queue/publicApi', () => ({
  getQueueItemDetailOptions: (itemId: number) => ({ queryFn: () => null, queryKey: ['queue', 'item-detail', itemId] }),
  getQueueReadModelOptions: () => ({ queryFn: () => null, queryKey: ['queue'], staleTime: Number.POSITIVE_INFINITY }),
  queueCommands: {},
}));

import { createTestQueueUiAdapter } from './queueUiAdapter.testing';
import { QueueUiProvider } from './QueueUiContext';
import { getPendingQueueItemReveal, requestQueueItemReveal } from './queueUiStore';
import { RecentSection } from './RecentSection';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

it('says once that an item it was asked to show is not in its list, instead of doing nothing', async () => {
  const notify = { error: vi.fn(), info: vi.fn(), success: vi.fn() };
  const client = new QueryClient();
  client.setQueryData(['queue'], {
    current: null,
    items: [],
    next: null,
    scope: {},
    status: {
      processor: { isProcessing: false, isStarted: true },
      queue: {
        canceled: 0,
        completed: 0,
        failed: 0,
        inProgress: 0,
        pending: 0,
        queueId: 'default',
        total: 0,
        waiting: 0,
      },
    },
  });
  requestQueueItemReveal(99);
  const request = getPendingQueueItemReveal();

  await act(() =>
    root.render(
      <StrictMode>
        <ChakraProvider value={system}>
          <QueryClientProvider client={client}>
            <QueueUiProvider adapter={createTestQueueUiAdapter({ notify })}>
              <RecentSection filter="all" revealRequest={request} />
            </QueueUiProvider>
          </QueryClientProvider>
        </ChakraProvider>
      </StrictMode>
    )
  );

  expect(notify.info).toHaveBeenCalledOnce();
  expect(notify.info).toHaveBeenCalledWith(
    'widgets.queue.revealUnavailable',
    'widgets.queue.revealUnavailableDescription'
  );
  expect(getPendingQueueItemReveal()).toBeNull();
});
