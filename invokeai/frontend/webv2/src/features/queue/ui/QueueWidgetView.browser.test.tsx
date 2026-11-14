import { ChakraProvider } from '@chakra-ui/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, expect, it, vi } from 'vitest';

// The filters' English labels, so widths are real; other keys stay as they are.
vi.mock('react-i18next', () => {
  const labels: Record<string, string> = {
    'common.active': 'Active',
    'common.all': 'All',
    'common.canceled': 'Canceled',
    'common.done': 'Done',
    'common.failed': 'Failed',
  };
  return { useTranslation: () => ({ i18n: { resolvedLanguage: 'en' }, t: (key: string) => labels[key] ?? key }) };
});
vi.mock('@features/queue/publicApi', () => ({
  getQueueItemDetailOptions: (itemId: number) => ({ queryFn: () => null, queryKey: ['queue', 'item-detail', itemId] }),
  getQueueReadModelOptions: () => ({ queryFn: () => null, queryKey: ['queue'], staleTime: Number.POSITIVE_INFINITY }),
  queueCommands: {},
}));

import { QUEUE_FILTER_TABS_ID } from './QueueFilterTabs';
import { createTestQueueUiAdapter } from './queueUiAdapter.testing';
import { QueueUiProvider } from './QueueUiContext';
import { QueueWidgetView } from './QueueWidgetView';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;
let root: Root | null = null;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

interface Counts {
  canceled: number;
  completed: number;
  failed: number;
  inProgress: number;
  pending: number;
}

const renderQueue = async (width: number, counts: Counts) => {
  const client = new QueryClient();
  client.setQueryData(['queue'], {
    current: null,
    items: [],
    next: null,
    scope: {},
    status: {
      processor: { isProcessing: false, isStarted: true },
      queue: {
        ...counts,
        queueId: 'default',
        total: Object.values(counts).reduce((sum, count) => sum + count, 0),
        waiting: 0,
      },
    },
  });
  host = document.createElement('div');
  host.style.height = '400px';
  host.style.width = `${width}px`;
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root!.render(
      <ChakraProvider value={system}>
        <QueryClientProvider client={client}>
          <QueueUiProvider adapter={createTestQueueUiAdapter()}>
            <QueueWidgetView region="right" />
          </QueueUiProvider>
        </QueryClientProvider>
      </ChakraProvider>
    )
  );
  return host.querySelector<HTMLElement>(`[role="tablist"]:has(#${QUEUE_FILTER_TABS_ID}-tab-all)`)!;
};

it('scrolls its filter strip in a narrow widget with large counts', async () => {
  const large = 12_345;
  const strip = await renderQueue(220, {
    canceled: large,
    completed: large,
    failed: large,
    inProgress: large,
    pending: large,
  });

  expect(strip.scrollWidth).toBeGreaterThan(strip.clientWidth);
  expect(strip.getBoundingClientRect().right).toBeLessThanOrEqual(host!.getBoundingClientRect().right);
});

it('fits all five filters without scrolling in a docked rail', async () => {
  // A rail's default width, with a busy session's counts.
  const strip = await renderQueue(450, { canceled: 83, completed: 83, failed: 83, inProgress: 83, pending: 168 });

  expect(strip.querySelectorAll('[role="tab"]')).toHaveLength(5);
  expect(strip.scrollWidth).toBe(strip.clientWidth);
});
