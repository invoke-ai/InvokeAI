import type { QueueItemReadModel } from '@features/queue/contracts';
import type * as QueuePublicApi from '@features/queue/publicApi';
import type { QueueItemDetailReadModel } from '@features/queue/react';

import { ChakraProvider } from '@chakra-ui/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { QueueItemActions } from './QueueItemActions';

vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));
const readDetail = vi.hoisted(() => vi.fn<(itemId: number) => Promise<QueueItemDetailReadModel>>());
vi.mock('@features/queue/publicApi', async (importOriginal) => ({
  ...(await importOriginal<typeof QueuePublicApi>()),
  getQueueItemDetailOptions: (itemId: number) => ({
    queryFn: () => readDetail(itemId),
    queryKey: ['queue', 'item-detail', itemId],
  }),
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const item = { batchId: 'batch-9', id: 9, status: 'failed' } as QueueItemReadModel;
const detail = (workflow: Record<string, unknown> | null): QueueItemDetailReadModel => ({
  errorMessage: 'Out of memory',
  errorTraceback: 'Traceback…',
  errorType: 'OutOfMemoryError',
  graph: { id: 'graph-9' },
  id: 9,
  priority: 0,
  results: {},
  sessionId: 'session-9',
  status: 'failed',
  workflow,
});

let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
  readDetail.mockReset();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

const render = async () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });

  await act(() =>
    root.render(
      <ChakraProvider value={system}>
        <QueryClientProvider client={client}>
          <QueueItemActions item={item} />
        </QueryClientProvider>
      </ChakraProvider>
    )
  );
};

const button = (label: string) =>
  [...document.querySelectorAll('button')].find((candidate) => candidate.textContent?.includes(label));
const dialogText = () => document.querySelector('[role="dialog"]')?.textContent ?? '';

describe('QueueItemActions View JSON', () => {
  it('reads the saved execution data only when opened, and offers a retry when the read fails', async () => {
    readDetail.mockRejectedValueOnce(new Error('Queue item 9 not found'));
    await render();

    expect(readDetail).not.toHaveBeenCalled();

    await act(() => button('common.viewJson')!.click());

    await vi.waitFor(() => expect(dialogText()).toContain('widgets.queue.detailLoadFailed'));
    expect(document.querySelector('[role="dialog"] [role="alert"]')?.textContent).toContain('Queue item 9 not found');

    readDetail.mockResolvedValueOnce(detail(null));
    const retry = button('common.retry')!;
    retry.focus();
    await act(() => retry.click());

    await vi.waitFor(() => expect(dialogText()).toContain('widgets.queue.detailViews.graph'));
    // The Retry button is gone; focus lands on the view now shown instead of falling to the page.
    expect(document.activeElement?.getAttribute('role')).toBe('tab');
    expect(document.activeElement?.getAttribute('aria-selected')).toBe('true');
    expect(dialogText()).not.toContain('widgets.queue.detailViews.workflow');
    expect(readDetail).toHaveBeenCalledTimes(2);
  });

  it('adds a Workflow view when the batch embedded one', async () => {
    readDetail.mockResolvedValue(detail({ name: 'Upscale' }));
    await render();

    await act(() => button('common.viewJson')!.click());

    await vi.waitFor(() => expect(dialogText()).toContain('widgets.queue.detailViews.workflow'));
  });
});
