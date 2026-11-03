import { ChakraProvider } from '@chakra-ui/react';
import { createQueueItemDTO, createQueueServer } from '@features/queue/data/queueServer.testing';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { focusManager, QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { QueueStatusBand } from './QueueStatusBand';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (_key: string, values: { inProgress: number; pending: number }) =>
      `${String(values.inProgress)} running, ${String(values.pending)} waiting`,
  }),
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const POLL_MS = 5_000;

let host: HTMLDivElement;
let root: Root;
let server: ReturnType<typeof createQueueServer>;
let queryClient: QueryClient;

const advance = (ms: number) => act(() => vi.advanceTimersByTimeAsync(ms));
/** Response bodies settle on real browser tasks: let each poll finish so it cannot dedupe the next tick. */
const settle = () => act(() => vi.waitFor(() => expect(queryClient.isFetching()).toBe(0)));
const pollOnce = async () => {
  await advance(POLL_MS);
  await settle();
};

beforeEach(async () => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'] });
  accountLifecycle.activate('queue-status-band');
  // A long queue: the band must not read any of it beyond the counts.
  server = createQueueServer(
    Array.from({ length: 1_000 }, (_, index) =>
      createQueueItemDTO(index + 1, { status: index === 999 ? 'in_progress' : index > 996 ? 'pending' : 'completed' })
    )
  );
  vi.stubGlobal('fetch', server.fetch);
  // Production disables focus refetching, so only the band's own interval may read again.
  queryClient = new QueryClient({ defaultOptions: { queries: { refetchOnWindowFocus: false, retry: false } } });
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root.render(
      <QueryClientProvider client={queryClient}>
        <ChakraProvider value={system}>
          <QueueStatusBand />
        </ChakraProvider>
      </QueryClientProvider>
    )
  );
  await settle();
  expect(host.textContent).toContain('1 running, 2 waiting');
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
  focusManager.setFocused(undefined);
  vi.unstubAllGlobals();
  vi.useRealTimers();
  accountLifecycle.invalidate();
});

describe('QueueStatusBand', () => {
  it('polls one status request per interval and reads nothing else', async () => {
    await pollOnce();
    await pollOnce();
    await pollOnce();

    expect(server.requests).toEqual(['GET status', 'GET status', 'GET status', 'GET status']);
  });

  it('stops polling while the document is hidden and resumes when it is shown', async () => {
    await act(() => focusManager.setFocused(false));
    await advance(POLL_MS * 3);

    expect(server.requests).toEqual(['GET status']);

    await act(() => focusManager.setFocused(true));
    await pollOnce();

    expect(server.requests).toEqual(['GET status', 'GET status']);
  });
});
