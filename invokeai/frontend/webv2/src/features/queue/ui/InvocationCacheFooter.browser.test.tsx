import type * as ServerApi from '@features/queue/data/serverApi';
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { InvocationCacheStatusDTO } from '@features/queue/data/serverTypes';

import { ChakraProvider } from '@chakra-ui/react';
import { queryClient } from '@platform/query/client';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { InvocationCacheFooter } from './InvocationCacheFooter';
import { QueueUiProvider, type QueueUiAdapter } from './QueueUiContext';

vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));
const server = vi.hoisted(() => ({
  clearInvocationCache: vi.fn<(signal?: AbortSignal) => Promise<unknown>>(),
  getInvocationCacheStatus: vi.fn<(signal?: AbortSignal) => Promise<InvocationCacheStatusDTO>>(),
  setInvocationCacheEnabled: vi.fn<(enabled: boolean, signal?: AbortSignal) => Promise<unknown>>(),
}));
vi.mock('@features/queue/data/serverApi', async (importOriginal) => ({
  ...(await importOriginal<typeof ServerApi>()),
  ...server,
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const status = (overrides: Partial<InvocationCacheStatusDTO> = {}): InvocationCacheStatusDTO => ({
  enabled: true,
  hits: 3,
  max_size: 512,
  misses: 1,
  size: 4,
  ...overrides,
});

const notify = { error: vi.fn(), info: vi.fn(), success: vi.fn() };
const adapter: QueueUiAdapter = {
  ItemActions: () => null,
  activeProjectId: null,
  canManageInvocationCache: true,
  canManageItem: () => true,
  canManageProcessor: true,
  canViewItemDetails: () => true,
  isConnected: true,
  isProjectOpen: () => true,
  notify,
  openQueue: () => undefined,
  queueJobsScope: 'all',
};

let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
  queryClient.clear();
  vi.clearAllMocks();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

const render = async () => {
  await act(() =>
    root.render(
      <ChakraProvider value={system}>
        <QueryClientProvider client={queryClient}>
          <QueueUiProvider adapter={adapter}>
            <InvocationCacheFooter />
          </QueueUiProvider>
        </QueryClientProvider>
      </ChakraProvider>
    )
  );
  // The status read retries once before it reports a failure.
  await vi.waitFor(() => expect(host.textContent).not.toContain('widgets.queue.invocationCache.loading'), {
    timeout: 5_000,
  });
  await act(() => host.querySelector<HTMLButtonElement>('button[aria-expanded]')!.click());
};

const button = (label: string) =>
  [...host.querySelectorAll('button')].find((candidate) => candidate.textContent?.includes(label))!;

describe('InvocationCacheFooter', () => {
  it('disables the cache controls and says why when the server keeps no cache', async () => {
    server.getInvocationCacheStatus.mockResolvedValue(status({ enabled: false, max_size: 0, size: 0 }));

    await render();

    expect(host.textContent).toContain('widgets.queue.invocationCache.off');
    expect(host.textContent).toContain('widgets.queue.invocationCache.noCapacity');
    expect(button('widgets.queue.invocationCache.clear').disabled).toBe(true);
    expect(button('widgets.queue.invocationCache.enable').disabled).toBe(true);
    for (const label of ['widgets.queue.invocationCache.clear', 'widgets.queue.invocationCache.enable']) {
      const description = document.getElementById(button(label).getAttribute('aria-describedby') ?? '');
      expect(description?.textContent).toBe('widgets.queue.invocationCache.noCapacity');
    }
  });

  it('says the cache is unavailable, with the reason, and offers no command when its status cannot be read', async () => {
    server.getInvocationCacheStatus.mockRejectedValue(new Error('Server offline'));

    await render();

    expect(host.textContent).toContain('widgets.queue.invocationCache.unavailable');
    expect(host.textContent).toContain('Server offline');
    expect(button('widgets.queue.invocationCache.clear').disabled).toBe(true);
    expect(button('widgets.queue.invocationCache.enable').disabled).toBe(true);
    expect(server.clearInvocationCache).not.toHaveBeenCalled();
  });

  it('disables a running cache, refreshes its status and reports a failed clear', async () => {
    server.getInvocationCacheStatus.mockResolvedValue(status());
    server.setInvocationCacheEnabled.mockResolvedValue(null);
    server.clearInvocationCache.mockRejectedValue(new Error('Server offline'));

    await render();
    expect(host.textContent).toContain('widgets.queue.invocationCache.sizeOfMax');

    server.getInvocationCacheStatus.mockResolvedValue(status({ enabled: false }));
    await act(() => button('widgets.queue.invocationCache.disable').click());

    await vi.waitFor(() => expect(button('widgets.queue.invocationCache.enable')).toBeDefined());
    expect(server.setInvocationCacheEnabled).toHaveBeenCalledWith(false, expect.any(AbortSignal));
    expect(notify.success).toHaveBeenCalledWith('widgets.queue.invocationCache.disableDone');

    await act(() => button('widgets.queue.invocationCache.clear').click());

    await vi.waitFor(() =>
      expect(notify.error).toHaveBeenCalledWith('widgets.queue.invocationCache.clearFailed', 'Server offline')
    );
  });

  it('reports nothing for a command whose account signed out while it ran', async () => {
    server.getInvocationCacheStatus.mockResolvedValue(status());
    server.clearInvocationCache.mockImplementation(
      (signal) =>
        new Promise((_resolve, reject) => {
          signal?.addEventListener('abort', () => reject(signal.reason));
        })
    );

    await render();
    await act(() => button('widgets.queue.invocationCache.clear').click());
    await act(() => {
      accountLifecycle.invalidate();
    });

    await vi.waitFor(() => expect(button('widgets.queue.invocationCache.clear').disabled).toBe(false));
    expect(notify.error).not.toHaveBeenCalled();
    expect(notify.success).not.toHaveBeenCalled();
  });
});
