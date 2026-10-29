/* eslint-disable react-perf/jsx-no-new-object-as-prop */
import type { InvocationCacheStatusDTO } from '@features/queue/data/serverTypes';
import type * as Http from '@platform/transport/http';

import { ChakraProvider } from '@chakra-ui/react';
import { invocationCacheKey } from '@features/queue/data/invocationCacheQueries';
import { queryClient } from '@platform/query/client';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { refreshWatchedModelCacheStats, type ModelCacheStats } from './modelCacheStore';
import { QueueHeaderActions } from './QueueHeaderActions';
import { createTestQueueUiAdapter } from './queueUiAdapter.testing';
import { QueueUiProvider } from './QueueUiContext';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: Record<string, unknown>) => (values ? `${key} ${JSON.stringify(values)}` : key),
  }),
}));

/** One fake server behind `apiFetchJson`: both caches' reads and commands, by path and method. */
const server = vi.hoisted(() => ({
  clearInvocations: vi.fn<() => Promise<unknown>>(),
  clearModels: vi.fn<() => Promise<unknown>>(),
  invocationStatus: vi.fn<() => Promise<InvocationCacheStatusDTO>>(),
  modelStats: vi.fn<() => Promise<ModelCacheStats | null>>(),
  setInvocationsEnabled: vi.fn<(enabled: boolean) => Promise<unknown>>(),
}));
vi.mock('@platform/transport/http', async (importOriginal) => ({
  ...(await importOriginal<typeof Http>()),
  apiFetchJson: (path: string, init?: RequestInit) => {
    const method = init?.method ?? 'GET';
    if (path === '/api/v1/app/invocation_cache/status') {
      return server.invocationStatus();
    }
    if (path === '/api/v1/app/invocation_cache' && method === 'DELETE') {
      return server.clearInvocations();
    }
    if (path === '/api/v1/app/invocation_cache/enable') {
      return server.setInvocationsEnabled(true);
    }
    if (path === '/api/v1/app/invocation_cache/disable') {
      return server.setInvocationsEnabled(false);
    }
    if (path === '/api/v2/models/stats') {
      return server.modelStats();
    }
    if (path === '/api/v2/models/empty_model_cache') {
      return server.clearModels();
    }
    throw new Error(`Unexpected request ${method} ${path}`);
  },
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const GIB = 1024 ** 3;

const invocationStatus = (overrides: Partial<InvocationCacheStatusDTO> = {}): InvocationCacheStatusDTO => ({
  enabled: true,
  hits: 41,
  max_size: 512,
  misses: 9,
  size: 214,
  ...overrides,
});

const modelStats = (overrides: Partial<ModelCacheStats> = {}): ModelCacheStats => ({
  cache_size: 12 * GIB,
  cache_used: 6.2 * GIB,
  cleared: 0,
  high_watermark: 8 * GIB,
  hits: 93,
  in_cache: 3,
  misses: 7,
  ...overrides,
});

const notify = { error: vi.fn(), info: vi.fn(), success: vi.fn() };

let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
  queryClient.clear();
  // Resets the model cache store, which is account-owned.
  accountLifecycle.invalidate();
  vi.clearAllMocks();
  server.invocationStatus.mockResolvedValue(invocationStatus());
  server.modelStats.mockResolvedValue(modelStats());
  server.clearInvocations.mockResolvedValue(null);
  server.clearModels.mockResolvedValue({ bytes_freed: 2 * GIB, models_cleared: 2 });
  server.setInvocationsEnabled.mockResolvedValue(null);
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

const render = async ({ canManageInvocationCache = true } = {}) => {
  await act(() =>
    root.render(
      <ChakraProvider value={system}>
        <QueryClientProvider client={queryClient}>
          <QueueUiProvider adapter={createTestQueueUiAdapter({ canManageInvocationCache, notify })}>
            <QueueHeaderActions />
          </QueueUiProvider>
        </QueryClientProvider>
      </ChakraProvider>
    )
  );

  return host.querySelector<HTMLButtonElement>('button')!;
};

const dialog = () => document.querySelector<HTMLElement>('[role="dialog"]');

const open = async (options?: { canManageInvocationCache?: boolean }) => {
  const trigger = await render(options);
  await act(() => userEvent.click(trigger));
  await vi.waitFor(() => expect(dialog()).not.toBeNull());
  // The status read retries once before it reports a failure.
  await vi.waitFor(() => expect(dialog()!.textContent).not.toContain('widgets.queue.cache.loading'), {
    timeout: 5_000,
  });

  return trigger;
};

const button = (name: string) =>
  [...dialog()!.querySelectorAll('button')].find(
    (candidate) => (candidate.getAttribute('aria-label') ?? candidate.textContent) === name
  )!;

const progressbar = (name: string) =>
  [...dialog()!.querySelectorAll('[role="progressbar"]')].find((bar) => bar.getAttribute('aria-label') === name);

describe('QueueHeaderActions', () => {
  it('opens the Cache popover from its header button, moving focus in and back out', async () => {
    const trigger = await render();

    expect(host.querySelectorAll('button')).toHaveLength(1);
    expect(trigger.getAttribute('aria-label')).toBe('widgets.queue.cache.title');
    expect(trigger.getAttribute('aria-haspopup')).toBe('dialog');

    await act(() => userEvent.click(trigger));
    await vi.waitFor(() => expect(document.activeElement).toBe(dialog()));
    const title = document.getElementById(dialog()!.getAttribute('aria-labelledby') ?? '');
    expect(title?.textContent).toBe('widgets.queue.cache.title');

    await act(() => userEvent.keyboard('{Escape}'));
    await vi.waitFor(() => expect(dialog()).toBeNull());
    expect(document.activeElement).toBe(trigger);
  });

  it('shows usage and hit rate per cache, and reads both again each time it opens', async () => {
    const trigger = await open();
    const text = dialog()!.textContent;

    expect(dialog()!.querySelectorAll('[role="progressbar"]')).toHaveLength(2);

    expect(text).toContain('214/512');
    expect(text).toContain('82%');
    expect(text).toContain('widgets.queue.cache.hitRate {"percent":82}');
    expect(text).toContain('6.2/12.0 GiB');
    expect(text).toContain('93%');
    expect(
      progressbar('widgets.queue.cache.invocationUsage {"max":"512","size":"214"}')?.getAttribute('aria-valuenow')
    ).toBe(String(214 / 512));
    expect(progressbar('widgets.queue.cache.modelUsage {"total":"12.0 GiB","used":"6.2 GiB"}')).toBeDefined();

    await act(() => userEvent.keyboard('{Escape}'));
    await vi.waitFor(() => expect(dialog()).toBeNull());
    await act(() => userEvent.click(trigger));

    await vi.waitFor(() => expect(server.invocationStatus).toHaveBeenCalledTimes(2));
    await vi.waitFor(() => expect(server.modelStats).toHaveBeenCalledTimes(2));
  });

  it('reads no cache statistics while closed, and follows model loads only while open', async () => {
    const trigger = await render();
    // What the realtime runtime does on a model load or reconnect.
    await refreshWatchedModelCacheStats();

    expect(server.modelStats).not.toHaveBeenCalled();
    expect(server.invocationStatus).not.toHaveBeenCalled();

    await act(() => userEvent.click(trigger));
    await vi.waitFor(() => expect(server.modelStats).toHaveBeenCalledOnce());
    await act(() => refreshWatchedModelCacheStats());
    expect(server.modelStats).toHaveBeenCalledTimes(2);

    await act(() => userEvent.keyboard('{Escape}'));
    await vi.waitFor(() => expect(dialog()).toBeNull());
    await refreshWatchedModelCacheStats();
    expect(server.modelStats).toHaveBeenCalledTimes(2);
  });

  it('says a cache has had no lookups instead of a hit rate', async () => {
    server.invocationStatus.mockResolvedValue(invocationStatus({ hits: 0, misses: 0 }));

    await open();

    expect(dialog()!.textContent).toContain('widgets.queue.cache.noLookups');
    expect(dialog()!.textContent).not.toContain('NaN');
  });

  it('clears each cache from its own row and reports the result', async () => {
    await open();

    await act(() => button('widgets.queue.cache.clearInvocations').click());
    await vi.waitFor(() => expect(notify.success).toHaveBeenCalledWith('widgets.queue.invocationCache.clearDone'));
    expect(server.clearInvocations).toHaveBeenCalledOnce();

    await act(() => button('widgets.queue.cache.clearModels').click());
    await vi.waitFor(() =>
      expect(notify.success).toHaveBeenCalledWith(
        'widgets.queue.modelCache.cleared',
        'widgets.queue.modelCache.clearedDescription {"bytes":"2.0 GiB","count":2}'
      )
    );
    expect(server.clearModels).toHaveBeenCalledOnce();
  });

  it('reports a failed clear with the server reason', async () => {
    server.clearInvocations.mockRejectedValue(new Error('Server offline'));

    await open();
    await act(() => button('widgets.queue.cache.clearInvocations').click());

    await vi.waitFor(() =>
      expect(notify.error).toHaveBeenCalledWith('widgets.queue.invocationCache.clearFailed', 'Server offline')
    );
  });

  it('disables and re-enables the invocation cache from the footer', async () => {
    await open();
    expect(dialog()!.textContent).toContain('widgets.queue.cache.invocationsOn');

    server.invocationStatus.mockResolvedValue(invocationStatus({ enabled: false }));
    await act(() => button('widgets.queue.cache.disable').click());

    await vi.waitFor(() => expect(button('widgets.queue.cache.enable')).toBeDefined());
    expect(server.setInvocationsEnabled).toHaveBeenCalledWith(false);
    expect(notify.success).toHaveBeenCalledWith('widgets.queue.invocationCache.disableDone');
    expect(dialog()!.textContent).toContain('widgets.queue.cache.invocationsOff');

    server.invocationStatus.mockResolvedValue(invocationStatus());
    await act(() => button('widgets.queue.cache.enable').click());

    await vi.waitFor(() => expect(button('widgets.queue.cache.disable')).toBeDefined());
    expect(server.setInvocationsEnabled).toHaveBeenLastCalledWith(true);
  });

  it('shows a non-administrator only the model cache', async () => {
    await open({ canManageInvocationCache: false });

    expect(dialog()!.textContent).toContain('widgets.queue.cache.models');
    expect(dialog()!.textContent).not.toContain('widgets.queue.cache.invocations');
    expect(button('widgets.queue.cache.clearModels')).toBeDefined();
    expect(button('widgets.queue.cache.disable')).toBeUndefined();
    expect(server.invocationStatus).not.toHaveBeenCalled();
  });

  it('disables the invocation commands and says why when the server keeps no cache', async () => {
    server.invocationStatus.mockResolvedValue(invocationStatus({ enabled: false, max_size: 0, size: 0 }));

    await open();

    expect(dialog()!.textContent).toContain('widgets.queue.invocationCache.off');
    for (const name of ['widgets.queue.cache.clearInvocations', 'widgets.queue.cache.enable']) {
      expect(button(name).disabled).toBe(true);
      const description = document.getElementById(button(name).getAttribute('aria-describedby') ?? '');
      expect(description?.textContent).toBe('widgets.queue.invocationCache.noCapacity');
    }
  });

  it('offers no invocation command and claims no state once a refresh fails', async () => {
    await open();
    server.invocationStatus.mockRejectedValue(new Error('Server offline'));

    await act(() => queryClient.invalidateQueries({ queryKey: invocationCacheKey }));
    // The status read retries once before it reports a failure.
    await vi.waitFor(() => expect(dialog()!.textContent).toContain('Server offline'), { timeout: 5_000 });

    expect(button('widgets.queue.cache.clearInvocations').disabled).toBe(true);
    expect(button('widgets.queue.cache.disable').disabled).toBe(true);
    expect(dialog()!.textContent).not.toContain('widgets.queue.cache.invocationsOn');
  });

  it('shows why a cache cannot be read and offers no invocation command', async () => {
    server.invocationStatus.mockRejectedValue(new Error('Server offline'));
    server.modelStats.mockRejectedValue(new Error('Stats offline'));

    await open();
    await vi.waitFor(() => expect(dialog()!.textContent).toContain('Stats offline'));

    expect(dialog()!.textContent).toContain('Server offline');
    expect(button('widgets.queue.cache.clearInvocations').disabled).toBe(true);
    // With no reading there is no Enable/Disable to offer, nor a state to claim.
    expect(button('widgets.queue.cache.enable')).toBeUndefined();
    expect(dialog()!.textContent).not.toContain('widgets.queue.cache.invocationsO');
    expect(dialog()!.querySelectorAll('[role="progressbar"]')).toHaveLength(0);
  });
});
