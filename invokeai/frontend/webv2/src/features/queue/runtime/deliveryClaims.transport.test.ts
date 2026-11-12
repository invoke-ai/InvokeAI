import { queueBackend } from '@features/queue/data/httpRealtimeQueueBackend';
import { captureAccountScope } from '@platform/state/accountLifecycle';
import { afterEach, expect, it, vi } from 'vitest';

import { createDeliveryConfirmations } from './deliveryClaims';

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

it('replays a hung production confirmation with the exact delivery endpoint/body and keeps server delivery', async () => {
  const owner = captureAccountScope();
  const timeout = AbortSignal.timeout.bind(AbortSignal);
  vi.spyOn(AbortSignal, 'timeout').mockImplementation((ms) => timeout(Math.min(ms, 5)));
  let requests = 0;
  let delivered = false;
  let timeoutReason: unknown;
  vi.stubGlobal(
    'fetch',
    vi.fn((input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const body = init?.body ? JSON.parse(String(init.body)) : undefined;
      if (
        !String(input).endsWith('/api/v1/queue/default/i/99/delivery') ||
        init?.method !== 'POST' ||
        JSON.stringify(body) !== JSON.stringify({ action: 'confirm', claimant: 'tab:owner' })
      ) {
        return Promise.resolve(Response.json({ detail: 'Invalid delivery request' }, { status: 422 }));
      }
      delivered = true;
      if (++requests === 1) {
        return new Promise((_resolve, reject) => {
          init!.signal!.addEventListener(
            'abort',
            () => {
              timeoutReason = init!.signal!.reason;
              reject(timeoutReason);
            },
            { once: true }
          );
        });
      }
      return Promise.resolve(Response.json({ status: 'delivered' }));
    })
  );
  const reportError = vi.fn();
  const confirmations = createDeliveryConfirmations({
    backend: queueBackend,
    isActive: () => true,
    signal: owner.signal,
    reportError,
  });
  try {
    confirmations.confirm(99, 'tab:owner');
    await vi.waitFor(() => expect(requests).toBe(2), { timeout: 2_000 });
    expect(timeoutReason).toMatchObject({ name: 'TimeoutError' });
    expect(delivered).toBe(true);
    expect(reportError).not.toHaveBeenCalled();
  } finally {
    confirmations.dispose();
  }
});
