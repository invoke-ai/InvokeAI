import { accountLifecycle } from '@platform/state/accountLifecycle';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { enqueueUtility } from './submissionApi';

const request = {
  graph: { edges: [], id: 'utility', nodes: {} },
  origin: 'webv2:util:test',
  idempotencyKey: 'utility:test',
};

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('utility production transport', () => {
  it('recovers a committed hung POST after its actual timeout signal, and acknowledges the exact receipt', async () => {
    accountLifecycle.activate('utility-transport');
    const timeout = AbortSignal.timeout.bind(AbortSignal);
    vi.spyOn(AbortSignal, 'timeout').mockImplementation((ms) => timeout(Math.min(ms, 5)));
    let admitted = 0;
    let acknowledged = false;
    let timedOut: unknown;
    const fetch = vi.fn((input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = String(input);
      const body = init?.body ? JSON.parse(String(init.body)) : undefined;
      if (
        url.endsWith('/enqueue_batch') &&
        init?.method === 'POST' &&
        body.batch.idempotency_key === request.idempotencyKey
      ) {
        admitted += 1;
        return new Promise((_resolve, reject) => {
          const abort = () => {
            timedOut = init!.signal!.reason;
            reject(timedOut);
          };
          init!.signal!.addEventListener('abort', abort, { once: true });
        });
      }
      if (
        url.includes('/enqueue_batch/receipt?') &&
        new URL(url).searchParams.get('idempotency_key') === request.idempotencyKey
      ) {
        return Promise.resolve(Response.json({ batch_id: 'accepted', enqueued: 1, item_ids: [41], requested: 1 }));
      }
      if (
        url.endsWith('/enqueue_batch/acknowledge') &&
        init?.method === 'POST' &&
        JSON.stringify(body) === JSON.stringify({ idempotency_key: request.idempotencyKey })
      ) {
        acknowledged = true;
        return Promise.resolve(new Response(null, { status: 204 }));
      }
      return Promise.resolve(Response.json({ detail: 'Invalid endpoint or body' }, { status: 422 }));
    });
    vi.stubGlobal('fetch', fetch);
    const result = await enqueueUtility(request);
    expect(timedOut).toMatchObject({ name: 'TimeoutError' });
    expect(result.itemIds).toEqual([41]);
    expect(admitted).toBe(1);
    expect(acknowledged).toBe(false);
    result.acknowledge();
    await vi.waitFor(() => expect(acknowledged).toBe(true));
  });
});
