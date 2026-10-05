import type { QueueServerItemDTO } from './serverTypes';

const QUEUE_PATH = '/api/v1/queue/default/';

export const createQueueItemDTO = (
  itemId: number,
  overrides: Partial<QueueServerItemDTO> = {}
): QueueServerItemDTO => ({
  batch_id: 'batch',
  created_at: `2026-01-01T00:00:00.${String(itemId).padStart(3, '0')}`,
  item_id: itemId,
  session_id: `session-${String(itemId)}`,
  status: 'completed',
  updated_at: '2026-01-01T00:00:00',
  ...overrides,
});

/**
 * A `fetch` stand-in for the backend's queue read routes. It answers `item_ids` like the server (every id, newest
 * first, or the first `limit` of them) and records each request as `METHOD route?query` so tests can assert exactly
 * what crossed the transport. `predatesItemIdsLimit` answers like a server without `limit`, with every id.
 */
export const createQueueServer = (
  items: readonly QueueServerItemDTO[],
  { predatesItemIdsLimit = false }: { predatesItemIdsLimit?: boolean } = {}
) => {
  const requests: string[] = [];
  const hydratedIds: number[][] = [];
  const respond = (body: unknown, status = 200) =>
    Promise.resolve(new Response(JSON.stringify(body), { headers: { 'content-type': 'application/json' }, status }));

  const fetch = (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = new URL(input instanceof Request ? input.url : String(input), 'http://localhost');
    const method = init?.method ?? 'GET';

    if (!url.pathname.startsWith(QUEUE_PATH)) {
      requests.push(`${method} ${url.pathname}`);
      return respond({ detail: 'Not a queue route' }, 404);
    }

    const route = url.pathname.slice(QUEUE_PATH.length);
    requests.push(`${method} ${route}${url.search}`);

    if (route === 'status') {
      const count = (status: QueueServerItemDTO['status']) => items.filter((item) => item.status === status).length;

      return respond({
        processor: { is_processing: count('in_progress') > 0, is_started: true },
        queue: {
          canceled: count('canceled'),
          completed: count('completed'),
          failed: count('failed'),
          in_progress: count('in_progress'),
          pending: count('pending'),
          queue_id: 'default',
          total: items.length,
          waiting: count('waiting'),
        },
      });
    }
    if (route === 'current' || route === 'next') {
      const status = route === 'current' ? 'in_progress' : 'pending';

      return respond(items.find((item) => item.status === status) ?? null);
    }
    if (route === 'item_ids') {
      const ids = items.map((item) => item.item_id).sort((left, right) => right - left);
      const limit = predatesItemIdsLimit ? null : url.searchParams.get('limit');
      const returned = limit === null ? ids : ids.slice(0, Number(limit));

      return respond({ item_ids: returned, total_count: returned.length });
    }
    if (route === 'items_by_ids') {
      const { item_ids: requested } = JSON.parse(String(init?.body)) as { item_ids: number[] };
      hydratedIds.push(requested);

      return respond(requested.flatMap((itemId) => items.find((item) => item.item_id === itemId) ?? []));
    }

    return respond({ detail: `Unhandled queue route ${route}` }, 404);
  };

  return { fetch, hydratedIds, requests };
};
