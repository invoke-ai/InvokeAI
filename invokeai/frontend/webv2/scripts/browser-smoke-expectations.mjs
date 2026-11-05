// queueReadModelOptions owns one read pipeline: four parallel reads, then read-only POST hydration under the same
// query/owner signal. Keep the complete bounded read interface together; Queue commands remain critical.
const queueReadMethods = new Map([
  ['/api/v1/queue/default/status', 'GET'],
  ['/api/v1/queue/default/current', 'GET'],
  ['/api/v1/queue/default/next', 'GET'],
  ['/api/v1/queue/default/item_ids', 'GET'],
  ['/api/v1/queue/default/items_by_ids', 'POST'],
]);

export const isQueueReadRequest = (event, origin) => {
  const url = new URL(event.url);
  const method = queueReadMethods.get(url.pathname);
  return method !== undefined && url.origin === origin && method === event.method;
};

/** WebKit denies these old-document requests before dispatch during the deliberate reload recovery scenario. */
export const isExpectedWebKitUnloadError = (event, { engine, navigating, origin }) => {
  if (engine !== 'webkit' || !navigating || event.kind !== 'pageerror') {
    return false;
  }
  const match = /^(Fetch API|XMLHttpRequest) cannot load (https?:\/\/\S+) due to access control checks\.$/.exec(
    event.message.split('\n', 1)[0]
  );
  if (!match) {
    return false;
  }
  const url = new URL(match[2]);
  return (
    url.origin === origin &&
    ((match[1] === 'Fetch API' &&
      ((url.pathname === '/api/v1/projects/fixture-project-002/workspace' && !url.search) ||
        queueReadMethods.has(url.pathname))) ||
      (match[1] === 'XMLHttpRequest' &&
        url.pathname === '/ws/socket.io/' &&
        url.searchParams.get('transport') === 'polling'))
  );
};
