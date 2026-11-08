// Queue's POST hydration is a read; commands are deliberately outside owner-remount cancellation.
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

/** Explicit document exit can cancel reads and writes. Recovery must separately prove abandoned writes survive. */
export const isExpectedDocumentUnload = (event, { navigating, origin }) => {
  if (!navigating || event.kind === 'response' || event.responseStatus >= 400) {
    return false;
  }
  let url;
  if (event.kind === 'requestfailed' && /(?:ERR_ABORTED|NS_BINDING_ABORTED|cancelled)/i.test(event.message)) {
    url = new URL(event.url);
  } else if (event.kind === 'pageerror') {
    const match = /^(?:Fetch API|XMLHttpRequest) cannot load (https?:\/\/\S+) due to access control checks\.$/.exec(
      event.message.split('\n', 1)[0]
    );
    if (!match) {
      return false;
    }
    url = new URL(match[1]);
  } else {
    return false;
  }
  return (
    url.origin === origin &&
    (url.pathname.startsWith('/api/') ||
      (url.pathname === '/ws/socket.io/' && url.searchParams.get('transport') === 'polling'))
  );
};

/** Leaving the observed project retires its Gallery page reader, including a response whose body is still pending. */
export const isExpectedGalleryProjectSwitchAbort = (event, { origin }) => {
  if (
    event.kind !== 'requestfailed' ||
    event.method !== 'GET' ||
    event.responseStatus >= 400 ||
    !/(?:ERR_ABORTED|NS_BINDING_ABORTED|cancelled)/i.test(event.message)
  ) {
    return false;
  }
  const url = new URL(event.url);
  return url.origin === origin && url.pathname === '/api/v1/gallery/items/';
};
