import { createServer, request as requestHttp } from 'node:http';

/** Fault on the wire so the default production worker, not a page interception, handles these responses. */
export const startAssetFaultProxy = async (origin, chunk, failures) => {
  const preview = new URL(origin);
  const requests = [];
  const server = createServer((incoming, outgoing) => {
    const target = incoming.url;
    if (!target || !target.startsWith('/') || target.startsWith('//') || /[\\\s#]/.test(target)) {
      outgoing.writeHead(400);
      outgoing.end('Expected an origin-form request target');
      return;
    }
    if (target === `/${chunk}` && requests.length < failures) {
      requests.push(503);
      outgoing.writeHead(503, { 'Content-Type': 'text/javascript', 'Cache-Control': 'no-store' });
      outgoing.end('Temporary module outage');
      return;
    }
    const upstream = requestHttp(
      preview,
      {
        // Raw origin-form paths never participate in choosing the transport authority.
        path: target,
        method: incoming.method,
        headers: { ...incoming.headers, host: preview.host },
      },
      (response) => {
        if (target === `/${chunk}`) {
          requests.push(response.statusCode);
        }
        outgoing.writeHead(response.statusCode, response.headers);
        response.pipe(outgoing);
      }
    );
    upstream.on('error', (error) => outgoing.destroy(error));
    outgoing.on('close', () => upstream.destroy());
    incoming.pipe(upstream);
  });
  await new Promise((resolveListen, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolveListen);
  });
  return {
    origin: `http://127.0.0.1:${server.address().port}`,
    requests,
    close: () =>
      new Promise((resolveClose) => {
        server.closeAllConnections();
        server.close(resolveClose);
      }),
  };
};
