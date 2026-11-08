import assert from 'node:assert/strict';
import { createServer, request as requestHttp } from 'node:http';
import { test } from 'node:test';

import { startAssetFaultProxy } from './asset-fault-proxy.mjs';

const startUpstream = async (name, respond) => {
  const hits = [];
  const server = createServer(async (request, response) => {
    const chunks = [];
    for await (const chunk of request) {
      chunks.push(chunk);
    }
    const hit = {
      target: request.url,
      method: request.method,
      host: request.headers.host,
      body: Buffer.concat(chunks).toString(),
    };
    hits.push(hit);
    if (respond) {
      respond(response, hit);
    } else {
      response.writeHead(207, { 'Content-Type': 'text/plain', 'X-Upstream': name });
      response.write(`${name}:`);
      response.end(hit.body);
    }
  });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  return {
    origin: `http://127.0.0.1:${server.address().port}`,
    hits,
    close: () =>
      new Promise((resolve) => {
        server.closeAllConnections();
        server.close(resolve);
      }),
  };
};

const fixture = async (t, respond) => {
  const expected = await startUpstream('expected', respond);
  const trap = await startUpstream('trap');
  const proxy = await startAssetFaultProxy(expected.origin, 'assets/widget.js', 2);
  t.after(async () => {
    await proxy.close();
    await Promise.all([expected.close(), trap.close()]);
  });
  return { expected, trap, proxy };
};

/** Send the raw request target without WHATWG URL normalization at the client. */
const send = (origin, path, { method = 'GET', body = '', headers = {} } = {}) =>
  new Promise((resolve, reject) => {
    const request = requestHttp(origin, { path, method, headers, agent: false }, (response) => {
      const chunks = [];
      response.on('data', (chunk) => chunks.push(chunk));
      response.on('error', reject);
      response.on('end', () =>
        resolve({
          status: response.statusCode,
          headers: response.headers,
          body: Buffer.concat(chunks).toString(),
        })
      );
    });
    request.on('error', reject);
    request.write(body.slice(0, Math.floor(body.length / 2)));
    request.end(body.slice(Math.floor(body.length / 2)));
  });

test('rejects authority-bearing targets before reaching either upstream', async (t) => {
  const { expected, trap, proxy } = await fixture(t);
  const authority = new URL(trap.origin).host;
  for (const target of [
    `${trap.origin}/private`,
    `https://${authority}/private`,
    `http://fixture@${authority}/private`,
    `//${authority}/private`,
    `///${authority}/private`,
    `////${authority}/private`,
    `/\\${authority}/private`,
    `\\${authority}/private`,
    `\\\\${authority}/private`,
    `/\\//${authority}/private`,
  ]) {
    const response = await send(proxy.origin, target);
    assert.deepEqual(trap.hits, [], `${target} reached the trap`);
    assert.deepEqual(expected.hits, [], `${target} reached the expected upstream`);
    assert.equal(response.status, 400, target);
  }
  assert.deepEqual(proxy.requests, []);
});

test('preserves origin-form paths, query encodings, methods, bodies and upstream responses', async (t) => {
  const { expected, trap, proxy } = await fixture(t);
  const authority = new URL(trap.origin).host;
  const targets = [
    '/',
    '/index.html',
    '/sw.js',
    '/manifest.webmanifest',
    '/assets/other.js?x=1',
    '/assets//other.js?x=1',
    '/locales/en.json?v=content%2Fversion',
    `/api/v1/items?next=http://${authority}/private`,
    `/%2f%2f${authority}/private?next=https%3A%2F%2F${authority}`,
    `/%5c${authority}/private`,
    '/assets/%2e%2e/other.js?keep=%252f',
    '/api/v1/items?name=a%2Fb%3Fc%23d&plus=a+b',
  ];
  for (const target of targets) {
    const response = await send(proxy.origin, target, {
      method: 'POST',
      body: 'two-part request body',
      headers: { host: authority, 'Content-Type': 'text/plain' },
    });
    assert.equal(response.status, 207);
    assert.equal(response.headers['x-upstream'], 'expected');
    assert.equal(response.headers['content-type'], 'text/plain');
    assert.equal(response.body, 'expected:two-part request body');
    assert.deepEqual(expected.hits.at(-1), {
      target,
      method: 'POST',
      host: new URL(expected.origin).host,
      body: 'two-part request body',
    });
  }
  assert.equal(expected.hits.length, targets.length);
  assert.deepEqual(trap.hits, []);
  assert.deepEqual(proxy.requests, []);
});

test('faults only the exact manifest chunk twice, then streams its real response', async (t) => {
  const { expected, trap, proxy } = await fixture(t, (response) => {
    response.writeHead(200, { 'Content-Type': 'text/javascript', 'X-Upstream': 'expected' });
    response.write('export ');
    response.end('default true;');
  });
  assert.equal((await send(proxy.origin, '/assets/widget.js?v=1')).status, 200);
  assert.deepEqual(proxy.requests, []);
  for (let attempt = 0; attempt < 2; attempt += 1) {
    const response = await send(proxy.origin, '/assets/widget.js');
    assert.equal(response.status, 503);
    assert.equal(response.headers['content-type'], 'text/javascript');
    assert.equal(response.headers['cache-control'], 'no-store');
    assert.equal(response.body, 'Temporary module outage');
    assert.equal(expected.hits.length, 1);
  }
  const recovered = await send(proxy.origin, '/assets/widget.js');
  assert.equal(recovered.status, 200);
  assert.equal(recovered.headers['x-upstream'], 'expected');
  assert.equal(recovered.body, 'export default true;');
  assert.deepEqual(proxy.requests, [503, 503, 200]);
  assert.equal(expected.hits.length, 2);
  assert.deepEqual(trap.hits, []);
  await proxy.close();
  await assert.rejects(send(proxy.origin, '/index.html'), { code: 'ECONNREFUSED' });
});

test('returns upstream redirects without following their external authority', async (t) => {
  let destination;
  const { expected, trap, proxy } = await fixture(t, (response) => {
    response.writeHead(302, { Location: destination });
    response.end('redirect');
  });
  destination = `${trap.origin}/private`;
  const response = await send(proxy.origin, '/redirect');
  assert.equal(response.status, 302);
  assert.equal(response.headers.location, destination);
  assert.equal(response.body, 'redirect');
  assert.equal(expected.hits.length, 1);
  assert.deepEqual(trap.hits, []);
});

test('close releases an active response and its upstream connection', { timeout: 5000 }, async (t) => {
  let started;
  let closed;
  const streaming = new Promise((resolve) => {
    started = resolve;
  });
  const upstreamClosed = new Promise((resolve) => {
    closed = resolve;
  });
  const { expected, trap, proxy } = await fixture(t, (response) => {
    response.on('close', closed);
    response.writeHead(200);
    response.write('streaming');
    started();
  });
  const aborted = assert.rejects(send(proxy.origin, '/stream'), { code: 'ECONNRESET' });
  await streaming;
  await proxy.close();
  await aborted;
  await upstreamClosed;
  assert.equal(expected.hits.length, 1);
  assert.deepEqual(trap.hits, []);
  await assert.rejects(send(proxy.origin, '/index.html'), { code: 'ECONNREFUSED' });
});
