import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import test from 'node:test';
import { crc32, inflateSync } from 'node:zlib';

import { isExpectedWebKitUnloadError, isQueueReadRequest } from './browser-smoke-expectations.mjs';
import { assertRetirement, distribution, observeBrowserDiagnostics } from './browser-validation.mjs';
import { createMockBackendFixture } from './mock-backend-fixtures.mjs';
import { startMockBackend } from './mock-backend.mjs';
import { fixturePng, mediaDimensions } from './mock-media.mjs';

const decodePng = (bytes, width, height) => {
  assert.equal(bytes.subarray(0, 8).toString('hex'), '89504e470d0a1a0a');
  const idat = [];
  for (let offset = 8; offset < bytes.length;) {
    const size = bytes.readUInt32BE(offset);
    const type = bytes.toString('ascii', offset + 4, offset + 8);
    const data = bytes.subarray(offset + 8, offset + 8 + size);
    assert.equal(
      bytes.readUInt32BE(offset + 8 + size),
      crc32(bytes.subarray(offset + 4, offset + 8 + size)),
      `${type} CRC`
    );
    if (type === 'IHDR') {
      assert.equal(data.readUInt32BE(0), width);
      assert.equal(data.readUInt32BE(4), height);
      assert.equal(data[8], 8);
      assert.equal(data[9], 2);
    }
    if (type === 'IDAT') {
      idat.push(data);
    }
    offset += size + 12;
  }
  const pixels = inflateSync(Buffer.concat(idat));
  assert.equal(pixels.length, (width * 3 + 1) * height);
  assert.deepEqual([...pixels.subarray(0, 4)], [0, 64, 160, 128]);
  for (let y = 0; y < height; y += 1) {
    assert.equal(pixels[y * (width * 3 + 1)], 0);
  }
};

test('seeded media routes carry decodable PNG pixels at metadata and aspect-preserving thumbnail dimensions', async () => {
  const fixture = createMockBackendFixture('representative');
  const backend = await startMockBackend(0, { profile: 'representative' });
  try {
    // Every distinct seeded dimension, rather than downloading 1000 identical-size fixtures.
    const dimensions = new Map(fixture.images.map((image) => [`${image.width}x${image.height}`, image]));
    const thumbnails = {
      '512x512': [256, 256],
      '576x576': [256, 256],
      '640x640': [256, 256],
      '704x512': [256, 186],
      '512x576': [228, 256],
      '576x640': [230, 256],
      '640x512': [256, 205],
      '704x576': [256, 209],
      '512x640': [205, 256],
      '576x512': [256, 228],
      '640x576': [256, 230],
      '704x640': [256, 233],
    };
    for (const image of dimensions.values()) {
      for (const variant of ['full', 'thumbnail']) {
        const response = await fetch(`${backend.origin}/api/v1/images/i/${image.image_name}/${variant}`);
        assert.equal(response.status, 200);
        assert.equal(response.headers.get('content-type'), 'image/png');
        const [width, height] =
          variant === 'thumbnail' ? thumbnails[`${image.width}x${image.height}`] : [image.width, image.height];
        decodePng(Buffer.from(await response.arrayBuffer()), width, height);
      }
    }
    decodePng(fixturePng(4096, 4096), 4096, 4096);
    const corrupt = Buffer.from(fixturePng(64, 64));
    corrupt[45] ^= 1;
    assert.throws(() => decodePng(corrupt, 64, 64), /CRC/);
    assert.throws(() => fixturePng(8192, 1), /dimensions/);
    assert.deepEqual(mediaDimensions({ width: 832, height: 480 }, true), { width: 256, height: 148 });
  } finally {
    await backend.close();
  }
});

test('nominal diagnostics attribute only the mock socket 404 and reject failed media, HTTP and runtime errors', async () => {
  const page = new EventEmitter();
  const diagnostics = observeBrowserDiagnostics(page);
  const response = (url, status) => ({ url: () => url, status: () => status });
  page.emit('response', response('http://localhost/ws/socket.io/?transport=polling', 404));
  await diagnostics.assertHealthy();
  page.emit('response', response('http://localhost/api/v1/images/i/bad/full', 500));
  page.emit('requestfailed', {
    url: () => 'http://localhost/assets/app.js',
    failure: () => ({ errorText: 'connection refused' }),
  });
  page.emit('pageerror', new Error('raster failed'));
  page.emit('console', {
    type: () => 'error',
    location: () => ({ url: 'http://localhost/app.js' }),
    text: () => 'decode failed',
  });
  assert.equal(diagnostics.errors.length, 4);
  assert.equal(diagnostics.media.length, 1);
  await assert.rejects(() => diagnostics.assertHealthy(), /decode failed/);
});

test('retirement checks detect induced retained portals and sustained growth while allowing a stable plateau', () => {
  assertRetirement([{ portals: 0 }, { portals: 0 }], { portals: 0 });
  assert.throws(() => assertRetirement([{ portals: 0 }, { portals: 1 }], { portals: 0 }), /Retained portals/);
  assert.throws(
    () =>
      assertRetirement(
        [1, 2, 3, 4].map((portals) => ({ portals })),
        { portals: 10 }
      ),
    /Monotonic/
  );
  assert.deepEqual(distribution([4, 1, 3, 2]), { count: 4, min: 1, median: 3, p95: 4, max: 4 });
});

test('diagnostics await Error argument extraction when Firefox console text loses the error message', async () => {
  const page = new EventEmitter();
  const diagnostics = observeBrowserDiagnostics(page);
  let release;
  const available = new Promise((resolve) => {
    release = resolve;
  });
  page.emit('console', {
    type: () => 'error',
    location: () => ({ url: 'http://localhost/assets/vendor.js' }),
    text: () => 'Error',
    args: () => [
      {
        evaluate: async (extract) => {
          await available;
          return extract(new Error('authentication unavailable'));
        },
      },
    ],
  });
  const assertion = assert.rejects(() => diagnostics.assertHealthy(), /authentication unavailable/);
  release();
  await assertion;
  assert.equal(diagnostics.errors[0].arguments[0].name, 'Error');
});

test('WebKit native unload-denial expectation is limited to the exact fixture operation and document navigation', () => {
  const phase = { engine: 'webkit', navigating: true, origin: 'http://localhost:4180' };
  const error = (operation, path) => ({
    kind: 'pageerror',
    message: `${operation} cannot load http://localhost:4180${path} due to access control checks.\n    at native`,
  });
  const workspace = error('Fetch API', '/api/v1/projects/fixture-project-002/workspace');
  assert.equal(isExpectedWebKitUnloadError(workspace, phase), true);
  assert.equal(
    isExpectedWebKitUnloadError(error('XMLHttpRequest', '/ws/socket.io/?EIO=4&transport=polling&t=1'), phase),
    true
  );
  for (const event of [
    error('Fetch API', '/api/v1/projects/another-project/workspace'),
    error('Fetch API', '/api/v1/projects/fixture-project-002/workspace?unexpected=1'),
    error('Fetch API', '/api/v1/images/i/broken/full'),
    error('XMLHttpRequest', '/ws/socket.io/?transport=websocket'),
    error('XMLHttpRequest', '/api/v1/projects/fixture-project-002/workspace'),
    { ...workspace, kind: 'console' },
    { ...workspace, message: workspace.message.replace('localhost:4180', 'another-origin:4180') },
    { ...workspace, message: workspace.message.replace('due to access control checks.', 'failed with HTTP 500.') },
  ]) {
    assert.equal(isExpectedWebKitUnloadError(event, phase), false);
  }
  assert.equal(isExpectedWebKitUnloadError(workspace, { ...phase, navigating: false }), false);
  assert.equal(isExpectedWebKitUnloadError(workspace, { ...phase, engine: 'firefox' }), false);
});

test('Queue read-model hydration denied during WebKit document exit is expected without waiving writes or live-page errors', () => {
  const phase = { engine: 'webkit', navigating: true, origin: 'http://127.0.0.1:4180' };
  const event = {
    kind: 'pageerror',
    message:
      'Fetch API cannot load http://127.0.0.1:4180/api/v1/queue/default/items_by_ids due to access control checks.\n    at queryFn',
  };
  assert.equal(isExpectedWebKitUnloadError(event, phase), true);
  for (const path of ['enqueue_batch', 'cancel', 'clear', 'prune', 'item_summaries_by_ids']) {
    assert.equal(
      isExpectedWebKitUnloadError({ ...event, message: event.message.replace('items_by_ids', path) }, phase),
      false
    );
  }
  assert.equal(isExpectedWebKitUnloadError(event, { ...phase, navigating: false }), false);
  assert.equal(isExpectedWebKitUnloadError({ ...event, kind: 'response', status: 500 }, phase), false);
  assert.equal(
    isExpectedWebKitUnloadError(
      { ...event, message: event.message.replace('127.0.0.1:4180', 'another-origin:4180') },
      phase
    ),
    false
  );
});

test('Queue remount cancellation attribution covers the read pipeline with its actual GET and POST methods', () => {
  const origin = 'http://127.0.0.1:4180';
  for (const [operation, method] of [
    ['status', 'GET'],
    ['current', 'GET'],
    ['next', 'GET'],
    ['item_ids', 'GET'],
    ['items_by_ids', 'POST'],
  ]) {
    const event = { method, url: `${origin}/api/v1/queue/default/${operation}` };
    assert.equal(isQueueReadRequest(event, origin), true);
    assert.equal(isQueueReadRequest({ ...event, method: method === 'GET' ? 'POST' : 'GET' }, origin), false);
    assert.equal(
      isExpectedWebKitUnloadError(
        { kind: 'pageerror', message: `Fetch API cannot load ${event.url} due to access control checks.` },
        { engine: 'webkit', navigating: true, origin }
      ),
      true
    );
  }
  assert.equal(
    isQueueReadRequest({ method: 'POST', url: `${origin}/api/v1/queue/default/enqueue_batch` }, origin),
    false
  );
  assert.equal(
    isQueueReadRequest({ method: 'POST', url: 'http://another-origin:4180/api/v1/queue/default/items_by_ids' }, origin),
    false
  );
  assert.equal(isQueueReadRequest({ url: `${origin}/api/v1/queue/default/unexpected` }, origin), false);
});

test('nominal diagnostics still reject a real Queue HTTP failure after an expected WebKit unload denial', async () => {
  const page = new EventEmitter();
  const phase = { engine: 'webkit', navigating: true, origin: 'http://127.0.0.1:4180' };
  const diagnostics = observeBrowserDiagnostics(page, {
    expected: (event) => isExpectedWebKitUnloadError(event, phase),
  });
  const url = `${phase.origin}/api/v1/queue/default/items_by_ids`;
  const message = `Fetch API cannot load ${url} due to access control checks.`;
  const nativeError = { message, stack: `${message}\n    at queryFn` };
  page.emit('pageerror', nativeError);
  await diagnostics.assertHealthy();
  page.emit('response', { url: () => url, status: () => 500 });
  await assert.rejects(() => diagnostics.assertHealthy(), /500/);
  phase.navigating = false;
  page.emit('pageerror', nativeError);
  assert.equal(
    diagnostics.errors.length,
    2,
    'A live-page native error and real HTTP failure must both remain critical'
  );
});
