import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import test from 'node:test';
import { crc32, inflateSync } from 'node:zlib';

import {
  isExpectedDocumentUnload,
  isQueueReadRequest,
  isExpectedGalleryProjectSwitchAbort,
} from './browser-smoke-expectations.mjs';
import {
  assertRetirement,
  distribution,
  observeBrowserDiagnostics,
  withBrowserNavigation,
} from './browser-validation.mjs';
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
  const diagnostics = await observeBrowserDiagnostics(page);
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
  const diagnostics = await observeBrowserDiagnostics(page);
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

test('document teardown is same-origin and active only; real failures and live mutations stay critical', async () => {
  const phase = { navigating: true, origin: 'http://localhost:4180' };
  const canceled = {
    kind: 'requestfailed',
    method: 'PUT',
    url: `${phase.origin}/api/v1/projects/p/workspace`,
    message: 'net::ERR_ABORTED',
  };
  const native = {
    kind: 'pageerror',
    message: `Fetch API cannot load ${phase.origin}/api/v1/new-read due to access control checks.`,
  };
  assert.equal(isExpectedDocumentUnload(canceled, phase), true);
  assert.equal(isExpectedDocumentUnload(native, phase), true);
  for (const event of [
    { ...canceled, message: 'connection refused' },
    { ...canceled, responseStatus: 500 },
    { ...canceled, kind: 'response', status: 500 },
    { ...canceled, url: 'http://foreign/api/v1/projects/p/workspace' },
    { ...canceled, url: `${phase.origin}/assets/app.js` },
    { ...native, kind: 'console' },
    { ...native, message: 'Unexpected runtime error' },
  ]) {
    assert.equal(isExpectedDocumentUnload(event, phase), false);
  }
  assert.equal(isExpectedDocumentUnload(canceled, { ...phase, navigating: false }), false);
  assert.equal(isExpectedDocumentUnload(native, { ...phase, navigating: false }), false);
  const page = new EventEmitter();
  const diagnostics = await observeBrowserDiagnostics(page, {
    expected: (event) => isExpectedDocumentUnload(event, phase),
  });
  page.emit('pageerror', { message: native.message, stack: native.message });
  await diagnostics.assertHealthy();
  page.emit('response', { url: () => canceled.url, status: () => 500 });
  await assert.rejects(() => diagnostics.assertHealthy(), /500/);
});

test('socket polling teardown excludes websocket, cross-origin and live-page failures', async () => {
  const phase = { navigating: true, origin: 'http://localhost:4180' };
  const pollingUrl = `${phase.origin}/ws/socket.io/?EIO=4&transport=polling&t=1`;
  const canceled = { kind: 'requestfailed', method: 'GET', url: pollingUrl, message: 'net::ERR_ABORTED' };
  const native = {
    kind: 'pageerror',
    message: `XMLHttpRequest cannot load ${pollingUrl} due to access control checks.`,
  };
  for (const event of [canceled, native]) {
    assert.equal(isExpectedDocumentUnload(event, phase), true);
    assert.equal(isExpectedDocumentUnload(event, { ...phase, navigating: false }), false);
  }
  for (const url of [pollingUrl.replace('polling', 'websocket'), pollingUrl.replace('localhost', 'foreign')]) {
    assert.equal(isExpectedDocumentUnload({ ...canceled, url }, phase), false);
    assert.equal(
      isExpectedDocumentUnload({ ...native, message: native.message.replace(pollingUrl, url) }, phase),
      false
    );
  }
  const page = new EventEmitter();
  const diagnostics = await observeBrowserDiagnostics(page, {
    expected: (event) => isExpectedDocumentUnload(event, phase),
  });
  const request = { url: () => pollingUrl, method: () => 'GET', failure: () => ({ errorText: canceled.message }) };
  page.emit('requestfailed', request);
  page.emit('pageerror', { message: native.message, stack: native.message });
  await diagnostics.assertHealthy();
  page.emit('response', { url: () => pollingUrl, status: () => 500 });
  await assert.rejects(() => diagnostics.assertHealthy(), /500/);
  phase.navigating = false;
  page.emit('requestfailed', request);
  page.emit('pageerror', { message: native.message, stack: native.message });
  assert.equal(diagnostics.errors.length, 3, 'The HTTP failure and both live-page errors remain fatal');
});

test('Queue owner cancellation includes its read methods and excludes commands, cross-origin requests and wrong methods', () => {
  const origin = 'http://localhost:4180';
  for (const [operation, method] of [
    ['status', 'GET'],
    ['current', 'GET'],
    ['next', 'GET'],
    ['item_ids', 'GET'],
    ['items_by_ids', 'POST'],
  ]) {
    const url = `${origin}/api/v1/queue/default/${operation}`;
    assert.equal(isQueueReadRequest({ method, url }, origin), true);
    assert.equal(isQueueReadRequest({ method: method === 'GET' ? 'POST' : 'GET', url }, origin), false);
    assert.equal(isQueueReadRequest({ method, url: url.replace('localhost', 'foreign') }, origin), false);
  }
  assert.equal(
    isQueueReadRequest({ method: 'POST', url: `${origin}/api/v1/queue/default/enqueue_batch` }, origin),
    false
  );
  assert.equal(isQueueReadRequest({ url: `${origin}/api/v1/queue/default/unexpected` }, origin), false);
});

test('natural navigation scopes cancellation to the owned transition and retires the expectation afterwards', async () => {
  const page = new EventEmitter();
  page.url = () => 'http://localhost:4180/#/app';
  const diagnostics = await observeBrowserDiagnostics(page);
  const cancellation = () =>
    page.emit('requestfailed', {
      url: () => 'http://localhost:4180/api/v1/projects/p/workspace',
      method: () => 'PUT',
      failure: () => ({ errorText: 'net::ERR_ABORTED' }),
    });
  await withBrowserNavigation(page, () => {
    cancellation();
  });
  await diagnostics.assertHealthy();
  cancellation();
  await assert.rejects(() => diagnostics.assertHealthy(), /workspace/);
});

test('Gallery project-switch cancellation is an exact same-origin read expectation, never a live-page or HTTP waiver', async () => {
  const phase = { origin: 'http://localhost:4180' };
  const url = `${phase.origin}/api/v1/gallery/items/?board_id=fixture-project-board-01&offset=0`;
  const aborted = { kind: 'requestfailed', method: 'GET', url, responseStatus: 200, message: 'net::ERR_ABORTED' };
  assert.equal(isExpectedGalleryProjectSwitchAbort(aborted, phase), true);
  for (const event of [
    { ...aborted, url: url.replace('localhost', 'foreign') },
    { ...aborted, url: `${phase.origin}/api/v1/gallery/items/location` },
    { ...aborted, url: `${phase.origin}/api/v1/gallery/items/names` },
    { ...aborted, url: `${phase.origin}/api/v1/images/i/broken/thumbnail` },
    { ...aborted, url: `${phase.origin}/assets/app.js` },
    { ...aborted, method: 'POST' },
    { ...aborted, method: 'PUT' },
    { ...aborted, message: 'net::ERR_CONNECTION_REFUSED' },
    { ...aborted, responseStatus: 404 },
    { ...aborted, responseStatus: 500 },
    { ...aborted, kind: 'response', status: 500 },
    { ...aborted, kind: 'pageerror' },
  ]) {
    assert.equal(isExpectedGalleryProjectSwitchAbort(event, phase), false);
  }
  const page = new EventEmitter();
  page.url = () => phase.origin;
  const diagnostics = await observeBrowserDiagnostics(page);
  const request = () => ({ url: () => url, method: () => 'GET', failure: () => ({ errorText: 'net::ERR_ABORTED' }) });
  const retired = request();
  page.emit('request', retired);
  page.emit('response', { url: () => url, status: () => 200, request: () => retired });
  await diagnostics.withGalleryProjectSwitch(() => {
    const newProject = request();
    page.emit('request', newProject);
    page.emit('requestfailed', newProject);
  });
  assert.equal(diagnostics.errors.length, 1, 'A new request with the very same URL must remain fatal');
  // The retired request's body cancellation can be delivered after the switch action has completed.
  page.emit('requestfailed', retired);
  assert.equal(diagnostics.errors.length, 1);
  for (const phase of ['setup', 'first autosave', 'live project']) {
    const live = request();
    page.emit('request', live);
    page.emit('requestfailed', live);
    assert.equal(diagnostics.errors.at(-1).message, 'net::ERR_ABORTED', `${phase} reads are strict`);
  }
  const finished = request();
  page.emit('request', finished);
  await diagnostics.withGalleryProjectSwitch(() => {});
  page.emit('requestfinished', finished);
  page.emit('requestfailed', finished);
  const failed = request();
  page.emit('request', failed);
  await diagnostics.withGalleryProjectSwitch(() => {
    page.emit('response', { url: () => url, status: () => 500, request: () => failed });
    page.emit('requestfailed', failed);
  });
  assert.equal(diagnostics.errors.length, 7, 'New, setup, autosave, live, settled and HTTP failures stay fatal');
  await assert.rejects(
    () => diagnostics.assertHealthy(),
    (error) => error.message.includes('500') && error.message.includes('ERR_ABORTED')
  );
});
