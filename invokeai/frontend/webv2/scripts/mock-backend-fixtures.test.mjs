import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

import {
  assertMockBackendFixture,
  createMockBackendFixture,
  getMockBackendFixtureCounts,
  MOCK_BACKEND_PROFILE_COUNTS,
  MOCK_BACKEND_PROFILE_NAMES,
  validateMockBackendFixture,
} from './mock-backend-fixtures.mjs';
import { startMockBackend } from './mock-backend.mjs';

test('empty and representative fixtures have exact, distinct workloads', () => {
  for (const profile of MOCK_BACKEND_PROFILE_NAMES) {
    const fixture = assertMockBackendFixture(createMockBackendFixture(profile));

    assert.deepEqual(getMockBackendFixtureCounts(fixture), MOCK_BACKEND_PROFILE_COUNTS[profile]);
    assert.deepEqual(validateMockBackendFixture(fixture), []);
  }

  assert.notDeepEqual(createMockBackendFixture('empty'), createMockBackendFixture('representative'));
});

test('video fixtures are additive without changing the profile-count contract', () => {
  const empty = createMockBackendFixture('empty');
  const representative = createMockBackendFixture('representative');

  assert.deepEqual(Object.keys(MOCK_BACKEND_PROFILE_COUNTS.empty), [
    'images',
    'layers',
    'models',
    'nodes',
    'projects',
    'queueItems',
    'workflowNodes',
  ]);
  assert.deepEqual(Object.keys(MOCK_BACKEND_PROFILE_COUNTS.representative), [
    'images',
    'layers',
    'models',
    'nodes',
    'projects',
    'queueItems',
    'workflowNodes',
  ]);
  assert.equal(empty.videos.length, 0);
  assert.ok(representative.videos.length >= 5);
  assert.equal(representative.images.length, 1_000);
  assert.equal(representative.images[0].image_name, 'fixture-image-0001.png');
  assert.ok(representative.videos.some((video) => video.video_name === 'fixture-image-0012.png'));
  assert.ok(representative.videos.some((video) => video.video_name === 'fixture-video-001.mp4'));
  assert.deepEqual(getMockBackendFixtureCounts(representative), MOCK_BACKEND_PROFILE_COUNTS.representative);
});

test('fixture generation is byte-deterministic', () => {
  for (const profile of MOCK_BACKEND_PROFILE_NAMES) {
    assert.equal(JSON.stringify(createMockBackendFixture(profile)), JSON.stringify(createMockBackendFixture(profile)));
  }
});

test('representative fixtures keep node discovery and heavy project data coherent', () => {
  const fixture = createMockBackendFixture('representative');
  const catalogTypes = fixture.nodeCatalog.node_packs.flatMap((pack) => pack.node_types);
  const invocationTypes = Object.values(fixture.openApiDocument.components.schemas)
    .filter((schema) => schema.class === 'invocation')
    .map((schema) => schema.properties.type.const);
  const project = fixture.projects[0];

  assert.deepEqual(catalogTypes, invocationTypes);
  assert.deepEqual(
    project.data.projectGraph.nodes.map((node) => node.data.type),
    invocationTypes
  );
  assert.equal(project.data.canvas.document.layers.length, MOCK_BACKEND_PROFILE_COUNTS.representative.layers);
  assert.equal(new Set(project.data.canvas.document.layers.map((layer) => layer.id)).size, 64);
});

test('the HTTP reset contract selects profiles explicitly and restores the startup profile by default', async () => {
  const backend = await startMockBackend(0, { profile: 'empty' });

  try {
    const readJson = async (path, init) => {
      const response = await fetch(`${backend.origin}${path}`, init);

      assert.equal(response.ok, true, `${init?.method ?? 'GET'} ${path} returned ${response.status}`);

      return response.json();
    };

    assert.deepEqual(await readJson('/__profile'), {
      counts: MOCK_BACKEND_PROFILE_COUNTS.empty,
      ok: true,
      profile: 'empty',
    });

    assert.deepEqual(await readJson('/__reset?profile=representative', { method: 'POST' }), {
      counts: MOCK_BACKEND_PROFILE_COUNTS.representative,
      ok: true,
      profile: 'representative',
    });

    const [images, itemIds, models, nodeCatalog, openApi, projects] = await Promise.all([
      readJson('/api/v1/images/?limit=17&offset=0'),
      readJson('/api/v1/queue/default/item_ids'),
      readJson('/api/v2/models/'),
      readJson('/api/v2/custom_nodes/'),
      readJson('/openapi.json'),
      readJson('/api/v1/projects/'),
    ]);

    assert.equal(images.items.length, 17);
    assert.equal(images.total, 1_000);
    assert.equal(itemIds.item_ids.length, 500);
    assert.equal(models.models.length, 100);
    assert.equal(nodeCatalog.node_packs.flatMap((pack) => pack.node_types).length, 100);
    assert.equal(
      Object.values(openApi.components.schemas).filter((schema) => schema.class === 'invocation').length,
      100
    );
    assert.equal(projects.length, 40);

    assert.deepEqual(await readJson('/__reset', { method: 'POST' }), {
      counts: MOCK_BACKEND_PROFILE_COUNTS.empty,
      ok: true,
      profile: 'empty',
    });

    assert.deepEqual(
      await readJson('/__reset', {
        body: JSON.stringify({ profile: 'representative' }),
        headers: { 'content-type': 'application/json' },
        method: 'POST',
      }),
      { counts: MOCK_BACKEND_PROFILE_COUNTS.representative, ok: true, profile: 'representative' }
    );
  } finally {
    await backend.close();
  }
});

const withRepresentativeBackend = async (run) => {
  const backend = await startMockBackend(0, { profile: 'representative' });

  try {
    await run(backend);
  } finally {
    await backend.close();
  }
};

const getJson = async (backend, path, init) => {
  const response = await fetch(`${backend.origin}${path}`, init);
  const body = await response.json();

  assert.equal(
    response.ok,
    true,
    `${init?.method ?? 'GET'} ${path} returned ${response.status}: ${JSON.stringify(body)}`
  );

  return body;
};

test('merged gallery list and names share qualified ordering, filters, ownership, totals, and pagination', async () => {
  await withRepresentativeBackend(async (backend) => {
    const defaultPage = await getJson(
      backend,
      '/api/v1/gallery/items/?categories=general&is_intermediate=false&starred_first=false&order_dir=DESC&limit=1&offset=0'
    );
    const backendDefaultPage = await getJson(
      backend,
      '/api/v1/gallery/items/?categories=general&is_intermediate=false&order_dir=DESC&limit=100&offset=0'
    );

    assert.equal(defaultPage.items[0]?.kind, 'image');
    assert.deepEqual(
      backendDefaultPage.items.map((item) => item.starred),
      backendDefaultPage.items.map((item) => item.starred).toSorted((left, right) => Number(right) - Number(left))
    );

    const query =
      'categories=general&is_intermediate=false&starred_first=true&order_dir=DESC&created_from=2026-01-15&created_to=2026-01-15';
    const page = await getJson(backend, `/api/v1/gallery/items/?${query}&limit=3&offset=0`);
    const names = await getJson(backend, `/api/v1/gallery/items/names?${query}`);
    const chronologicalNames = await getJson(
      backend,
      '/api/v1/gallery/items/names?categories=general&is_intermediate=false&starred_first=false&order_dir=DESC'
    );

    assert.equal(page.items.length, 3);
    assert.equal(page.total, names.total_count);
    assert.equal(chronologicalNames.starred_count, 0);
    assert.deepEqual(
      page.items.map(({ kind, name }) => ({ kind, name })),
      names.items.slice(0, 3)
    );
    assert.deepEqual(names.items.slice(1, 3), [
      { kind: 'video', name: 'fixture-image-0012.png' },
      { kind: 'image', name: 'fixture-image-0012.png' },
    ]);
    assert.ok(names.starred_count >= 2);
    assert.ok(!names.items.some((item) => item.name === 'fixture-video-foreign.mp4'));
    assert.ok(!names.items.some((item) => item.name === 'fixture-video-intermediate.mp4'));

    const secondPage = await getJson(backend, `/api/v1/gallery/items/?${query}&limit=2&offset=1`);

    assert.deepEqual(secondPage.items, page.items.slice(1, 3));

    const assetPage = await getJson(
      backend,
      '/api/v1/gallery/items/?categories=control&categories=mask&categories=user&is_intermediate=false&limit=100&offset=0'
    );

    assert.ok(assetPage.items.some((item) => item.kind === 'video' && item.name === 'fixture-video-asset.mp4'));
    assert.ok(!assetPage.items.some((item) => item.kind === 'video' && item.category === 'general'));

    const searched = await getJson(
      backend,
      '/api/v1/gallery/items/?categories=general&is_intermediate=false&search_term=wan%20fixture&limit=100&offset=0'
    );

    assert.deepEqual(
      searched.items.map(({ kind, name }) => ({ kind, name })),
      [{ kind: 'video', name: 'fixture-video-001.mp4' }]
    );

    const ascending = await getJson(
      backend,
      '/api/v1/gallery/items/?categories=general&is_intermediate=false&starred_first=false&order_dir=ASC&limit=100&offset=0'
    );
    const ascendingKeys = ascending.items.map((item) => `${item.created_at}|${item.kind}|${item.name}`);

    assert.deepEqual(ascendingKeys, [...ascendingKeys].sort());
  });
});

test('date-board item names preserve the mixed gallery ordering and filter contract', async () => {
  await withRepresentativeBackend(async (backend) => {
    const result = await getJson(
      backend,
      '/api/v1/virtual_boards/by_date/2026-01-15/item_names?categories=general&is_intermediate=false&starred_first=true&order_dir=DESC'
    );

    assert.ok(result.items.some((item) => item.kind === 'image'));
    assert.ok(result.items.some((item) => item.kind === 'video'));
    assert.deepEqual(result.items.slice(1, 3), [
      { kind: 'video', name: 'fixture-image-0012.png' },
      { kind: 'image', name: 'fixture-image-0012.png' },
    ]);
    assert.equal(result.total_count, result.items.length);
  });
});

test('video DTO, Details, poster, full media, HEAD, and Range routes use the checked-in assets', async () => {
  await withRepresentativeBackend(async (backend) => {
    const name = 'fixture-video-001.mp4';
    const encodedName = encodeURIComponent(name);
    const dto = await getJson(backend, `/api/v1/videos/i/${encodedName}`);

    assert.equal(dto.video_name, name);
    assert.equal(dto.video_category, 'general');
    assert.equal(dto.video_url, `/api/v1/videos/i/${name}/full`);
    assert.equal(dto.thumbnail_url, `/api/v1/videos/i/${name}/thumbnail`);
    assert.equal(Number.isFinite(dto.duration), true);
    assert.equal(dto.width, 64);
    assert.equal(dto.height, 64);

    assert.deepEqual(await getJson(backend, `/api/v1/videos/i/${encodedName}/metadata`), {
      codec: 'h264',
      prompt: 'wan fixture',
    });
    assert.deepEqual(await getJson(backend, `/api/v1/videos/i/${encodedName}/workflow`), {
      graph: '{"nodes":[{"id":"fixture-video-output","type":"wan_l2v"}]}',
      workflow: '{"name":"Fixture video workflow","nodes":[]}',
    });

    const poster = await fetch(`${backend.origin}/api/v1/videos/i/${encodedName}/thumbnail`);

    assert.equal(poster.status, 200);
    assert.equal(poster.headers.get('content-type'), 'image/webp');
    assert.ok((await poster.arrayBuffer()).byteLength > 0);

    const head = await fetch(`${backend.origin}/api/v1/videos/i/${encodedName}/full`, { method: 'HEAD' });
    const length = Number(head.headers.get('content-length'));

    assert.equal(head.status, 200);
    assert.equal(head.headers.get('accept-ranges'), 'bytes');
    assert.equal(head.headers.get('content-type'), 'video/mp4');
    assert.ok(length > 0);
    assert.equal((await head.arrayBuffer()).byteLength, 0);

    const full = await fetch(`${backend.origin}/api/v1/videos/i/${encodedName}/full`);

    assert.equal(full.status, 200);
    assert.equal(full.headers.get('accept-ranges'), 'bytes');
    assert.equal(Number(full.headers.get('content-length')), length);
    assert.equal((await full.arrayBuffer()).byteLength, length);

    const range = await fetch(`${backend.origin}/api/v1/videos/i/${encodedName}/full`, {
      headers: { range: 'bytes=0-15' },
    });

    assert.equal(range.status, 206);
    assert.equal(range.headers.get('accept-ranges'), 'bytes');
    assert.equal(range.headers.get('content-length'), '16');
    assert.equal(range.headers.get('content-range'), `bytes 0-15/${String(length)}`);
    assert.equal((await range.arrayBuffer()).byteLength, 16);

    for (const invalidRange of ['bytes=bad', `bytes=${String(length)}-`, 'bytes=9-1']) {
      const unsatisfiable = await fetch(`${backend.origin}/api/v1/videos/i/${encodedName}/full`, {
        headers: { range: invalidRange },
      });

      assert.equal(unsatisfiable.status, 416);
      assert.equal(unsatisfiable.headers.get('accept-ranges'), 'bytes');
      assert.equal(unsatisfiable.headers.get('content-range'), `bytes */${String(length)}`);
      assert.equal((await unsatisfiable.arrayBuffer()).byteLength, 0);
    }
  });
});

test('video upload, star, delete, board movement, counts, covers, and board deletion return authoritative outcomes', async () => {
  await withRepresentativeBackend(async (backend) => {
    const boards = await getJson(backend, '/api/v1/boards/?all=true');
    const board = boards.find((candidate) => candidate.board_id === 'fixture-board-02');

    assert.ok(board);
    assert.ok(board.image_count > 0);
    assert.ok(board.video_count > 0);
    assert.equal(board.cover_image_name, null);
    assert.equal(board.cover_video_name, 'fixture-video-board.mp4');
    assert.equal((await getJson(backend, '/api/v1/videos/i/fixture-video-board.mp4')).starred, true);

    const form = new FormData();
    form.append('file', new Blob(['fixture upload'], { type: 'video/mp4' }), 'uploaded-fixture.mp4');
    const uploadResponse = await fetch(
      `${backend.origin}/api/v1/videos/upload?video_category=general&is_intermediate=false&board_id=fixture-board-02`,
      { body: form, method: 'POST' }
    );
    const uploaded = await uploadResponse.json();

    assert.equal(uploadResponse.status, 201);
    assert.equal(uploaded.video_category, 'general');
    assert.equal(uploaded.is_intermediate, false);
    assert.equal(uploaded.board_id, 'fixture-board-02');

    assert.deepEqual(
      await getJson(backend, '/api/v1/videos/star', {
        body: JSON.stringify({ video_names: [uploaded.video_name, 'missing.mp4'] }),
        headers: { 'content-type': 'application/json' },
        method: 'POST',
      }),
      {
        affected_boards: ['fixture-board-02'],
        failed_videos: [],
        starred_videos: [uploaded.video_name],
      }
    );

    assert.deepEqual(
      await getJson(backend, '/api/v1/videos/board', {
        body: JSON.stringify({ board_id: 'fixture-board-03', video_name: uploaded.video_name }),
        headers: { 'content-type': 'application/json' },
        method: 'POST',
      }),
      {
        added_videos: [uploaded.video_name],
        affected_boards: ['fixture-board-02', 'fixture-board-03'],
      }
    );

    assert.deepEqual(
      await getJson(backend, '/api/v1/videos/board', {
        body: JSON.stringify({ video_name: uploaded.video_name }),
        headers: { 'content-type': 'application/json' },
        method: 'DELETE',
      }),
      {
        affected_boards: ['fixture-board-03', 'none'],
        removed_videos: [uploaded.video_name],
      }
    );

    assert.deepEqual(
      await getJson(backend, '/api/v1/videos/delete', {
        body: JSON.stringify({ video_names: [uploaded.video_name, 'missing.mp4'] }),
        headers: { 'content-type': 'application/json' },
        method: 'POST',
      }),
      {
        affected_boards: ['none'],
        deleted_videos: [uploaded.video_name],
        failed_videos: [],
      }
    );

    const detachResult = await getJson(backend, '/api/v1/boards/fixture-board-02?include_images=false', {
      method: 'DELETE',
    });

    assert.ok(detachResult.deleted_board_images.length > 0);
    assert.deepEqual(detachResult.deleted_board_videos, ['fixture-video-board.mp4']);
    assert.deepEqual(detachResult.deleted_images, []);
    assert.deepEqual(detachResult.deleted_videos, []);
    assert.equal((await getJson(backend, '/api/v1/videos/i/fixture-video-board.mp4')).board_id, null);
  });

  await withRepresentativeBackend(async (backend) => {
    const deleteResult = await getJson(backend, '/api/v1/boards/fixture-board-02?include_images=true', {
      method: 'DELETE',
    });

    assert.ok(deleteResult.deleted_images.length > 0);
    assert.deepEqual(deleteResult.deleted_videos, ['fixture-video-board.mp4']);
    assert.deepEqual(deleteResult.failed_images, []);
    assert.deepEqual(deleteResult.failed_videos, []);
    assert.equal((await fetch(`${backend.origin}/api/v1/videos/i/fixture-video-board.mp4`)).status, 404);
  });
});

test('the representative video accessibility journey owns the only generated-media caption exception', async () => {
  const source = await readFile(new URL('./run-accessibility-journeys.mjs', import.meta.url), 'utf8');
  const captionExceptions = source.match(/'video-caption': \{ enabled: false \}/g) ?? [];

  assert.equal(captionExceptions.length, 1);
  assert.match(source, /generated media has no caption track/i);
  assert.match(source, /workbench-video-preview-representative/);
  assert.match(source, /getByRole\('list', \{ exact: true, name: 'Gallery items' \}\)/);
  assert.match(source, /INVOKEAI_ACCESSIBILITY_JOURNEY/);
});
