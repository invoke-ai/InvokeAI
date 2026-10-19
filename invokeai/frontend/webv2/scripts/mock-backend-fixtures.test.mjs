import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

import {
  PROJECT_FILE_MEMBER_BOARD,
  assertMockBackendFixture,
  collectCanvasLeaves,
  createMockBackendFixture,
  getFixtureProjectWorkflowDocument,
  getMockBackendFixtureCounts,
  MOCK_BACKEND_PROFILE_COUNTS,
  MOCK_BACKEND_PROFILE_NAMES,
  MOCK_BACKEND_REPRESENTATIVE_VIDEO_NAME,
  PROJECT_FILE_BOARD,
  PROJECT_FILE_BOARD_ID,
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
  assert.equal(project.data.documentSchemaVersion, 3);
  assert.deepEqual(
    getFixtureProjectWorkflowDocument(project.data).nodes.map((node) => node.data.type),
    invocationTypes
  );
  assert.equal(project.data.workflows.entries.length, 2);
  assert.equal(project.data.workflows.activeWorkflowId, project.data.workflows.entries[0].document.id);
  const leaves = collectCanvasLeaves(project.data.canvas.document);
  assert.equal(leaves.length, MOCK_BACKEND_PROFILE_COUNTS.representative.layers);
  assert.equal(new Set(leaves.map((layer) => layer.id)).size, 64);
  assert.equal(project.data.canvas.document.version, 3);
  assert.equal(project.data.canvas.version, 3);
  assert.equal(project.minimum_canvas_schema_version, 3);
  // The raster forest nests: three groups, one two levels deep, one disabled, one locked.
  const roots = project.data.canvas.document.stacks.raster;
  const groups = roots.filter((node) => node.type === 'group');
  assert.equal(groups.length, 2);
  assert.equal(groups[0].children.filter((node) => node.type === 'group').length, 1);
  assert.equal(groups[0].children.find((node) => node.type === 'group').isEnabled, false);
  assert.equal(groups[1].isLocked, true);
});

test('Fixture Project 002 carries the image and video references used by the project-file journey', () => {
  const fixture = createMockBackendFixture('representative');
  const project = fixture.projects[1];

  assert.equal(project.project_id, 'fixture-project-002');
  assert.deepEqual(
    collectCanvasLeaves(project.data.canvas.document).map((layer) => layer.source.image.imageName),
    ['fixture-image-0001.png', 'fixture-image-0002.png', 'fixture-image-0003.png', 'fixture-image-0004.png']
  );
  assert.equal(
    collectCanvasLeaves(project.data.canvas.document).every((layer) => layer.type === 'raster'),
    true
  );
  // A schema-2 document on purpose: the project-file journey migrates it on load and exports schema 3.
  assert.equal(project.data.documentSchemaVersion, 2);
  assert.deepEqual(project.data.projectGraph.nodes[0]?.data.inputs.video?.value, {
    video_name: MOCK_BACKEND_REPRESENTATIVE_VIDEO_NAME,
  });
});

/**
 * Cover board-only and referenced media, all visible categories, exclusions, and external references in the
 * round-trip fixture.
 */
test('Fixture Project 002 owns a board carrying every case the project-file journey exercises', () => {
  const fixture = createMockBackendFixture('representative');
  const project = fixture.projects[1];
  const byName = new Map(fixture.images.map((image) => [image.image_name, image]));
  const onBoard = (name) => byName.get(name)?.board_id;

  assert.equal(project.board_id, PROJECT_FILE_BOARD_ID);
  assert.equal(onBoard(PROJECT_FILE_BOARD.referencedImage), PROJECT_FILE_BOARD_ID);
  assert.equal(onBoard(PROJECT_FILE_BOARD.unreferencedImage), PROJECT_FILE_BOARD_ID);
  assert.equal(byName.get(PROJECT_FILE_BOARD.userAsset)?.image_category, 'user');
  assert.equal(byName.get(PROJECT_FILE_BOARD.maskAsset)?.image_category, 'mask');
  assert.equal(byName.get(PROJECT_FILE_BOARD.starredImage)?.starred, true);
  assert.equal(byName.get(PROJECT_FILE_BOARD.canvasOwnedImage)?.image_category, 'other');
  assert.equal(byName.get(PROJECT_FILE_BOARD.intermediateImage)?.is_intermediate, true);
  assert.equal(
    fixture.videos.find((video) => video.video_name === PROJECT_FILE_BOARD.video)?.board_id,
    PROJECT_FILE_BOARD_ID
  );

  // External references deduplicate on import and remain shared on duplication.
  const layerNames = collectCanvasLeaves(project.data.canvas.document).map((layer) => layer.source.image.imageName);

  for (const name of PROJECT_FILE_BOARD.externalImages) {
    assert.ok(layerNames.includes(name), `${name} must stay a canvas reference`);
    assert.notEqual(onBoard(name), PROJECT_FILE_BOARD_ID);
  }

  assert.ok(layerNames.includes(PROJECT_FILE_BOARD.referencedImage));
  assert.ok(!layerNames.includes(PROJECT_FILE_BOARD.unreferencedImage));
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

    const [generationDevices, images, itemIds, models, nodeCatalog, openApi, projects, runtimeConfig] =
      await Promise.all([
        readJson('/api/v1/app/generation_device_options'),
        readJson('/api/v1/images/?limit=17&offset=0'),
        readJson('/api/v1/queue/default/item_ids'),
        readJson('/api/v2/models/'),
        readJson('/api/v2/custom_nodes/'),
        readJson('/openapi.json'),
        readJson('/api/v1/projects/'),
        readJson('/api/v1/app/runtime_config'),
      ]);

    assert.deepEqual(generationDevices, [{ device: 'cpu', name: 'CPU' }]);
    assert.equal(images.items.length, 17);
    // One of the 1,000 fixture images is intermediate and excluded from gallery/export counts.
    assert.equal(images.total, 999);
    assert.equal(itemIds.item_ids.length, 500);
    assert.equal(models.models.length, 100);
    assert.equal(nodeCatalog.node_packs.flatMap((pack) => pack.node_types).length, 100);
    assert.equal(
      Object.values(openApi.components.schemas).filter((schema) => schema.class === 'invocation').length,
      100
    );
    assert.equal(projects.length, 40);
    assert.deepEqual(runtimeConfig, {
      config: { generation_devices: 'auto', max_queue_history: null },
      set_fields: [],
    });

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

test('queue item ids take the backend limit: the head of the order, counted as returned, bounded', async () => {
  await withRepresentativeBackend(async (backend) => {
    for (const orderDir of ['DESC', 'ASC']) {
      const all = await getJson(backend, `/api/v1/queue/default/item_ids?order_dir=${orderDir}`);
      const limited = await getJson(backend, `/api/v1/queue/default/item_ids?order_dir=${orderDir}&limit=50`);

      assert.equal(all.total_count, 500);
      assert.deepEqual(limited, { item_ids: all.item_ids.slice(0, 50), total_count: 50 });
    }
    for (const limit of ['0', '-1', '1001', 'fifty']) {
      const response = await fetch(`${backend.origin}/api/v1/queue/default/item_ids?limit=${limit}`);

      assert.equal(response.status, 422, `limit=${limit}`);
    }
  });
});

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
    const countOnlyPage = await getJson(
      backend,
      '/api/v1/gallery/items/?categories=general&is_intermediate=false&order_dir=DESC&limit=0&offset=0'
    );

    assert.equal(defaultPage.items[0]?.kind, 'image');
    assert.deepEqual(countOnlyPage.items, []);
    assert.equal(countOnlyPage.total, backendDefaultPage.total);
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

test('gallery item location shares listing filters and order, and reports missing items as not found', async () => {
  await withRepresentativeBackend(async (backend) => {
    const query =
      'categories=general&is_intermediate=false&origin=internal&starred=false&order_dir=ASC&starred_first=false&created_from=2026-01-15&created_to=2026-01-15';
    const listing = await getJson(backend, `/api/v1/gallery/items/?${query}&limit=1000&offset=0`);
    const target = listing.items.find((item) => item.kind === 'video');

    assert.ok(target, 'the filtered fixture listing should contain a video');

    const location = await getJson(
      backend,
      `/api/v1/gallery/items/location?${query}&kind=${target.kind}&name=${encodeURIComponent(target.name)}`
    );

    assert.deepEqual(location, {
      index: listing.items.findIndex((item) => item.kind === target.kind && item.name === target.name),
      kind: target.kind,
      name: target.name,
      total: listing.total,
    });

    const excludedQuery = query.replace('starred=false', 'starred=true');
    const notFound = await fetch(
      `${backend.origin}/api/v1/gallery/items/location?${excludedQuery}&kind=${target.kind}&name=${encodeURIComponent(target.name)}`
    );

    assert.equal(notFound.status, 404);
    assert.deepEqual(await notFound.json(), { detail: 'Gallery item not found' });
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

test('starred filter narrows gallery items, names, and date boards consistently', async () => {
  await withRepresentativeBackend(async (backend) => {
    const query = 'categories=general&is_intermediate=false&limit=1000&offset=0';
    const all = await getJson(backend, `/api/v1/gallery/items/?${query}`);
    const starredOnly = await getJson(backend, `/api/v1/gallery/items/?${query}&starred=true`);
    const unstarred = await getJson(backend, `/api/v1/gallery/items/?${query}&starred=false`);

    assert.ok(starredOnly.items.every((item) => item.starred === true));
    assert.ok(starredOnly.items.some((item) => item.kind === 'video'));
    assert.ok(unstarred.items.every((item) => item.starred === false));
    assert.equal(starredOnly.total, all.items.filter((item) => item.starred).length);
    assert.equal(starredOnly.total + unstarred.total, all.total);

    const names = await getJson(
      backend,
      '/api/v1/gallery/items/names?categories=general&is_intermediate=false&starred_first=true&starred=false'
    );
    assert.equal(names.starred_count, 0);
    assert.equal(names.total_count, unstarred.total);

    const dateQuery = '/api/v1/virtual_boards/by_date/2026-01-15/item_names?categories=general&is_intermediate=false';
    const dated = await getJson(backend, dateQuery);
    const datedStarred = await getJson(backend, `${dateQuery}&starred=true`);
    assert.equal(datedStarred.total_count, dated.starred_count);
    assert.ok(datedStarred.total_count > 0);
  });
});

test('media_kind narrows gallery items, names, location, date boards, and the image map consistently', async () => {
  await withRepresentativeBackend(async (backend) => {
    const query = 'categories=general&is_intermediate=false&starred_first=false&limit=1000&offset=0';
    const all = await getJson(backend, `/api/v1/gallery/items/?${query}`);
    const videos = await getJson(backend, `/api/v1/gallery/items/?${query}&media_kind=video`);
    const images = await getJson(backend, `/api/v1/gallery/items/?${query}&media_kind=image`);

    assert.ok(videos.total > 0, 'the fixture listing should contain videos');
    assert.ok(videos.items.every((item) => item.kind === 'video'));
    assert.ok(images.items.every((item) => item.kind === 'image'));
    assert.equal(videos.total + images.total, all.total);

    const names = await getJson(
      backend,
      '/api/v1/gallery/items/names?categories=general&is_intermediate=false&starred_first=false&media_kind=video'
    );
    assert.equal(names.total_count, videos.total);

    const video = videos.items[0];
    const locationQuery = `categories=general&is_intermediate=false&starred_first=false&kind=video&name=${encodeURIComponent(video.name)}`;
    const location = await getJson(backend, `/api/v1/gallery/items/location?${locationQuery}&media_kind=video`);
    assert.deepEqual(location, { index: 0, kind: 'video', name: video.name, total: videos.total });
    const excluded = await fetch(`${backend.origin}/api/v1/gallery/items/location?${locationQuery}&media_kind=image`);
    assert.equal(excluded.status, 404);
    await excluded.body?.cancel();

    const dated = await getJson(
      backend,
      '/api/v1/virtual_boards/by_date/2026-01-15/item_names?categories=general&is_intermediate=false&media_kind=video'
    );
    assert.ok(dated.total_count > 0);
    assert.ok(dated.items.every((item) => item.kind === 'video'));

    const points = await getJson(backend, '/api/v1/image_map/points?include_videos=true&media_kind=video');
    const labels = await getJson(backend, '/api/v1/image_map/cluster_labels?include_videos=true&media_kind=video');
    const bothPoints = await getJson(backend, '/api/v1/image_map/points?include_videos=true');
    const status = await getJson(backend, '/api/v1/image_map/status?include_videos=true&media_kind=video');

    assert.ok(points.points.length > 0);
    assert.ok(points.points.every((point) => point.kind === 'video'));
    assert.equal(labels.visible_hash, points.visible_hash);
    assert.notEqual(points.visible_hash, bothPoints.visible_hash);
    assert.equal(status.projection.point_count, points.points.length);
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

test('image upload preserves restore metadata and is retrievable through every image route', async () => {
  await withRepresentativeBackend(async (backend) => {
    const form = new FormData();
    form.append('file', new Blob(['fixture upload'], { type: 'image/webp' }), 'uploaded fixture.webp');
    const response = await fetch(
      `${backend.origin}/api/v1/images/upload?image_category=other&is_intermediate=false&board_id=fixture-board-02`,
      { body: form, method: 'POST' }
    );
    const expectedName = 'fixture-upload-1001-uploaded-fixture.webp.png';
    const expectedDto = {
      board_id: 'fixture-board-02',
      created_at: '2026-01-15T12:00:00.000Z',
      deleted_at: null,
      has_workflow: false,
      height: 1,
      image_category: 'other',
      image_name: expectedName,
      image_origin: 'external',
      image_subfolder: '',
      image_url: `/api/v1/images/i/${expectedName}/full`,
      is_intermediate: false,
      node_id: null,
      session_id: null,
      starred: false,
      thumbnail_url: `/api/v1/images/i/${expectedName}/thumbnail`,
      updated_at: '2026-01-15T12:00:00.000Z',
      width: 1,
    };

    assert.equal(response.status, 201);
    assert.equal(response.headers.get('location'), expectedDto.image_url);
    assert.deepEqual(await response.json(), expectedDto);
    assert.deepEqual(
      await getJson(backend, '/api/v1/images/images_by_names', {
        body: JSON.stringify({ image_names: [expectedName, 'missing.png'] }),
        headers: { 'content-type': 'application/json' },
        method: 'POST',
      }),
      [expectedDto]
    );
    assert.deepEqual(await getJson(backend, `/api/v1/images/i/${encodeURIComponent(expectedName)}`), expectedDto);

    for (const variant of ['full', 'thumbnail']) {
      const asset = await fetch(`${backend.origin}/api/v1/images/i/${encodeURIComponent(expectedName)}/${variant}`);

      assert.equal(asset.status, 200);
      assert.equal(asset.headers.get('content-type'), 'image/png');
      assert.ok((await asset.arrayBuffer()).byteLength > 0);
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

test('project inboxes are protected from the generic board routes, and members are not', async () => {
  await withRepresentativeBackend(async (backend) => {
    const projects = await getJson(backend, '/api/v1/projects/');
    const project = projects.find((entry) => entry.project_id === 'fixture-project-002');

    // Every project has exactly one inbox, and no two share one.
    assert.equal(new Set(projects.map((entry) => entry.board_id)).size, projects.length);
    assert.ok(project.board_id);

    const board = await getJson(backend, `/api/v1/boards/${project.board_id}`);
    assert.equal(board.project_id, project.project_id);
    assert.equal(board.is_inbox, true);
    assert.equal(board.board_name, project.name);

    // A Library board belongs to no project and is nobody's inbox.
    const plainBoard = await getJson(backend, '/api/v1/boards/fixture-board-02');
    assert.equal(plainBoard.project_id, null);
    assert.equal(plainBoard.is_inbox, false);

    // A member is an ordinary board that happens to belong to the project.
    const member = await getJson(backend, '/api/v1/boards/fixture-member-board-02');
    assert.equal(member.project_id, project.project_id);
    assert.equal(member.is_inbox, false);

    // The generic routes refuse to rename, move or delete an inbox; only the project APIs may touch it.
    for (const [method, path, body] of [
      ['PATCH', `/api/v1/boards/${project.board_id}`, { board_name: 'Renamed' }],
      ['PATCH', `/api/v1/boards/${project.board_id}`, { project_id: null }],
      ['DELETE', `/api/v1/boards/${project.board_id}?include_images=true`, undefined],
    ]) {
      const refused = await fetch(`${backend.origin}${path}`, {
        body: body === undefined ? undefined : JSON.stringify(body),
        headers: { 'content-type': 'application/json' },
        method,
      });
      assert.equal(refused.status, 409, `${method} ${path} should be refused`);
    }
    assert.equal((await getJson(backend, `/api/v1/boards/${project.board_id}`)).board_name, project.name);

    // Renaming the project renames its board; the two never disagree.
    await getJson(backend, `/api/v1/projects/${project.project_id}`, {
      body: JSON.stringify({ data: project.data, expected_revision: project.revision, name: 'Renamed Project' }),
      headers: { 'content-type': 'application/json' },
      method: 'PUT',
    });
    assert.equal((await getJson(backend, `/api/v1/boards/${project.board_id}`)).board_name, 'Renamed Project');
  });
});

test('boards are created in a project or the Library, move between them, and follow a deleted project', async () => {
  await withRepresentativeBackend(async (backend) => {
    const projects = await getJson(backend, '/api/v1/projects/');
    const [first, second] = projects;

    const created = await getJson(backend, `/api/v1/boards/?board_name=Kitchen&project_id=${first.project_id}`, {
      method: 'POST',
    });
    assert.equal(created.project_id, first.project_id);
    assert.equal(created.is_inbox, false);
    assert.equal(
      (
        await fetch(`${backend.origin}/api/v1/boards/?board_name=Nowhere&project_id=no-such-project`, {
          method: 'POST',
        })
      ).status,
      404
    );

    const move = (projectId) =>
      getJson(backend, `/api/v1/boards/${created.board_id}`, {
        body: JSON.stringify({ project_id: projectId }),
        headers: { 'content-type': 'application/json' },
        method: 'PATCH',
      });
    assert.equal((await move(second.project_id)).project_id, second.project_id);
    assert.equal((await move(null)).project_id, null);
    // A rename without the key leaves membership alone.
    const renamed = await getJson(backend, `/api/v1/boards/${created.board_id}`, {
      body: JSON.stringify({ board_name: 'Pantry' }),
      headers: { 'content-type': 'application/json' },
      method: 'PATCH',
    });
    assert.equal(renamed.project_id, null);
    assert.equal(renamed.board_name, 'Pantry');

    // Deleting a project releases its other boards by default, and takes them when asked.
    await move(second.project_id);
    await fetch(`${backend.origin}/api/v1/projects/${second.project_id}`, { method: 'DELETE' });
    assert.equal((await getJson(backend, `/api/v1/boards/${created.board_id}`)).project_id, null);
    assert.equal((await fetch(`${backend.origin}/api/v1/boards/${second.board_id}`)).status, 404);

    await fetch(`${backend.origin}/api/v1/projects/${first.project_id}?boards=delete`, { method: 'DELETE' });
    assert.equal((await fetch(`${backend.origin}/api/v1/boards/fixture-member-board-01`)).status, 404);
    assert.equal((await fetch(`${backend.origin}/api/v1/boards/${created.board_id}`)).status, 200);
  });
});

test('the board snapshot lists only what the gallery would show on a project board', async () => {
  await withRepresentativeBackend(async (backend) => {
    const [project] = await getJson(backend, '/api/v1/projects/');
    const boardId = project.board_id;
    const upload = async (category, isIntermediate) => {
      const form = new FormData();
      form.append('file', new Blob(['x'], { type: 'image/png' }), `${category}-${isIntermediate}.png`);
      const created = await fetch(
        `${backend.origin}/api/v1/images/upload?image_category=${category}` +
          `&is_intermediate=${isIntermediate}&board_id=${boardId}`,
        { body: form, method: 'POST' }
      );
      return (await created.json()).image_name;
    };

    const general = await upload('general', false);
    const control = await upload('control', false);
    await upload('other', false);
    await upload('general', true);

    const snapshot = await getJson(backend, `/api/v1/projects/${project.project_id}/board-snapshot`);

    // `other` is the canvas's private category and intermediates are hidden — neither travels.
    const [inbox] = snapshot.boards;
    assert.equal(inbox.is_inbox, true);
    assert.deepEqual(inbox.items.map((item) => item.name).sort(), [control, general].sort());
    assert.deepEqual(inbox.items.map((item) => item.category).sort(), ['control', 'general']);
    assert.equal(
      inbox.items.every((item) => item.kind === 'image' && item.starred === false),
      true
    );

    await getJson(backend, '/api/v1/images/star', {
      body: JSON.stringify({ image_names: [general] }),
      headers: { 'content-type': 'application/json' },
      method: 'POST',
    });
    const starred = await getJson(backend, `/api/v1/projects/${project.project_id}/board-snapshot`);
    assert.equal(starred.boards[0].items.find((item) => item.name === general).starred, true);

    assert.equal((await fetch(`${backend.origin}/api/v1/projects/nope/board-snapshot`)).status, 404);
  });
});

test('the board snapshot lists the project other boards after the inbox, with their own items', async () => {
  await withRepresentativeBackend(async (backend) => {
    const snapshot = await getJson(backend, '/api/v1/projects/fixture-project-002/board-snapshot');

    assert.deepEqual(
      snapshot.boards.map((board) => [board.board_id, board.is_inbox, board.name, board.items.length]),
      [
        [PROJECT_FILE_BOARD_ID, true, 'Fixture Project 002', 6],
        [PROJECT_FILE_MEMBER_BOARD.id, false, PROJECT_FILE_MEMBER_BOARD.name, 2],
      ]
    );
    assert.deepEqual(
      snapshot.boards[1].items.map((item) => item.name).sort(),
      [...PROJECT_FILE_BOARD.memberImages].sort()
    );
  });
});

test('copying media mints fresh identities and deleting a project keeps its media', async () => {
  await withRepresentativeBackend(async (backend) => {
    const [project] = await getJson(backend, '/api/v1/projects/');
    const source = 'fixture-image-0002.png';

    const staging = await getJson(backend, '/api/v1/boards/?board_name=Staging', { method: 'POST' });
    const copied = await getJson(backend, '/api/v1/images/copy', {
      body: JSON.stringify({ board_id: staging.board_id, image_names: [source, 'missing.png'] }),
      headers: { 'content-type': 'application/json' },
      method: 'POST',
    });

    assert.deepEqual(copied.failed, ['missing.png']);
    assert.equal(copied.copied.length, 1);
    const [{ image_name: copyName }] = copied.copied;
    // A copy is a new identity, because board membership keys on the name.
    assert.notEqual(copyName, source);
    const copy = await getJson(backend, `/api/v1/images/i/${copyName}`);
    assert.equal(copy.board_id, staging.board_id);
    assert.equal(copy.image_category, (await getJson(backend, `/api/v1/images/i/${source}`)).image_category);
    // The source keeps its own board.
    assert.notEqual((await getJson(backend, `/api/v1/images/i/${source}`)).board_id, staging.board_id);

    // The staging board can then be claimed, which is how an import commits.
    const claimed = await getJson(backend, '/api/v1/projects/', {
      body: JSON.stringify({ board_id: staging.board_id, data: {}, name: 'Imported', project_id: 'imported-1' }),
      headers: { 'content-type': 'application/json' },
      method: 'POST',
    });
    assert.equal(claimed.board_id, staging.board_id);
    // ...but only once.
    const second = await fetch(`${backend.origin}/api/v1/projects/`, {
      body: JSON.stringify({ board_id: staging.board_id, data: {}, name: 'Again', project_id: 'imported-2' }),
      headers: { 'content-type': 'application/json' },
      method: 'POST',
    });
    assert.equal(second.status, 409);

    // Deleting a project removes its board but leaves the media uncategorized.
    await fetch(`${backend.origin}/api/v1/projects/${claimed.project_id}`, { method: 'DELETE' });
    assert.equal((await fetch(`${backend.origin}/api/v1/boards/${staging.board_id}`)).status, 404);
    assert.equal((await getJson(backend, `/api/v1/images/i/${copyName}`)).board_id, null);

    assert.equal(
      (await getJson(backend, '/api/v1/projects/')).some((entry) => entry.project_id === project.project_id),
      true
    );
  });
});

test('a project claims only an unshared Library board of the caller’s own', async () => {
  await withRepresentativeBackend(async (backend) => {
    const call = (path, init) =>
      fetch(`${backend.origin}${path}`, {
        ...init,
        headers: { 'content-type': 'application/json' },
        ...(init?.body === undefined ? {} : { body: JSON.stringify(init.body) }),
      });
    const claim = (boardId) => call('/api/v1/projects/', { body: { board_id: boardId, data: {} }, method: 'POST' });
    const listed = await getJson(backend, '/api/v1/boards/');
    const staging = listed.find((board) => board.project_id === null && !board.is_inbox);

    // Seeded boards carry their general access, and the deprecated visibility alias is derived from it.
    assert.equal(staging.general_access, 'restricted');
    assert.equal(staging.board_visibility, 'private');

    // Shared with everyone: not claimable, until restricted again.
    await call('/__board-access', {
      body: { board_id: staging.board_id, general_access: 'contributor' },
      method: 'POST',
    });
    assert.equal((await getJson(backend, `/api/v1/boards/${staging.board_id}`)).board_visibility, 'public');
    assert.equal((await claim(staging.board_id)).status, 409);
    await call('/__board-access', {
      body: { board_id: staging.board_id, general_access: 'restricted' },
      method: 'POST',
    });

    // Shared with a member: not claimable either.
    await call('/__board-access', { body: { board_id: staging.board_id, member_role: 'viewer' }, method: 'POST' });
    assert.equal((await claim(staging.board_id)).status, 409);
    await call('/__board-access', { body: { board_id: staging.board_id, member_role: null }, method: 'POST' });

    // Someone else's board, even one the caller may read, is never theirs to claim.
    await call('/__faults', { body: { sharedBoards: true }, method: 'POST' });
    assert.equal((await claim('shared-contributor')).status, 409);

    const claimed = await claim(staging.board_id);
    assert.equal(claimed.ok, true);
    assert.equal((await claimed.json()).board_id, staging.board_id);
  });
});

test('a project save may carry the caller’s workspace, stored with the content or refused with it', async () => {
  await withRepresentativeBackend(async (backend) => {
    const call = (path, init) =>
      fetch(`${backend.origin}${path}`, {
        ...init,
        headers: { 'content-type': 'application/json' },
        ...(init?.body === undefined ? {} : { body: JSON.stringify(init.body) }),
      });
    const [project] = await getJson(backend, '/api/v1/projects/');
    const save = (body) =>
      call(`/api/v1/projects/${project.project_id}`, {
        body: { max_document_schema_version: 4, ...body },
        method: 'PUT',
      });

    // Content alone: the answer names no workspace.
    const contentOnly = await save({ data: project.data, expected_revision: project.revision });
    assert.equal(contentOnly.status, 200);
    const contentOnlyRecord = await contentOnly.json();
    assert.equal(contentOnlyRecord.revision, project.revision + 1);
    assert.equal(contentOnlyRecord.workspace, null);

    // A workspace beside a whole document, or the split without one, is refused as the server refuses it.
    const beforeSplit = await save({
      data: project.data,
      expected_revision: contentOnlyRecord.revision,
      workspace: { data: { layout: 'early' }, expected_revision: null },
    });
    assert.equal(beforeSplit.status, 400);
    assert.equal((await beforeSplit.json()).detail.code, 'workspace_needs_split_document');
    const contentOnlySplit = await save({
      data: project.data,
      expected_revision: contentOnlyRecord.revision,
      minimum_document_schema_version: 4,
    });
    assert.equal(contentOnlySplit.status, 400);
    assert.deepEqual((await contentOnlySplit.json()).detail.code, 'split_needs_workspace');
    assert.equal(
      (await getJson(backend, `/api/v1/projects/${project.project_id}`)).revision,
      contentOnlyRecord.revision
    );

    // Content and the first workspace, in one write.
    const split = await save({
      data: project.data,
      expected_revision: contentOnlyRecord.revision,
      minimum_document_schema_version: 4,
      workspace: { data: { layout: 'split' }, expected_revision: null },
    });
    assert.equal(split.status, 200);
    const splitRecord = await split.json();
    assert.equal(splitRecord.revision, contentOnlyRecord.revision + 1);
    assert.deepEqual(splitRecord.workspace.data, { layout: 'split' });
    assert.equal(splitRecord.workspace.revision, 1);
    assert.equal(typeof splitRecord.workspace.updated_at, 'string');
    assert.equal(splitRecord.workspace_updated_at, splitRecord.workspace.updated_at);

    // A stale workspace revision refuses the whole save: the content revision does not move either.
    const stale = await save({
      data: { ...project.data, changed: true },
      expected_revision: splitRecord.revision,
      workspace: { data: { layout: 'stale' }, expected_revision: null },
    });
    assert.equal(stale.status, 409);
    assert.deepEqual(await stale.json(), {
      detail: { current_revision: 1, message: 'Workspace revision conflict' },
    });
    const unchanged = await getJson(backend, `/api/v1/projects/${project.project_id}?max_document_schema_version=4`);
    assert.equal(unchanged.revision, splitRecord.revision);
    assert.deepEqual(unchanged.workspace.data, { layout: 'split' });

    // A stale content revision answers with the current one, as the server does.
    const staleContent = await save({ data: project.data, expected_revision: 0 });
    assert.equal(staleContent.status, 409);
    assert.deepEqual(await staleContent.json(), {
      detail: { current_revision: splitRecord.revision, message: 'Revision conflict' },
    });

    // The workspace route answers the canvas floor as the content routes do.
    const belowFloor = await call(`/api/v1/projects/${project.project_id}/workspace`, {
      body: { data: {}, expected_revision: 1, max_canvas_schema_version: 1, max_document_schema_version: 4 },
      method: 'PUT',
    });
    assert.equal(belowFloor.status, 412);
    const refusal = await belowFloor.json();
    assert.equal(refusal.detail.code, 'canvas_schema_unsupported');
    assert.equal(refusal.detail.max_canvas_schema_version, 1);
    assert.equal(refusal.detail.minimum_canvas_schema_version, unchanged.minimum_canvas_schema_version);
    const saved = await call(`/api/v1/projects/${project.project_id}/workspace`, {
      body: { data: { layout: 'own-route' }, expected_revision: 1, max_document_schema_version: 4 },
      method: 'PUT',
    });
    assert.equal(saved.status, 200);
    assert.equal((await saved.json()).revision, 2);
  });
});

test('a project create may carry the caller’s first workspace, stored with the content', async () => {
  await withRepresentativeBackend(async (backend) => {
    const call = (path, init) =>
      fetch(`${backend.origin}${path}`, {
        ...init,
        headers: { 'content-type': 'application/json' },
        ...(init?.body === undefined ? {} : { body: JSON.stringify(init.body) }),
      });
    const create = (body) =>
      call('/api/v1/projects/', { body: { max_document_schema_version: 4, ...body }, method: 'POST' });

    // Content alone: the answer names no workspace.
    const contentOnly = await create({ data: { documentSchemaVersion: 4 }, minimum_document_schema_version: 4 });
    assert.equal(contentOnly.status, 200);
    assert.equal((await contentOnly.json()).workspace, null);

    // Both halves in one request, answered with the workspace it stored, which a read then carries.
    const created = await create({
      data: { documentSchemaVersion: 4 },
      minimum_document_schema_version: 4,
      project_id: 'created-with-workspace',
      workspace: { layout: 'first' },
    });
    assert.equal(created.status, 200);
    const record = await created.json();
    assert.deepEqual(record.workspace.data, { layout: 'first' });
    assert.equal(record.workspace.revision, 1);
    assert.equal(record.workspace_updated_at, record.workspace.updated_at);
    const read = await getJson(backend, '/api/v1/projects/created-with-workspace?max_document_schema_version=4');
    assert.deepEqual(read.workspace, record.workspace);

    // A workspace beside a whole document is refused, as the server refuses it, and the refused create claims no
    // board: the staging board is still the caller's to delete or claim.
    const staging = await getJson(backend, '/api/v1/boards/?board_name=Staging', { method: 'POST' });
    const refused = await create({
      board_id: staging.board_id,
      data: { documentSchemaVersion: 3 },
      minimum_document_schema_version: 3,
      project_id: 'refused-whole',
      workspace: { layout: 'early' },
    });
    assert.equal(refused.status, 400);
    assert.deepEqual(await refused.json(), {
      detail: {
        code: 'workspace_needs_split_document',
        message: 'A workspace is saved only with content at document schema 4 or newer.',
        minimum_document_schema_version: 3,
      },
    });
    assert.equal((await getJson(backend, `/api/v1/boards/${staging.board_id}`)).project_id, null);
    assert.equal((await call('/api/v1/projects/refused-whole')).status, 404);
  });
});

test('the image map serves videos only on request and keeps its two endpoints comparable', async () => {
  await withRepresentativeBackend(async (backend) => {
    const imagesOnly = await getJson(backend, '/api/v1/image_map/points');
    const withVideos = await getJson(backend, '/api/v1/image_map/points?include_videos=true');

    assert.equal(
      imagesOnly.points.some((point) => point.kind === 'video'),
      false
    );
    assert.equal(
      withVideos.points.some((point) => point.kind === 'video'),
      true
    );
    // The footer reports point_count, so it has to match what was served.
    assert.equal(imagesOnly.point_count, imagesOnly.points.length);
    assert.equal(withVideos.point_count, withVideos.points.length);
    assert.equal(withVideos.points.length > imagesOnly.points.length, true);

    // Labels must match the points' projection and visible set or the client discards them.
    const labels = await getJson(backend, '/api/v1/image_map/cluster_labels?include_videos=true');
    assert.equal(labels.updated_at, withVideos.updated_at);
    assert.equal(labels.visible_hash, withVideos.visible_hash);

    // Use a video name absent from images so the cross-namespace lookup genuinely misses.
    const imageNames = new Set(imagesOnly.points.map((point) => point.image_name));
    const video = withVideos.points.find((point) => point.kind === 'video' && !imageNames.has(point.image_name));
    assert.equal((await getJson(backend, `/api/v1/videos/i/${video.image_name}`)).video_name, video.image_name);

    const videoLabels = await getJson(
      backend,
      `/api/v1/image_map/image_labels?image_name=${video.image_name}&kind=video`
    );
    assert.equal(typeof videoLabels.label, 'string');
    // A name from the other namespace is a miss, not another item's labels.
    const crossed = await fetch(
      `${backend.origin}/api/v1/image_map/image_labels?image_name=${video.image_name}&kind=image`
    );
    assert.equal(crossed.status, 404);
  });
});

test('workflow thumbnails round-trip their bytes and refresh the listed URL on every replacement', async () => {
  await withRepresentativeBackend(async (backend) => {
    const path = '/api/v1/workflows/i/fixture-workflow-002/thumbnail';
    const upload = (bytes, type) => {
      const body = new FormData();

      body.append('image', new Blob([bytes], { type }), 'thumbnail.png');
      return fetch(`${backend.origin}${path}`, { body, method: 'PUT' });
    };
    const listedUrl = async () =>
      (await getJson(backend, '/api/v1/workflows/?categories=user&per_page=100')).items.find(
        (item) => item.workflow_id === 'fixture-workflow-002'
      ).thumbnail_url;

    assert.equal(await listedUrl(), null);
    assert.equal((await fetch(`${backend.origin}${path}`)).status, 404);

    assert.equal((await upload(new Uint8Array([1, 2, 3]), 'image/png')).status, 200);
    const first = await listedUrl();
    const served = await fetch(`${backend.origin}${first}`);
    assert.equal(served.headers.get('content-type'), 'image/png');
    assert.deepEqual(new Uint8Array(await served.arrayBuffer()), new Uint8Array([1, 2, 3]));

    assert.equal((await upload(new Uint8Array([4]), 'image/webp')).status, 200);
    const second = await listedUrl();
    assert.notEqual(second, first);
    assert.equal((await getJson(backend, '/api/v1/workflows/i/fixture-workflow-002')).thumbnail_url, second);

    assert.equal((await upload(new Uint8Array([5]), 'text/plain')).status, 415);
    const misnamed = new FormData();
    misnamed.append('file', new Blob([new Uint8Array([6])], { type: 'image/png' }), 'image');
    assert.equal((await fetch(`${backend.origin}${path}`, { body: misnamed, method: 'PUT' })).status, 415);
    assert.equal(
      (await fetch(`${backend.origin}/api/v1/workflows/i/default_fixture-workflow-001/thumbnail`, { method: 'DELETE' }))
        .status,
      403
    );

    await getJson(backend, path, { method: 'DELETE' });
    assert.equal(await listedUrl(), null);
  });
});

test('shared boards answer as the backend role table does', async () => {
  const backend = await startMockBackend(0, { profile: 'representative' });
  const call = (path, init) =>
    fetch(`${backend.origin}${path}`, {
      ...init,
      headers: { 'content-type': 'application/json' },
      ...(init?.body === undefined ? {} : { body: JSON.stringify(init.body) }),
    });

  try {
    await call('/__faults', { body: { intermediatesCaller: 'user', sharedBoards: true }, method: 'POST' });
    const listed = await (await call('/api/v1/boards/')).json();
    const byId = new Map(listed.map((board) => [board.board_id, board]));

    // Membership, general access, and nothing for a restricted board of someone else's.
    assert.deepEqual(byId.get('shared-editor').allowed_actions, [
      'read',
      'add_media',
      'manage_items',
      'rename',
      'set_cover',
      'archive',
    ]);
    assert.deepEqual(byId.get('shared-viewer').allowed_actions, ['read']);
    assert.equal(byId.get('shared-contributor').my_role, 'contributor');
    assert.equal(byId.has('others-private'), false);
    // A board that exists but the caller may not read answers 403; only a missing board answers 404.
    const unreadable = await call('/api/v1/boards/others-private');
    assert.equal(unreadable.status, 403);
    assert.deepEqual(await unreadable.json(), {
      detail: { code: 'board_unreadable', message: 'Not authorized to access this board' },
    });
    assert.equal((await call('/api/v1/boards/no-such-board')).status, 404);
    assert.equal(
      (await call('/api/v1/images/copy', { body: { board_id: 'others-private', image_names: [] }, method: 'POST' }))
        .status,
      403
    );

    // The role is refused before anything else; an inbox's own refusal comes after it.
    const forbidden = await call('/api/v1/boards/shared-viewer', { body: { board_name: 'x' }, method: 'PATCH' });
    assert.equal(forbidden.status, 403);
    assert.deepEqual(await forbidden.json(), {
      detail: { code: 'board_action_denied', message: 'Not authorized to change this board' },
    });
    const inbox = listed.find((board) => board.is_inbox);
    assert.equal(
      (await call(`/api/v1/boards/${inbox.board_id}`, { body: { board_name: 'x' }, method: 'PATCH' })).status,
      409
    );

    // A revocation behind the page's back makes the board unreadable: it still exists, so it answers 403.
    await call('/__board-access', { body: { board_id: 'shared-editor', member_role: null }, method: 'POST' });
    assert.equal(
      (await call('/api/v1/boards/shared-editor', { body: { board_name: 'x' }, method: 'PATCH' })).status,
      403
    );
    assert.equal((await call('/api/v1/boards/shared-editor')).status, 403);
    const relisted = await (await call('/api/v1/boards/')).json();
    assert.equal(
      relisted.some((board) => board.board_id === 'shared-editor'),
      false
    );
  } finally {
    await backend.close();
  }
});

test('queue retry clones failed or canceled roots once, cancel-by-batch spares other batches, prune keeps active work', async () => {
  await withRepresentativeBackend(async (backend) => {
    const items = await getJson(backend, '/api/v1/queue/default/list_all');
    const failed = items.find((item) => item.status === 'failed');
    const pending = items.find((item) => item.status === 'pending');
    const completed = items.find((item) => item.status === 'completed');

    assert.ok(failed && pending && completed, 'the representative queue has failed, pending and completed items');

    const summaries = await getJson(backend, '/api/v1/queue/default/item_summaries_by_ids', {
      body: JSON.stringify({ item_ids: [completed.item_id, 999_999, failed.item_id] }),
      method: 'POST',
    });
    // In the requested order, unknown ids skipped, with the lineage and status a scoped clear chooses roots by.
    assert.deepEqual(
      summaries.map(({ item_id, parent_item_id, status }) => ({ item_id, parent_item_id, status })),
      [
        { item_id: completed.item_id, parent_item_id: null, status: 'completed' },
        { item_id: failed.item_id, parent_item_id: null, status: 'failed' },
      ]
    );
    assert.equal('session' in summaries[0], false);

    const retried = await getJson(backend, '/api/v1/queue/default/retry_items_by_id', {
      body: JSON.stringify([failed.item_id, failed.item_id, completed.item_id]),
      method: 'PUT',
    });
    const reruns = (await getJson(backend, '/api/v1/queue/default/list_all')).filter(
      (item) => item.retried_from_item_id === failed.item_id
    );

    // Like the backend, the result names the retried root, not its rerun; the pairing names the rerun.
    assert.deepEqual(retried.retried_item_ids, [failed.item_id]);
    assert.equal(reruns.length, 1);
    const [clone] = reruns;
    assert.deepEqual(retried.retried_items, [{ item_id: clone.item_id, source_item_id: failed.item_id }]);
    assert.equal(clone.status, 'pending');
    assert.notEqual(clone.session_id, failed.session_id);
    assert.deepEqual(clone.session.results, {});

    // Retrying the rerun pairs the new rerun with it while the lineage still names the original.
    await getJson(backend, `/api/v1/queue/default/i/${clone.item_id}/cancel`, { method: 'PUT' });
    const again = await getJson(backend, '/api/v1/queue/default/retry_items_by_id', {
      body: JSON.stringify([clone.item_id]),
      method: 'PUT',
    });
    assert.deepEqual(again.retried_item_ids, [clone.item_id]);
    const secondRerun = await getJson(backend, `/api/v1/queue/default/i/${again.retried_items[0].item_id}`);
    assert.equal(secondRerun.retried_from_item_id, failed.item_id);

    // One claimant at a time stages a rerun's results; a confirmed rerun stays delivered.
    const deliver = (action, claimant) =>
      getJson(backend, `/api/v1/queue/default/i/${clone.item_id}/delivery`, {
        body: JSON.stringify({ action, claimant }),
        method: 'POST',
      });
    assert.deepEqual(await deliver('claim', 'tab:a'), { status: 'claimed' });
    const held = await deliver('claim', 'tab:b');
    assert.equal(held.status, 'held');
    assert.ok(held.retry_after_ms > 0);
    assert.deepEqual(await deliver('release', 'tab:a'), { status: 'released' });
    assert.deepEqual(await deliver('claim', 'tab:b'), { status: 'claimed' });
    assert.deepEqual(await deliver('confirm', 'tab:b'), { status: 'delivered' });
    assert.deepEqual(await deliver('claim', 'tab:a'), { status: 'delivered' });

    const canceled = await getJson(backend, '/api/v1/queue/default/cancel_by_batch_ids', {
      body: JSON.stringify({ batch_ids: [pending.batch_id] }),
      method: 'PUT',
    });

    assert.ok(canceled.canceled >= 1);
    assert.equal((await getJson(backend, `/api/v1/queue/default/i/${pending.item_id}`)).status, 'canceled');
    if (secondRerun.batch_id !== pending.batch_id) {
      assert.equal((await getJson(backend, `/api/v1/queue/default/i/${secondRerun.item_id}`)).status, 'pending');
    }

    const pruned = await getJson(backend, '/api/v1/queue/default/prune', { method: 'PUT' });
    const remaining = await getJson(backend, '/api/v1/queue/default/list_all');

    assert.ok(pruned.deleted >= 1);
    assert.ok(remaining.every((item) => ['pending', 'waiting', 'in_progress'].includes(item.status)));
  });
});

test('queue chain rules: prune keeps finished children of an active root, and canceling a child cancels its chain', async () => {
  await withRepresentativeBackend(async (backend) => {
    const items = await getJson(backend, '/api/v1/queue/default/list_all');
    const pending = items.find((item) => item.status === 'pending');
    const completed = items.find((item) => item.status === 'completed');
    assert.ok(pending && completed, 'the representative queue has pending and completed items');

    const nextId = Math.max(...items.map((item) => item.item_id)) + 1;
    const chained = (item_id, root, status) => ({
      ...root,
      item_id,
      parent_item_id: root.item_id,
      root_item_id: root.item_id,
      status,
    });
    const activeRoot = { ...pending, item_id: nextId, parent_item_id: null, root_item_id: null };
    const finishedChildOfActive = chained(nextId + 1, activeRoot, 'completed');
    const waitingChildOfActive = chained(nextId + 2, activeRoot, 'pending');
    const finishedChildOfFinished = chained(nextId + 3, completed, 'completed');
    await getJson(backend, '/__queue_items', {
      body: JSON.stringify({
        items: [activeRoot, finishedChildOfActive, waitingChildOfActive, finishedChildOfFinished],
      }),
      method: 'POST',
    });

    await getJson(backend, '/api/v1/queue/default/prune', { method: 'PUT' });
    const afterPrune = new Set((await getJson(backend, '/api/v1/queue/default/list_all')).map((item) => item.item_id));

    // Like the backend, a finished child stays while its root is still active; one under a finished root goes.
    assert.ok(afterPrune.has(finishedChildOfActive.item_id));
    assert.ok(!afterPrune.has(finishedChildOfFinished.item_id));

    await getJson(backend, `/api/v1/queue/default/i/${waitingChildOfActive.item_id}/cancel`, { method: 'PUT' });
    const status = async (itemId) => (await getJson(backend, `/api/v1/queue/default/i/${itemId}`)).status;

    assert.equal(await status(activeRoot.item_id), 'canceled');
    assert.equal(await status(waitingChildOfActive.item_id), 'canceled');
    assert.equal(await status(finishedChildOfActive.item_id), 'completed');
    assert.equal(await status(pending.item_id), 'pending');
  });
});

test('runtime config PATCH keeps the other writable field and validates the queue history limit', async () => {
  await withRepresentativeBackend(async (backend) => {
    await getJson(backend, '/api/v1/app/runtime_config', {
      body: JSON.stringify({ generation_devices: ['cpu'] }),
      method: 'PATCH',
    });
    const withHistory = await getJson(backend, '/api/v1/app/runtime_config', {
      body: JSON.stringify({ max_queue_history: 0 }),
      method: 'PATCH',
    });

    assert.deepEqual(withHistory.config, { generation_devices: ['cpu'], max_queue_history: 0 });

    const invalid = await fetch(`${backend.origin}/api/v1/app/runtime_config`, {
      body: JSON.stringify({ max_queue_history: -1 }),
      headers: { 'Content-Type': 'application/json' },
      method: 'PATCH',
    });

    assert.equal(invalid.status, 422);

    const cache = await getJson(backend, '/api/v1/app/invocation_cache/disable', { method: 'PUT' });
    assert.equal(cache, null);
    assert.equal((await getJson(backend, '/api/v1/app/invocation_cache/status')).enabled, false);
  });
});
