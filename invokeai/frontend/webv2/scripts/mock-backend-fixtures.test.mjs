import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

import {
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
      config: { generation_devices: 'auto', image_subfolder_strategy: 'flat', max_queue_history: null },
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

const withRepresentativeBackend = async (run, options = {}) => {
  const backend = await startMockBackend(0, { ...options, profile: 'representative' });

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
      cfg_scale: 5,
      fps: 16,
      generation_mode: 'wan_t2v',
      height: 480,
      model: {
        base: 'wan',
        hash: 'blake3:fixture-wan-2-2-t2v-a14b',
        key: 'fixture-wan-t2v-a14b',
        name: 'Wan 2.2 T2V A14B',
        type: 'main',
      },
      negative_prompt: 'blurry, low quality, static frame',
      num_frames: 81,
      positive_prompt: 'wan fixture',
      seed: 1234567,
      steps: 30,
      width: 832,
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

    assert.equal(new Set(projects.map((entry) => entry.board_id)).size, projects.length);
    assert.ok(project.board_id);

    const board = await getJson(backend, `/api/v1/boards/${project.board_id}`);
    assert.equal(board.project_id, project.project_id);
    assert.equal(board.is_inbox, true);
    assert.equal(board.board_name, project.name);

    const plainBoard = await getJson(backend, '/api/v1/boards/fixture-board-02');
    assert.equal(plainBoard.project_id, null);
    assert.equal(plainBoard.is_inbox, false);

    const member = await getJson(backend, '/api/v1/boards/fixture-member-board-02');
    assert.equal(member.project_id, project.project_id);
    assert.equal(member.is_inbox, false);

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
    const renamed = await getJson(backend, `/api/v1/boards/${created.board_id}`, {
      body: JSON.stringify({ board_name: 'Pantry' }),
      headers: { 'content-type': 'application/json' },
      method: 'PATCH',
    });
    assert.equal(renamed.project_id, null);
    assert.equal(renamed.board_name, 'Pantry');

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
    assert.notEqual(copyName, source);
    const copy = await getJson(backend, `/api/v1/images/i/${copyName}`);
    assert.equal(copy.board_id, staging.board_id);
    assert.equal(copy.image_category, (await getJson(backend, `/api/v1/images/i/${source}`)).image_category);
    assert.notEqual((await getJson(backend, `/api/v1/images/i/${source}`)).board_id, staging.board_id);

    const claimed = await getJson(backend, '/api/v1/projects/', {
      body: JSON.stringify({ board_id: staging.board_id, data: {}, name: 'Imported', project_id: 'imported-1' }),
      headers: { 'content-type': 'application/json' },
      method: 'POST',
    });
    assert.equal(claimed.board_id, staging.board_id);
    const second = await fetch(`${backend.origin}/api/v1/projects/`, {
      body: JSON.stringify({ board_id: staging.board_id, data: {}, name: 'Again', project_id: 'imported-2' }),
      headers: { 'content-type': 'application/json' },
      method: 'POST',
    });
    assert.equal(second.status, 409);

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

    assert.equal(staging.general_access, 'restricted');
    assert.equal(staging.board_visibility, 'private');

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

    await call('/__board-access', { body: { board_id: staging.board_id, member_role: 'viewer' }, method: 'POST' });
    assert.equal((await claim(staging.board_id)).status, 409);
    await call('/__board-access', { body: { board_id: staging.board_id, member_role: null }, method: 'POST' });

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

    const contentOnly = await save({ data: project.data, expected_revision: project.revision });
    assert.equal(contentOnly.status, 200);
    const contentOnlyRecord = await contentOnly.json();
    assert.equal(contentOnlyRecord.revision, project.revision + 1);
    assert.equal(contentOnlyRecord.workspace, null);

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

    const staleContent = await save({ data: project.data, expected_revision: 0 });
    assert.equal(staleContent.status, 409);
    assert.deepEqual(await staleContent.json(), {
      detail: { current_revision: splitRecord.revision, message: 'Revision conflict' },
    });

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
    assert.equal(imagesOnly.point_count, imagesOnly.points.length);
    assert.equal(withVideos.point_count, withVideos.points.length);
    assert.equal(withVideos.points.length > imagesOnly.points.length, true);

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
    const crossed = await fetch(
      `${backend.origin}/api/v1/image_map/image_labels?image_name=${video.image_name}&kind=image`
    );
    assert.equal(crossed.status, 404);
  });
});

test('image metadata is empty until the test seam seeds it, and a reset clears it', async () => {
  await withRepresentativeBackend(async (backend) => {
    const name = 'fixture-image-0001.png';
    const metadata = { model: { key: 'm', name: 'Seeded model' }, steps: 12 };
    const post = (path, body) =>
      fetch(`${backend.origin}${path}`, {
        body: JSON.stringify(body),
        headers: { 'Content-Type': 'application/json' },
        method: 'POST',
      });
    const served = (imageName) => getJson(backend, `/api/v1/images/i/${imageName}/metadata`);

    assert.deepEqual(await served(name), {});
    assert.equal((await post('/__image-metadata', { image_name: 'missing.png', metadata })).status, 404);
    assert.equal((await post('/__image-metadata', { image_name: name, metadata: [] })).status, 400);
    assert.equal((await post('/__image-metadata', { image_name: name, metadata })).status, 200);

    assert.deepEqual(await served(name), metadata);
    assert.deepEqual(await served('fixture-image-0002.png'), {});

    await getJson(backend, '/__reset', { method: 'POST' });
    assert.deepEqual(await served(name), {});
  });
});

test('board_id=all lists unboarded items and those on active readable boards, alike in items, names and location', async () => {
  await withRepresentativeBackend(async (backend) => {
    const query = 'categories=general&is_intermediate=false';
    const everything = await getJson(backend, `/api/v1/gallery/items/?${query}&limit=5000`);
    const looseBoards = new Set(
      (await getJson(backend, '/api/v1/boards/?all=true'))
        .filter((board) => board.project_id === null)
        .map((board) => board.board_id)
    );
    const archivedBoardId = everything.items.find((item) => looseBoards.has(item.board_id)).board_id;
    const archive = await fetch(`${backend.origin}/api/v1/boards/${archivedBoardId}`, {
      body: JSON.stringify({ archived: true }),
      headers: { 'content-type': 'application/json' },
      method: 'PATCH',
    });
    assert.equal(archive.ok, true);

    const all = await getJson(backend, `/api/v1/gallery/items/?${query}&board_id=all&limit=5000`);
    const names = await getJson(backend, `/api/v1/gallery/items/names?${query}&board_id=all`);
    const expected = everything.items.filter((item) => item.board_id !== archivedBoardId);

    assert.ok(expected.some((item) => item.board_id === null));
    assert.ok(expected.some((item) => item.board_id !== null));
    assert.deepEqual(all.items.map((item) => item.name).sort(), expected.map((item) => item.name).sort());
    assert.equal(names.total_count, all.total);

    const hidden = everything.items.find((item) => item.board_id === archivedBoardId);
    const shown = all.items.at(-1);
    const location = (item) =>
      fetch(
        `${backend.origin}/api/v1/gallery/items/location?${query}&board_id=all&kind=${item.kind}&name=${encodeURIComponent(item.name)}`
      );
    assert.equal((await location(hidden)).status, 404);
    const found = await (await location(shown)).json();
    assert.equal(found.total, all.total);
  });
});

test('image PATCH adopts an intermediate as hidden durable Canvas pixels on every read', async () => {
  await withRepresentativeBackend(async (backend) => {
    const form = new FormData();
    form.append('file', new Blob(['paint'], { type: 'image/png' }), 'paint.png');
    const response = await fetch(`${backend.origin}/api/v1/images/upload?image_category=general&is_intermediate=true`, {
      method: 'POST',
      body: form,
    });
    const temporary = await response.json();
    assert.equal(temporary.is_intermediate, true);
    const path = `/api/v1/images/i/${encodeURIComponent(temporary.image_name)}`;
    const adopted = await getJson(backend, path, {
      method: 'PATCH',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ is_intermediate: false, image_category: 'other' }),
    });
    assert.deepEqual(adopted, { ...temporary, is_intermediate: false, image_category: 'other' });
    assert.deepEqual(await getJson(backend, path), adopted);
    assert.deepEqual(
      await getJson(backend, '/api/v1/images/images_by_names', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ image_names: [temporary.image_name] }),
      }),
      [adopted]
    );
    const missing = await fetch(`${backend.origin}/api/v1/images/i/missing.png`, {
      method: 'PATCH',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ is_intermediate: false }),
    });
    assert.equal(missing.status, 404);
  });
});
