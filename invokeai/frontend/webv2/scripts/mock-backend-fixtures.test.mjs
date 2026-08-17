import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

import {
  assertMockBackendFixture,
  createMockBackendFixture,
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
  assert.deepEqual(
    project.data.projectGraph.nodes.map((node) => node.data.type),
    invocationTypes
  );
  assert.equal(project.data.canvas.document.layers.length, MOCK_BACKEND_PROFILE_COUNTS.representative.layers);
  assert.equal(new Set(project.data.canvas.document.layers.map((layer) => layer.id)).size, 64);
});

test('Fixture Project 002 carries the image and video references used by the project-file journey', () => {
  const fixture = createMockBackendFixture('representative');
  const project = fixture.projects[1];

  assert.equal(project.project_id, 'fixture-project-002');
  assert.deepEqual(
    project.data.canvas.document.layers.map((layer) => layer.source.image.imageName),
    ['fixture-image-0001.png', 'fixture-image-0002.png', 'fixture-image-0003.png', 'fixture-image-0004.png']
  );
  assert.equal(
    project.data.canvas.document.layers.every((layer) => layer.type === 'raster'),
    true
  );
  assert.deepEqual(project.data.projectGraph.nodes[0]?.data.inputs.video?.value, {
    video_name: MOCK_BACKEND_REPRESENTATIVE_VIDEO_NAME,
  });
});

/**
 * The board path can only be proved end to end if the fixture contains the cases: an item that is
 * both board membership and a canvas reference, one that is only membership, each visible category,
 * the two kinds of media that must be excluded, and references that live outside the board.
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

  // The canvas draws with these, and no project owns them: on import they deduplicate against the
  // destination, and on duplication they are not copied at all.
  const layerNames = project.data.canvas.document.layers.map((layer) => layer.source.image.imageName);

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
    // One of the thousand is an intermediate on Fixture Project 002's board, which every gallery
    // listing hides — it exists so the project-file journey can prove it never travels.
    assert.equal(images.total, 999);
    assert.equal(itemIds.item_ids.length, 500);
    assert.equal(models.models.length, 100);
    assert.equal(nodeCatalog.node_packs.flatMap((pack) => pack.node_types).length, 100);
    assert.equal(
      Object.values(openApi.components.schemas).filter((schema) => schema.class === 'invocation').length,
      100
    );
    assert.equal(projects.length, 40);
    assert.deepEqual(runtimeConfig, { config: { generation_devices: 'auto' }, set_fields: [] });

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

test('project boards are owned, protected from the generic board routes, and enumerable', async () => {
  await withRepresentativeBackend(async (backend) => {
    const projects = await getJson(backend, '/api/v1/projects/');
    const project = projects.find((entry) => entry.project_id === 'fixture-project-002');

    // Every project owns exactly one board, and no two share one.
    assert.equal(new Set(projects.map((entry) => entry.board_id)).size, projects.length);
    assert.ok(project.board_id);

    const board = await getJson(backend, `/api/v1/boards/${project.board_id}`);
    assert.equal(board.project_id, project.project_id);
    assert.equal(board.board_name, project.name);

    // An unclaimed board omits the key entirely, matching the backend's null-excluding DTO.
    const plainBoard = await getJson(backend, '/api/v1/boards/fixture-board-02');
    assert.equal('project_id' in plainBoard, false);

    // The generic routes refuse a claimed board; only the project APIs may touch it.
    for (const [method, path, body] of [
      ['PATCH', `/api/v1/boards/${project.board_id}`, { board_name: 'Renamed' }],
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
    assert.deepEqual(snapshot.items.map((item) => item.name).sort(), [control, general].sort());
    assert.deepEqual(snapshot.items.map((item) => item.category).sort(), ['control', 'general']);
    assert.equal(
      snapshot.items.every((item) => item.kind === 'image' && item.starred === false),
      true
    );

    await getJson(backend, '/api/v1/images/star', {
      body: JSON.stringify({ image_names: [general] }),
      headers: { 'content-type': 'application/json' },
      method: 'POST',
    });
    const starred = await getJson(backend, `/api/v1/projects/${project.project_id}/board-snapshot`);
    assert.equal(starred.items.find((item) => item.name === general).starred, true);

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
