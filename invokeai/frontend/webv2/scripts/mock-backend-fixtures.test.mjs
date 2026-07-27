import assert from 'node:assert/strict';
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
