import { createServer } from 'node:http';

import {
  assertMockBackendFixture,
  assertMockBackendProfileName,
  createMockBackendFixture,
  getMockBackendFixtureCounts,
  MOCK_BACKEND_FIXED_EPOCH,
} from './mock-backend-fixtures.mjs';

/**
 * Disposable in-memory InvokeAI backend for browser release/performance tests.
 *
 * `empty` and `representative` are explicit workload profiles. A reset without
 * a profile restores the profile selected at server startup; callers may switch
 * deterministically with `POST /__reset?profile=representative`. No socket.io
 * server is provided, so realtime transport remains stably disconnected.
 */

const FIXED_EPOCH_MS = Date.parse(MOCK_BACKEND_FIXED_EPOCH);
const TINY_PNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M/wHwAF/gL+XcX1WQAAAABJRU5ErkJggg==',
  'base64'
);

const clone = (value) => structuredClone(value);

const createState = (profile) => {
  const fixture = assertMockBackendFixture(createMockBackendFixture(profile));

  return {
    boards: new Map(fixture.boards.map((board) => [board.board_id, clone(board)])),
    clientState: new Map(),
    images: new Map(fixture.images.map((image) => [image.image_name, clone(image)])),
    models: new Map(fixture.models.map((model) => [model.key, clone(model)])),
    mutationClock: 0,
    nextProjectNumber: fixture.projects.length + 1,
    nodeCatalog: clone(fixture.nodeCatalog),
    openApiDocument: clone(fixture.openApiDocument),
    profile,
    projects: new Map(fixture.projects.map((project) => [project.project_id, clone(project)])),
    queueItems: new Map(fixture.queueItems.map((item) => [item.item_id, clone(item)])),
    workflows: new Map(fixture.workflows.map((workflow) => [workflow.workflow_id, clone(workflow)])),
  };
};

const timestamp = (state) => {
  const value = new Date(FIXED_EPOCH_MS + state.mutationClock * 1_000).toISOString();

  state.mutationClock += 1;

  return value;
};

const summaryOf = (project) => ({
  created_at: project.created_at,
  name: project.name,
  project_id: project.project_id,
  revision: project.revision,
  updated_at: project.updated_at,
});

const readBody = (request) =>
  new Promise((resolve, reject) => {
    const chunks = [];

    request.on('data', (chunk) => chunks.push(chunk));
    request.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    request.on('error', reject);
  });

const readJsonBody = async (request, fallback = {}) => {
  const body = await readBody(request);

  return body ? JSON.parse(body) : fallback;
};

const invocationNodeCount = (state) =>
  Object.values(state.openApiDocument.components?.schemas ?? {}).filter((schema) => schema?.class === 'invocation')
    .length;

const getStateCounts = (state) => ({
  images: state.images.size,
  layers: state.projects.values().next().value?.data?.canvas?.document?.layers?.length ?? 0,
  models: state.models.size,
  nodes: invocationNodeCount(state),
  projects: state.projects.size,
  queueItems: state.queueItems.size,
  workflowNodes: state.projects.values().next().value?.data?.projectGraph?.nodes?.length ?? 0,
});

const getProfileInfo = (state) => ({
  counts: getStateCounts(state),
  profile: state.profile,
});

const queueItemsForScope = (state, url) => {
  const originPrefix = url.searchParams.get('origin_prefix');
  const items = [...state.queueItems.values()];

  return originPrefix ? items.filter((item) => item.origin?.startsWith(originPrefix)) : items;
};

const queueStatus = (items) => {
  const count = (status) => items.filter((item) => item.status === status).length;

  return {
    canceled: count('canceled'),
    completed: count('completed'),
    failed: count('failed'),
    in_progress: count('in_progress'),
    pending: count('pending'),
    queue_id: 'default',
    total: items.length,
    waiting: count('waiting'),
  };
};

const getRequestedCategories = (url) => {
  const values = url.searchParams.getAll('categories');

  return values.flatMap((value) => value.split(',')).filter(Boolean);
};

const listImages = (state, url) => {
  const boardId = url.searchParams.get('board_id');
  const categories = getRequestedCategories(url);
  const createdFrom = url.searchParams.get('created_from');
  const createdTo = url.searchParams.get('created_to');
  const orderDir = url.searchParams.get('order_dir')?.toUpperCase() === 'ASC' ? 'ASC' : 'DESC';
  const searchTerm = url.searchParams.get('search_term')?.trim().toLocaleLowerCase() ?? '';
  const starredFirst = url.searchParams.get('starred_first') === 'true';
  const offset = Math.max(0, Number(url.searchParams.get('offset') ?? 0) || 0);
  const limit = Math.max(0, Number(url.searchParams.get('limit') ?? 100) || 0);
  let items = [...state.images.values()].filter((image) => {
    if (image.is_intermediate) {
      return false;
    }
    if (boardId && boardId !== 'all') {
      if (boardId === 'none' ? image.board_id !== null : image.board_id !== boardId) {
        return false;
      }
    }
    if (categories.length > 0 && !categories.includes(image.image_category)) {
      return false;
    }
    if (createdFrom && image.created_at.slice(0, 10) < createdFrom) {
      return false;
    }
    if (createdTo && image.created_at.slice(0, 10) > createdTo) {
      return false;
    }

    return !searchTerm || image.image_name.toLocaleLowerCase().includes(searchTerm);
  });

  items.sort((left, right) => {
    if (starredFirst && Boolean(left.starred) !== Boolean(right.starred)) {
      return left.starred ? -1 : 1;
    }

    return orderDir === 'ASC'
      ? left.created_at.localeCompare(right.created_at)
      : right.created_at.localeCompare(left.created_at);
  });

  const total = items.length;
  items = limit === 0 ? [] : items.slice(offset, offset + limit);

  return { items, limit, offset, total };
};

const listVirtualDateBoards = (state) => {
  const groups = new Map();

  for (const image of state.images.values()) {
    const date = image.created_at.slice(0, 10);
    const group = groups.get(date) ?? [];

    group.push(image);
    groups.set(date, group);
  }

  return [...groups.entries()]
    .sort(([left], [right]) => right.localeCompare(left))
    .map(([date, images]) => ({
      asset_count: images.filter((image) => image.image_category !== 'general').length,
      board_name: date,
      cover_image_name: images[0]?.image_name ?? null,
      date,
      image_count: images.filter((image) => image.image_category === 'general').length,
      virtual_board_id: `by_date:${date}`,
    }));
};

const writeJson = (response, status, value) => {
  const body = JSON.stringify(value ?? null);

  response.writeHead(status, {
    'cache-control': 'no-store',
    'content-length': Buffer.byteLength(body),
    'content-type': 'application/json',
  });
  response.end(body);
};

const writePng = (response) => {
  response.writeHead(200, {
    'cache-control': 'public, max-age=3600',
    'content-length': TINY_PNG.length,
    'content-type': 'image/png',
  });
  response.end(TINY_PNG);
};

export const startMockBackend = async (port, { profile = 'empty' } = {}) => {
  const initialProfile = assertMockBackendProfileName(profile);
  let state = createState(initialProfile);

  const server = createServer(async (request, response) => {
    const url = new URL(request.url, `http://127.0.0.1:${String(port)}`);
    const path = url.pathname;
    const method = request.method ?? 'GET';
    const json = (status, value) => writeJson(response, status, value);

    try {
      if (method === 'GET' && (path === '/__health' || path === '/__profile')) {
        return json(200, { ok: true, ...getProfileInfo(state) });
      }

      if (method === 'POST' && path === '/__reset') {
        const body = await readJsonBody(request);
        const requestedProfile = url.searchParams.get('profile') ?? body.profile ?? initialProfile;

        try {
          state = createState(assertMockBackendProfileName(requestedProfile));
        } catch (error) {
          return json(400, { detail: error instanceof Error ? error.message : String(error) });
        }

        return json(200, { ok: true, ...getProfileInfo(state) });
      }

      if (method === 'GET' && path === '/openapi.json') {
        return json(200, state.openApiDocument);
      }

      if (method === 'GET' && path === '/api/v1/auth/status') {
        return json(200, {
          admin_email: null,
          multiuser_enabled: false,
          setup_required: false,
          strict_password_checking: false,
        });
      }

      if (method === 'GET' && path === '/api/v1/app/version') {
        return json(200, { version: 'fixture' });
      }

      if (path.startsWith('/api/v1/app/external_providers')) {
        return json(200, []);
      }

      if (path === '/api/v1/projects/' || path === '/api/v1/projects') {
        if (method === 'GET') {
          return json(200, [...state.projects.values()].map(summaryOf));
        }
        if (method === 'POST') {
          const requested = await readJsonBody(request);

          if (requested.project_id && state.projects.has(requested.project_id)) {
            return json(409, { detail: 'Project already exists' });
          }

          const now = timestamp(state);
          const projectNumber = state.nextProjectNumber;
          const project = {
            created_at: now,
            data: requested.data ?? {},
            name: requested.name ?? `Project Name #${projectNumber}`,
            project_id: requested.project_id ?? `mock-project-${projectNumber}`,
            revision: 1,
            updated_at: now,
          };

          state.nextProjectNumber += 1;
          state.projects.set(project.project_id, project);

          return json(200, project);
        }
      }

      const projectMatch = /^\/api\/v1\/projects\/([^/]+)$/.exec(path);
      if (projectMatch) {
        const projectId = decodeURIComponent(projectMatch[1]);
        const project = state.projects.get(projectId);

        if (method === 'GET') {
          return project ? json(200, project) : json(404, { detail: 'Project not found' });
        }
        if (method === 'PUT') {
          if (!project) {
            return json(404, { detail: 'Project not found' });
          }

          const requested = await readJsonBody(request);

          if (requested.expected_revision !== undefined && requested.expected_revision !== project.revision) {
            return json(409, { detail: 'Revision conflict' });
          }

          project.data = requested.data ?? project.data;
          project.name = requested.name ?? project.name;
          project.revision += 1;
          project.updated_at = timestamp(state);

          return json(200, project);
        }
        if (method === 'DELETE') {
          state.projects.delete(projectId);
          return json(200, { ok: true });
        }
      }

      if (path.startsWith('/api/v1/client_state/')) {
        const key = url.searchParams.get('key') ?? '';

        if (path.endsWith('/get_by_key')) {
          return json(200, state.clientState.get(key) ?? null);
        }
        if (path.endsWith('/set_by_key')) {
          const value = await readJsonBody(request, null);

          state.clientState.set(key, value);
          return json(200, value);
        }
        if (path.endsWith('/delete_by_key')) {
          state.clientState.delete(key);
          return json(200, { ok: true });
        }
      }

      const queueItemMatch = /^\/api\/v1\/queue\/[^/]+\/i\/(\d+)$/.exec(path);
      if (queueItemMatch) {
        const itemId = Number(queueItemMatch[1]);
        const item = state.queueItems.get(itemId);

        if (method === 'GET') {
          return item ? json(200, item) : json(404, { detail: 'Queue item not found' });
        }
        if (method === 'DELETE') {
          state.queueItems.delete(itemId);
          return json(200, { ok: true });
        }
      }

      const queueMatch = /^\/api\/v1\/queue\/[^/]+\/(.+)$/.exec(path);
      if (queueMatch) {
        const action = queueMatch[1];
        const scopedItems = queueItemsForScope(state, url);

        if (action === 'status') {
          return json(200, {
            processor: { is_processing: scopedItems.some((item) => item.status === 'in_progress'), is_started: true },
            queue: queueStatus(scopedItems),
          });
        }
        if (action === 'current') {
          return json(200, scopedItems.find((item) => item.status === 'in_progress') ?? null);
        }
        if (action === 'next') {
          return json(200, scopedItems.find((item) => item.status === 'pending' || item.status === 'waiting') ?? null);
        }
        if (action === 'list_all') {
          return json(200, scopedItems);
        }
        if (action === 'items_by_ids') {
          const body = await readJsonBody(request);
          const ids = Array.isArray(body.item_ids) ? new Set(body.item_ids) : new Set();

          return json(
            200,
            scopedItems.filter((item) => ids.has(item.item_id))
          );
        }
        if (action === 'item_ids' || action === 'list') {
          const descending = url.searchParams.get('order_dir')?.toUpperCase() !== 'ASC';
          const itemIds = scopedItems.map((item) => item.item_id).sort((left, right) => left - right);

          if (descending) {
            itemIds.reverse();
          }

          return json(200, { item_ids: itemIds, total_count: itemIds.length });
        }
        if (action === 'clear' && method === 'PUT') {
          state.queueItems.clear();
          return json(200, { deleted: true });
        }
        if (action === 'prune' && method === 'PUT') {
          for (const item of state.queueItems.values()) {
            if (item.status === 'completed' || item.status === 'failed' || item.status === 'canceled') {
              state.queueItems.delete(item.item_id);
            }
          }
          return json(200, { deleted: true });
        }

        return json(200, null);
      }

      if (method === 'GET' && path === '/api/v2/models/stats') {
        return json(200, null);
      }
      if (method === 'GET' && path === '/api/v2/models/missing') {
        return json(200, { models: [] });
      }
      if (method === 'GET' && path === '/api/v2/models/models_dir') {
        return json(200, '/opt/invokeai/models');
      }
      if (method === 'GET' && path === '/api/v2/models/install') {
        return json(200, []);
      }
      if (method === 'GET' && path === '/api/v2/models/starter_models') {
        return json(200, { starter_bundles: {}, starter_models: [] });
      }
      if (method === 'GET' && path === '/api/v2/models/hf_login') {
        return json(200, 'unknown');
      }
      if (method === 'GET' && path === '/api/v2/models/sync/orphaned') {
        return json(200, []);
      }

      const modelMatch = /^\/api\/v2\/models\/i\/([^/]+)$/.exec(path);
      if (modelMatch) {
        const key = decodeURIComponent(modelMatch[1]);
        const model = state.models.get(key);

        return model ? json(200, model) : json(404, { detail: 'Model not found' });
      }

      if (method === 'GET' && (path === '/api/v2/models' || path === '/api/v2/models/')) {
        return json(200, { models: [...state.models.values()] });
      }

      if (path.startsWith('/api/v1/model_relationships')) {
        return json(200, []);
      }

      // Dynamic prompt expansion. Enough of the `{a|b}` grammar for journeys to
      // exercise the preview and the batch dimension; the real generator lives
      // in the backend.
      if (method === 'POST' && path === '/api/v1/utilities/dynamicprompts') {
        const requested = await readJsonBody(request);
        const prompt = typeof requested?.prompt === 'string' ? requested.prompt : '';
        const maxPrompts = typeof requested?.max_prompts === 'number' ? requested.max_prompts : 100;
        const match = /\{([^{}]*)\}/.exec(prompt);
        const prompts = match ? match[1].split('|').map((value) => prompt.replace(match[0], value.trim())) : [prompt];

        return json(200, { error: null, prompts: prompts.slice(0, Math.max(1, maxPrompts)) });
      }

      if (method === 'GET' && (path === '/api/v2/custom_nodes' || path === '/api/v2/custom_nodes/')) {
        return json(200, state.nodeCatalog);
      }

      if (method === 'GET' && (path === '/api/v1/boards' || path === '/api/v1/boards/')) {
        return json(200, [...state.boards.values()]);
      }

      if (method === 'GET' && path === '/api/v1/virtual_boards/by_date') {
        return json(200, listVirtualDateBoards(state));
      }

      const virtualBoardNamesMatch = /^\/api\/v1\/virtual_boards\/by_date\/([^/]+)\/image_names$/.exec(path);
      if (method === 'GET' && virtualBoardNamesMatch) {
        const date = decodeURIComponent(virtualBoardNamesMatch[1]);
        const imageNames = [...state.images.values()]
          .filter((image) => image.created_at.startsWith(date))
          .map((image) => image.image_name);

        return json(200, { image_names: imageNames, total_count: imageNames.length });
      }

      if (method === 'POST' && path === '/api/v1/images/images_by_names') {
        const body = await readJsonBody(request);
        const names = Array.isArray(body.image_names) ? body.image_names : [];

        return json(
          200,
          names.flatMap((name) => (state.images.has(name) ? [state.images.get(name)] : []))
        );
      }

      if (method === 'GET' && (path === '/api/v1/images' || path === '/api/v1/images/')) {
        return json(200, listImages(state, url));
      }

      const imageAssetMatch = /^\/api\/v1\/images\/i\/([^/]+)\/(full|thumbnail)$/.exec(path);
      if (method === 'GET' && imageAssetMatch) {
        const imageName = decodeURIComponent(imageAssetMatch[1]);

        return state.images.has(imageName) ? writePng(response) : json(404, { detail: 'Image not found' });
      }

      const imageMetadataMatch = /^\/api\/v1\/images\/i\/([^/]+)\/metadata$/.exec(path);
      if (method === 'GET' && imageMetadataMatch) {
        return json(200, {});
      }

      const imageMatch = /^\/api\/v1\/images\/i\/([^/]+)$/.exec(path);
      if (imageMatch) {
        const imageName = decodeURIComponent(imageMatch[1]);
        const image = state.images.get(imageName);

        return image ? json(200, image) : json(404, { detail: 'Image not found' });
      }

      const workflowMatch = /^\/api\/v1\/workflows\/i\/([^/]+)(?:\/opened_at)?$/.exec(path);
      if (workflowMatch) {
        const workflowId = decodeURIComponent(workflowMatch[1]);
        const workflow = state.workflows.get(workflowId);

        if (!workflow) {
          return json(404, { detail: 'Workflow not found' });
        }

        return json(200, {
          name: workflow.name,
          workflow: workflow.workflow,
          workflow_id: workflow.workflow_id,
        });
      }

      if (method === 'GET' && (path === '/api/v1/workflows' || path === '/api/v1/workflows/')) {
        const categories = url.searchParams.getAll('categories');
        const query = url.searchParams.get('query')?.trim().toLocaleLowerCase() ?? '';
        const page = Math.max(1, Number(url.searchParams.get('page') ?? 1) || 1);
        const perPage = Math.max(1, Number(url.searchParams.get('per_page') ?? 20) || 20);
        const items = [...state.workflows.values()].filter(
          (workflow) =>
            (categories.length === 0 || categories.includes(workflow.category)) &&
            (!query || `${workflow.name} ${workflow.description}`.toLocaleLowerCase().includes(query))
        );
        const pages = Math.max(1, Math.ceil(items.length / perPage));
        const pageItems = items
          .slice((page - 1) * perPage, page * perPage)
          .map(({ workflow: _workflow, ...item }) => item);

        return json(200, { items: pageItems, page, pages, total: items.length });
      }

      return json(404, { detail: `No mock for ${method} ${path}` });
    } catch (error) {
      return json(500, { detail: error instanceof Error ? error.message : String(error) });
    }
  });

  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(port, '127.0.0.1', resolve);
  });

  const address = server.address();
  const actualPort = typeof address === 'object' && address ? address.port : port;

  return {
    close: () =>
      new Promise((resolve) => {
        server.close(resolve);
        server.closeAllConnections?.();
      }),
    counts: () => getStateCounts(state),
    origin: `http://127.0.0.1:${String(actualPort)}`,
    port: actualPort,
    profile: () => state.profile,
  };
};

export { getMockBackendFixtureCounts };
