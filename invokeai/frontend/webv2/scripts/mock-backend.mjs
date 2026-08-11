import { readFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { resolve } from 'node:path';

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
const FIXTURE_VIDEO = readFileSync(resolve(import.meta.dirname, 'mock-assets/fixture-video.mp4'));
const FIXTURE_VIDEO_POSTER = readFileSync(resolve(import.meta.dirname, 'mock-assets/fixture-video.webp'));
const MOCK_USER_ID = 'fixture-user';

const clone = (value) => structuredClone(value);

/**
 * Two starter bundles, so the Launchpad's "no models installed" onboarding can
 * actually be exercised. An empty response renders no call to action at all,
 * which made the most important path on a fresh install unverifiable.
 */
const STARTER_MODEL = {
  base: 'sd-1',
  description: 'Fixture starter model',
  format: 'diffusers',
  is_installed: false,
  name: 'Fixture Starter',
  source: 'https://example.invalid/fixture-starter.safetensors',
  type: 'main',
};

const STARTER_MODELS_RESPONSE = {
  starter_bundles: {
    'sd-1': { models: [STARTER_MODEL], name: 'Stable Diffusion 1.5' },
    sdxl: {
      models: [{ ...STARTER_MODEL, base: 'sdxl', name: 'Fixture Starter XL' }],
      name: 'SDXL',
    },
  },
  starter_models: [STARTER_MODEL],
};

const createState = (profile) => {
  const fixture = assertMockBackendFixture(createMockBackendFixture(profile));

  return {
    boards: new Map(fixture.boards.map((board) => [board.board_id, clone(board)])),
    clientState: new Map(),
    images: new Map(fixture.images.map((image) => [image.image_name, clone(image)])),
    models: new Map(fixture.models.map((model) => [model.key, clone(model)])),
    mutationClock: 0,
    nextBoardNumber: fixture.boards.length + 1,
    nextImageNumber: fixture.images.length + 1,
    nextProjectNumber: fixture.projects.length + 1,
    nextVideoNumber: fixture.videos.length + 1,
    nodeCatalog: clone(fixture.nodeCatalog),
    openApiDocument: clone(fixture.openApiDocument),
    profile,
    projects: new Map(fixture.projects.map((project) => [project.project_id, clone(project)])),
    queueItems: new Map(fixture.queueItems.map((item) => [item.item_id, clone(item)])),
    videos: new Map(fixture.videos.map((video) => [video.video_name, clone(video)])),
    workflows: new Map(fixture.workflows.map((workflow) => [workflow.workflow_id, clone(workflow)])),
  };
};

/**
 * Put media back under names a fresh state does not have, without its projects or boards.
 *
 * A restore has to be provably free of name adoption: board media must take a new identity even
 * when the destination already holds an image with the archived name — on the same server it always
 * does, and reusing it would move a stranger's picture onto the restored board. Reset alone cannot
 * express that, because it clears everything, so the journey names the collisions it wants kept.
 * The media lands unboarded, exactly as media whose board was deleted would.
 */
const seedCollisionMedia = (state, names) => {
  const source = createMockBackendFixture('representative');
  const requested = new Set(names);

  for (const image of source.images) {
    if (requested.has(image.image_name)) {
      state.images.set(image.image_name, { ...clone(image), board_id: null, is_intermediate: false });
    }
  }

  for (const video of source.videos) {
    if (requested.has(video.video_name)) {
      state.videos.set(video.video_name, { ...clone(video), board_id: null, is_intermediate: false });
    }
  }
};

const timestamp = (state) => {
  const value = new Date(FIXED_EPOCH_MS + state.mutationClock * 1_000).toISOString();

  state.mutationClock += 1;

  return value;
};

const summaryOf = (project) => ({
  board_id: project.board_id,
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

const getOptionalBoolean = (url, name) => {
  const value = url.searchParams.get(name);

  return value === null ? undefined : value === 'true';
};

const toVideoDto = (video) => ({
  board_id: video.board_id,
  created_at: video.created_at,
  duration: video.duration,
  fps: video.fps,
  height: video.height,
  is_intermediate: video.is_intermediate,
  starred: video.starred,
  thumbnail_url: video.thumbnail_url,
  video_category: video.video_category,
  video_name: video.video_name,
  video_url: video.video_url,
  width: video.width,
});

const toGalleryItem = (kind, value) =>
  kind === 'image'
    ? {
        board_id: value.board_id,
        category: value.image_category,
        created_at: value.created_at,
        full_url: value.image_url,
        height: value.height,
        is_intermediate: value.is_intermediate,
        kind,
        name: value.image_name,
        starred: value.starred ?? false,
        thumbnail_url: value.thumbnail_url,
        width: value.width,
      }
    : {
        board_id: value.board_id,
        category: value.video_category,
        created_at: value.created_at,
        duration: value.duration,
        fps: value.fps,
        full_url: value.video_url,
        height: value.height,
        is_intermediate: value.is_intermediate,
        kind,
        name: value.video_name,
        starred: value.starred,
        thumbnail_url: value.thumbnail_url,
        width: value.width,
      };

const getGalleryCandidates = (state) => [
  ...[...state.images.values()].map((image) => ({ item: toGalleryItem('image', image), searchable: image.metadata })),
  ...[...state.videos.values()]
    .filter((video) => video.owner_user_id === MOCK_USER_ID)
    .map((video) => ({ item: toGalleryItem('video', video), searchable: video.metadata })),
];

const compareGalleryItems = (left, right, orderDir, starredFirst) => {
  if (starredFirst && left.starred !== right.starred) {
    return left.starred ? -1 : 1;
  }

  const direction = orderDir === 'ASC' ? 1 : -1;

  return (
    direction * left.created_at.localeCompare(right.created_at) ||
    direction * left.kind.localeCompare(right.kind) ||
    direction * left.name.localeCompare(right.name)
  );
};

const filterGalleryItems = (state, url, { createdDate } = {}) => {
  const boardId = url.searchParams.get('board_id');
  const categories = getRequestedCategories(url);
  const createdFrom = url.searchParams.get('created_from');
  const createdTo = url.searchParams.get('created_to');
  const intermediate = getOptionalBoolean(url, 'is_intermediate');
  const searchTerm = url.searchParams.get('search_term')?.trim().toLocaleLowerCase() ?? '';

  return getGalleryCandidates(state)
    .filter(({ item, searchable }) => {
      if (boardId && (boardId === 'none' ? item.board_id !== null : item.board_id !== boardId)) {
        return false;
      }
      if (categories.length > 0 && !categories.includes(item.category)) {
        return false;
      }
      if (intermediate !== undefined && item.is_intermediate !== intermediate) {
        return false;
      }
      if (createdDate && item.created_at.slice(0, 10) !== createdDate) {
        return false;
      }
      if (createdFrom && item.created_at < createdFrom) {
        return false;
      }
      if (createdTo && item.created_at >= `${createdTo}T24:00:00`) {
        return false;
      }
      if (searchTerm) {
        const document = `${JSON.stringify(searchable ?? {})} ${item.created_at}`.toLocaleLowerCase();

        if (!document.includes(searchTerm)) {
          return false;
        }
      }

      return true;
    })
    .map(({ item }) => item)
    .sort((left, right) =>
      compareGalleryItems(
        left,
        right,
        url.searchParams.get('order_dir')?.toUpperCase() === 'ASC' ? 'ASC' : 'DESC',
        url.searchParams.get('starred_first') !== 'false'
      )
    );
};

const listGalleryItems = (state, url) => {
  const offset = Math.max(0, Number(url.searchParams.get('offset') ?? 0) || 0);
  const limit = Math.max(0, Number(url.searchParams.get('limit') ?? 10) || 0);
  const filtered = filterGalleryItems(state, url);

  return {
    items: limit === 0 ? [] : filtered.slice(offset, offset + limit),
    limit,
    offset,
    total: filtered.length,
  };
};

const listGalleryItemNames = (state, url, options) => {
  const items = filterGalleryItems(state, url, options);

  return {
    items: items.map(({ kind, name }) => ({ kind, name })),
    starred_count: url.searchParams.get('starred_first') !== 'false' ? items.filter((item) => item.starred).length : 0,
    total_count: items.length,
  };
};

const listVideos = (state, url) => {
  const boardId = url.searchParams.get('board_id');
  const categories = getRequestedCategories(url);
  const intermediate = getOptionalBoolean(url, 'is_intermediate');
  const searchTerm = url.searchParams.get('search_term')?.trim().toLocaleLowerCase() ?? '';
  const orderDir = url.searchParams.get('order_dir')?.toUpperCase() === 'ASC' ? 'ASC' : 'DESC';
  const starredFirst = url.searchParams.get('starred_first') !== 'false';
  const offset = Math.max(0, Number(url.searchParams.get('offset') ?? 0) || 0);
  const limit = Math.max(0, Number(url.searchParams.get('limit') ?? 10) || 0);
  const filtered = [...state.videos.values()]
    .filter((video) => {
      if (video.owner_user_id !== MOCK_USER_ID) {
        return false;
      }
      if (boardId && (boardId === 'none' ? video.board_id !== null : video.board_id !== boardId)) {
        return false;
      }
      if (categories.length > 0 && !categories.includes(video.video_category)) {
        return false;
      }
      if (intermediate !== undefined && video.is_intermediate !== intermediate) {
        return false;
      }

      return (
        !searchTerm ||
        `${JSON.stringify(video.metadata ?? {})} ${video.created_at}`.toLocaleLowerCase().includes(searchTerm)
      );
    })
    .map(toVideoDto)
    .sort((left, right) =>
      compareGalleryItems(
        {
          created_at: left.created_at,
          kind: 'video',
          name: left.video_name,
          starred: left.starred,
        },
        {
          created_at: right.created_at,
          kind: 'video',
          name: right.video_name,
          starred: right.starred,
        },
        orderDir,
        starredFirst
      )
    );

  return {
    items: limit === 0 ? [] : filtered.slice(offset, offset + limit),
    limit,
    offset,
    total: filtered.length,
  };
};

/** The project that owns this board, if any. Derived, exactly as the backend derives it. */
const projectIdForBoard = (state, boardId) =>
  [...state.projects.values()].find((project) => project.board_id === boardId)?.project_id ?? null;

/**
 * Visible board membership, matching `GET /projects/{id}/board-snapshot`: intermediates and the
 * canvas's private `other` category are excluded, and the result is sorted by kind then name.
 */
const boardSnapshotItems = (state, boardId) => {
  const visible = (category) => ['general', 'control', 'mask', 'user'].includes(category);
  const items = [
    ...[...state.images.values()]
      .filter((image) => image.board_id === boardId && !image.is_intermediate && visible(image.image_category))
      .map((image) => ({
        category: image.image_category,
        kind: 'image',
        name: image.image_name,
        starred: Boolean(image.starred),
      })),
    ...[...state.videos.values()]
      .filter(
        (video) =>
          video.board_id === boardId &&
          video.owner_user_id === MOCK_USER_ID &&
          !video.is_intermediate &&
          visible(video.video_category)
      )
      .map((video) => ({
        category: video.video_category,
        kind: 'video',
        name: video.video_name,
        starred: Boolean(video.starred),
      })),
  ];

  return items.sort((left, right) => left.kind.localeCompare(right.kind) || left.name.localeCompare(right.name));
};

const boardDto = (state, board) => {
  const images = [...state.images.values()].filter((image) => image.board_id === board.board_id);
  const videos = [...state.videos.values()].filter(
    (video) => video.board_id === board.board_id && video.owner_user_id === MOCK_USER_ID
  );
  const cover = [
    ...images.map((image) => ({
      createdAt: image.created_at,
      kind: 'image',
      name: image.image_name,
      starred: image.starred,
    })),
    ...videos.map((video) => ({
      createdAt: video.created_at,
      kind: 'video',
      name: video.video_name,
      starred: video.starred,
    })),
  ].sort(
    (left, right) =>
      Number(right.starred) - Number(left.starred) ||
      right.createdAt.localeCompare(left.createdAt) ||
      right.kind.localeCompare(left.kind) ||
      right.name.localeCompare(left.name)
  )[0];

  const projectId = projectIdForBoard(state, board.board_id);

  return {
    ...board,
    asset_count: images.filter((image) => image.image_category !== 'general').length,
    cover_image_name: cover?.kind === 'image' ? cover.name : null,
    cover_video_name: cover?.kind === 'video' ? cover.name : null,
    image_count: images.filter((image) => image.image_category === 'general').length,
    // The backend's BoardRecord excludes nulls, so an unclaimed board omits the key entirely.
    ...(projectId === null ? {} : { project_id: projectId }),
    video_count: videos.length,
  };
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

  for (const item of getGalleryCandidates(state).map(({ item }) => item)) {
    if (item.is_intermediate) {
      continue;
    }
    const date = item.created_at.slice(0, 10);
    const group = groups.get(date) ?? [];

    group.push(item);
    groups.set(date, group);
  }

  return [...groups.entries()]
    .sort(([left], [right]) => right.localeCompare(left))
    .map(([date, items]) => {
      const ordered = [...items].sort((left, right) => compareGalleryItems(left, right, 'DESC', false));
      const cover = ordered[0];

      return {
        asset_count: items.filter((item) => item.kind === 'image' && item.category !== 'general').length,
        board_name: date,
        cover_image_name: cover?.kind === 'image' ? cover.name : null,
        cover_video_name: cover?.kind === 'video' ? cover.name : null,
        date,
        image_count: items.filter((item) => item.kind === 'image' && item.category === 'general').length,
        video_count: items.filter((item) => item.kind === 'video').length,
        virtual_board_id: `by_date:${date}`,
      };
    });
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

const writeBinary = (response, status, body, headers) => {
  response.writeHead(status, {
    'cache-control': 'public, max-age=3600',
    'content-length': body.length,
    ...headers,
  });
  response.end(body);
};

const parseByteRange = (value, size) => {
  const match = /^bytes=(\d*)-(\d*)$/.exec(value.trim());

  if (!match || size <= 0 || (!match[1] && !match[2])) {
    return null;
  }
  if (!match[1]) {
    const suffixLength = Number(match[2]);

    return suffixLength > 0 ? [Math.max(size - suffixLength, 0), size - 1] : null;
  }

  const start = Number(match[1]);
  const end = match[2] ? Number(match[2]) : size - 1;

  return Number.isSafeInteger(start) && Number.isSafeInteger(end) && start <= end && start < size
    ? [start, Math.min(end, size - 1)]
    : null;
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

        const collisions = [...url.searchParams.getAll('collide'), ...(body.collide ?? [])]
          .flatMap((value) => String(value).split(','))
          .map((value) => value.trim())
          .filter(Boolean);

        if (collisions.length > 0) {
          seedCollisionMedia(state, collisions);
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

      if (method === 'GET' && path === '/api/v1/app/generation_device_options') {
        return json(200, [{ device: 'cpu', name: 'CPU' }]);
      }

      if (method === 'GET' && path === '/api/v1/app/runtime_config') {
        return json(200, { config: { generation_devices: 'auto' }, set_fields: [] });
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
          const name = requested.name ?? `Project Name #${projectNumber}`;

          // Creating a project either adopts the board it was given or makes one. Adopting is what
          // lets an import upload its media before the project exists, so that creating the project
          // is the import's single commit point.
          let boardId = requested.board_id ?? null;
          if (boardId === null) {
            boardId = `mock-project-board-${projectNumber}`;
            state.boards.set(boardId, {
              archived: false,
              board_id: boardId,
              board_name: name,
              board_visibility: 'private',
              created_at: now,
              deleted_at: null,
              owner_username: null,
              updated_at: now,
              user_id: MOCK_USER_ID,
            });
          } else {
            const board = state.boards.get(boardId);
            if (!board) {
              return json(404, { detail: 'Board not found' });
            }
            if (board.board_visibility !== 'private' || projectIdForBoard(state, boardId) !== null) {
              return json(409, { detail: 'Board is not available to be claimed by a project' });
            }
            board.board_name = name;
            board.updated_at = now;
          }

          const project = {
            board_id: boardId,
            created_at: now,
            data: requested.data ?? {},
            name,
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

          // The board's name tracks the project's; the two commit together.
          const board = state.boards.get(project.board_id);
          if (board) {
            board.board_name = project.name;
            board.updated_at = project.updated_at;
          }

          return json(200, project);
        }
        if (method === 'DELETE') {
          if (project) {
            state.projects.delete(projectId);
            // The board goes with the project; its media survives as uncategorized.
            state.boards.delete(project.board_id);
            for (const image of state.images.values()) {
              if (image.board_id === project.board_id) {
                image.board_id = null;
              }
            }
            for (const video of state.videos.values()) {
              if (video.board_id === project.board_id) {
                video.board_id = null;
              }
            }
          }
          return json(200, { ok: true });
        }
      }

      const boardSnapshotMatch = /^\/api\/v1\/projects\/([^/]+)\/board-snapshot$/.exec(path);
      if (method === 'GET' && boardSnapshotMatch) {
        const project = state.projects.get(decodeURIComponent(boardSnapshotMatch[1]));

        return project
          ? json(200, { items: boardSnapshotItems(state, project.board_id) })
          : json(404, { detail: 'Project not found' });
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
        return json(200, STARTER_MODELS_RESPONSE);
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

      if (method === 'GET' && (path === '/api/v1/wildcards' || path === '/api/v1/wildcards/')) {
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

      if (path === '/api/v1/boards' || path === '/api/v1/boards/') {
        if (method === 'GET') {
          return json(
            200,
            [...state.boards.values()].map((board) => boardDto(state, board))
          );
        }
        if (method === 'POST') {
          const now = timestamp(state);
          const boardId = `mock-board-${state.nextBoardNumber}`;
          const board = {
            archived: false,
            board_id: boardId,
            board_name: url.searchParams.get('board_name') ?? `Board ${state.nextBoardNumber}`,
            board_visibility: 'private',
            created_at: now,
            deleted_at: null,
            owner_username: null,
            updated_at: now,
            user_id: MOCK_USER_ID,
          };

          state.nextBoardNumber += 1;
          state.boards.set(boardId, board);

          return json(201, boardDto(state, board));
        }
      }

      const boardMatch = /^\/api\/v1\/boards\/([^/]+)$/.exec(path);
      if (boardMatch) {
        const boardId = decodeURIComponent(boardMatch[1]);
        const board = state.boards.get(boardId);

        if (!board) {
          return json(404, { detail: 'Board not found' });
        }
        if (method === 'GET') {
          return json(200, boardDto(state, board));
        }
        if (method === 'PATCH') {
          const changes = await readJsonBody(request);

          // A project's board takes its name, archived state and visibility from the project.
          if (
            projectIdForBoard(state, boardId) !== null &&
            (changes.board_name !== undefined ||
              changes.archived !== undefined ||
              changes.board_visibility !== undefined)
          ) {
            return json(409, { detail: 'This board belongs to a project' });
          }

          for (const key of ['archived', 'board_name', 'board_visibility', 'cover_image_name']) {
            if (changes[key] !== undefined) {
              board[key] = changes[key];
            }
          }
          board.updated_at = timestamp(state);

          return json(201, boardDto(state, board));
        }
        if (method === 'DELETE') {
          if (projectIdForBoard(state, boardId) !== null) {
            // Refused before any media is touched — the whole point of the ordering.
            return json(409, { detail: 'This board belongs to a project' });
          }

          const includeImages = url.searchParams.get('include_images') === 'true';
          const boardImages = [...state.images.values()].filter((image) => image.board_id === boardId);
          const boardVideos = [...state.videos.values()].filter(
            (video) => video.board_id === boardId && video.owner_user_id === MOCK_USER_ID
          );
          const imageNames = boardImages.map((image) => image.image_name);
          const videoNames = boardVideos.map((video) => video.video_name);

          state.boards.delete(boardId);
          if (includeImages) {
            for (const imageName of imageNames) {
              state.images.delete(imageName);
            }
            for (const videoName of videoNames) {
              state.videos.delete(videoName);
            }

            return json(200, {
              board_id: boardId,
              deleted_board_images: [],
              deleted_board_videos: [],
              deleted_images: imageNames,
              deleted_videos: videoNames,
              failed_images: [],
              failed_videos: [],
            });
          }

          for (const image of boardImages) {
            image.board_id = null;
          }
          for (const video of boardVideos) {
            video.board_id = null;
          }

          return json(200, {
            board_id: boardId,
            deleted_board_images: imageNames,
            deleted_board_videos: videoNames,
            deleted_images: [],
            deleted_videos: [],
            failed_images: [],
            failed_videos: [],
          });
        }
      }

      if (method === 'GET' && path === '/api/v1/virtual_boards/by_date') {
        return json(200, listVirtualDateBoards(state));
      }

      const virtualBoardItemNamesMatch = /^\/api\/v1\/virtual_boards\/by_date\/([^/]+)\/item_names$/.exec(path);
      if (method === 'GET' && virtualBoardItemNamesMatch) {
        const date = decodeURIComponent(virtualBoardItemNamesMatch[1]);

        return json(200, listGalleryItemNames(state, url, { createdDate: date }));
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

      if (method === 'POST' && path === '/api/v1/images/upload') {
        const multipartBody = await readBody(request);
        const requestedName = /filename="([^"]+)"/.exec(multipartBody)?.[1] ?? 'uploaded-fixture';
        const suffix = String(state.nextImageNumber).padStart(3, '0');
        const imageName = `fixture-upload-${suffix}-${requestedName.replaceAll(/[^a-zA-Z0-9._-]/g, '-')}.png`;
        const boardId = url.searchParams.get('board_id');
        const now = timestamp(state);
        const image = {
          board_id: boardId && state.boards.has(boardId) ? boardId : null,
          created_at: now,
          deleted_at: null,
          has_workflow: false,
          height: 1,
          image_category: url.searchParams.get('image_category') ?? 'general',
          image_name: imageName,
          image_origin: 'external',
          image_subfolder: '',
          image_url: `/api/v1/images/i/${imageName}/full`,
          is_intermediate: url.searchParams.get('is_intermediate') === 'true',
          node_id: null,
          session_id: null,
          starred: false,
          thumbnail_url: `/api/v1/images/i/${imageName}/thumbnail`,
          updated_at: now,
          width: 1,
        };

        state.nextImageNumber += 1;
        state.images.set(imageName, image);
        response.setHeader('location', image.image_url);
        return json(201, image);
      }

      if (method === 'POST' && (path === '/api/v1/images/star' || path === '/api/v1/images/unstar')) {
        const body = await readJsonBody(request);
        const names = Array.isArray(body.image_names) ? [...new Set(body.image_names)] : [];
        const succeeded = [];
        const affectedBoards = [];
        const starred = path.endsWith('/star');

        for (const name of names) {
          const image = state.images.get(name);

          if (!image) {
            continue;
          }
          image.starred = starred;
          succeeded.push(name);
          affectedBoards.push(image.board_id ?? 'none');
        }

        return json(200, {
          affected_boards: [...new Set(affectedBoards)],
          ...(starred ? { starred_images: succeeded } : { unstarred_images: succeeded }),
        });
      }

      if (method === 'POST' && path === '/api/v1/images/copy') {
        const body = await readJsonBody(request);
        const names = Array.isArray(body.image_names) ? body.image_names : [];
        const boardId = body.board_id ?? null;

        if (boardId !== null && !state.boards.has(boardId)) {
          return json(404, { detail: 'Board not found' });
        }

        const copied = [];
        const failed = [];

        for (const name of names) {
          const source = state.images.get(name);

          if (!source) {
            failed.push(name);
            continue;
          }

          const suffix = String(state.nextImageNumber).padStart(3, '0');
          const imageName = `fixture-copy-${suffix}.png`;
          const now = timestamp(state);

          state.nextImageNumber += 1;
          // A genuinely new identity: `board_images` keys on the name, so a copy must own its own.
          // Category and provenance travel; starring does not (callers use /images/star).
          state.images.set(imageName, {
            ...clone(source),
            board_id: boardId,
            created_at: now,
            image_name: imageName,
            image_url: `/api/v1/images/i/${imageName}/full`,
            is_intermediate: false,
            starred: false,
            thumbnail_url: `/api/v1/images/i/${imageName}/thumbnail`,
            updated_at: now,
          });
          copied.push({ image_name: imageName, source_image_name: name });
        }

        return json(200, { copied, failed });
      }

      if (method === 'POST' && path === '/api/v1/videos/copy') {
        const body = await readJsonBody(request);
        const names = Array.isArray(body.video_names) ? body.video_names : [];
        const boardId = body.board_id ?? null;

        if (boardId !== null && !state.boards.has(boardId)) {
          return json(404, { detail: 'Board not found' });
        }

        const copied = [];
        const failed = [];

        for (const name of names) {
          const source = state.videos.get(name);

          if (!source || source.owner_user_id !== MOCK_USER_ID) {
            failed.push(name);
            continue;
          }

          const suffix = String(state.nextVideoNumber).padStart(3, '0');
          const videoName = `fixture-copy-${suffix}.mp4`;
          const now = timestamp(state);

          state.nextVideoNumber += 1;
          state.videos.set(videoName, {
            ...clone(source),
            board_id: boardId,
            created_at: now,
            is_intermediate: false,
            starred: false,
            updated_at: now,
            video_name: videoName,
          });
          copied.push({ source_video_name: name, video_name: videoName });
        }

        return json(200, { copied, failed });
      }

      if (method === 'GET' && (path === '/api/v1/images' || path === '/api/v1/images/')) {
        return json(200, listImages(state, url));
      }

      if (method === 'GET' && path === '/api/v1/gallery/items/') {
        return json(200, listGalleryItems(state, url));
      }

      if (method === 'GET' && path === '/api/v1/gallery/items/names') {
        return json(200, listGalleryItemNames(state, url));
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

      if (method === 'POST' && path === '/api/v1/videos/upload') {
        const multipartBody = await readBody(request);
        const requestedName = /filename="([^"]+)"/.exec(multipartBody)?.[1] ?? 'uploaded-fixture.mp4';
        const suffix = String(state.nextVideoNumber).padStart(3, '0');
        const videoName = `fixture-upload-${suffix}-${requestedName.replaceAll(/[^a-zA-Z0-9._-]/g, '-')}`;
        const boardId = url.searchParams.get('board_id');
        const video = {
          board_id: boardId && state.boards.has(boardId) ? boardId : null,
          created_at: timestamp(state),
          duration: 1,
          fps: 10,
          graph: null,
          height: 64,
          is_intermediate: url.searchParams.get('is_intermediate') === 'true',
          metadata: null,
          owner_user_id: MOCK_USER_ID,
          starred: false,
          thumbnail_url: `/api/v1/videos/i/${videoName}/thumbnail`,
          video_category: url.searchParams.get('video_category') ?? 'general',
          video_name: videoName,
          video_origin: 'external',
          video_url: `/api/v1/videos/i/${videoName}/full`,
          width: 64,
          workflow: null,
        };

        state.nextVideoNumber += 1;
        state.videos.set(videoName, video);
        response.setHeader('location', video.video_url);
        return json(201, toVideoDto(video));
      }

      if (method === 'GET' && (path === '/api/v1/videos' || path === '/api/v1/videos/')) {
        return json(200, listVideos(state, url));
      }

      if (method === 'POST' && (path === '/api/v1/videos/star' || path === '/api/v1/videos/unstar')) {
        const body = await readJsonBody(request);
        const names = Array.isArray(body.video_names) ? [...new Set(body.video_names)] : [];
        const succeeded = [];
        const affectedBoards = [];
        const starred = path.endsWith('/star');

        for (const name of names) {
          const video = state.videos.get(name);

          if (!video || video.owner_user_id !== MOCK_USER_ID) {
            continue;
          }
          video.starred = starred;
          succeeded.push(name);
          affectedBoards.push(video.board_id ?? 'none');
        }

        return json(200, {
          affected_boards: [...new Set(affectedBoards)],
          failed_videos: [],
          [starred ? 'starred_videos' : 'unstarred_videos']: succeeded,
        });
      }

      if (method === 'POST' && path === '/api/v1/videos/delete') {
        const body = await readJsonBody(request);
        const names = Array.isArray(body.video_names) ? [...new Set(body.video_names)] : [];
        const deletedVideos = [];
        const affectedBoards = [];

        for (const name of names) {
          const video = state.videos.get(name);

          if (!video || video.owner_user_id !== MOCK_USER_ID) {
            continue;
          }
          state.videos.delete(name);
          deletedVideos.push(name);
          affectedBoards.push(video.board_id ?? 'none');
        }

        return json(200, {
          affected_boards: [...new Set(affectedBoards)],
          deleted_videos: deletedVideos,
          failed_videos: [],
        });
      }

      if (path === '/api/v1/videos/board' && (method === 'POST' || method === 'DELETE')) {
        const body = await readJsonBody(request);
        const video = state.videos.get(body.video_name);

        if (!video || video.owner_user_id !== MOCK_USER_ID) {
          return json(404, { detail: 'Video not found' });
        }

        const oldBoardId = video.board_id ?? 'none';

        if (method === 'POST') {
          if (!body.board_id || !state.boards.has(body.board_id)) {
            return json(404, { detail: 'Board not found' });
          }
          video.board_id = body.board_id;
          return json(200, {
            added_videos: [video.video_name],
            affected_boards: [...new Set([oldBoardId, body.board_id])],
          });
        }

        video.board_id = null;
        return json(200, {
          affected_boards: [...new Set([oldBoardId, 'none'])],
          removed_videos: [video.video_name],
        });
      }

      const videoMetadataMatch = /^\/api\/v1\/videos\/i\/([^/]+)\/metadata$/.exec(path);
      if (method === 'GET' && videoMetadataMatch) {
        const video = state.videos.get(decodeURIComponent(videoMetadataMatch[1]));

        return video && video.owner_user_id === MOCK_USER_ID
          ? json(200, video.metadata)
          : json(404, { detail: 'Video not found' });
      }

      const videoWorkflowMatch = /^\/api\/v1\/videos\/i\/([^/]+)\/workflow$/.exec(path);
      if (method === 'GET' && videoWorkflowMatch) {
        const video = state.videos.get(decodeURIComponent(videoWorkflowMatch[1]));

        return video && video.owner_user_id === MOCK_USER_ID
          ? json(200, { graph: video.graph, workflow: video.workflow })
          : json(404, { detail: 'Video not found' });
      }

      const videoThumbnailMatch = /^\/api\/v1\/videos\/i\/([^/]+)\/thumbnail$/.exec(path);
      if (method === 'GET' && videoThumbnailMatch) {
        const video = state.videos.get(decodeURIComponent(videoThumbnailMatch[1]));

        return video && video.owner_user_id === MOCK_USER_ID
          ? writeBinary(response, 200, FIXTURE_VIDEO_POSTER, { 'content-type': 'image/webp' })
          : json(404, { detail: 'Video not found' });
      }

      const videoFullMatch = /^\/api\/v1\/videos\/i\/([^/]+)\/full$/.exec(path);
      if ((method === 'GET' || method === 'HEAD') && videoFullMatch) {
        const video = state.videos.get(decodeURIComponent(videoFullMatch[1]));

        if (!video || video.owner_user_id !== MOCK_USER_ID) {
          return json(404, { detail: 'Video not found' });
        }

        const commonHeaders = {
          'accept-ranges': 'bytes',
          'content-type': 'video/mp4',
        };

        if (method === 'HEAD') {
          response.writeHead(200, { ...commonHeaders, 'content-length': FIXTURE_VIDEO.length });
          return response.end();
        }

        const rangeHeader = request.headers.range;

        if (!rangeHeader) {
          return writeBinary(response, 200, FIXTURE_VIDEO, commonHeaders);
        }

        const range = parseByteRange(rangeHeader, FIXTURE_VIDEO.length);

        if (!range) {
          response.writeHead(416, {
            ...commonHeaders,
            'content-length': 0,
            'content-range': `bytes */${String(FIXTURE_VIDEO.length)}`,
          });
          return response.end();
        }

        const [start, end] = range;
        const body = FIXTURE_VIDEO.subarray(start, end + 1);

        return writeBinary(response, 206, body, {
          ...commonHeaders,
          'content-range': `bytes ${String(start)}-${String(end)}/${String(FIXTURE_VIDEO.length)}`,
        });
      }

      const videoMatch = /^\/api\/v1\/videos\/i\/([^/]+)$/.exec(path);
      if (method === 'GET' && videoMatch) {
        const video = state.videos.get(decodeURIComponent(videoMatch[1]));

        return video && video.owner_user_id === MOCK_USER_ID
          ? json(200, toVideoDto(video))
          : json(404, { detail: 'Video not found' });
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
