/**
 * Deterministic synthetic data for browser release and performance journeys.
 *
 * Keep this module free of browser and server dependencies: harnesses can
 * import the same builders to validate their workload without starting HTTP.
 * Every generated value is derived from an index and a fixed epoch, so resets
 * are byte-identical and checked-in baselines never depend on wall-clock time.
 */

export const MOCK_BACKEND_PROFILE_NAMES = Object.freeze(['empty', 'representative']);

export const MOCK_BACKEND_PROFILE_COUNTS = Object.freeze({
  empty: Object.freeze({
    images: 0,
    layers: 0,
    models: 0,
    nodes: 0,
    projects: 0,
    queueItems: 0,
    workflowNodes: 0,
  }),
  representative: Object.freeze({
    images: 1_000,
    layers: 64,
    models: 100,
    nodes: 100,
    projects: 40,
    queueItems: 500,
    workflowNodes: 100,
  }),
});

export const MOCK_BACKEND_FIXED_EPOCH = '2026-01-15T12:00:00.000Z';
export const MOCK_BACKEND_REPRESENTATIVE_VIDEO_NAME = 'fixture-video-001.mp4';

const FIXED_EPOCH_MS = Date.parse(MOCK_BACKEND_FIXED_EPOCH);
const range = (length, build) => Array.from({ length }, (_, index) => build(index));
const ordinal = (index, width = 3) => String(index + 1).padStart(width, '0');

/** The single user every fixture belongs to; matches `MOCK_USER_ID` in the mock backend. */
const FIXTURE_USER_ID = 'fixture-user';
const timestampAt = (index) => new Date(FIXED_EPOCH_MS - index * 60_000).toISOString();

export const isMockBackendProfileName = (value) => MOCK_BACKEND_PROFILE_NAMES.includes(value);

export const assertMockBackendProfileName = (value) => {
  if (!isMockBackendProfileName(value)) {
    throw new TypeError(
      `Unknown mock-backend profile ${JSON.stringify(value)}. Expected one of ${MOCK_BACKEND_PROFILE_NAMES.join(', ')}.`
    );
  }

  return value;
};

/**
 * What Fixture Project 002's own board holds, and what sits just outside it.
 *
 * The project-file journey is the only place the whole board path runs end to end, and it can only
 * prove the interesting rules if the fixture actually contains them: a result the canvas draws with
 * *and* the board owns (so a restore must copy it and rewrite the layer), a result the document
 * never mentions (which is the entire reason `.invk` carries a board at all), assets under each
 * visible category, media that must be excluded, and references that live outside the board and so
 * must be deduplicated rather than copied.
 *
 * Every name here is an existing fixture image reassigned to the project's board — the image count
 * is a pinned dimension of the representative profile, so this composition must not change it.
 */
export const PROJECT_FILE_BOARD = Object.freeze({
  /** Generated, on the board, and drawn by the canvas: the overlap case. */
  referencedImage: 'fixture-image-0002.png',
  /** Generated, on the board, referenced by nothing. Travels only because v3 enumerates the board. */
  unreferencedImage: 'fixture-image-0005.png',
  /** An upload, filed under the `user` asset category. */
  userAsset: 'fixture-image-0006.png',
  /** A control-layer asset, so the restore has more than one category to preserve. */
  maskAsset: 'fixture-image-0007.png',
  /** Already starred in the base composition, so starring survives without perturbing any ordering. */
  starredImage: 'fixture-image-0012.png',
  /** The canvas's private category. On the board, and it must never travel as board membership. */
  canvasOwnedImage: 'fixture-image-0008.png',
  /** Hidden from every gallery view, and from the snapshot with it. */
  intermediateImage: 'fixture-image-0009.png',
  /** A visible video on the board, which is a separate namespace and a separate copy path. */
  video: 'fixture-video-project.mp4',
  /** Drawn by the canvas but owned by no project: reused on import, never copied. */
  externalImages: Object.freeze(['fixture-image-0001.png', 'fixture-image-0003.png', 'fixture-image-0004.png']),
});

/** Category and visibility overrides that put the board composition above onto the project's board. */
const PROJECT_BOARD_IMAGES = new Map([
  [PROJECT_FILE_BOARD.referencedImage, { image_category: 'general' }],
  [PROJECT_FILE_BOARD.unreferencedImage, { image_category: 'general' }],
  [PROJECT_FILE_BOARD.starredImage, { image_category: 'general', starred: true }],
  [PROJECT_FILE_BOARD.userAsset, { image_category: 'user' }],
  [PROJECT_FILE_BOARD.maskAsset, { image_category: 'mask' }],
  [PROJECT_FILE_BOARD.canvasOwnedImage, { image_category: 'other' }],
  [PROJECT_FILE_BOARD.intermediateImage, { image_category: 'general', is_intermediate: true }],
]);

const createImages = (count) =>
  range(count, (index) => {
    const id = ordinal(index, 4);
    const imageName = `fixture-image-${id}.png`;
    const boardIndex = index % 10;
    const imageCategory = index % 10 === 0 ? 'control' : 'general';
    const projectBoardMembership = PROJECT_BOARD_IMAGES.get(imageName);

    return {
      board_id: index % 8 === 0 ? null : `fixture-board-${ordinal(boardIndex, 2)}`,
      created_at: timestampAt(index),
      height: 512 + (index % 3) * 64,
      image_category: imageCategory,
      image_name: imageName,
      image_url: `/api/v1/images/i/${imageName}/full`,
      is_intermediate: false,
      starred: index % 11 === 0,
      thumbnail_url: `/api/v1/images/i/${imageName}/thumbnail`,
      width: 512 + (index % 4) * 64,
      ...(projectBoardMembership === undefined
        ? {}
        : { board_id: PROJECT_FILE_BOARD_ID, starred: false, ...projectBoardMembership }),
    };
  });

const createVideos = () => [
  {
    board_id: null,
    created_at: timestampAt(2),
    duration: 1,
    fps: 10,
    graph: '{"nodes":[{"id":"fixture-video-output","type":"wan_l2v"}]}',
    height: 64,
    is_intermediate: false,
    metadata: { codec: 'h264', prompt: 'wan fixture' },
    owner_user_id: 'fixture-user',
    starred: true,
    thumbnail_url: `/api/v1/videos/i/${MOCK_BACKEND_REPRESENTATIVE_VIDEO_NAME}/thumbnail`,
    video_category: 'general',
    video_name: MOCK_BACKEND_REPRESENTATIVE_VIDEO_NAME,
    video_origin: 'external',
    video_url: `/api/v1/videos/i/${MOCK_BACKEND_REPRESENTATIVE_VIDEO_NAME}/full`,
    width: 64,
    workflow: '{"name":"Fixture video workflow","nodes":[]}',
  },
  {
    board_id: 'fixture-board-02',
    created_at: timestampAt(0),
    duration: 1,
    fps: 10,
    graph: null,
    height: 64,
    is_intermediate: false,
    metadata: { prompt: 'board cover fixture' },
    owner_user_id: 'fixture-user',
    starred: true,
    thumbnail_url: '/api/v1/videos/i/fixture-video-board.mp4/thumbnail',
    video_category: 'control',
    video_name: 'fixture-video-board.mp4',
    video_origin: 'internal',
    video_url: '/api/v1/videos/i/fixture-video-board.mp4/full',
    width: 64,
    workflow: null,
  },
  {
    board_id: null,
    created_at: timestampAt(11),
    duration: 1,
    fps: 10,
    graph: null,
    height: 64,
    is_intermediate: false,
    metadata: { prompt: 'same-name collision fixture' },
    owner_user_id: 'fixture-user',
    starred: true,
    thumbnail_url: '/api/v1/videos/i/fixture-image-0012.png/thumbnail',
    video_category: 'general',
    video_name: 'fixture-image-0012.png',
    video_origin: 'internal',
    video_url: '/api/v1/videos/i/fixture-image-0012.png/full',
    width: 64,
    workflow: null,
  },
  {
    board_id: 'fixture-board-03',
    created_at: timestampAt(2),
    duration: 1,
    fps: 10,
    graph: null,
    height: 64,
    is_intermediate: false,
    metadata: { prompt: 'asset category fixture' },
    owner_user_id: 'fixture-user',
    starred: false,
    thumbnail_url: '/api/v1/videos/i/fixture-video-asset.mp4/thumbnail',
    video_category: 'control',
    video_name: 'fixture-video-asset.mp4',
    video_origin: 'internal',
    video_url: '/api/v1/videos/i/fixture-video-asset.mp4/full',
    width: 64,
    workflow: null,
  },
  {
    board_id: null,
    created_at: timestampAt(3),
    duration: 1,
    fps: 10,
    graph: null,
    height: 64,
    is_intermediate: true,
    metadata: { prompt: 'intermediate fixture' },
    owner_user_id: 'fixture-user',
    starred: false,
    thumbnail_url: '/api/v1/videos/i/fixture-video-intermediate.mp4/thumbnail',
    video_category: 'general',
    video_name: 'fixture-video-intermediate.mp4',
    video_origin: 'internal',
    video_url: '/api/v1/videos/i/fixture-video-intermediate.mp4/full',
    width: 64,
    workflow: null,
  },
  {
    // On Fixture Project 002's own board: videos are a separate namespace with their own copy and
    // upload routes, so a project file that only ever carried images would prove half the path.
    board_id: 'fixture-project-board-02',
    created_at: timestampAt(5),
    duration: 1,
    fps: 10,
    graph: null,
    height: 64,
    is_intermediate: false,
    metadata: { prompt: 'project board fixture' },
    owner_user_id: 'fixture-user',
    starred: false,
    thumbnail_url: '/api/v1/videos/i/fixture-video-project.mp4/thumbnail',
    video_category: 'general',
    video_name: 'fixture-video-project.mp4',
    video_origin: 'internal',
    video_url: '/api/v1/videos/i/fixture-video-project.mp4/full',
    width: 64,
    workflow: null,
  },
  {
    board_id: null,
    created_at: timestampAt(4),
    duration: 1,
    fps: 10,
    graph: null,
    height: 64,
    is_intermediate: false,
    metadata: { prompt: 'foreign fixture' },
    owner_user_id: 'other-user',
    starred: false,
    thumbnail_url: '/api/v1/videos/i/fixture-video-foreign.mp4/thumbnail',
    video_category: 'general',
    video_name: 'fixture-video-foreign.mp4',
    video_origin: 'external',
    video_url: '/api/v1/videos/i/fixture-video-foreign.mp4/full',
    width: 64,
    workflow: null,
  },
  {
    board_id: null,
    created_at: timestampAt(800),
    duration: 1,
    fps: 10,
    graph: null,
    height: 64,
    is_intermediate: false,
    metadata: { prompt: 'previous-day fixture' },
    owner_user_id: 'fixture-user',
    starred: false,
    thumbnail_url: '/api/v1/videos/i/fixture-video-old.mp4/thumbnail',
    video_category: 'general',
    video_name: 'fixture-video-old.mp4',
    video_origin: 'internal',
    video_url: '/api/v1/videos/i/fixture-video-old.mp4/full',
    width: 64,
    workflow: null,
  },
];

/**
 * The board a project owns. Every project has exactly one, and only project APIs may rename or
 * delete it — the generic board routes refuse a claimed board.
 */
export const projectBoardId = (index) => `fixture-project-board-${ordinal(index, 2)}`;

/**
 * The board owned by Fixture Project 002 — the project the project-file journey exports, imports
 * and duplicates. Declared here rather than inlined so the composition above and the journey's
 * assertions cannot drift from the project they describe.
 */
export const PROJECT_FILE_BOARD_ID = projectBoardId(1);

const buildBoard = (boardId, boardName, images, videos, createdAt) => {
  const boardImages = images.filter((image) => image.board_id === boardId);
  const boardVideos = videos.filter((video) => video.board_id === boardId);
  const cover = [
    ...boardImages.map((image) => ({
      createdAt: image.created_at,
      kind: 'image',
      name: image.image_name,
      starred: image.starred,
    })),
    ...boardVideos.map((video) => ({
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

  return {
    archived: false,
    asset_count: boardImages.filter((image) => image.image_category !== 'general').length,
    board_id: boardId,
    board_name: boardName,
    board_visibility: 'private',
    cover_image_name: cover?.kind === 'image' ? cover.name : null,
    cover_video_name: cover?.kind === 'video' ? cover.name : null,
    created_at: createdAt,
    image_count: boardImages.filter((image) => image.image_category === 'general').length,
    owner_username: null,
    user_id: FIXTURE_USER_ID,
    video_count: boardVideos.length,
  };
};

const createBoards = (images, videos, projectCount) => [
  ...range(10, (index) =>
    buildBoard(
      `fixture-board-${ordinal(index, 2)}`,
      `Fixture Board ${ordinal(index, 2)}`,
      images,
      videos,
      timestampAt(index * 10)
    )
  ),
  ...range(projectCount, (index) =>
    buildBoard(projectBoardId(index), `Fixture Project ${ordinal(index, 3)}`, images, videos, timestampAt(index))
  ),
];

const QUEUE_STATUSES = ['pending', 'waiting', 'in_progress', 'completed', 'failed', 'canceled'];

const createQueueItems = (count) =>
  range(count, (index) => {
    const itemId = index + 1;
    const status = QUEUE_STATUSES[index % QUEUE_STATUSES.length];
    const createdAt = timestampAt(index);
    const isSettled = status === 'completed' || status === 'failed' || status === 'canceled';

    return {
      batch_id: `fixture-batch-${ordinal(Math.floor(index / 4), 3)}`,
      completed_at: isSettled ? createdAt : null,
      created_at: createdAt,
      destination: index % 2 === 0 ? 'gallery' : 'canvas',
      error_message: status === 'failed' ? 'Synthetic fixture failure' : null,
      error_traceback: null,
      error_type: status === 'failed' ? 'FixtureError' : null,
      field_values: [
        {
          field_name: 'prompt',
          node_path: 'fixture_prompt',
          value: `Representative prompt ${itemId}`,
        },
      ],
      item_id: itemId,
      origin: `webv2:fixture-project-${ordinal(index % 40, 3)}:generate`,
      retried_from_item_id: null,
      session: { prepared_source_mapping: {}, results: {} },
      session_id: `fixture-session-${ordinal(index, 4)}`,
      started_at: status === 'pending' || status === 'waiting' ? null : createdAt,
      status,
      updated_at: createdAt,
      user_display_name: 'Fixture User',
      user_email: 'fixture@example.test',
      user_id: 'fixture-user',
    };
  });

const MODEL_BASES = ['sd-1', 'sdxl', 'flux', 'flux2', 'qwen-image'];
const MODEL_TYPES = ['main', 'lora', 'controlnet', 'vae', 'embedding'];

const createModels = (count) =>
  range(count, (index) => {
    const id = ordinal(index);
    const type = MODEL_TYPES[index % MODEL_TYPES.length];

    return {
      base: MODEL_BASES[index % MODEL_BASES.length],
      cover_image: null,
      default_settings: type === 'main' ? { cfg_scale: 7, height: 1024, steps: 30, width: 1024 } : null,
      description: `Deterministic representative ${type} model ${id}.`,
      file_size: 512_000_000 + index * 1_024,
      format: type === 'embedding' ? 'embedding_file' : 'checkpoint',
      hash: `fixture-hash-${id}`,
      key: `fixture-model-${id}`,
      name: `Fixture Model ${id}`,
      path: `fixtures/models/fixture-model-${id}.safetensors`,
      source: `fixtures/models/fixture-model-${id}.safetensors`,
      source_type: 'path',
      source_url: null,
      trigger_phrases: type === 'lora' ? [`fixture-style-${id}`] : null,
      type,
    };
  });

const createNodePacks = (nodeCount) =>
  range(Math.ceil(nodeCount / 5), (packIndex) => {
    const firstNodeIndex = packIndex * 5;
    const nodeTypes = range(Math.min(5, nodeCount - firstNodeIndex), (nodeOffset) => {
      const nodeIndex = firstNodeIndex + nodeOffset;

      return `mock_node_${ordinal(nodeIndex)}`;
    });

    return {
      name: `fixture-pack-${ordinal(packIndex, 2)}`,
      node_count: nodeTypes.length,
      node_types: nodeTypes,
      path: `/opt/invokeai/nodes/fixture-pack-${ordinal(packIndex, 2)}`,
    };
  });

const createOpenApiDocument = (nodeCount) => {
  const outputSchema = {
    class: 'output',
    properties: {
      type: { const: 'fixture_output', default: 'fixture_output' },
      value: { field_kind: 'output', title: 'Value', type: 'string' },
    },
    type: 'object',
  };
  const invocationSchemas = Object.fromEntries(
    range(nodeCount, (index) => {
      const id = ordinal(index);
      const type = `mock_node_${id}`;

      return [
        `FixtureNode${id}Invocation`,
        {
          category: `fixture-${(index % 5) + 1}`,
          class: 'invocation',
          classification: 'stable',
          description: `Synthetic invocation template ${id}.`,
          node_pack: `fixture-pack-${ordinal(Math.floor(index / 5), 2)}`,
          output: { $ref: '#/components/schemas/FixtureOutput' },
          properties: {
            prompt: {
              default: '',
              field_kind: 'input',
              input: 'direct',
              orig_required: false,
              title: 'Prompt',
              type: 'string',
              ui_hidden: false,
            },
            type: { const: type, default: type, title: 'type' },
            use_cache: { default: true, field_kind: 'internal', type: 'boolean' },
          },
          tags: ['fixture'],
          title: `Fixture Node ${id}`,
          type: 'object',
          version: '1.0.0',
        },
      ];
    })
  );

  return {
    components: { schemas: { FixtureOutput: outputSchema, ...invocationSchemas } },
    info: { title: 'InvokeAI deterministic mock', version: '1.0.0' },
    openapi: '3.1.0',
    paths: {},
  };
};

const createWorkflowNodes = (count) =>
  range(count, (index) => {
    const id = ordinal(index);

    return {
      data: {
        inputs: { prompt: { label: '', name: 'prompt', value: `Fixture node ${id}` } },
        isIntermediate: true,
        isOpen: true,
        label: `Fixture Node ${id}`,
        nodePack: `fixture-pack-${ordinal(Math.floor(index / 5), 2)}`,
        notes: '',
        type: `mock_node_${id}`,
        useCache: true,
        version: '1.0.0',
      },
      id: `fixture-workflow-node-${id}`,
      position: { x: (index % 10) * 320, y: Math.floor(index / 10) * 240 },
      type: 'invocation',
    };
  });

const createCanvasLayers = (count) =>
  range(count, (index) => {
    const id = ordinal(index);
    const imageName = `fixture-image-${ordinal(index, 4)}.png`;

    return {
      blendMode: 'normal',
      id: `fixture-layer-${id}`,
      isEnabled: true,
      isLocked: false,
      name: `Fixture Layer ${id}`,
      opacity: 1,
      source: {
        image: { height: 512, imageName, width: 512 },
        type: 'image',
      },
      transform: {
        rotation: 0,
        scaleX: 1,
        scaleY: 1,
        x: (index % 8) * 32,
        y: Math.floor(index / 8) * 32,
      },
      type: 'raster',
    };
  });

const createProjectFileWorkflowNodes = () => {
  const [node] = createWorkflowNodes(1);

  return [
    {
      ...node,
      data: {
        ...node.data,
        inputs: {
          ...node.data.inputs,
          video: {
            label: '',
            name: 'video',
            value: { video_name: MOCK_BACKEND_REPRESENTATIVE_VIDEO_NAME },
          },
        },
      },
    },
  ];
};

const createProjectDocument = ({ index, layers = [], workflowNodes = [] }) => {
  const id = `fixture-project-${ordinal(index, 3)}`;
  const graphId = `${id}-graph`;
  const formRootId = `${graphId}-form-root`;

  return {
    canvas: {
      document: {
        background: 'transparent',
        bbox: { height: 1024, width: 1024, x: 0, y: 0 },
        height: 1024,
        layers,
        selectedLayerId: layers[0]?.id ?? null,
        version: 2,
        width: 1024,
      },
      documentRevision: 0,
      snapshots: [],
      stagingArea: {
        areThumbnailsVisible: true,
        autoSwitchMode: 'off',
        isVisible: false,
        pendingImageIds: [],
        pendingImages: [],
        selectedImageIndex: 0,
      },
      version: 2,
    },
    events: [],
    graphHistory: [],
    id,
    invocation: {
      destination: 'gallery',
      destinationLocked: false,
      sourceId: workflowNodes.length > 0 ? 'workflow' : 'generate',
      sourceLocked: false,
    },
    layout: {
      centerViewId: 'preview',
      panels: { isBottomOpen: false, isLeftOpen: true, isRightOpen: true },
      presetId: 'canvas-default',
    },
    name: `Fixture Project ${ordinal(index, 3)}`,
    promptHistory: [],
    projectGraph: {
      author: 'InvokeAI',
      contact: '',
      description: index === 0 ? 'Representative 100-node workflow.' : '',
      edges: [],
      form: {
        elements: {
          [formRootId]: {
            data: { children: [], layout: 'column' },
            id: formRootId,
            type: 'container',
          },
        },
        rootElementId: formRootId,
      },
      id: graphId,
      name: index === 0 ? 'Representative Workflow' : 'Empty Workflow',
      nodes: workflowNodes,
      notes: '',
      tags: 'fixture',
      updatedAt: timestampAt(index),
      version: 2,
      workflowVersion: '1.0.0',
    },
    queue: { items: [] },
    settings: {},
    widgetGraphs: {},
  };
};

const createProjects = (count, workflowNodeCount, layerCount) =>
  range(count, (index) => {
    const data = createProjectDocument({
      index,
      layers: index === 0 ? createCanvasLayers(layerCount) : index === 1 ? createCanvasLayers(4) : [],
      workflowNodes:
        index === 0 ? createWorkflowNodes(workflowNodeCount) : index === 1 ? createProjectFileWorkflowNodes() : [],
    });
    const timestamp = timestampAt(index);

    return {
      // Every project owns exactly one private board; the server is authoritative for which.
      board_id: projectBoardId(index),
      created_at: timestamp,
      data,
      name: data.name,
      project_id: data.id,
      revision: 1,
      updated_at: timestamp,
    };
  });

const createWorkflows = (count) =>
  range(count, (index) => {
    const id = ordinal(index);

    return {
      category: index % 4 === 0 ? 'default' : 'user',
      created_at: timestampAt(index),
      description: `Synthetic workflow library entry ${id}.`,
      name: `Fixture Workflow ${id}`,
      opened_at: index % 3 === 0 ? timestampAt(index) : null,
      tags: 'fixture,representative',
      thumbnail_url: null,
      updated_at: timestampAt(index),
      workflow: {
        author: 'InvokeAI',
        contact: '',
        description: `Synthetic workflow library entry ${id}.`,
        edges: [],
        exposedFields: [],
        form: { elements: {}, rootElementId: '' },
        meta: { category: 'user', version: '3.0.0' },
        name: `Fixture Workflow ${id}`,
        nodes: [],
        notes: '',
        tags: 'fixture',
        version: '3.0.0',
        workflowVersion: '1.0.0',
      },
      workflow_id: `fixture-workflow-${id}`,
    };
  });

const createEmptyFixture = () => ({
  boards: [],
  images: [],
  models: [],
  nodeCatalog: { custom_nodes_path: '/opt/invokeai/nodes', node_packs: [] },
  openApiDocument: createOpenApiDocument(0),
  profile: 'empty',
  projects: [],
  queueItems: [],
  videos: [],
  workflows: [],
});

const createRepresentativeFixture = () => {
  const counts = MOCK_BACKEND_PROFILE_COUNTS.representative;
  const images = createImages(counts.images);
  const videos = createVideos();

  return {
    boards: createBoards(images, videos, counts.projects),
    images,
    models: createModels(counts.models),
    nodeCatalog: {
      custom_nodes_path: '/opt/invokeai/nodes',
      node_packs: createNodePacks(counts.nodes),
    },
    openApiDocument: createOpenApiDocument(counts.nodes),
    profile: 'representative',
    projects: createProjects(counts.projects, counts.workflowNodes, counts.layers),
    queueItems: createQueueItems(counts.queueItems),
    videos,
    workflows: createWorkflows(100),
  };
};

export const createMockBackendFixture = (profile = 'empty') => {
  assertMockBackendProfileName(profile);

  return profile === 'representative' ? createRepresentativeFixture() : createEmptyFixture();
};

const countInvocationSchemas = (fixture) =>
  Object.values(fixture.openApiDocument?.components?.schemas ?? {}).filter((schema) => schema?.class === 'invocation')
    .length;

export const getMockBackendFixtureCounts = (fixture) => ({
  images: fixture.images.length,
  layers: fixture.projects[0]?.data?.canvas?.document?.layers?.length ?? 0,
  models: fixture.models.length,
  nodes: countInvocationSchemas(fixture),
  projects: fixture.projects.length,
  queueItems: fixture.queueItems.length,
  workflowNodes: fixture.projects[0]?.data?.projectGraph?.nodes?.length ?? 0,
});

const findDuplicates = (values) => {
  const seen = new Set();
  const duplicates = new Set();

  for (const value of values) {
    if (seen.has(value)) {
      duplicates.add(value);
    }
    seen.add(value);
  }

  return [...duplicates];
};

export const validateMockBackendFixture = (fixture) => {
  const failures = [];

  if (!isMockBackendProfileName(fixture?.profile)) {
    failures.push(`fixture.profile must be one of ${MOCK_BACKEND_PROFILE_NAMES.join(', ')}`);
    return failures;
  }

  const expected = MOCK_BACKEND_PROFILE_COUNTS[fixture.profile];
  const received = getMockBackendFixtureCounts(fixture);

  for (const [name, expectedCount] of Object.entries(expected)) {
    if (received[name] !== expectedCount) {
      failures.push(`${fixture.profile}.${name}: expected ${expectedCount}, received ${received[name]}`);
    }
  }

  const identities = [
    ['image names', fixture.images.map((image) => image.image_name)],
    ['model keys', fixture.models.map((model) => model.key)],
    ['project ids', fixture.projects.map((project) => project.project_id)],
    ['queue item ids', fixture.queueItems.map((item) => item.item_id)],
    ['video names', fixture.videos.map((video) => video.video_name)],
  ];

  for (const [label, values] of identities) {
    const duplicates = findDuplicates(values);

    if (duplicates.length > 0) {
      failures.push(`${label} must be unique; duplicated ${duplicates.join(', ')}`);
    }
  }

  const nodeTypes = fixture.nodeCatalog.node_packs.flatMap((pack) => pack.node_types);
  const invocationTypes = Object.values(fixture.openApiDocument.components.schemas)
    .filter((schema) => schema?.class === 'invocation')
    .map((schema) => schema.properties?.type?.const);

  if (JSON.stringify(nodeTypes) !== JSON.stringify(invocationTypes)) {
    failures.push('custom-node catalog types and OpenAPI invocation types must match in deterministic order');
  }

  return failures;
};

export const assertMockBackendFixture = (fixture) => {
  const failures = validateMockBackendFixture(fixture);

  if (failures.length > 0) {
    throw new Error(`Invalid mock-backend fixture:\n- ${failures.join('\n- ')}`);
  }

  return fixture;
};
