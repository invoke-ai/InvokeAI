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

const FIXED_EPOCH_MS = Date.parse(MOCK_BACKEND_FIXED_EPOCH);
const range = (length, build) => Array.from({ length }, (_, index) => build(index));
const ordinal = (index, width = 3) => String(index + 1).padStart(width, '0');
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

const createImages = (count) =>
  range(count, (index) => {
    const id = ordinal(index, 4);
    const imageName = `fixture-image-${id}.png`;
    const boardIndex = index % 10;
    const imageCategory = index % 10 === 0 ? 'control' : 'general';

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
    };
  });

const createBoards = (images) =>
  range(10, (index) => {
    const boardId = `fixture-board-${ordinal(index, 2)}`;
    const boardImages = images.filter((image) => image.board_id === boardId);

    return {
      archived: false,
      asset_count: boardImages.filter((image) => image.image_category !== 'general').length,
      board_id: boardId,
      board_name: `Fixture Board ${ordinal(index, 2)}`,
      cover_image_name: boardImages[0]?.image_name ?? null,
      created_at: timestampAt(index * 10),
      image_count: boardImages.filter((image) => image.image_category === 'general').length,
      owner_username: null,
    };
  });

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
      layers: index === 0 ? createCanvasLayers(layerCount) : [],
      workflowNodes: index === 0 ? createWorkflowNodes(workflowNodeCount) : [],
    });
    const timestamp = timestampAt(index);

    return {
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
  workflows: [],
});

const createRepresentativeFixture = () => {
  const counts = MOCK_BACKEND_PROFILE_COUNTS.representative;
  const images = createImages(counts.images);

  return {
    boards: createBoards(images),
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
