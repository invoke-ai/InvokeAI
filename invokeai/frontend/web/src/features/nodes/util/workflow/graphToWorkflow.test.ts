import { $templates } from 'features/nodes/store/nodesSlice';
import type { Templates } from 'features/nodes/store/types';
import type { InvocationTemplate } from 'features/nodes/types/invocation';
import { isWorkflowInvocationNode } from 'features/nodes/types/workflow';
import type { NonNullableGraph } from 'services/api/types';
import { afterEach, beforeAll, beforeEach, describe, expect, it } from 'vitest';

import { graphToWorkflow } from './graphToWorkflow';
import { parseAndMigrateWorkflow } from './migrations';

// Minimal templates needed to render the user's graph. We use the same shape
// the real parseSchema would produce for these nodes' fields - only the
// `core_metadata.loras` and `lora_selector.{lora,weight}` templates need to
// be accurate enough for the roundtrip; other inputs are mocked permissively.
const looseStringInput = {
  name: '',
  title: '',
  description: '',
  ui_hidden: false,
  fieldKind: 'input' as const,
  input: 'any' as const,
  required: false,
  default: undefined,
};
const passthroughTemplate = (typeName: string) => ({
  type: 'invocation',
  title: typeName,
  version: '1.0.0',
  tags: [],
  description: '',
  outputType: `${typeName}_output`,
  inputs: new Proxy(
    {},
    {
      get: (_, prop: string) => ({
        ...looseStringInput,
        name: prop,
        title: prop,
        type: { name: 'StringField', cardinality: 'SINGLE', batch: false },
      }),
      has: () => true,
    }
  ),
  outputs: {},
  useCache: true,
  nodePack: 'invokeai',
  classification: 'stable',
  category: 'misc',
});

const templates: Templates = new Proxy(
  {},
  {
    get: (_, prop: string) => {
      if (prop === 'core_metadata') {
        return {
          ...passthroughTemplate('core_metadata'),
          inputs: new Proxy(
            {},
            {
              get: (_, key: string) => ({
                ...looseStringInput,
                name: key,
                title: key,
                type:
                  key === 'loras'
                    ? { name: 'LoRAMetadataField', cardinality: 'COLLECTION', batch: false }
                    : { name: 'StringField', cardinality: 'SINGLE', batch: false },
              }),
              has: () => true,
            }
          ),
          classification: 'internal',
        };
      }
      if (prop === 'lora_selector') {
        return {
          ...passthroughTemplate('lora_selector'),
          version: '1.0.3',
          inputs: new Proxy(
            {},
            {
              get: (_, key: string) => ({
                ...looseStringInput,
                name: key,
                title: key,
                type:
                  key === 'lora'
                    ? { name: 'ModelIdentifierField', cardinality: 'SINGLE', batch: false }
                    : { name: 'FloatField', cardinality: 'SINGLE', batch: false },
              }),
              has: () => true,
            }
          ),
        };
      }
      return passthroughTemplate(typeof prop === 'string' ? prop : 'unknown');
    },
  }
) as unknown as Templates;

const expectedLorasValue = [
  {
    model: {
      key: '1bce536a-a770-49bd-a58e-c6576a7fe41d',
      hash: 'blake3:5dc8b2234366702e22410e5f2de079d24c13eb58aa9597fa6a4d6688ce9f1da2',
      name: 'hina_zImageTurbo_anime_v2.52-dream',
      base: 'z-image',
      type: 'lora',
      submodel_type: null,
    },
    weight: 0.75,
  },
];

const userFirstGraph = {
  id: 'z_image_graph:EnnoIJwImk',
  nodes: {
    'core_metadata:sbmUlPbCpY': {
      id: 'core_metadata:sbmUlPbCpY',
      type: 'core_metadata',
      is_intermediate: true,
      use_cache: true,
      generation_mode: 'z_image_txt2img',
      loras: expectedLorasValue,
      // Backend `extra='allow'` extras - not declared in OpenAPI schema:
      z_image_seed_variance_enabled: false,
      z_image_seed_variance_strength: 0.1,
      z_image_seed_variance_randomize_percent: 50,
    },
    'lora_selector:lXZkTpWiQQ': {
      id: 'lora_selector:lXZkTpWiQQ',
      type: 'lora_selector',
      is_intermediate: true,
      use_cache: true,
      lora: expectedLorasValue[0]?.model,
      weight: 0.75,
    },
  },
  edges: [],
} as unknown as NonNullableGraph;

const findInput = (workflow: ReturnType<typeof graphToWorkflow>, nodeId: string, fieldName: string) => {
  const node = workflow.nodes.find((n) => n.id === nodeId);
  if (!node || node.type !== 'invocation') {
    return undefined;
  }
  return node.data.inputs[fieldName];
};

describe('issue #9151: graphToWorkflow + zod validation roundtrip', () => {
  beforeAll(() => {
    $templates.set(templates);
  });

  it('graphToWorkflow preserves core_metadata.loras value', () => {
    const workflow = graphToWorkflow(userFirstGraph, false);
    const lorasInput = findInput(workflow, 'core_metadata:sbmUlPbCpY', 'loras');
    expect(lorasInput).toBeDefined();
    expect(lorasInput?.value).toEqual(expectedLorasValue);
  });

  it('parseAndMigrateWorkflow does not strip core_metadata.loras', () => {
    const workflow = graphToWorkflow(userFirstGraph, false);
    const migrated = parseAndMigrateWorkflow(workflow);
    const lorasInput = findInput(migrated, 'core_metadata:sbmUlPbCpY', 'loras');
    expect(lorasInput).toBeDefined();
    expect(lorasInput?.value).toEqual(expectedLorasValue);
  });

  it('graphToWorkflow preserves core_metadata extra fields (z_image_seed_variance_*)', () => {
    const workflow = graphToWorkflow(userFirstGraph, false);
    expect(findInput(workflow, 'core_metadata:sbmUlPbCpY', 'z_image_seed_variance_enabled')?.value).toBe(false);
    expect(findInput(workflow, 'core_metadata:sbmUlPbCpY', 'z_image_seed_variance_strength')?.value).toBe(0.1);
    expect(findInput(workflow, 'core_metadata:sbmUlPbCpY', 'z_image_seed_variance_randomize_percent')?.value).toBe(50);
  });

  it('parseAndMigrateWorkflow does not strip core_metadata extra fields', () => {
    const workflow = graphToWorkflow(userFirstGraph, false);
    const migrated = parseAndMigrateWorkflow(workflow);
    expect(findInput(migrated, 'core_metadata:sbmUlPbCpY', 'z_image_seed_variance_enabled')?.value).toBe(false);
    expect(findInput(migrated, 'core_metadata:sbmUlPbCpY', 'z_image_seed_variance_strength')?.value).toBe(0.1);
    expect(findInput(migrated, 'core_metadata:sbmUlPbCpY', 'z_image_seed_variance_randomize_percent')?.value).toBe(50);
  });
});

const imageCollectionTemplate = {
  title: 'Image Collection Primitive',
  type: 'image_collection',
  version: '1.0.2',
  tags: ['primitives', 'image', 'collection'],
  description: 'A collection of image primitive values',
  outputType: 'image_collection_output',
  inputs: {
    collection: {
      name: 'collection',
      title: 'Collection',
      required: false,
      description: 'An optional image collection to append to',
      fieldKind: 'input',
      input: 'connection',
      ui_hidden: false,
      type: { name: 'ImageField', cardinality: 'COLLECTION', batch: false },
      default: undefined,
    },
    images: {
      name: 'images',
      title: 'Images',
      required: false,
      description: 'The images to append to the collection',
      fieldKind: 'input',
      input: 'direct',
      ui_hidden: false,
      type: { name: 'ImageField', cardinality: 'COLLECTION', batch: false },
      default: undefined,
    },
  },
  outputs: {
    collection: {
      fieldKind: 'output',
      name: 'collection',
      title: 'Collection',
      description: 'The output images',
      type: { name: 'ImageField', cardinality: 'COLLECTION', batch: false },
      ui_hidden: false,
    },
  },
  useCache: true,
  nodePack: 'invokeai',
  classification: 'stable',
  category: 'primitives',
} satisfies InvocationTemplate;

describe('graphToWorkflow', () => {
  const originalTemplates = $templates.get();

  beforeEach(() => {
    $templates.set({ image_collection: imageCollectionTemplate });
  });

  afterEach(() => {
    $templates.set(originalTemplates);
  });

  it('moves legacy image_collection graph collection values to the visible images field', () => {
    const images = [{ image_name: 'legacy.png' }];
    const graph: NonNullableGraph = {
      id: 'graph',
      nodes: {
        image_collection: {
          id: 'image_collection',
          type: 'image_collection',
          collection: images,
        },
      },
      edges: [],
    };

    const workflow = graphToWorkflow(graph, false);
    const node = workflow.nodes[0];

    if (!node || !isWorkflowInvocationNode(node)) {
      throw new Error('Expected an image_collection workflow node');
    }
    expect(node.data.inputs.images?.value).toEqual(images);
    expect(node.data.inputs.collection?.value).toEqual([]);
  });

  it('preserves legacy image_collection graph collection values when collection is connected', () => {
    const images = [{ image_name: 'shadowed.png' }];
    const graph: NonNullableGraph = {
      id: 'graph',
      nodes: {
        source: {
          id: 'source',
          type: 'image_collection',
        },
        target: {
          id: 'target',
          type: 'image_collection',
          collection: images,
        },
      },
      edges: [
        {
          source: { node_id: 'source', field: 'collection' },
          destination: { node_id: 'target', field: 'collection' },
        },
      ],
    };

    const workflow = graphToWorkflow(graph, false);
    const node = workflow.nodes[1];

    if (!node || !isWorkflowInvocationNode(node)) {
      throw new Error('Expected an image_collection workflow node');
    }
    expect(node.data.inputs.images?.value).toBeUndefined();
    expect(node.data.inputs.collection?.value).toEqual(images);
  });
});
