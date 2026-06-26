/**
 * Reproduction of issue #9151 using the user's actual first-image graph
 * extracted from a real generation. Verifies the LoRA metadata survives
 * the graph -> workflow roundtrip.
 */

import { $templates } from 'features/nodes/store/nodesSlice';
import type { Templates } from 'features/nodes/store/types';
import type { NonNullableGraph } from 'services/api/types';
import { beforeAll, describe, expect, it } from 'vitest';

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
