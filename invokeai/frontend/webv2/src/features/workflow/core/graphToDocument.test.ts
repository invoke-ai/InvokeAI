import { describe, expect, it } from 'vitest';

import type { FieldInputTemplate, InvocationTemplate } from './types';

import { compileProjectGraph, getProjectGraphReadiness } from './buildGraph';
import { updateWorkflowNodes } from './document';
import { previewGraphToDocument, storedGraphToDocument, type PreviewGraphLike } from './graphToDocument';
import { isInvocationNode } from './types';
import { parseWorkflowJson, serializeWorkflowJson } from './workflowJson';

const input = (name: string, overrides: Partial<FieldInputTemplate> = {}): FieldInputTemplate => ({
  default: undefined,
  description: '',
  exclusiveMaximum: null,
  exclusiveMinimum: null,
  fieldKind: 'input',
  input: 'any',
  maximum: null,
  minimum: null,
  multipleOf: null,
  name,
  options: null,
  required: false,
  title: name,
  type: { batch: false, cardinality: 'SINGLE', name: 'IntegerField' },
  uiChoiceLabels: null,
  uiComponent: null,
  uiHidden: false,
  uiModelBase: null,
  uiModelFormat: null,
  uiModelType: null,
  uiOrder: null,
  ...overrides,
});

const template = (type: string, inputs: Record<string, FieldInputTemplate>): InvocationTemplate => ({
  category: 'test',
  classification: 'stable',
  description: '',
  inputs,
  nodePack: 'invokeai',
  outputs: {
    value: {
      description: '',
      name: 'value',
      title: 'Value',
      type: { batch: false, cardinality: 'SINGLE', name: 'IntegerField' },
    },
  },
  outputType: `${type}_output`,
  tags: [],
  title: type,
  type,
  useCache: true,
  version: '1.0.0',
});

const templates = {
  denoise: template('denoise', {
    seed: input('seed', { default: 0 }),
    steps: input('steps', { default: 30 }),
  }),
  integer: template('integer', {
    value: input('value', { default: 0 }),
  }),
};

const graph: PreviewGraphLike = {
  edges: [
    {
      id: 'ignored',
      sourceField: 'value',
      sourceNodeId: 'seed',
      targetField: 'seed',
      targetNodeId: 'denoise_latents',
    },
    {
      id: 'ignored-2',
      sourceField: 'value',
      sourceNodeId: 'seed',
      targetField: 'value',
      targetNodeId: 'mystery',
    },
  ],
  nodes: [
    { id: 'seed', inputs: { use_cache: false, value: 123 }, type: 'integer' },
    { id: 'denoise_latents', inputs: { is_intermediate: true, steps: 28 }, type: 'denoise' },
    { id: 'mystery', inputs: {}, type: 'unknown_type' },
  ],
};

describe('storedGraphToDocument', () => {
  it('loads unversioned frame counts as current editable choices', () => {
    const frameTemplates = {
      minimax_h3_denoise: {
        ...template('minimax_h3_denoise', {
          num_frames: input('num_frames', {
            default: '124',
            options: ['90', '124'],
            type: { batch: false, cardinality: 'SINGLE', name: 'EnumField' },
          }),
        }),
        version: '1.4.0',
      },
    };
    const { document } = storedGraphToDocument(
      {
        nodes: {
          valid: { id: 'valid', type: 'minimax_h3_denoise', num_frames: 90 },
          invalid: { id: 'invalid', type: 'minimax_h3_denoise', num_frames: 100 },
          current: { id: 'current', type: 'minimax_h3_denoise', num_frames: '90' },
        },
        edges: [],
      },
      frameTemplates
    );
    const loaded = updateWorkflowNodes(parseWorkflowJson(serializeWorkflowJson(document)).document, frameTemplates);
    expect(
      getProjectGraphReadiness(loaded.document, { status: 'loaded', error: null, templates: frameTemplates })
    ).toMatchObject({ canInvoke: true, reasons: [] });
    const compiled = compileProjectGraph(loaded.document, frameTemplates).backendGraph;
    expect(compiled.nodes.valid?.num_frames).toBe('90');
    expect(compiled.nodes.invalid?.num_frames).toBe('124');
    expect(compiled.nodes.current?.num_frames).toBe('90');
  });

  it('loads old image lists as editable images without replacing current lists or connected collections', () => {
    const imageList = { batch: false, cardinality: 'COLLECTION' as const, name: 'ImageField' };
    const collectionTemplates = {
      image_collection: {
        ...template('image_collection', {
          collection: input('collection', { input: 'connection', type: imageList }),
          images: input('images', { default: [], type: imageList }),
        }),
        outputs: { collection: { description: '', name: 'collection', title: 'Collection', type: imageList } },
        version: '1.0.2',
      },
    };
    const oldImages = [{ image_name: 'old.png' }];
    const currentImages = [{ image_name: 'current.png' }];
    const edge = {
      source: { node_id: 'legacy', field: 'collection' },
      destination: { node_id: 'connected', field: 'collection' },
    };
    const { document } = storedGraphToDocument(
      {
        nodes: {
          legacy: { id: 'legacy', type: 'image_collection', collection: oldImages },
          current: { id: 'current', type: 'image_collection', collection: oldImages, images: currentImages },
          connected: { id: 'connected', type: 'image_collection', collection: oldImages },
        },
        edges: [edge],
      },
      collectionTemplates
    );
    const loaded = updateWorkflowNodes(
      parseWorkflowJson(serializeWorkflowJson(document)).document,
      collectionTemplates
    );
    expect(
      getProjectGraphReadiness(loaded.document, { status: 'loaded', error: null, templates: collectionTemplates })
    ).toMatchObject({ canInvoke: true, reasons: [] });
    const compiled = compileProjectGraph(loaded.document, collectionTemplates).backendGraph;
    expect(compiled.nodes.legacy?.images).toEqual(oldImages);
    expect(compiled.nodes.legacy).not.toHaveProperty('collection');
    expect(compiled.nodes.current?.images).toEqual(currentImages);
    expect(compiled.nodes.connected?.images).toEqual([]);
    expect(compiled.nodes.connected).not.toHaveProperty('collection');
    expect(compiled.edges).toEqual([{ ...edge, type: 'default' }]);
  });

  it('recalls generation values, flags, metadata extras and edges through serialization and compilation', () => {
    const storedTemplates = { ...templates, core_metadata: template('core_metadata', {}) };
    const stored = {
      id: 'generated',
      nodes: {
        seed: { id: 'seed', type: 'integer', value: 42, use_cache: false },
        denoise: { id: 'denoise', type: 'denoise', steps: 12, is_intermediate: false, unrelated: 'drop' },
        metadata: {
          id: 'metadata',
          type: 'core_metadata',
          is_intermediate: true,
          loras: [{ model: { key: 'lora-1' }, weight: 0.75 }],
          z_image_seed_variance_enabled: true,
        },
      },
      edges: [
        { source: { node_id: 'seed', field: 'value' }, destination: { node_id: 'denoise', field: 'seed' } },
        { source: { node_id: 'seed', field: 'value' }, destination: { node_id: 'metadata', field: 'custom_seed' } },
      ],
    };
    const { document, skippedNodeTypes } = storedGraphToDocument(stored, storedTemplates);
    expect(skippedNodeTypes).toEqual([]);
    const recalled = parseWorkflowJson(serializeWorkflowJson(document)).document;
    const compiled = compileProjectGraph(recalled, storedTemplates).backendGraph;

    expect(compiled.nodes.seed).toMatchObject({ value: 42, use_cache: false });
    expect(compiled.nodes.denoise).toMatchObject({ steps: 12, is_intermediate: false });
    expect(compiled.nodes.denoise).not.toHaveProperty('unrelated');
    expect(compiled.nodes.metadata).toMatchObject({
      loras: [{ model: { key: 'lora-1' }, weight: 0.75 }],
      z_image_seed_variance_enabled: true,
      is_intermediate: true,
    });
    expect(compiled.edges).toEqual(stored.edges.map((edge) => ({ ...edge, type: 'default' })));
    expect(recalled.nodes[0]?.position).not.toEqual(recalled.nodes[1]?.position);

    const metadata = recalled.nodes.find((node) => node.id === 'metadata');
    if (!metadata || !isInvocationNode(metadata)) {
      throw new Error('Missing metadata node');
    }
    for (const name of ['id', 'type', 'is_intermediate', 'use_cache']) {
      metadata.data.inputs[name] = { name, label: '', value: 'injected' };
    }
    expect(compileProjectGraph(recalled, storedTemplates).backendGraph.nodes.metadata).toMatchObject({
      id: 'metadata',
      type: 'core_metadata',
      is_intermediate: true,
      use_cache: true,
    });
  });

  it('reports unknown nodes and omits their connections', () => {
    const { document, skippedNodeTypes } = storedGraphToDocument(
      {
        nodes: { seed: { id: 'seed', type: 'integer', value: 42 }, absent: { id: 'absent', type: 'missing' } },
        edges: [{ source: { node_id: 'seed', field: 'value' }, destination: { node_id: 'absent', field: 'seed' } }],
      },
      templates
    );
    expect(skippedNodeTypes).toEqual(['missing']);
    expect(document.nodes.map((node) => node.id)).toEqual(['seed']);
    expect(document.edges).toEqual([]);
  });

  it.each([null, {}, { nodes: [], edges: [] }, { nodes: { seed: { type: 'integer' } }, edges: [] }])(
    'rejects malformed stored graphs: %j',
    (raw) => {
      expect(() => storedGraphToDocument(raw, templates)).toThrow();
    }
  );
});

describe('previewGraphToDocument', () => {
  it('treats prototype-named metadata fields as own values without changing object prototypes', () => {
    const metadataTemplates = { core_metadata: template('core_metadata', {}) };
    const prototypeBefore = Object.getOwnPropertyDescriptors(Object.prototype);
    const { document } = previewGraphToDocument(
      {
        nodes: [
          {
            id: 'metadata',
            type: 'core_metadata',
            inputs: JSON.parse('{"__proto__":{"marker":42},"constructor":"custom"}'),
          },
        ],
        edges: [],
      },
      metadataTemplates
    );
    const node = document.nodes[0];
    expect(node && isInvocationNode(node) && Object.hasOwn(node.data.inputs, '__proto__')).toBe(true);

    const compiled = compileProjectGraph(document, metadataTemplates).backendGraph.nodes.metadata;
    expect(JSON.parse(JSON.stringify(compiled))).toMatchObject(
      JSON.parse('{"__proto__":{"marker":42},"constructor":"custom"}')
    );
    expect(Object.getPrototypeOf(compiled)).toBe(Object.prototype);
    expect(Object.getOwnPropertyDescriptors(Object.prototype)).toEqual(prototypeBefore);
  });

  it('converts a preview graph into an editable document, skipping unknown node types', () => {
    const { document, skippedNodeTypes } = previewGraphToDocument(graph, templates);

    expect(skippedNodeTypes).toEqual(['unknown_type']);
    expect(document.nodes.map((node) => node.id)).toEqual(['seed', 'denoise_latents']);

    const denoise = document.nodes.filter(isInvocationNode).find((node) => node.id === 'denoise_latents');

    expect(denoise?.data.inputs.steps.value).toBe(28); // literal copied
    expect(denoise?.data.inputs.seed.value).toBeUndefined(); // connected input cleared
    expect(document.edges).toHaveLength(1);
    expect(document.edges[0]).toMatchObject({
      source: 'seed',
      sourceHandle: 'value',
      target: 'denoise_latents',
      targetHandle: 'seed',
    });

    const seed = document.nodes.filter(isInvocationNode).find((node) => node.id === 'seed');

    expect(seed?.data.useCache).toBe(false); // is_intermediate/use_cache lifted out of inputs
    expect(seed?.data.inputs.value.value).toBe(123);
    expect(seed?.position.x).toBe(0);
    expect(denoise?.position.x).toBe(300); // layered layout
  });

  it('round-trips through compileProjectGraph back to the fixture edge', () => {
    const { document } = previewGraphToDocument(graph, templates);
    const compiled = compileProjectGraph(document, templates);

    expect(compiled.edges).toHaveLength(1);
    expect(compiled.edges[0]).toMatchObject({ sourceNodeId: 'seed', targetField: 'seed' });
  });

  it('preserves a direct loop-linkage edge when opening a preview as a document', () => {
    const loopTemplates = {
      for: template('for', { collection: input('collection', { input: 'connection' }) }),
      for_return: template('for_return', {
        loop_linkage: input('loop_linkage', { input: 'connection' }),
        output: input('output'),
      }),
    };
    const loopGraph: PreviewGraphLike = {
      edges: [
        {
          id: 'loop-edge',
          sourceField: 'loop_linkage',
          sourceNodeId: 'for',
          targetField: 'loop_linkage',
          targetNodeId: 'for-return',
          type: 'loop_linkage',
        },
      ],
      nodes: [
        { id: 'for', inputs: {}, type: 'for' },
        { id: 'for-return', inputs: {}, type: 'for_return' },
      ],
    };

    const { document } = previewGraphToDocument(loopGraph, loopTemplates);

    expect(document.edges).toEqual([
      {
        id: expect.any(String),
        source: 'for',
        sourceHandle: 'loop_linkage',
        target: 'for-return',
        targetHandle: 'loop_linkage',
        type: 'loop_linkage',
      },
    ]);
  });
});
