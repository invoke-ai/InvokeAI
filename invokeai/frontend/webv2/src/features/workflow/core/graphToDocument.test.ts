import { describe, expect, it } from 'vitest';

import type { FieldInputTemplate, InvocationTemplate } from './types';

import { compileProjectGraph } from './buildGraph';
import { previewGraphToDocument, type PreviewGraphLike } from './graphToDocument';
import { isInvocationNode } from './types';

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

describe('previewGraphToDocument', () => {
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
