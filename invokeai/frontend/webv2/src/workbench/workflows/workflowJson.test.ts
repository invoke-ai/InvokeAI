import { describe, expect, it } from 'vitest';

import type { InvocationTemplate } from './types';

import {
  buildCurrentImageNode,
  buildConnectorNode,
  buildInvocationNode,
  buildNotesNode,
  createProjectGraph,
  getFormChildren,
  projectGraphReducer,
} from './document';
import { parseWorkflowJson, serializeWorkflowJson } from './workflowJson';

const template: InvocationTemplate = {
  category: 'test',
  classification: 'stable',
  description: '',
  inputs: {
    prompt: {
      default: 'a cat',
      description: '',
      exclusiveMaximum: null,
      exclusiveMinimum: null,
      input: 'any',
      maximum: null,
      minimum: null,
      multipleOf: null,
      name: 'prompt',
      options: null,
      required: true,
      title: 'Prompt',
      type: { batch: false, cardinality: 'SINGLE', name: 'StringField' },
      uiChoiceLabels: null,
      uiComponent: 'textarea',
      uiHidden: false,
      uiModelBase: null,
      uiModelType: null,
      uiOrder: null,
    },
  },
  nodePack: 'invokeai',
  outputs: {},
  outputType: 'string_output',
  tags: [],
  title: 'Prompt',
  type: 'prompt',
  useCache: true,
  version: '1.2.0',
};

describe('workflow JSON round-trip', () => {
  it('serializes and re-parses a document, preserving nodes, edges, form, and metadata', () => {
    const node = buildInvocationNode(template, { x: 10, y: 20 });
    let doc = createProjectGraph('roundtrip');

    doc = projectGraphReducer(doc, { node, type: 'addNode' });
    doc = projectGraphReducer(doc, {
      description: 'Custom description',
      fieldName: 'prompt',
      nodeId: node.id,
      type: 'setFieldDescription',
    });
    doc = projectGraphReducer(doc, { fieldIdentifier: { fieldName: 'prompt', nodeId: node.id }, type: 'exposeField' });
    doc = projectGraphReducer(doc, {
      patch: { author: 'josh', description: 'desc', name: 'Round Trip', tags: 'a,b' },
      type: 'setMetadata',
    });

    const serialized = serializeWorkflowJson(doc);
    const { document: parsed, warnings } = parseWorkflowJson(serialized);

    expect(warnings).toEqual([]);
    expect(parsed.name).toBe('Round Trip');
    expect(parsed.author).toBe('josh');
    expect(parsed.nodes).toHaveLength(1);

    const parsedNode = parsed.nodes[0];

    expect(parsedNode?.id).toBe(node.id);
    expect(parsedNode?.position).toEqual({ x: 10, y: 20 });
    expect(parsedNode?.type === 'invocation' && parsedNode.data.inputs.prompt?.value).toBe('a cat');
    expect(parsedNode?.type === 'invocation' && parsedNode.data.inputs.prompt?.description).toBe('Custom description');

    const formChildren = getFormChildren(parsed.form);

    expect(formChildren).toHaveLength(1);
    expect(formChildren[0]?.type).toBe('node-field');
  });

  it('keeps the serialized shape legacy-compatible', () => {
    const doc = createProjectGraph('legacy-shape');
    const serialized = serializeWorkflowJson(doc);

    expect(serialized.meta).toEqual({ category: 'user', version: '3.0.0' });
    expect(serialized.exposedFields).toEqual([]);
    expect(serialized).toHaveProperty('form');
  });

  it('round-trips notes, current_image, and connector UI nodes', () => {
    let doc = createProjectGraph('ui-nodes');

    doc = projectGraphReducer(doc, { node: buildNotesNode({ x: 1, y: 2 }), type: 'addNode' });
    doc = projectGraphReducer(doc, { node: buildCurrentImageNode({ x: 3, y: 4 }), type: 'addNode' });
    doc = projectGraphReducer(doc, { node: buildConnectorNode({ x: 5, y: 6 }), type: 'addNode' });

    const { document: parsed, warnings } = parseWorkflowJson(serializeWorkflowJson(doc));

    expect(warnings).toEqual([]);
    expect(parsed.nodes.map((node) => node.type).sort()).toEqual(['connector', 'current_image', 'notes']);

    const currentImage = parsed.nodes.find((node) => node.type === 'current_image');

    expect(currentImage?.position).toEqual({ x: 3, y: 4 });
    expect(currentImage?.type === 'current_image' && currentImage.data.label).toBe('Current Image');

    const connector = parsed.nodes.find((node) => node.type === 'connector');

    expect(connector?.position).toEqual({ x: 5, y: 6 });
    expect(connector?.type === 'connector' && connector.data.label).toBe('');
  });
});

describe('parseWorkflowJson tolerance', () => {
  it('migrates pre-form exposedFields into form elements', () => {
    const { document, warnings } = parseWorkflowJson({
      edges: [],
      exposedFields: [{ fieldName: 'prompt', nodeId: 'n1' }],
      name: 'Old Workflow',
      nodes: [
        {
          data: { id: 'n1', inputs: { prompt: { label: '', name: 'prompt', value: 'hi' } }, type: 'prompt' },
          id: 'n1',
          position: { x: 0, y: 0 },
          type: 'invocation',
        },
      ],
      version: '1.0.0',
    });

    expect(warnings).toEqual([]);

    const children = getFormChildren(document.form);

    expect(children).toHaveLength(1);
    expect(children[0]?.type === 'node-field' && children[0].data.fieldIdentifier).toEqual({
      fieldName: 'prompt',
      nodeId: 'n1',
    });
  });

  it('preserves connector nodes and edges', () => {
    const { document, warnings } = parseWorkflowJson({
      edges: [
        { id: 'e1', source: 'n1', sourceHandle: 'out', target: 'conn1', targetHandle: 'in', type: 'default' },
        { id: 'e2', source: 'conn1', sourceHandle: 'out', target: 'n2', targetHandle: 'text', type: 'default' },
      ],
      name: 'Connectors',
      nodes: [
        { data: { id: 'n1', inputs: {}, type: 'a' }, id: 'n1', position: { x: 0, y: 0 }, type: 'invocation' },
        { data: { id: 'n2', inputs: {}, type: 'b' }, id: 'n2', position: { x: 0, y: 0 }, type: 'invocation' },
        { data: { label: 'Connector' }, id: 'conn1', position: { x: 4, y: 5 }, type: 'connector' },
      ],
    });

    expect(warnings).toEqual([]);
    expect(document.nodes.map((node) => node.type).sort()).toEqual(['connector', 'invocation', 'invocation']);
    expect(document.nodes.find((node) => node.type === 'connector')?.position).toEqual({ x: 4, y: 5 });
    expect(document.edges).toEqual([
      { id: 'e1', source: 'n1', sourceHandle: 'out', target: 'conn1', targetHandle: 'in', type: 'default' },
      { id: 'e2', source: 'conn1', sourceHandle: 'out', target: 'n2', targetHandle: 'text', type: 'default' },
    ]);
  });

  it('drops dangling edges and unknown form elements with warnings', () => {
    const { document, warnings } = parseWorkflowJson({
      edges: [{ id: 'e1', source: 'missing', sourceHandle: 'out', target: 'n1', targetHandle: 'in', type: 'default' }],
      form: { elements: { weird: { id: 'weird', type: 'mystery' } }, rootElementId: 'missing-root' },
      name: 'Broken',
      nodes: [{ data: { id: 'n1', inputs: {}, type: 'a' }, id: 'n1', position: { x: 0, y: 0 }, type: 'invocation' }],
    });

    expect(document.edges).toEqual([]);
    expect(document.form.elements[document.form.rootElementId]?.type).toBe('container');
    expect(warnings.length).toBeGreaterThan(0);
  });

  it('rejects non-workflow payloads', () => {
    expect(() => parseWorkflowJson('not a workflow')).toThrow(/not a recognizable/);
  });

  it('accepts form: null, as stored by pre-form-builder library workflows', () => {
    const { document, warnings } = parseWorkflowJson({
      edges: [],
      form: null,
      name: 'No Form',
      nodes: [{ data: { id: 'n1', inputs: {}, type: 'a' }, id: 'n1', position: { x: 0, y: 0 }, type: 'invocation' }],
    });

    expect(warnings).toEqual([]);
    expect(document.form.elements[document.form.rootElementId]?.type).toBe('container');
  });
});
