import { describe, expect, it } from 'vitest';

import {
  buildInvocationNode,
  createProjectGraph,
  getFormChildren,
  isFieldExposed,
  normalizeProjectGraph,
  projectGraphReducer,
} from './document';
import type { InvocationTemplate, ProjectGraphState, WorkflowEdge } from './types';

const template: InvocationTemplate = {
  category: 'math',
  classification: 'stable',
  description: '',
  inputs: {
    a: {
      default: 1,
      description: '',
      exclusiveMaximum: null,
      exclusiveMinimum: null,
      input: 'any',
      maximum: null,
      minimum: null,
      multipleOf: null,
      name: 'a',
      options: null,
      required: false,
      title: 'A',
      type: { batch: false, cardinality: 'SINGLE', name: 'IntegerField' },
      uiChoiceLabels: null,
      uiComponent: null,
      uiHidden: false,
      uiModelBase: null,
      uiModelType: null,
      uiOrder: null,
    },
  },
  nodePack: 'invokeai',
  outputs: {
    value: {
      description: '',
      name: 'value',
      title: 'Value',
      type: { batch: false, cardinality: 'SINGLE', name: 'IntegerField' },
    },
  },
  outputType: 'integer_output',
  tags: [],
  title: 'Add',
  type: 'add',
  useCache: true,
  version: '1.0.0',
};

const createDocWithNodes = (): { doc: ProjectGraphState; nodeAId: string; nodeBId: string } => {
  const nodeA = buildInvocationNode(template, { x: 0, y: 0 });
  const nodeB = buildInvocationNode(template, { x: 100, y: 0 });
  let doc = createProjectGraph('test-graph');

  doc = projectGraphReducer(doc, { node: nodeA, type: 'addNode' });
  doc = projectGraphReducer(doc, { node: nodeB, type: 'addNode' });

  return { doc, nodeAId: nodeA.id, nodeBId: nodeB.id };
};

const createEdge = (source: string, target: string): WorkflowEdge => ({
  id: `edge-${source}-${target}`,
  source,
  sourceHandle: 'value',
  target,
  targetHandle: 'a',
  type: 'default',
});

describe('projectGraphReducer', () => {
  it('builds nodes from templates with default input values', () => {
    const node = buildInvocationNode(template, { x: 5, y: 6 });

    expect(node.data.inputs.a?.value).toBe(1);
    expect(node.data.useCache).toBe(true);
    expect(node.position).toEqual({ x: 5, y: 6 });
  });

  it('removing nodes drops their edges and exposed form fields', () => {
    const { doc, nodeAId, nodeBId } = createDocWithNodes();
    let next = projectGraphReducer(doc, { edge: createEdge(nodeAId, nodeBId), type: 'addEdge' });

    next = projectGraphReducer(next, { fieldIdentifier: { fieldName: 'a', nodeId: nodeBId }, type: 'exposeField' });

    expect(next.edges).toHaveLength(1);
    expect(isFieldExposed(next.form, { fieldName: 'a', nodeId: nodeBId })).toBe(true);

    next = projectGraphReducer(next, { nodeIds: [nodeBId], type: 'removeNodes' });

    expect(next.nodes.map((node) => node.id)).toEqual([nodeAId]);
    expect(next.edges).toEqual([]);
    expect(isFieldExposed(next.form, { fieldName: 'a', nodeId: nodeBId })).toBe(false);
  });

  it('adds pasted nodes and their internal edges in one action', () => {
    const { doc, nodeAId, nodeBId } = createDocWithNodes();
    const pastedA = buildInvocationNode(template, { x: 32, y: 32 });
    const pastedB = buildInvocationNode(template, { x: 132, y: 32 });
    const next = projectGraphReducer(doc, {
      edges: [createEdge(pastedA.id, pastedB.id)],
      nodes: [pastedA, pastedB],
      type: 'addGraphElements',
    });

    expect(next.nodes.map((node) => node.id)).toEqual([nodeAId, nodeBId, pastedA.id, pastedB.id]);
    expect(next.edges).toHaveLength(1);
  });

  it('adds a new node connected to an existing node in one action', () => {
    const { doc, nodeAId, nodeBId } = createDocWithNodes();
    const insertedNode = buildInvocationNode(template, { x: 200, y: 0 });
    const next = projectGraphReducer(doc, {
      edge: createEdge(nodeAId, insertedNode.id),
      node: insertedNode,
      type: 'addNodeAndEdge',
    });

    expect(next.nodes.map((node) => node.id)).toEqual([nodeAId, nodeBId, insertedNode.id]);
    expect(next.edges).toEqual([createEdge(nodeAId, insertedNode.id)]);
  });

  it('addGraphElements drops colliding node ids and their edges', () => {
    const { doc, nodeAId, nodeBId } = createDocWithNodes();
    const existingNode = doc.nodes[0];
    const freshNode = buildInvocationNode(template, { x: 64, y: 64 });
    const next = projectGraphReducer(doc, {
      edges: [createEdge(nodeAId, freshNode.id)],
      nodes: [existingNode!, freshNode],
      type: 'addGraphElements',
    });

    expect(next.nodes.map((node) => node.id)).toEqual([nodeAId, nodeBId, freshNode.id]);
    // The edge referenced a dropped duplicate, so it is dropped too.
    expect(next.edges).toEqual([]);
  });

  it('connecting an already-connected input replaces the existing edge', () => {
    const { doc, nodeAId, nodeBId } = createDocWithNodes();
    let next = projectGraphReducer(doc, { edge: createEdge(nodeAId, nodeBId), type: 'addEdge' });
    const replacement = { ...createEdge(nodeAId, nodeBId), id: 'edge-replacement' };

    next = projectGraphReducer(next, { edge: replacement, type: 'addEdge' });

    expect(next.edges).toHaveLength(1);
    expect(next.edges[0]?.id).toBe('edge-replacement');
  });

  it('sets field values without disturbing other inputs', () => {
    const { doc, nodeAId } = createDocWithNodes();
    const next = projectGraphReducer(doc, { fieldName: 'a', nodeId: nodeAId, type: 'setFieldValue', value: 42 });
    const node = next.nodes.find((candidate) => candidate.id === nodeAId);

    expect(node?.type === 'invocation' && node.data.inputs.a?.value).toBe(42);
  });

  it('sets and clears field description overrides', () => {
    const { doc, nodeAId } = createDocWithNodes();
    let next = projectGraphReducer(doc, {
      description: 'Custom help text',
      fieldName: 'a',
      nodeId: nodeAId,
      type: 'setFieldDescription',
    });
    const getInstance = (document: typeof next) => {
      const node = document.nodes.find((candidate) => candidate.id === nodeAId);

      return node?.type === 'invocation' ? node.data.inputs.a : undefined;
    };

    expect(getInstance(next)?.description).toBe('Custom help text');

    next = projectGraphReducer(next, { description: '', fieldName: 'a', nodeId: nodeAId, type: 'setFieldDescription' });

    expect(getInstance(next)?.description).toBeUndefined();
  });

  it('exposing a field twice is a no-op, and form elements reorder', () => {
    const { doc, nodeAId, nodeBId } = createDocWithNodes();
    let next = projectGraphReducer(doc, { fieldIdentifier: { fieldName: 'a', nodeId: nodeAId }, type: 'exposeField' });

    next = projectGraphReducer(next, { fieldIdentifier: { fieldName: 'a', nodeId: nodeAId }, type: 'exposeField' });
    next = projectGraphReducer(next, { fieldIdentifier: { fieldName: 'a', nodeId: nodeBId }, type: 'exposeField' });

    const childrenBefore = getFormChildren(next.form).map((element) => element.id);

    expect(childrenBefore).toHaveLength(2);

    const lastElementId = childrenBefore[1] as string;

    next = projectGraphReducer(next, { direction: -1, elementId: lastElementId, type: 'moveFormElement' });

    expect(getFormChildren(next.form).map((element) => element.id)).toEqual([lastElementId, childrenBefore[0]]);
  });

  it('reparents form elements via moveFormElementTo with index adjustment and cycle guards', () => {
    const { doc, nodeAId, nodeBId } = createDocWithNodes();
    let next = projectGraphReducer(doc, { fieldIdentifier: { fieldName: 'a', nodeId: nodeAId }, type: 'exposeField' });

    next = projectGraphReducer(next, { fieldIdentifier: { fieldName: 'a', nodeId: nodeBId }, type: 'exposeField' });
    next = projectGraphReducer(next, { elementType: 'container', layout: 'row', type: 'addFormElement' });

    const rootId = next.form.rootElementId;
    const [fieldA, fieldB, container] = getFormChildren(next.form);

    expect(container?.type).toBe('container');

    // Reorder within the root: move fieldA after fieldB (index adjusts for removal).
    next = projectGraphReducer(next, {
      elementId: fieldA?.id ?? '',
      index: 2,
      parentId: rootId,
      type: 'moveFormElementTo',
    });

    expect(getFormChildren(next.form).map((element) => element.id)).toEqual([fieldB?.id, fieldA?.id, container?.id]);

    // Reparent fieldB into the container.
    next = projectGraphReducer(next, {
      elementId: fieldB?.id ?? '',
      index: 0,
      parentId: container?.id ?? '',
      type: 'moveFormElementTo',
    });

    const containerElement = next.form.elements[container?.id ?? ''];

    expect(containerElement?.type === 'container' && containerElement.data.children).toEqual([fieldB?.id]);
    expect(next.form.elements[fieldB?.id ?? '']?.parentId).toBe(container?.id);

    // A container cannot be dropped into its own subtree.
    const guarded = projectGraphReducer(next, {
      elementId: container?.id ?? '',
      index: 0,
      parentId: container?.id ?? '',
      type: 'moveFormElementTo',
    });

    expect(guarded).toBe(next);
  });

  it('toggles container layout and node-field descriptions', () => {
    const { doc, nodeAId } = createDocWithNodes();
    let next = projectGraphReducer(doc, { elementType: 'container', layout: 'column', type: 'addFormElement' });

    next = projectGraphReducer(next, { fieldIdentifier: { fieldName: 'a', nodeId: nodeAId }, type: 'exposeField' });

    const [container, field] = getFormChildren(next.form);

    next = projectGraphReducer(next, { elementId: container?.id ?? '', layout: 'row', type: 'setContainerLayout' });

    const updatedContainer = next.form.elements[container?.id ?? ''];

    expect(updatedContainer?.type === 'container' && updatedContainer.data.layout).toBe('row');

    next = projectGraphReducer(next, {
      elementId: field?.id ?? '',
      showDescription: true,
      type: 'setNodeFieldShowDescription',
    });

    const updatedField = next.form.elements[field?.id ?? ''];

    expect(updatedField?.type === 'node-field' && updatedField.data.showDescription).toBe(true);
  });

  it('updates metadata via patch', () => {
    const { doc } = createDocWithNodes();
    const next = projectGraphReducer(doc, { patch: { author: 'josh', name: 'My Flow' }, type: 'setMetadata' });

    expect(next.name).toBe('My Flow');
    expect(next.author).toBe('josh');
  });
});

describe('normalizeProjectGraph', () => {
  it('passes through current documents', () => {
    const doc = createProjectGraph('keep-me');

    expect(normalizeProjectGraph(doc)).toBe(doc);
  });

  it('replaces the Phase-1 placeholder graph with an empty document, preserving the id', () => {
    const normalized = normalizeProjectGraph({ edges: [], id: 'legacy-id', label: 'Old', nodes: [], version: 1 });

    expect(normalized.version).toBe(2);
    expect(normalized.id).toBe('legacy-id');
    expect(normalized.nodes).toEqual([]);
    expect(normalized.form.elements[normalized.form.rootElementId]?.type).toBe('container');
  });
});
