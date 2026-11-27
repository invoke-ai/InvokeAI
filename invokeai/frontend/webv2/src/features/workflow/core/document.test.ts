import { describe, expect, it } from 'vitest';

import type { InvocationTemplate, ProjectGraphState, WorkflowEdge, WorkflowInvocationNode } from './types';

import {
  buildInvocationNode,
  createProjectGraph,
  getFormChildren,
  getProjectGraphUndoEntry,
  isFieldExposed,
  projectGraphReducer,
  updateWorkflowNodes,
} from './document';

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
      fieldKind: 'input',
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
      uiModelFormat: null,
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

  it('adds a new node and multiple edges in one action', () => {
    const { doc, nodeAId, nodeBId } = createDocWithNodes();
    const insertedNode = buildInvocationNode(template, { x: 200, y: 0 });
    const firstEdge = createEdge(nodeAId, insertedNode.id);
    const secondEdge = { ...createEdge(nodeBId, insertedNode.id), targetHandle: 'b' };
    const next = projectGraphReducer(doc, {
      edge: [firstEdge, secondEdge],
      node: insertedNode,
      type: 'addNodeAndEdge',
    });

    expect(next.edges).toEqual([firstEdge, secondEdge]);
  });

  it('addGraphElements connects pasted nodes to existing sources but never fills one input twice', () => {
    const { doc, nodeAId, nodeBId } = createDocWithNodes();
    const pasted = buildInvocationNode(template, { x: 64, y: 64 });
    const collect = buildInvocationNode({ ...template, type: 'collect' }, { x: 96, y: 96 });
    const fromA = { ...createEdge(nodeAId, pasted.id), id: 'edge-from-a' };
    const fromB = { ...createEdge(nodeBId, pasted.id), id: 'edge-from-b' };
    const fromGone = { ...createEdge('deleted-node', pasted.id), id: 'edge-from-gone', targetHandle: 'b' };
    const itemFromA = { ...createEdge(nodeAId, collect.id), id: 'item-from-a', targetHandle: 'item' };
    const itemFromB = { ...createEdge(nodeBId, collect.id), id: 'item-from-b', targetHandle: 'item' };
    const next = projectGraphReducer(doc, {
      edges: [fromA, fromB, fromGone, itemFromA, itemFromB],
      nodes: [pasted, collect],
      type: 'addGraphElements',
    });

    // The first edge into `a` wins; a collect's `item` keeps every connection; a missing source is dropped.
    expect(next.edges.map((edge) => edge.id)).toEqual(['edge-from-a', 'item-from-a', 'item-from-b']);
  });

  it('connecting an already-connected input replaces the existing edge', () => {
    const { doc, nodeAId, nodeBId } = createDocWithNodes();
    let next = projectGraphReducer(doc, { edge: createEdge(nodeAId, nodeBId), type: 'addEdge' });
    const replacement = { ...createEdge(nodeAId, nodeBId), id: 'edge-replacement' };

    next = projectGraphReducer(next, { edge: replacement, type: 'addEdge' });

    expect(next.edges).toHaveLength(1);
    expect(next.edges[0]?.id).toBe('edge-replacement');
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
});

describe('seed modes', () => {
  it('advances only fields still holding the planned value under the planned mode', () => {
    const { doc, nodeAId, nodeBId } = createDocWithNodes();
    let next = projectGraphReducer(doc, {
      fieldName: 'a',
      nodeId: nodeAId,
      seedMode: 'increment',
      type: 'setFieldSeedMode',
    });
    next = projectGraphReducer(next, {
      fieldName: 'a',
      nodeId: nodeBId,
      seedMode: 'increment',
      type: 'setFieldSeedMode',
    });
    // B was edited after the plan was captured, so its advance is stale.
    next = projectGraphReducer(next, { fieldName: 'a', nodeId: nodeBId, type: 'setFieldValue', value: 50 });

    const advanced = projectGraphReducer(next, {
      advances: [
        { fieldName: 'a', fromSeed: 1, nodeId: nodeAId, seedMode: 'increment', toSeed: 4 },
        { fieldName: 'a', fromSeed: 1, nodeId: nodeBId, seedMode: 'increment', toSeed: 4 },
        { fieldName: 'a', fromSeed: 4, nodeId: 'missing', seedMode: 'increment', toSeed: 5 },
      ],
      type: 'advanceSeedFields',
    });

    expect(advanced.nodes[0]).toMatchObject({ data: { inputs: { a: { seedMode: 'increment', value: 4 } } } });
    expect(advanced.nodes[1]).toMatchObject({ data: { inputs: { a: { seedMode: 'increment', value: 50 } } } });

    // A mode switched since the plan keeps the value the user now expects to hold.
    const refixed = projectGraphReducer(next, {
      fieldName: 'a',
      nodeId: nodeAId,
      seedMode: 'fixed',
      type: 'setFieldSeedMode',
    });
    const held = projectGraphReducer(refixed, {
      advances: [{ fieldName: 'a', fromSeed: 1, nodeId: nodeAId, seedMode: 'increment', toSeed: 4 }],
      type: 'advanceSeedFields',
    });

    expect(held).toBe(refixed);
  });
});

describe('getProjectGraphUndoEntry', () => {
  it('keys streamed edits per field so a burst folds into one step, and leaves moves out of history', () => {
    expect(getProjectGraphUndoEntry({ fieldName: 'a', nodeId: 'n', type: 'setFieldValue', value: 1 })).toEqual({
      label: 'Edit workflow field value',
      mergeKey: 'setFieldValue:n:a',
    });
    expect(getProjectGraphUndoEntry({ fieldName: 'b', nodeId: 'n', type: 'setFieldValue', value: 1 })?.mergeKey).toBe(
      'setFieldValue:n:b'
    );
    expect(getProjectGraphUndoEntry({ fieldName: 'a', nodeId: 'n', type: 'setFieldValue', value: true })).toEqual({
      label: 'Edit workflow field value',
    });
    // Typing into a scalar list row streams like a scalar; picking images into a list does not.
    expect(
      getProjectGraphUndoEntry({ fieldName: 'a', nodeId: 'n', type: 'setFieldValue', value: [1, null, 'x'] })?.mergeKey
    ).toBe('setFieldValue:n:a');
    expect(
      getProjectGraphUndoEntry({ fieldName: 'a', nodeId: 'n', type: 'setFieldValue', value: [{ image_name: 'a' }] })
        ?.mergeKey
    ).toBeUndefined();
    // Typing generator settings folds per variant; switching variants starts a new step.
    expect(
      getProjectGraphUndoEntry({
        fieldName: 'generator',
        nodeId: 'n',
        type: 'setFieldValue',
        value: { input: 'a,b', splitOn: ',', type: 'string_generator_parse_string' },
      })?.mergeKey
    ).toBe('setFieldValue:n:generator:string_generator_parse_string');
    expect(
      getProjectGraphUndoEntry({ fieldName: 'workflow_id', nodeId: 'n', type: 'setFieldValue', value: 'workflow-a' })
    ).toEqual({
      label: 'Edit workflow field value',
    });
    expect(getProjectGraphUndoEntry({ nodeId: 'n', type: 'setNodeUseCache', useCache: false })).toEqual({
      label: 'Change workflow node caching',
    });
    expect(getProjectGraphUndoEntry({ nodeId: 'n', position: { x: 1, y: 1 }, type: 'setNodePosition' })).toBeNull();
    expect(getProjectGraphUndoEntry({ isOpen: false, nodeId: 'n', type: 'setNodeIsOpen' })).toBeNull();
  });
});

describe('node updates', () => {
  const input = (
    name: string,
    overrides: Partial<InvocationTemplate['inputs'][string]> = {}
  ): InvocationTemplate['inputs'][string] => ({
    ...(template.inputs.a as InvocationTemplate['inputs'][string]),
    name,
    title: name.toUpperCase(),
    ...overrides,
  });
  const templateV110: InvocationTemplate = {
    ...template,
    inputs: { a: input('a'), b: input('b', { default: 5 }) },
    version: '1.1.0',
  };
  const oldNode = (): WorkflowInvocationNode => ({
    ...buildInvocationNode(template, { x: 4, y: 8 }),
    data: {
      ...buildInvocationNode(template, { x: 4, y: 8 }).data,
      inputs: {
        a: { label: 'Custom A', name: 'a', value: 3 },
        z: { label: '', name: 'z', value: 'obsolete' },
      },
      label: 'My add',
      notes: 'keep me',
      useCache: false,
    },
    id: 'add-1',
  });

  it('merges fresh defaults under the stored node, drops inputs the template lost, and cleans their edges and form elements', () => {
    const node = oldNode();
    let doc: ProjectGraphState = {
      ...createProjectGraph('update-test'),
      edges: [
        { id: 'e-a', source: 'src', sourceHandle: 'value', target: 'add-1', targetHandle: 'a', type: 'default' },
        { id: 'e-z', source: 'src', sourceHandle: 'value', target: 'add-1', targetHandle: 'z', type: 'default' },
      ],
      nodes: [buildInvocationNode({ ...template, type: 'src' }, { x: 0, y: 0 }), node],
    };

    doc = { ...doc, nodes: doc.nodes.map((n) => (n.id === doc.nodes[0]?.id ? { ...n, id: 'src' } : n)) };
    doc = projectGraphReducer(doc, { fieldIdentifier: { fieldName: 'a', nodeId: 'add-1' }, type: 'exposeField' });
    doc = projectGraphReducer(doc, { fieldIdentifier: { fieldName: 'z', nodeId: 'add-1' }, type: 'exposeField' });

    const update = updateWorkflowNodes(doc, { add: templateV110 });
    const updated = update.document.nodes.find((n) => n.id === 'add-1');

    expect(update.updatedNodeIds).toEqual(['add-1']);
    expect(update.skippedNodeIds).toEqual([]);
    expect(update.droppedEdgeIds).toEqual(['e-z']);
    expect(update.droppedFormElementIds).toHaveLength(1);
    expect(updated?.type === 'invocation' && updated.data).toMatchObject({
      inputs: {
        a: { label: 'Custom A', name: 'a', value: 3 },
        b: { label: '', name: 'b', value: 5 },
      },
      label: 'My add',
      notes: 'keep me',
      useCache: false,
      version: '1.1.0',
    });
    expect(updated?.type === 'invocation' ? Object.keys(updated.data.inputs) : []).toEqual(['a', 'b']);
    expect(updated?.position).toEqual({ x: 4, y: 8 });
    expect(update.document.edges.map((edge) => edge.id)).toEqual(['e-a']);
    expect(isFieldExposed(update.document.form, { fieldName: 'a', nodeId: 'add-1' })).toBe(true);
    expect(isFieldExposed(update.document.form, { fieldName: 'z', nodeId: 'add-1' })).toBe(false);
  });

  it('skips nodes newer than or a major away from their template and returns the same document when nothing changes', () => {
    const newer: WorkflowInvocationNode = { ...oldNode(), data: { ...oldNode().data, version: '1.5.0' }, id: 'newer' };
    const major: WorkflowInvocationNode = { ...oldNode(), data: { ...oldNode().data, version: '0.9.0' }, id: 'major' };
    const current: WorkflowInvocationNode = {
      ...oldNode(),
      data: { ...oldNode().data, version: '1.1.0' },
      id: 'current',
    };
    const doc = { ...createProjectGraph('update-test'), nodes: [newer, major, current] };
    const update = updateWorkflowNodes(doc, { add: templateV110 });

    expect(update.document).toBe(doc);
    expect(update.updatedNodeIds).toEqual([]);
    expect(update.skippedNodeIds).toEqual(['newer', 'major']);
    expect(updateWorkflowNodes(doc, {}).document).toBe(doc);
  });

  it('moves an old image_collection list from collection to images unless collection is connected', () => {
    const imageList = { batch: false, cardinality: 'COLLECTION' as const, name: 'ImageField' };
    const collectionTemplate: InvocationTemplate = {
      ...template,
      inputs: {
        collection: input('collection', { default: undefined, input: 'connection', type: imageList }),
        images: input('images', { default: [], type: imageList }),
      },
      type: 'image_collection',
      version: '1.0.2',
    };
    const images = [{ image_name: 'a.png' }];
    const node = (id: string): WorkflowInvocationNode => ({
      ...oldNode(),
      data: {
        ...oldNode().data,
        inputs: { collection: { label: '', name: 'collection', value: images } },
        type: 'image_collection',
        version: '1.0.1',
      },
      id,
    });
    let doc: ProjectGraphState = {
      ...createProjectGraph('update-test'),
      edges: [
        { id: 'e', source: 'src', sourceHandle: 'x', target: 'wired', targetHandle: 'collection', type: 'default' },
      ],
      nodes: [node('free'), node('wired')],
    };

    doc = projectGraphReducer(doc, {
      fieldIdentifier: { fieldName: 'collection', nodeId: 'free' },
      type: 'exposeField',
    });

    const update = updateWorkflowNodes(doc, { image_collection: collectionTemplate });
    const free = update.document.nodes.find((n) => n.id === 'free');
    const wired = update.document.nodes.find((n) => n.id === 'wired');

    expect(free?.type === 'invocation' ? free.data.inputs : {}).toMatchObject({
      collection: { value: undefined },
      images: { value: images },
    });
    expect(wired?.type === 'invocation' ? wired.data.inputs : {}).toMatchObject({
      collection: { value: images },
      images: { value: [] },
    });
    expect(isFieldExposed(update.document.form, { fieldName: 'images', nodeId: 'free' })).toBe(true);
    expect(isFieldExposed(update.document.form, { fieldName: 'collection', nodeId: 'free' })).toBe(false);
  });

  it('turns a stored minimax_h3_denoise frame count into the matching choice or the default', () => {
    const framesTemplate: InvocationTemplate = {
      ...template,
      inputs: {
        num_frames: input('num_frames', {
          default: '124',
          options: ['90', '124'],
          type: { batch: false, cardinality: 'SINGLE', name: 'EnumField' },
        }),
      },
      type: 'minimax_h3_denoise',
      version: '1.4.0',
    };
    const node = (id: string, frames: number): WorkflowInvocationNode => ({
      ...oldNode(),
      data: {
        ...oldNode().data,
        inputs: { num_frames: { label: '', name: 'num_frames', value: frames } },
        type: 'minimax_h3_denoise',
        version: '1.2.0',
      },
      id,
    });
    const update = updateWorkflowNodes(
      { ...createProjectGraph('update-test'), nodes: [node('on-grid', 90), node('off-grid', 100)] },
      { minimax_h3_denoise: framesTemplate }
    );
    const value = (id: string) => {
      const n = update.document.nodes.find((candidate) => candidate.id === id);

      return n?.type === 'invocation' ? n.data.inputs.num_frames?.value : null;
    };

    expect(value('on-grid')).toBe('90');
    expect(value('off-grid')).toBe('124');
  });
});
