import type {
  FieldType,
  InvocationTemplate,
  InvocationTemplates,
  ProjectGraphState,
  WorkflowConnectorNode,
  WorkflowEdge,
  WorkflowInvocationNode,
} from '@features/workflow/contracts';

import {
  buildCurrentImageNode,
  CONNECTOR_INPUT_HANDLE,
  CONNECTOR_OUTPUT_HANDLE,
  createProjectGraph,
} from '@features/workflow/utility';
import { describe, expect, it } from 'vitest';

import { getWorkflowEdgeData, toFlowEdges, toFlowNodes, withNodeSelection } from './flowAdapters';

const single = (name: string): FieldType => ({ batch: false, cardinality: 'SINGLE', name });
const collection = (name: string): FieldType => ({ batch: false, cardinality: 'COLLECTION', name });

const createTemplate = (outputType: FieldType = single('IntegerField')): InvocationTemplate => ({
  category: 'test',
  classification: 'stable',
  description: '',
  inputs: {
    a: {
      default: undefined,
      description: '',
      exclusiveMaximum: null,
      exclusiveMinimum: null,
      input: 'any',
      maximum: null,
      minimum: null,
      multipleOf: null,
      name: 'a',
      options: null,
      required: true,
      title: 'A',
      type: outputType,
      uiChoiceLabels: null,
      uiComponent: null,
      uiHidden: false,
      uiModelBase: null,
      uiModelType: null,
      uiOrder: null,
    },
  },
  nodePack: 'invokeai',
  outputType: 'test',
  outputs: {
    value: { description: '', name: 'value', title: 'Value', type: outputType },
  },
  tags: [],
  title: 'Add',
  type: 'add',
  useCache: true,
  version: '1.0.0',
});

const createTemplates = (outputType?: FieldType): InvocationTemplates => ({ add: createTemplate(outputType) });

const createNode = (id: string): WorkflowInvocationNode => ({
  data: {
    inputs: {},
    isIntermediate: true,
    isOpen: true,
    label: '',
    nodePack: 'invokeai',
    notes: '',
    type: 'add',
    useCache: true,
    version: '1.0.0',
  },
  id,
  position: { x: 0, y: 0 },
  type: 'invocation',
});

const createEdge = (
  id: string,
  source: string,
  target: string,
  sourceHandle = 'value',
  targetHandle = 'a'
): WorkflowEdge => ({
  id,
  source,
  sourceHandle,
  target,
  targetHandle,
  type: 'default',
});

const createConnector = (id: string): WorkflowConnectorNode => ({
  data: { label: '' },
  id,
  position: { x: 0, y: 0 },
  type: 'connector',
});

const createDoc = (overrides?: Partial<ProjectGraphState>): ProjectGraphState => ({
  ...createProjectGraph('adapters-test'),
  edges: [createEdge('e1', 'a', 'b')],
  nodes: [createNode('a'), createNode('b')],
  ...overrides,
});

describe('flowAdapters identity preservation', () => {
  it('reuses unchanged node objects across rebuilds (memo-friendly)', () => {
    const doc = createDoc();
    const first = toFlowNodes(doc);
    const second = toFlowNodes(doc, first);

    expect(second[0]).toBe(first[0]);
    expect(second[1]).toBe(first[1]);
  });

  it('replaces only the node whose document node changed', () => {
    const doc = createDoc();
    const first = toFlowNodes(doc);
    const movedNodeB = { ...doc.nodes[1]!, position: { x: 50, y: 50 } };
    const second = toFlowNodes({ ...doc, nodes: [doc.nodes[0]!, movedNodeB] }, first);

    expect(second[0]).toBe(first[0]);
    expect(second[1]).not.toBe(first[1]);
  });

  it('recomputes a node when its incoming connections change', () => {
    const doc = createDoc();
    const first = toFlowNodes(doc);
    // Drop the edge into node b; its connectedTargetHandles must update.
    const second = toFlowNodes({ ...doc, edges: [] }, first);
    const nodeB = second[1];

    expect(nodeB?.type === 'invocation' && nodeB.data.connectedTargetHandles).toEqual([]);
    expect(second[1]).not.toBe(first[1]);
    // Node a lost its outgoing handle metadata, so it must also update.
    expect(second[0]).not.toBe(first[0]);
    expect(second[0]?.type === 'invocation' && second[0].data.connectedSourceHandles).toEqual([]);
  });

  it('precomputes connected source handles onto invocation node data', () => {
    const doc = createDoc();
    const nodes = toFlowNodes(doc);
    const source = nodes[0];
    const target = nodes[1];

    expect(source?.type === 'invocation' ? source.data.connectedSourceHandles : []).toEqual(['value']);
    expect(target?.type === 'invocation' ? target.data.connectedSourceHandles : []).toEqual([]);
  });

  it('precomputes exposed field names onto invocation node data', () => {
    const doc = createDoc();
    const exposed = {
      ...doc,
      form: {
        ...doc.form,
        elements: {
          ...doc.form.elements,
          'field-1': {
            data: { fieldIdentifier: { fieldName: 'a', nodeId: 'b' }, showDescription: false },
            id: 'field-1',
            parentId: doc.form.rootElementId,
            type: 'node-field' as const,
          },
        },
      },
    };
    const nodes = toFlowNodes(exposed);
    const nodeB = nodes[1];

    expect(nodeB?.type === 'invocation' && nodeB.data.exposedFieldNames).toEqual(['a']);
  });

  it('marks invocation nodes compact for large workflow rendering', () => {
    const doc = createDoc();
    const normal = toFlowNodes(doc);
    const compact = toFlowNodes(doc, normal, undefined, undefined, true);
    const compactAgain = toFlowNodes(doc, compact, undefined, undefined, true);

    expect(normal[0]?.type === 'invocation' && normal[0].data.isCompact).toBe(false);
    expect(compact[0]?.type === 'invocation' && compact[0].data.isCompact).toBe(true);
    expect(compact[0]).not.toBe(normal[0]);
    expect(compactAgain[0]).toBe(compact[0]);
  });

  it('precomputes resolved connector field types onto connector node data', () => {
    const fieldType = single('ImageField');
    const templates = createTemplates(fieldType);
    const doc = createDoc({
      edges: [
        createEdge('e1', 'a', 'c', 'value', CONNECTOR_INPUT_HANDLE),
        createEdge('e2', 'c', 'b', CONNECTOR_OUTPUT_HANDLE, 'a'),
      ],
      nodes: [createNode('a'), createConnector('c'), createNode('b')],
    });
    const untyped = toFlowNodes(doc);
    const typed = toFlowNodes(doc, untyped, templates);
    const connector = typed[1];

    expect(connector?.type).toBe('connector');
    expect(connector?.type === 'connector' && connector.data.inputFieldType).toBe(fieldType);
    expect(connector?.type === 'connector' && connector.data.outputFieldType).toBe(fieldType);
    expect(connector).not.toBe(untyped[1]);
    expect(toFlowNodes(doc, typed, templates)[1]).toBe(connector);
  });

  it('precomputes stable invocation template views onto invocation node data', () => {
    const doc = createDoc();
    const templates = createTemplates();
    const first = toFlowNodes(doc, [], templates);
    const second = toFlowNodes(doc, first, templates);
    const node = first[0];

    expect(node?.type === 'invocation' ? node.data.template?.template.title : null).toBe('Add');
    expect(node?.type === 'invocation' ? node.data.template?.inputTemplates.map((input) => input.name) : []).toEqual([
      'a',
    ]);
    expect(second[0]).toBe(first[0]);
  });

  it('does not mark a target as connected through an unresolved connector output', () => {
    const doc = createDoc({
      edges: [createEdge('e1', 'c', 'b', CONNECTOR_OUTPUT_HANDLE, 'a')],
      nodes: [createConnector('c'), createNode('b')],
    });
    const nodes = toFlowNodes(doc);
    const target = nodes.find((node) => node.id === 'b');

    expect(target?.type).toBe('invocation');
    expect(target?.type === 'invocation' ? target.data.connectedTargetHandles : []).toEqual([]);
  });

  it('carries selection across rebuilds and toggles it without copying unchanged nodes', () => {
    const doc = createDoc();
    const selected = withNodeSelection(toFlowNodes(doc), new Set(['a']));

    expect(selected[0]?.selected).toBe(true);

    const rebuilt = toFlowNodes(doc, selected);

    expect(rebuilt[0]?.selected).toBe(true);

    // Re-applying the same selection set returns identical node objects.
    const reselected = withNodeSelection(selected, new Set(['a']));

    expect(reselected[0]).toBe(selected[0]);
  });

  it('builds current_image nodes and reuses them when unchanged', () => {
    const currentImage = buildCurrentImageNode({ x: 0, y: 0 });
    const doc = createDoc({ nodes: [createNode('a'), currentImage] });
    const first = toFlowNodes(doc);

    expect(first[1]?.type).toBe('current_image');
    expect(toFlowNodes(doc, first)[1]).toBe(first[1]);
  });

  it('reuses unchanged edge objects but replaces edges on type change', () => {
    const doc = createDoc();
    const first = toFlowEdges(doc, [], 'default');
    const same = toFlowEdges(doc, first, 'default');

    expect(same[0]).toBe(first[0]);

    const restyled = toFlowEdges(doc, first, 'step');

    expect(restyled[0]).not.toBe(first[0]);
    expect(restyled[0]?.type).toBe('step');
  });

  it('adds field type metadata for custom edge rendering', () => {
    const doc = createDoc();
    const edge = toFlowEdges(doc, [], 'default', new Set(), createTemplates())[0];

    expect(edge?.data).toMatchObject({
      fieldTypeLabel: 'Integer',
      pathType: 'default',
      stroke: '#f87171',
      strokeWidth: 2,
      tooltip: 'Integer',
    });
  });

  it('uses line pattern metadata for collection field types', () => {
    const doc = createDoc();
    const data = getWorkflowEdgeData(doc, doc.edges[0]!, 'step', createTemplates(collection('ImageField')));

    expect(data).toMatchObject({
      fieldTypeLabel: 'Image Collection',
      pathType: 'step',
      stroke: '#c4b5fd',
      strokeDasharray: '8 4',
      strokeWidth: 2.5,
      tooltip: 'Image Collection',
    });
  });

  it('replaces edges when templates resolve their field type styling', () => {
    const doc = createDoc();
    const untyped = toFlowEdges(doc, [], 'default');
    const typed = toFlowEdges(doc, untyped, 'default', new Set(), createTemplates());

    expect(typed[0]).not.toBe(untyped[0]);
    expect(typed[0]?.data?.tooltip).toBe('Integer');
  });

  it('styles edges connected to selected nodes without persisting the highlight after deselection', () => {
    const doc = createDoc();
    const highlighted = toFlowEdges(doc, [], 'default', new Set(['a']));

    expect(highlighted[0]?.animated).toBe(true);
    expect(highlighted[0]?.className).toBe('workflow-selected-node-edge');
    expect(highlighted[0]?.zIndex).toBe(1000);
    expect(highlighted[0]?.style).toEqual({ strokeWidth: 2 });
    expect(toFlowEdges(doc, highlighted, 'default', new Set(['a']))[0]).toBe(highlighted[0]);

    const cleared = toFlowEdges(doc, highlighted, 'default');

    expect(cleared[0]).not.toBe(highlighted[0]);
    expect(cleared[0]?.animated).toBeUndefined();
    expect(cleared[0]?.className).toBeUndefined();
    expect(cleared[0]?.zIndex).toBeUndefined();
    expect(cleared[0]?.style).toBeUndefined();
  });

  it('keeps selected-node edge styling but disables edge animation when motion is reduced', () => {
    const doc = createDoc();
    const highlighted = toFlowEdges(doc, [], 'default', new Set(['a']), undefined, true);

    expect(highlighted[0]?.animated).toBeUndefined();
    expect(highlighted[0]?.className).toBe('workflow-selected-node-edge');
    expect(highlighted[0]?.zIndex).toBe(1000);
    expect(highlighted[0]?.style).toEqual({ strokeWidth: 2 });
  });
});
