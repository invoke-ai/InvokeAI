import { describe, expect, it } from 'vitest';

import type { InvocationTemplatesSnapshot } from './templates';
import type { FieldInputTemplate, InvocationTemplate, ProjectGraphState } from './types';

import { compileProjectGraph, getProjectGraphReadiness } from './buildGraph';
import {
  buildConnectorNode,
  buildInvocationNode,
  buildNotesNode,
  createProjectGraph,
  projectGraphReducer,
} from './document';

const input = (name: string, overrides: Partial<FieldInputTemplate> = {}): FieldInputTemplate => ({
  default: undefined,
  description: '',
  exclusiveMaximum: null,
  exclusiveMinimum: null,
  input: 'any',
  maximum: null,
  minimum: null,
  multipleOf: null,
  name,
  options: null,
  required: false,
  title: name,
  type: { batch: false, cardinality: 'SINGLE', name: 'StringField' },
  uiChoiceLabels: null,
  uiComponent: null,
  uiHidden: false,
  uiModelBase: null,
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
    out: {
      description: '',
      name: 'out',
      title: 'Out',
      type: { batch: false, cardinality: 'SINGLE', name: 'StringField' },
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
  sink: template('sink', {
    board: input('board', { type: { batch: false, cardinality: 'SINGLE', name: 'BoardField' } }),
    text: input('text', { required: true }),
  }),
  source: template('source', { value: input('value', { default: 'hello', required: true }) }),
};

const loadedSnapshot: InvocationTemplatesSnapshot = { error: null, status: 'loaded', templates };

const buildDocument = (): { doc: ProjectGraphState; sinkId: string; sourceId: string } => {
  const sourceNode = buildInvocationNode(templates.source, { x: 0, y: 0 });
  const sinkNode = buildInvocationNode(templates.sink, { x: 10, y: 0 });
  let doc = createProjectGraph('compile-test');

  doc = projectGraphReducer(doc, { node: sourceNode, type: 'addNode' });
  doc = projectGraphReducer(doc, { node: sinkNode, type: 'addNode' });
  doc = projectGraphReducer(doc, { node: buildNotesNode({ x: 0, y: 100 }), type: 'addNode' });
  doc = projectGraphReducer(doc, {
    edge: {
      id: 'e1',
      source: sourceNode.id,
      sourceHandle: 'out',
      target: sinkNode.id,
      targetHandle: 'text',
      type: 'default',
    },
    type: 'addEdge',
  });

  return { doc, sinkId: sinkNode.id, sourceId: sourceNode.id };
};

describe('getProjectGraphReadiness', () => {
  it('requires loaded templates and at least one node', () => {
    const empty = createProjectGraph('empty');

    expect(getProjectGraphReadiness(empty, { error: null, status: 'loading', templates: {} }).reasons[0]).toMatch(
      /still loading/
    );
    expect(getProjectGraphReadiness(empty, loadedSnapshot).reasons[0]).toMatch(/no nodes/);
  });

  it('is ready when required inputs are connected or filled', () => {
    const { doc } = buildDocument();

    expect(getProjectGraphReadiness(doc, loadedSnapshot)).toEqual({ canInvoke: true, reasons: [] });
  });

  it('treats resolved connector output edges as connected required inputs', () => {
    const sourceNode = buildInvocationNode(templates.source, { x: 0, y: 0 });
    const sinkNode = buildInvocationNode(templates.sink, { x: 10, y: 0 });
    const connector = buildConnectorNode({ x: 5, y: 0 });
    let doc = createProjectGraph('connector-readiness');

    doc = projectGraphReducer(doc, { node: sourceNode, type: 'addNode' });
    doc = projectGraphReducer(doc, { node: sinkNode, type: 'addNode' });
    doc = projectGraphReducer(doc, { node: connector, type: 'addNode' });
    doc = projectGraphReducer(doc, {
      edge: {
        id: 'connector-in',
        source: sourceNode.id,
        sourceHandle: 'out',
        target: connector.id,
        targetHandle: 'in',
        type: 'default',
      },
      type: 'addEdge',
    });
    doc = projectGraphReducer(doc, {
      edge: {
        id: 'connector-out',
        source: connector.id,
        sourceHandle: 'out',
        target: sinkNode.id,
        targetHandle: 'text',
        type: 'default',
      },
      type: 'addEdge',
    });

    expect(getProjectGraphReadiness(doc, loadedSnapshot)).toEqual({ canInvoke: true, reasons: [] });
  });

  it('reports missing required inputs and unknown node types', () => {
    const { doc, sourceId } = buildDocument();
    const withEmptyValue = projectGraphReducer(doc, {
      fieldName: 'value',
      nodeId: sourceId,
      type: 'setFieldValue',
      value: '',
    });

    expect(getProjectGraphReadiness(withEmptyValue, loadedSnapshot).reasons[0]).toMatch(/missing required input/);

    const unknownTemplates: InvocationTemplatesSnapshot = {
      error: null,
      status: 'loaded',
      templates: { source: templates.source },
    };

    expect(getProjectGraphReadiness(doc, unknownTemplates).reasons[0]).toMatch(/Unknown node type "sink"/);
  });
});

describe('compileProjectGraph', () => {
  it('compiles nodes and edges, omitting notes nodes and connected direct values', () => {
    const { doc, sinkId, sourceId } = buildDocument();
    const withStaleValue = projectGraphReducer(doc, {
      fieldName: 'text',
      nodeId: sinkId,
      type: 'setFieldValue',
      value: 'stale direct value',
    });
    const compiled = compileProjectGraph(withStaleValue, templates);
    const backendGraph = compiled.backendGraph;

    expect(backendGraph).toBeDefined();
    expect(Object.keys(backendGraph?.nodes ?? {}).sort()).toEqual([sinkId, sourceId].sort());
    expect(backendGraph?.nodes[sourceId]).toMatchObject({ type: 'source', use_cache: true, value: 'hello' });
    // The connected input's direct value must not be sent alongside the edge.
    expect(backendGraph?.nodes[sinkId]).not.toHaveProperty('text');
    expect(backendGraph?.edges).toEqual([
      { destination: { field: 'text', node_id: sinkId }, source: { field: 'out', node_id: sourceId } },
    ]);
  });

  it('omits auto/none board sentinels and keeps explicit boards', () => {
    const { doc, sinkId } = buildDocument();
    const withAutoBoard = projectGraphReducer(doc, {
      fieldName: 'board',
      nodeId: sinkId,
      type: 'setFieldValue',
      value: 'auto',
    });

    expect(compileProjectGraph(withAutoBoard, templates).backendGraph?.nodes[sinkId]).not.toHaveProperty('board');

    const withExplicitBoard = projectGraphReducer(doc, {
      fieldName: 'board',
      nodeId: sinkId,
      type: 'setFieldValue',
      value: { board_id: 'board-1' },
    });

    expect(compileProjectGraph(withExplicitBoard, templates).backendGraph?.nodes[sinkId]).toMatchObject({
      board: { board_id: 'board-1' },
    });
  });

  it('resolves connector chains into executable backend edges', () => {
    const sourceNode = buildInvocationNode(templates.source, { x: 0, y: 0 });
    const sinkNode = buildInvocationNode(templates.sink, { x: 10, y: 0 });
    const connector = buildConnectorNode({ x: 5, y: 0 });
    let doc = createProjectGraph('connector-compile');

    doc = projectGraphReducer(doc, { node: sourceNode, type: 'addNode' });
    doc = projectGraphReducer(doc, { node: sinkNode, type: 'addNode' });
    doc = projectGraphReducer(doc, { node: connector, type: 'addNode' });
    doc = projectGraphReducer(doc, {
      edge: {
        id: 'connector-in',
        source: sourceNode.id,
        sourceHandle: 'out',
        target: connector.id,
        targetHandle: 'in',
        type: 'default',
      },
      type: 'addEdge',
    });
    doc = projectGraphReducer(doc, {
      edge: {
        id: 'connector-out',
        source: connector.id,
        sourceHandle: 'out',
        target: sinkNode.id,
        targetHandle: 'text',
        type: 'default',
      },
      type: 'addEdge',
    });

    expect(compileProjectGraph(doc, templates).backendGraph?.edges).toEqual([
      { destination: { field: 'text', node_id: sinkNode.id }, source: { field: 'out', node_id: sourceNode.id } },
    ]);
  });
});
