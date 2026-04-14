import { deepClone } from 'common/util/deepClone';
import { CONNECTOR_INPUT_HANDLE, CONNECTOR_OUTPUT_HANDLE } from 'features/nodes/store/util/connectorTopology';
import { add, buildEdge, buildNode, img_resize, sub, templates } from 'features/nodes/store/util/testUtils';
import { describe, expect, it } from 'vitest';

import { buildNodesGraph } from './buildNodesGraph';

const buildConnectorNode = (id: string) => ({
  id,
  type: 'connector' as const,
  position: { x: 0, y: 0 },
  data: {
    id,
    type: 'connector' as const,
    label: 'Connector',
    isOpen: true,
  },
});

const buildState = (nodes: unknown[], edges: unknown[]) =>
  ({
    nodes: {
      present: {
        _version: 1,
        nodes,
        edges,
        formFieldInitialValues: {},
        id: undefined,
        name: '',
        author: '',
        description: '',
        version: '',
        contact: '',
        tags: '',
        notes: '',
        exposedFields: [],
        meta: { version: '4.0.0', category: 'user' },
        form: {
          rootElementId: 'root',
          elements: {
            root: {
              id: 'root',
              type: 'container',
              data: { layout: 'column', children: [] },
            },
          },
        },
      },
    },
    gallery: {
      autoAddBoardId: 'none',
      selection: [],
    },
  }) as unknown as Parameters<typeof buildNodesGraph>[0];

describe('buildNodesGraph', () => {
  it('flattens a single connector to one direct execution edge', () => {
    const source = buildNode(add);
    const target = buildNode(sub);
    const connector = buildConnectorNode('connector-1');
    const state = buildState(
      [source, target, connector],
      [
        buildEdge(source.id, 'value', connector.id, CONNECTOR_INPUT_HANDLE),
        buildEdge(connector.id, CONNECTOR_OUTPUT_HANDLE, target.id, 'a'),
      ]
    );

    const graph = buildNodesGraph(state, templates);

    expect(graph.nodes).not.toHaveProperty(connector.id);
    expect(graph.edges).toEqual([
      {
        source: { node_id: source.id, field: 'value' },
        destination: { node_id: target.id, field: 'a' },
      },
    ]);
  });

  it('flattens chained connectors transitively', () => {
    const source = buildNode(add);
    const target = buildNode(sub);
    const connectorA = buildConnectorNode('connector-a');
    const connectorB = buildConnectorNode('connector-b');
    const state = buildState(
      [source, target, connectorA, connectorB],
      [
        buildEdge(source.id, 'value', connectorA.id, CONNECTOR_INPUT_HANDLE),
        buildEdge(connectorA.id, CONNECTOR_OUTPUT_HANDLE, connectorB.id, CONNECTOR_INPUT_HANDLE),
        buildEdge(connectorB.id, CONNECTOR_OUTPUT_HANDLE, target.id, 'a'),
      ]
    );

    const graph = buildNodesGraph(state, templates);

    expect(graph.edges).toEqual([
      {
        source: { node_id: source.id, field: 'value' },
        destination: { node_id: target.id, field: 'a' },
      },
    ]);
  });

  it('fans out through a connector into multiple execution edges', () => {
    const source = buildNode(add);
    const targetA = buildNode(sub);
    const targetB = buildNode(img_resize);
    const connector = buildConnectorNode('connector-1');
    const state = buildState(
      [source, targetA, targetB, connector],
      [
        buildEdge(source.id, 'value', connector.id, CONNECTOR_INPUT_HANDLE),
        buildEdge(connector.id, CONNECTOR_OUTPUT_HANDLE, targetA.id, 'a'),
        buildEdge(connector.id, CONNECTOR_OUTPUT_HANDLE, targetB.id, 'width'),
      ]
    );

    const graph = buildNodesGraph(state, templates);

    expect(graph.edges).toEqual([
      {
        source: { node_id: source.id, field: 'value' },
        destination: { node_id: targetA.id, field: 'a' },
      },
      {
        source: { node_id: source.id, field: 'value' },
        destination: { node_id: targetB.id, field: 'width' },
      },
    ]);
  });

  it('drops unresolved connector paths from the execution graph', () => {
    const target = buildNode(sub);
    const connector = buildConnectorNode('connector-1');
    const state = buildState([target, connector], [buildEdge(connector.id, CONNECTOR_OUTPUT_HANDLE, target.id, 'a')]);

    const graph = buildNodesGraph(state, templates);

    expect(graph.nodes).not.toHaveProperty(connector.id);
    expect(graph.edges).toEqual([]);
  });

  it('deduplicates effective execution edges created by flattening', () => {
    const source = buildNode(add);
    const target = buildNode(sub);
    const connector = buildConnectorNode('connector-1');
    const state = buildState(
      [source, target, connector],
      [
        buildEdge(source.id, 'value', connector.id, CONNECTOR_INPUT_HANDLE),
        buildEdge(connector.id, CONNECTOR_OUTPUT_HANDLE, target.id, 'a'),
        buildEdge(source.id, 'value', target.id, 'a'),
      ]
    );

    const graph = buildNodesGraph(state, templates);

    expect(graph.edges).toEqual([
      {
        source: { node_id: source.id, field: 'value' },
        destination: { node_id: target.id, field: 'a' },
      },
    ]);
  });

  it('still omits explicit destination input values when the flattened edge exists', () => {
    const source = buildNode(add);
    const target = deepClone(buildNode(sub));
    const connector = buildConnectorNode('connector-1');
    const inputA = target.data.inputs.a;
    expect(inputA).toBeDefined();
    if (!inputA) {
      throw new Error('Missing input a');
    }
    inputA.value = 'not-an-integer' as never;
    const state = buildState(
      [source, target, connector],
      [
        buildEdge(source.id, 'value', connector.id, CONNECTOR_INPUT_HANDLE),
        buildEdge(connector.id, CONNECTOR_OUTPUT_HANDLE, target.id, 'a'),
      ]
    );

    const graph = buildNodesGraph(state, templates);

    expect(graph.edges).toEqual([
      {
        source: { node_id: source.id, field: 'value' },
        destination: { node_id: target.id, field: 'a' },
      },
    ]);
    expect(graph.nodes[target.id]).not.toHaveProperty('a');
  });
});
