import { deepClone } from 'common/util/deepClone';
import { buildConnectorNode } from 'features/nodes/util/node/buildConnectorNode';
import { describe, expect, it } from 'vitest';

import { connectorInserted, nodesChanged, nodesSliceConfig } from './nodesSlice';
import { CONNECTOR_INPUT_HANDLE, CONNECTOR_OUTPUT_HANDLE } from './util/connectorTopology';
import { add, buildEdge, buildNode, sub } from './util/testUtils';

const buildFixedConnectorNode = (id: string) => {
  const connectorNode = buildConnectorNode({ x: 0, y: 0 });
  return {
    ...connectorNode,
    id,
    data: {
      ...connectorNode.data,
      id,
    },
  };
};

describe('nodesSlice connector actions', () => {
  it('splits a direct edge into source -> connector -> target edges when inserting a connector', () => {
    const source = buildNode(add);
    const target = buildNode(sub);
    const connector = buildFixedConnectorNode('connector-1');
    const directEdge = buildEdge(source.id, 'value', target.id, 'a');

    const initialState = deepClone(nodesSliceConfig.slice.reducer(undefined, { type: 'test/init' }));
    initialState.nodes = [source, target];
    initialState.edges = [directEdge];

    const nextState = nodesSliceConfig.slice.reducer(
      initialState,
      connectorInserted({
        edgeId: directEdge.id,
        connector,
      })
    );

    expect(nextState.nodes.map((node) => node.id)).toEqual([source.id, target.id, connector.id]);
    expect(nextState.edges).toEqual([
      buildEdge(source.id, 'value', connector.id, CONNECTOR_INPUT_HANDLE),
      buildEdge(connector.id, CONNECTOR_OUTPUT_HANDLE, target.id, 'a'),
    ]);
  });

  it('splices connector outputs back to the resolved upstream source when removed', () => {
    const source = buildNode(add);
    const target = buildNode(sub);
    const connector = buildFixedConnectorNode('connector-1');

    const initialState = deepClone(nodesSliceConfig.slice.reducer(undefined, { type: 'test/init' }));
    initialState.nodes = [source, connector, target];
    initialState.edges = [
      buildEdge(source.id, 'value', connector.id, CONNECTOR_INPUT_HANDLE),
      buildEdge(connector.id, CONNECTOR_OUTPUT_HANDLE, target.id, 'a'),
    ];

    const nextState = nodesSliceConfig.slice.reducer(
      initialState,
      nodesChanged([{ type: 'remove', id: connector.id }])
    );

    expect(nextState.nodes.map((node) => node.id)).toEqual([source.id, target.id]);
    expect(nextState.edges).toEqual([buildEdge(source.id, 'value', target.id, 'a')]);
  });

  it('splices one connector source back to multiple downstream targets when removed', () => {
    const source = buildNode(add);
    const targetA = buildNode(sub);
    const targetB = buildNode(sub);
    const connector = buildFixedConnectorNode('connector-1');

    const initialState = deepClone(nodesSliceConfig.slice.reducer(undefined, { type: 'test/init' }));
    initialState.nodes = [source, connector, targetA, targetB];
    initialState.edges = [
      buildEdge(source.id, 'value', connector.id, CONNECTOR_INPUT_HANDLE),
      buildEdge(connector.id, CONNECTOR_OUTPUT_HANDLE, targetA.id, 'a'),
      buildEdge(connector.id, CONNECTOR_OUTPUT_HANDLE, targetB.id, 'b'),
    ];

    const nextState = nodesSliceConfig.slice.reducer(
      initialState,
      nodesChanged([{ type: 'remove', id: connector.id }])
    );

    expect(nextState.nodes.map((node) => node.id)).toEqual([source.id, targetA.id, targetB.id]);
    expect(nextState.edges).toEqual([
      buildEdge(source.id, 'value', targetA.id, 'a'),
      buildEdge(source.id, 'value', targetB.id, 'b'),
    ]);
  });

  it('does not create any edges when removing a connector with no downstream targets', () => {
    const source = buildNode(add);
    const connector = buildFixedConnectorNode('connector-1');

    const initialState = deepClone(nodesSliceConfig.slice.reducer(undefined, { type: 'test/init' }));
    initialState.nodes = [source, connector];
    initialState.edges = [buildEdge(source.id, 'value', connector.id, CONNECTOR_INPUT_HANDLE)];

    const nextState = nodesSliceConfig.slice.reducer(
      initialState,
      nodesChanged([{ type: 'remove', id: connector.id }])
    );

    expect(nextState.nodes.map((node) => node.id)).toEqual([source.id]);
    expect(nextState.edges).toEqual([]);
  });

  it('removes a connector while preserving downstream connector edges in a chained splice case', () => {
    const source = buildNode(add);
    const connectorA = buildFixedConnectorNode('connector-a');
    const connectorB = buildFixedConnectorNode('connector-b');
    const target = buildNode(sub);

    const initialState = deepClone(nodesSliceConfig.slice.reducer(undefined, { type: 'test/init' }));
    initialState.nodes = [source, connectorA, connectorB, target];
    initialState.edges = [
      buildEdge(source.id, 'value', connectorA.id, CONNECTOR_INPUT_HANDLE),
      buildEdge(connectorA.id, CONNECTOR_OUTPUT_HANDLE, connectorB.id, CONNECTOR_INPUT_HANDLE),
      buildEdge(connectorB.id, CONNECTOR_OUTPUT_HANDLE, target.id, 'a'),
    ];

    const nextState = nodesSliceConfig.slice.reducer(
      initialState,
      nodesChanged([{ type: 'remove', id: connectorA.id }])
    );

    expect(nextState.nodes.map((node) => node.id)).toEqual([source.id, connectorB.id, target.id]);
    expect(nextState.edges).toHaveLength(2);
    expect(nextState.edges).toEqual(
      expect.arrayContaining([
        buildEdge(source.id, 'value', connectorB.id, CONNECTOR_INPUT_HANDLE),
        buildEdge(connectorB.id, CONNECTOR_OUTPUT_HANDLE, target.id, 'a'),
      ])
    );
  });

  it('splices connector edges when the connector is removed through generic node removal', () => {
    const source = buildNode(add);
    const target = buildNode(sub);
    const connector = buildFixedConnectorNode('connector-1');

    const initialState = deepClone(nodesSliceConfig.slice.reducer(undefined, { type: 'test/init' }));
    initialState.nodes = [source, connector, target];
    initialState.edges = [
      buildEdge(source.id, 'value', connector.id, CONNECTOR_INPUT_HANDLE),
      buildEdge(connector.id, CONNECTOR_OUTPUT_HANDLE, target.id, 'a'),
    ];

    const nextState = nodesSliceConfig.slice.reducer(
      initialState,
      nodesChanged([{ type: 'remove', id: connector.id }])
    );

    expect(nextState.nodes.map((node) => node.id)).toEqual([source.id, target.id]);
    expect(nextState.edges).toEqual([buildEdge(source.id, 'value', target.id, 'a')]);
  });
});
