import type { AnyNode, ConnectorNode } from 'features/nodes/types/invocation';
import { describe, expect, it } from 'vitest';

import {
  CONNECTOR_INPUT_HANDLE,
  CONNECTOR_OUTPUT_HANDLE,
  getConnectorDeletionSpliceConnections,
  getConnectorInputEdge,
  getConnectorOutputEdges,
  resolveConnectorSource,
  resolveConnectorSourceFieldType,
} from './connectorTopology';
import { add, buildEdge, buildNode, img_resize, sub, templates } from './testUtils';

const buildConnectorNode = (id: string): ConnectorNode => ({
  id,
  type: 'connector',
  position: { x: 0, y: 0 },
  data: {
    id,
    type: 'connector',
    label: 'Connector',
    isOpen: true,
  },
});

describe('connectorTopology', () => {
  it('resolves the effective upstream source through one connector', () => {
    const source = buildNode(add);
    const connector = buildConnectorNode('connector-1');
    const target = buildNode(sub);
    const nodes: AnyNode[] = [source, connector, target];
    const edges = [
      buildEdge(source.id, 'value', connector.id, CONNECTOR_INPUT_HANDLE),
      buildEdge(connector.id, CONNECTOR_OUTPUT_HANDLE, target.id, 'a'),
    ];

    expect(resolveConnectorSource(connector.id, nodes, edges)).toEqual({
      nodeId: source.id,
      fieldName: 'value',
    });
    expect(resolveConnectorSourceFieldType(connector.id, nodes, edges, templates)).toEqual(add.outputs.value?.type);
  });

  it('resolves the effective upstream source through chained connectors', () => {
    const source = buildNode(add);
    const connectorA = buildConnectorNode('connector-a');
    const connectorB = buildConnectorNode('connector-b');
    const nodes: AnyNode[] = [source, connectorA, connectorB];
    const edges = [
      buildEdge(source.id, 'value', connectorA.id, CONNECTOR_INPUT_HANDLE),
      buildEdge(connectorA.id, CONNECTOR_OUTPUT_HANDLE, connectorB.id, CONNECTOR_INPUT_HANDLE),
    ];

    expect(resolveConnectorSource(connectorB.id, nodes, edges)).toEqual({
      nodeId: source.id,
      fieldName: 'value',
    });
  });

  it('returns no source or type for an unresolved connector chain', () => {
    const connectorA = buildConnectorNode('connector-a');
    const connectorB = buildConnectorNode('connector-b');
    const nodes: AnyNode[] = [connectorA, connectorB];
    const edges = [buildEdge(connectorA.id, CONNECTOR_OUTPUT_HANDLE, connectorB.id, CONNECTOR_INPUT_HANDLE)];

    expect(resolveConnectorSource(connectorB.id, nodes, edges)).toBe(null);
    expect(resolveConnectorSourceFieldType(connectorB.id, nodes, edges, templates)).toBe(null);
  });

  it('enumerates multiple outgoing edges for a connector', () => {
    const source = buildNode(add);
    const connector = buildConnectorNode('connector-1');
    const targetA = buildNode(sub);
    const targetB = buildNode(img_resize);
    const incoming = buildEdge(source.id, 'value', connector.id, CONNECTOR_INPUT_HANDLE);
    const outgoingA = buildEdge(connector.id, CONNECTOR_OUTPUT_HANDLE, targetA.id, 'a');
    const outgoingB = buildEdge(connector.id, CONNECTOR_OUTPUT_HANDLE, targetB.id, 'width');
    const edges = [incoming, outgoingA, outgoingB];

    expect(getConnectorInputEdge(connector.id, edges)).toEqual(incoming);
    expect(getConnectorOutputEdges(connector.id, edges)).toEqual([outgoingA, outgoingB]);
  });

  it('rejects connector deletion splice-through when any downstream target would be invalid', () => {
    const source = buildNode(add);
    const connector = buildConnectorNode('connector-1');
    const target = buildNode(img_resize);
    const nodes: AnyNode[] = [source, connector, target];
    const edges = [
      buildEdge(source.id, 'value', connector.id, CONNECTOR_INPUT_HANDLE),
      buildEdge(connector.id, CONNECTOR_OUTPUT_HANDLE, target.id, 'image'),
    ];

    expect(getConnectorDeletionSpliceConnections(connector.id, nodes, edges, templates)).toBe(null);
  });

  it('builds connector deletion splice-through edges when every downstream target remains valid', () => {
    const source = buildNode(add);
    const connector = buildConnectorNode('connector-1');
    const target = buildNode(sub);
    const nodes: AnyNode[] = [source, connector, target];
    const edges = [
      buildEdge(source.id, 'value', connector.id, CONNECTOR_INPUT_HANDLE),
      buildEdge(connector.id, CONNECTOR_OUTPUT_HANDLE, target.id, 'a'),
    ];

    expect(getConnectorDeletionSpliceConnections(connector.id, nodes, edges, templates)).toEqual([
      {
        source: source.id,
        sourceHandle: 'value',
        target: target.id,
        targetHandle: 'a',
      },
    ]);
  });

  it('returns no splice-through edges when a connector has downstream targets but no upstream source', () => {
    const connector = buildConnectorNode('connector-1');
    const target = buildNode(sub);
    const nodes: AnyNode[] = [connector, target];
    const edges = [buildEdge(connector.id, CONNECTOR_OUTPUT_HANDLE, target.id, 'a')];

    expect(getConnectorDeletionSpliceConnections(connector.id, nodes, edges, templates)).toBe(null);
  });
});
