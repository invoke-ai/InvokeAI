import { deepClone } from 'common/util/deepClone';
import { unset } from 'es-toolkit/compat';
import { CONNECTOR_INPUT_HANDLE, CONNECTOR_OUTPUT_HANDLE } from 'features/nodes/store/util/connectorTopology';
import {
  getFirstValidConnection,
  getSourceCandidateFields,
  getTargetCandidateFields,
} from 'features/nodes/store/util/getFirstValidConnection';
import { add, buildEdge, buildNode, img_resize, sub, templates } from 'features/nodes/store/util/testUtils';
import { describe, expect, it } from 'vitest';

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

describe('getFirstValidConnection', () => {
  it('should return null if the pending and candidate nodes are the same node', () => {
    const n = buildNode(add);
    expect(getFirstValidConnection(n.id, 'value', n.id, null, [n], [], templates, null)).toBe(null);
  });

  it('should return null if the sourceHandle and targetHandle are null', () => {
    const n1 = buildNode(add);
    const n2 = buildNode(add);
    expect(getFirstValidConnection(n1.id, null, n2.id, null, [n1, n2], [], templates, null)).toBe(null);
  });

  it('should return itself if both sourceHandle and targetHandle are provided', () => {
    const n1 = buildNode(add);
    const n2 = buildNode(add);
    expect(getFirstValidConnection(n1.id, 'value', n2.id, 'a', [n1, n2], [], templates, null)).toEqual({
      source: n1.id,
      sourceHandle: 'value',
      target: n2.id,
      targetHandle: 'a',
    });
  });

  describe('connecting from a source to a target', () => {
    const n1 = buildNode(img_resize);
    const n2 = buildNode(img_resize);

    it('should return the first valid connection if there are no connected fields', () => {
      const r = getFirstValidConnection(n1.id, 'width', n2.id, null, [n1, n2], [], templates, null);
      const c = {
        source: n1.id,
        sourceHandle: 'width',
        target: n2.id,
        targetHandle: 'width',
      };
      expect(r).toEqual(c);
    });
    it('should return the first valid connection if there is a connected field', () => {
      const e = buildEdge(n1.id, 'height', n2.id, 'width');
      const r = getFirstValidConnection(n1.id, 'width', n2.id, null, [n1, n2], [e], templates, null);
      const c = {
        source: n1.id,
        sourceHandle: 'width',
        target: n2.id,
        targetHandle: 'height',
      };
      expect(r).toEqual(c);
    });
    it('should return the first valid connection if there is an edgePendingUpdate', () => {
      const e = buildEdge(n1.id, 'width', n2.id, 'width');
      const r = getFirstValidConnection(n1.id, 'width', n2.id, null, [n1, n2], [e], templates, e);
      const c = {
        source: n1.id,
        sourceHandle: 'width',
        target: n2.id,
        targetHandle: 'width',
      };
      expect(r).toEqual(c);
    });
    it('should return null if the target has no valid fields', () => {
      const e1 = buildEdge(n1.id, 'width', n2.id, 'width');
      const e2 = buildEdge(n1.id, 'height', n2.id, 'height');
      const n3 = buildNode(add);
      const r = getFirstValidConnection(n3.id, 'value', n2.id, null, [n1, n2, n3], [e1, e2], templates, null);
      expect(r).toEqual(null);
    });
  });

  describe('connecting from a target to a source', () => {
    const n1 = buildNode(img_resize);
    const n2 = buildNode(img_resize);

    it('should return the first valid connection if there are no connected fields', () => {
      const r = getFirstValidConnection(n1.id, null, n2.id, 'width', [n1, n2], [], templates, null);
      const c = {
        source: n1.id,
        sourceHandle: 'width',
        target: n2.id,
        targetHandle: 'width',
      };
      expect(r).toEqual(c);
    });
    it('should return the first valid connection if there is a connected field', () => {
      const e = buildEdge(n1.id, 'height', n2.id, 'width');
      const r = getFirstValidConnection(n1.id, null, n2.id, 'height', [n1, n2], [e], templates, null);
      const c = {
        source: n1.id,
        sourceHandle: 'width',
        target: n2.id,
        targetHandle: 'height',
      };
      expect(r).toEqual(c);
    });
    it('should return the first valid connection if there is an edgePendingUpdate', () => {
      const e = buildEdge(n1.id, 'width', n2.id, 'width');
      const r = getFirstValidConnection(n1.id, null, n2.id, 'width', [n1, n2], [e], templates, e);
      const c = {
        source: n1.id,
        sourceHandle: 'width',
        target: n2.id,
        targetHandle: 'width',
      };
      expect(r).toEqual(c);
    });
    it('should return null if the target has no valid fields', () => {
      const e1 = buildEdge(n1.id, 'width', n2.id, 'width');
      const e2 = buildEdge(n1.id, 'height', n2.id, 'height');
      const n3 = buildNode(add);
      const r = getFirstValidConnection(n3.id, null, n2.id, 'a', [n1, n2, n3], [e1, e2], templates, null);
      expect(r).toEqual(null);
    });
  });

  it('should resolve connector target candidates when connecting an invocation output to a connector', () => {
    const n1 = buildNode(add);
    const connector = buildConnectorNode('connector-1');
    expect(getFirstValidConnection(n1.id, 'value', connector.id, null, [n1, connector], [], templates, null)).toEqual({
      source: n1.id,
      sourceHandle: 'value',
      target: connector.id,
      targetHandle: CONNECTOR_INPUT_HANDLE,
    });
  });

  it('should resolve connector source candidates when connecting a connector to a typed invocation input', () => {
    const n1 = buildNode(add);
    const connector = buildConnectorNode('connector-1');
    const n2 = buildNode(img_resize);
    const edges = [buildEdge(n1.id, 'value', connector.id, CONNECTOR_INPUT_HANDLE)];

    expect(
      getFirstValidConnection(connector.id, null, n2.id, 'width', [n1, connector, n2], edges, templates, null)
    ).toEqual({
      source: connector.id,
      sourceHandle: CONNECTOR_OUTPUT_HANDLE,
      target: n2.id,
      targetHandle: 'width',
    });
  });
});

describe('getTargetCandidateFields', () => {
  it('should return an empty array if the nodes canot be found', () => {
    const r = getTargetCandidateFields('missing', 'value', 'missing', [], [], templates, null);
    expect(r).toEqual([]);
  });
  it('should return an empty array if the templates cannot be found', () => {
    const n1 = buildNode(add);
    const n2 = buildNode(add);
    const nodes = [n1, n2];
    const r = getTargetCandidateFields(n1.id, 'value', n2.id, nodes, [], {}, null);
    expect(r).toEqual([]);
  });
  it('should return an empty array if the source field template cannot be found', () => {
    const n1 = buildNode(add);
    const n2 = buildNode(add);
    const nodes = [n1, n2];

    const addWithoutOutputValue = deepClone(add);
    unset(addWithoutOutputValue, 'outputs.value');

    const r = getTargetCandidateFields(n1.id, 'value', n2.id, nodes, [], { add: addWithoutOutputValue }, null);
    expect(r).toEqual([]);
  });
  it('should return all valid target fields if there are no connected fields', () => {
    const n1 = buildNode(img_resize);
    const n2 = buildNode(img_resize);
    const nodes = [n1, n2];
    const r = getTargetCandidateFields(n1.id, 'width', n2.id, nodes, [], templates, null);
    expect(r).toEqual([img_resize.inputs['width'], img_resize.inputs['height']]);
  });
  it('should ignore the edgePendingUpdate if provided', () => {
    const n1 = buildNode(img_resize);
    const n2 = buildNode(img_resize);
    const nodes = [n1, n2];
    const edgePendingUpdate = buildEdge(n1.id, 'width', n2.id, 'width');
    const r = getTargetCandidateFields(n1.id, 'width', n2.id, nodes, [], templates, edgePendingUpdate);
    expect(r).toEqual([img_resize.inputs['width'], img_resize.inputs['height']]);
  });
  it('should return the connector input handle when the target is a connector', () => {
    const n1 = buildNode(add);
    const connector = buildConnectorNode('connector-1');
    const r = getTargetCandidateFields(n1.id, 'value', connector.id, [n1, connector], [], templates, null);
    expect(r.map((field) => field.name)).toEqual([CONNECTOR_INPUT_HANDLE]);
  });
  it('should advertise typed target candidates for an unresolved connector output when no downstream constraint exists', () => {
    const connector = buildConnectorNode('connector-1');
    const n2 = buildNode(sub);
    const r = getTargetCandidateFields(
      connector.id,
      CONNECTOR_OUTPUT_HANDLE,
      n2.id,
      [connector, n2],
      [],
      templates,
      null
    );
    expect(r.map((field) => field.name)).toEqual(['a', 'b']);
  });
  it('should only advertise compatible typed target candidates for an unresolved connector output with downstream constraints', () => {
    const connector = buildConnectorNode('connector-1');
    const n1 = buildNode(sub);
    const n2 = buildNode(img_resize);
    const edges = [buildEdge(connector.id, CONNECTOR_OUTPUT_HANDLE, n1.id, 'a')];
    const r = getTargetCandidateFields(
      connector.id,
      CONNECTOR_OUTPUT_HANDLE,
      n2.id,
      [connector, n1, n2],
      edges,
      templates,
      null
    );
    expect(r.map((field) => field.name)).toEqual(['width', 'height']);
  });
  it('should resolve chained connector sources like the direct upstream source', () => {
    const n1 = buildNode(add);
    const connectorA = buildConnectorNode('connector-a');
    const connectorB = buildConnectorNode('connector-b');
    const n2 = buildNode(img_resize);
    const edges = [
      buildEdge(n1.id, 'value', connectorA.id, CONNECTOR_INPUT_HANDLE),
      buildEdge(connectorA.id, CONNECTOR_OUTPUT_HANDLE, connectorB.id, CONNECTOR_INPUT_HANDLE),
    ];
    const r = getTargetCandidateFields(
      connectorB.id,
      CONNECTOR_OUTPUT_HANDLE,
      n2.id,
      [n1, connectorA, connectorB, n2],
      edges,
      templates,
      null
    );
    expect(r).toEqual([img_resize.inputs['width'], img_resize.inputs['height']]);
  });
});

describe('getSourceCandidateFields', () => {
  it('should return an empty array if the nodes canot be found', () => {
    const r = getSourceCandidateFields('missing', 'value', 'missing', [], [], templates, null);
    expect(r).toEqual([]);
  });
  it('should return an empty array if the templates cannot be found', () => {
    const n1 = buildNode(add);
    const n2 = buildNode(add);
    const nodes = [n1, n2];
    const r = getSourceCandidateFields(n2.id, 'a', n1.id, nodes, [], {}, null);
    expect(r).toEqual([]);
  });
  it('should return an empty array if the source field template cannot be found', () => {
    const n1 = buildNode(add);
    const n2 = buildNode(add);
    const nodes = [n1, n2];

    const addWithoutInputA = deepClone(add);
    unset(addWithoutInputA, 'inputs.a');

    const r = getSourceCandidateFields(n1.id, 'a', n2.id, nodes, [], { add: addWithoutInputA }, null);
    expect(r).toEqual([]);
  });
  it('should return all valid source fields if there are no connected fields', () => {
    const n1 = buildNode(img_resize);
    const n2 = buildNode(img_resize);
    const nodes = [n1, n2];
    const r = getSourceCandidateFields(n2.id, 'width', n1.id, nodes, [], templates, null);
    expect(r).toEqual([img_resize.outputs['width'], img_resize.outputs['height']]);
  });
  it('should ignore the edgePendingUpdate if provided', () => {
    const n1 = buildNode(img_resize);
    const n2 = buildNode(img_resize);
    const nodes = [n1, n2];
    const edgePendingUpdate = buildEdge(n1.id, 'width', n2.id, 'width');
    const r = getSourceCandidateFields(n2.id, 'width', n1.id, nodes, [], templates, edgePendingUpdate);
    expect(r).toEqual([img_resize.outputs['width'], img_resize.outputs['height']]);
  });
  it('should return the connector output handle when the source is a connector with a typed upstream source', () => {
    const n1 = buildNode(add);
    const connector = buildConnectorNode('connector-1');
    const n2 = buildNode(img_resize);
    const edges = [buildEdge(n1.id, 'value', connector.id, CONNECTOR_INPUT_HANDLE)];
    const r = getSourceCandidateFields(n2.id, 'width', connector.id, [n1, connector, n2], edges, templates, null);
    expect(r.map((field) => field.name)).toEqual([CONNECTOR_OUTPUT_HANDLE]);
  });
  it('should return a target-constrained connector source candidate when the connector chain is unresolved', () => {
    const connector = buildConnectorNode('connector-1');
    const n2 = buildNode(img_resize);
    const r = getSourceCandidateFields(n2.id, 'width', connector.id, [connector, n2], [], templates, null);
    expect(r.map((field) => field.name)).toEqual([CONNECTOR_OUTPUT_HANDLE]);
  });
});
