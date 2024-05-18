import { deepClone } from 'common/util/deepClone';
import {
  getFirstValidConnection,
  getSourceCandidateFields,
  getTargetCandidateFields,
} from 'features/nodes/store/util/getFirstValidConnection';
import { add, buildEdge, buildNode, img_resize, templates } from 'features/nodes/store/util/testUtils';
import { unset } from 'lodash-es';
import { describe, expect, it } from 'vitest';

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
});
