import { deepClone } from 'common/util/deepClone';
import { set } from 'lodash-es';
import { describe, expect, it } from 'vitest';

import { add, buildEdge, buildNode, collect, img_resize, main_model_loader, sub, templates } from './testUtils';
import { buildAcceptResult, buildRejectResult, validateConnection } from './validateConnection';

describe(validateConnection.name, () => {
  it('should reject invalid connection to self', () => {
    const c = { source: 'add', sourceHandle: 'value', target: 'add', targetHandle: 'a' };
    const r = validateConnection(c, [], [], templates, null);
    expect(r).toEqual(buildRejectResult('nodes.cannotConnectToSelf'));
  });

  describe('missing nodes', () => {
    const n1 = buildNode(add);
    const n2 = buildNode(sub);
    const c = { source: n1.id, sourceHandle: 'value', target: n2.id, targetHandle: 'a' };

    it('should reject missing source node', () => {
      const r = validateConnection(c, [n2], [], templates, null);
      expect(r).toEqual(buildRejectResult('nodes.missingNode'));
    });

    it('should reject missing target node', () => {
      const r = validateConnection(c, [n1], [], templates, null);
      expect(r).toEqual(buildRejectResult('nodes.missingNode'));
    });
  });

  describe('missing invocation templates', () => {
    const n1 = buildNode(add);
    const n2 = buildNode(sub);
    const c = { source: n1.id, sourceHandle: 'value', target: n2.id, targetHandle: 'a' };
    const nodes = [n1, n2];

    it('should reject missing source template', () => {
      const r = validateConnection(c, nodes, [], { sub }, null);
      expect(r).toEqual(buildRejectResult('nodes.missingInvocationTemplate'));
    });

    it('should reject missing target template', () => {
      const r = validateConnection(c, nodes, [], { add }, null);
      expect(r).toEqual(buildRejectResult('nodes.missingInvocationTemplate'));
    });
  });

  describe('missing field templates', () => {
    const n1 = buildNode(add);
    const n2 = buildNode(sub);
    const nodes = [n1, n2];

    it('should reject missing source field template', () => {
      const c = { source: n1.id, sourceHandle: 'invalid', target: n2.id, targetHandle: 'a' };
      const r = validateConnection(c, nodes, [], templates, null);
      expect(r).toEqual(buildRejectResult('nodes.missingFieldTemplate'));
    });

    it('should reject missing target field template', () => {
      const c = { source: n1.id, sourceHandle: 'value', target: n2.id, targetHandle: 'invalid' };
      const r = validateConnection(c, nodes, [], templates, null);
      expect(r).toEqual(buildRejectResult('nodes.missingFieldTemplate'));
    });
  });

  describe('duplicate connections', () => {
    const n1 = buildNode(add);
    const n2 = buildNode(sub);
    it('should accept non-duplicate connections', () => {
      const c = { source: n1.id, sourceHandle: 'value', target: n2.id, targetHandle: 'a' };
      const r = validateConnection(c, [n1, n2], [], templates, null);
      expect(r).toEqual(buildAcceptResult());
    });
    it('should reject duplicate connections', () => {
      const c = { source: n1.id, sourceHandle: 'value', target: n2.id, targetHandle: 'a' };
      const e = buildEdge(n1.id, 'value', n2.id, 'a');
      const r = validateConnection(c, [n1, n2], [e], templates, null);
      expect(r).toEqual(buildRejectResult('nodes.cannotDuplicateConnection'));
    });
    it('should accept duplicate connections if the duplicate is an ignored edge', () => {
      const c = { source: n1.id, sourceHandle: 'value', target: n2.id, targetHandle: 'a' };
      const e = buildEdge(n1.id, 'value', n2.id, 'a');
      const r = validateConnection(c, [n1, n2], [e], templates, e);
      expect(r).toEqual(buildAcceptResult());
    });
  });

  it('should reject connection to direct input', () => {
    // Create cloned add template w/ a direct input
    const addWithDirectAField = deepClone(add);
    set(addWithDirectAField, 'inputs.a.input', 'direct');
    set(addWithDirectAField, 'type', 'addWithDirectAField');

    const n1 = buildNode(add);
    const n2 = buildNode(addWithDirectAField);
    const c = { source: n1.id, sourceHandle: 'value', target: n2.id, targetHandle: 'a' };
    const r = validateConnection(c, [n1, n2], [], { add, addWithDirectAField }, null);
    expect(r).toEqual(buildRejectResult('nodes.cannotConnectToDirectInput'));
  });

  it('should reject connection to a collect node with mismatched item types', () => {
    const n1 = buildNode(add);
    const n2 = buildNode(collect);
    const n3 = buildNode(main_model_loader);
    const nodes = [n1, n2, n3];
    const e1 = buildEdge(n1.id, 'value', n2.id, 'item');
    const edges = [e1];
    const c = { source: n3.id, sourceHandle: 'vae', target: n2.id, targetHandle: 'item' };
    const r = validateConnection(c, nodes, edges, templates, null);
    expect(r).toEqual(buildRejectResult('nodes.cannotMixAndMatchCollectionItemTypes'));
  });

  it('should accept connection to a collect node with matching item types', () => {
    const n1 = buildNode(add);
    const n2 = buildNode(collect);
    const n3 = buildNode(sub);
    const nodes = [n1, n2, n3];
    const e1 = buildEdge(n1.id, 'value', n2.id, 'item');
    const edges = [e1];
    const c = { source: n3.id, sourceHandle: 'value', target: n2.id, targetHandle: 'item' };
    const r = validateConnection(c, nodes, edges, templates, null);
    expect(r).toEqual(buildAcceptResult());
  });

  it('should reject connections to target field that is already connected', () => {
    const n1 = buildNode(add);
    const n2 = buildNode(add);
    const n3 = buildNode(add);
    const nodes = [n1, n2, n3];
    const e1 = buildEdge(n1.id, 'value', n2.id, 'a');
    const edges = [e1];
    const c = { source: n3.id, sourceHandle: 'value', target: n2.id, targetHandle: 'a' };
    const r = validateConnection(c, nodes, edges, templates, null);
    expect(r).toEqual(buildRejectResult('nodes.inputMayOnlyHaveOneConnection'));
  });

  it('should accept connections to target field that is already connected (ignored edge)', () => {
    const n1 = buildNode(add);
    const n2 = buildNode(add);
    const n3 = buildNode(add);
    const nodes = [n1, n2, n3];
    const e1 = buildEdge(n1.id, 'value', n2.id, 'a');
    const edges = [e1];
    const c = { source: n3.id, sourceHandle: 'value', target: n2.id, targetHandle: 'a' };
    const r = validateConnection(c, nodes, edges, templates, e1);
    expect(r).toEqual(buildAcceptResult());
  });

  it('should reject connections between invalid types', () => {
    const n1 = buildNode(add);
    const n2 = buildNode(img_resize);
    const nodes = [n1, n2];
    const c = { source: n1.id, sourceHandle: 'value', target: n2.id, targetHandle: 'image' };
    const r = validateConnection(c, nodes, [], templates, null);
    expect(r).toEqual(buildRejectResult('nodes.fieldTypesMustMatch'));
  });

  it('should reject connections that would create cycles', () => {
    const n1 = buildNode(add);
    const n2 = buildNode(sub);
    const nodes = [n1, n2];
    const e1 = buildEdge(n1.id, 'value', n2.id, 'a');
    const edges = [e1];
    const c = { source: n2.id, sourceHandle: 'value', target: n1.id, targetHandle: 'a' };
    const r = validateConnection(c, nodes, edges, templates, null);
    expect(r).toEqual(buildRejectResult('nodes.connectionWouldCreateCycle'));
  });

  describe('non-strict mode', () => {
    it('should reject connections from self to self in non-strict mode', () => {
      const c = { source: 'add', sourceHandle: 'value', target: 'add', targetHandle: 'a' };
      const r = validateConnection(c, [], [], templates, null, false);
      expect(r).toEqual(buildRejectResult('nodes.cannotConnectToSelf'));
    });
    it('should reject connections that create cycles in non-strict mode', () => {
      const n1 = buildNode(add);
      const n2 = buildNode(sub);
      const nodes = [n1, n2];
      const e1 = buildEdge(n1.id, 'value', n2.id, 'a');
      const edges = [e1];
      const c = { source: n2.id, sourceHandle: 'value', target: n1.id, targetHandle: 'a' };
      const r = validateConnection(c, nodes, edges, templates, null, false);
      expect(r).toEqual(buildRejectResult('nodes.connectionWouldCreateCycle'));
    });
    it('should otherwise allow invalid connections in non-strict mode', () => {
      const n1 = buildNode(add);
      const n2 = buildNode(img_resize);
      const nodes = [n1, n2];
      const c = { source: n1.id, sourceHandle: 'value', target: n2.id, targetHandle: 'image' };
      const r = validateConnection(c, nodes, [], templates, null, false);
      expect(r).toEqual(buildAcceptResult());
    });
  });
});
