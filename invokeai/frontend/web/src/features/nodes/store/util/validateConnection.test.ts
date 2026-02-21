import { deepClone } from 'common/util/deepClone';
import { set } from 'es-toolkit/compat';
import type { InvocationTemplate } from 'features/nodes/types/invocation';
import { describe, expect, it } from 'vitest';

import { add, buildEdge, buildNode, collect, img_resize, main_model_loader, sub, templates } from './testUtils';
import { validateConnection } from './validateConnection';

const ifTemplate: InvocationTemplate = {
  title: 'If',
  type: 'if',
  version: '1.0.0',
  tags: [],
  description: 'Selects between two inputs based on a boolean condition',
  outputType: 'if_output',
  inputs: {
    condition: {
      name: 'condition',
      title: 'Condition',
      required: true,
      description: 'The condition used to select an input',
      fieldKind: 'input',
      input: 'connection',
      ui_hidden: false,
      ui_type: 'BooleanField',
      type: {
        name: 'BooleanField',
        cardinality: 'SINGLE',
        batch: false,
      },
      default: false,
    },
    true_input: {
      name: 'true_input',
      title: 'True Input',
      required: false,
      description: 'Selected when condition is true',
      fieldKind: 'input',
      input: 'connection',
      ui_hidden: false,
      ui_type: 'AnyField',
      type: {
        name: 'AnyField',
        cardinality: 'SINGLE',
        batch: false,
      },
      default: undefined,
    },
    false_input: {
      name: 'false_input',
      title: 'False Input',
      required: false,
      description: 'Selected when condition is false',
      fieldKind: 'input',
      input: 'connection',
      ui_hidden: false,
      ui_type: 'AnyField',
      type: {
        name: 'AnyField',
        cardinality: 'SINGLE',
        batch: false,
      },
      default: undefined,
    },
  },
  outputs: {
    value: {
      fieldKind: 'output',
      name: 'value',
      title: 'Output',
      description: 'The selected value',
      type: {
        name: 'AnyField',
        cardinality: 'SINGLE',
        batch: false,
      },
      ui_hidden: false,
      ui_type: 'AnyField',
    },
  },
  useCache: true,
  nodePack: 'invokeai',
  classification: 'stable',
};

const floatOutputTemplate: InvocationTemplate = {
  title: 'Float Output',
  type: 'float_output',
  version: '1.0.0',
  tags: [],
  description: 'Outputs a float',
  outputType: 'float_output',
  inputs: {},
  outputs: {
    value: {
      fieldKind: 'output',
      name: 'value',
      title: 'Value',
      description: 'Float value',
      type: {
        name: 'FloatField',
        cardinality: 'SINGLE',
        batch: false,
      },
      ui_hidden: false,
      ui_type: 'FloatField',
    },
  },
  useCache: true,
  nodePack: 'invokeai',
  classification: 'stable',
};

describe(validateConnection.name, () => {
  it('should reject invalid connection to self', () => {
    const c = { source: 'add', sourceHandle: 'value', target: 'add', targetHandle: 'a' };
    const r = validateConnection(c, [], [], templates, null);
    expect(r).toEqual('nodes.cannotConnectToSelf');
  });

  describe('missing nodes', () => {
    const n1 = buildNode(add);
    const n2 = buildNode(sub);
    const c = { source: n1.id, sourceHandle: 'value', target: n2.id, targetHandle: 'a' };

    it('should reject missing source node', () => {
      const r = validateConnection(c, [n2], [], templates, null);
      expect(r).toEqual('nodes.missingNode');
    });

    it('should reject missing target node', () => {
      const r = validateConnection(c, [n1], [], templates, null);
      expect(r).toEqual('nodes.missingNode');
    });
  });

  describe('missing invocation templates', () => {
    const n1 = buildNode(add);
    const n2 = buildNode(sub);
    const c = { source: n1.id, sourceHandle: 'value', target: n2.id, targetHandle: 'a' };
    const nodes = [n1, n2];

    it('should reject missing source template', () => {
      const r = validateConnection(c, nodes, [], { sub }, null);
      expect(r).toEqual('nodes.missingInvocationTemplate');
    });

    it('should reject missing target template', () => {
      const r = validateConnection(c, nodes, [], { add }, null);
      expect(r).toEqual('nodes.missingInvocationTemplate');
    });
  });

  describe('missing field templates', () => {
    const n1 = buildNode(add);
    const n2 = buildNode(sub);
    const nodes = [n1, n2];

    it('should reject missing source field template', () => {
      const c = { source: n1.id, sourceHandle: 'invalid', target: n2.id, targetHandle: 'a' };
      const r = validateConnection(c, nodes, [], templates, null);
      expect(r).toEqual('nodes.missingFieldTemplate');
    });

    it('should reject missing target field template', () => {
      const c = { source: n1.id, sourceHandle: 'value', target: n2.id, targetHandle: 'invalid' };
      const r = validateConnection(c, nodes, [], templates, null);
      expect(r).toEqual('nodes.missingFieldTemplate');
    });
  });

  describe('duplicate connections', () => {
    const n1 = buildNode(add);
    const n2 = buildNode(sub);
    it('should accept non-duplicate connections', () => {
      const c = { source: n1.id, sourceHandle: 'value', target: n2.id, targetHandle: 'a' };
      const r = validateConnection(c, [n1, n2], [], templates, null);
      expect(r).toEqual(null);
    });
    it('should reject duplicate connections', () => {
      const c = { source: n1.id, sourceHandle: 'value', target: n2.id, targetHandle: 'a' };
      const e = buildEdge(n1.id, 'value', n2.id, 'a');
      const r = validateConnection(c, [n1, n2], [e], templates, null);
      expect(r).toEqual('nodes.cannotDuplicateConnection');
    });
    it('should accept duplicate connections if the duplicate is an ignored edge', () => {
      const c = { source: n1.id, sourceHandle: 'value', target: n2.id, targetHandle: 'a' };
      const e = buildEdge(n1.id, 'value', n2.id, 'a');
      const r = validateConnection(c, [n1, n2], [e], templates, e);
      expect(r).toEqual(null);
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
    expect(r).toEqual('nodes.cannotConnectToDirectInput');
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
    expect(r).toEqual('nodes.cannotMixAndMatchCollectionItemTypes');
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
    expect(r).toEqual(null);
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
    expect(r).toEqual('nodes.inputMayOnlyHaveOneConnection');
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
    expect(r).toEqual(null);
  });

  it('should reject connections between invalid types', () => {
    const n1 = buildNode(add);
    const n2 = buildNode(img_resize);
    const nodes = [n1, n2];
    const c = { source: n1.id, sourceHandle: 'value', target: n2.id, targetHandle: 'image' };
    const r = validateConnection(c, nodes, [], templates, null);
    expect(r).toEqual('nodes.fieldTypesMustMatch');
  });

  it('should reject mismatched types between if node branch inputs', () => {
    const n1 = buildNode(add);
    const n2 = buildNode(img_resize);
    const n3 = buildNode(ifTemplate);
    const nodes = [n1, n2, n3];
    const e1 = buildEdge(n1.id, 'value', n3.id, 'true_input');
    const edges = [e1];
    const c = { source: n2.id, sourceHandle: 'image', target: n3.id, targetHandle: 'false_input' };
    const r = validateConnection(c, nodes, edges, { ...templates, if: ifTemplate }, null);
    expect(r).toEqual('nodes.fieldTypesMustMatch');
  });

  it('should reject mismatched types between if node branch inputs regardless of branch order', () => {
    const n1 = buildNode(add);
    const n2 = buildNode(img_resize);
    const n3 = buildNode(ifTemplate);
    const nodes = [n1, n2, n3];
    const e1 = buildEdge(n1.id, 'value', n3.id, 'false_input');
    const edges = [e1];
    const c = { source: n2.id, sourceHandle: 'image', target: n3.id, targetHandle: 'true_input' };
    const r = validateConnection(c, nodes, edges, { ...templates, if: ifTemplate }, null);
    expect(r).toEqual('nodes.fieldTypesMustMatch');
  });

  it('should accept convertible types between if node branch inputs', () => {
    const n1 = buildNode(add);
    const n2 = buildNode(sub);
    const n3 = buildNode(ifTemplate);
    const nodes = [n1, n2, n3];
    const e1 = buildEdge(n1.id, 'value', n3.id, 'true_input');
    const edges = [e1];
    const c = { source: n2.id, sourceHandle: 'value', target: n3.id, targetHandle: 'false_input' };
    const r = validateConnection(c, nodes, edges, { ...templates, if: ifTemplate }, null);
    expect(r).toEqual(null);
  });

  it('should accept one-way-convertible types between if node branch inputs in either connection order', () => {
    const n1 = buildNode(add);
    const n2 = buildNode(floatOutputTemplate);
    const n3 = buildNode(ifTemplate);
    const nodes = [n1, n2, n3];
    const e1 = buildEdge(n1.id, 'value', n3.id, 'false_input');
    const edges = [e1];
    const c = { source: n2.id, sourceHandle: 'value', target: n3.id, targetHandle: 'true_input' };
    const r = validateConnection(
      c,
      nodes,
      edges,
      { ...templates, if: ifTemplate, float_output: floatOutputTemplate },
      null
    );
    expect(r).toEqual(null);
  });

  it('should reject connections that would create cycles', () => {
    const n1 = buildNode(add);
    const n2 = buildNode(sub);
    const nodes = [n1, n2];
    const e1 = buildEdge(n1.id, 'value', n2.id, 'a');
    const edges = [e1];
    const c = { source: n2.id, sourceHandle: 'value', target: n1.id, targetHandle: 'a' };
    const r = validateConnection(c, nodes, edges, templates, null);
    expect(r).toEqual('nodes.connectionWouldCreateCycle');
  });

  describe('non-strict mode', () => {
    it('should reject connections from self to self in non-strict mode', () => {
      const c = { source: 'add', sourceHandle: 'value', target: 'add', targetHandle: 'a' };
      const r = validateConnection(c, [], [], templates, null, false);
      expect(r).toEqual('nodes.cannotConnectToSelf');
    });
    it('should reject connections that create cycles in non-strict mode', () => {
      const n1 = buildNode(add);
      const n2 = buildNode(sub);
      const nodes = [n1, n2];
      const e1 = buildEdge(n1.id, 'value', n2.id, 'a');
      const edges = [e1];
      const c = { source: n2.id, sourceHandle: 'value', target: n1.id, targetHandle: 'a' };
      const r = validateConnection(c, nodes, edges, templates, null, false);
      expect(r).toEqual('nodes.connectionWouldCreateCycle');
    });
    it('should otherwise allow invalid connections in non-strict mode', () => {
      const n1 = buildNode(add);
      const n2 = buildNode(img_resize);
      const nodes = [n1, n2];
      const c = { source: n1.id, sourceHandle: 'value', target: n2.id, targetHandle: 'image' };
      const r = validateConnection(c, nodes, [], templates, null, false);
      expect(r).toEqual(null);
    });
  });
});
