import { deepClone } from 'common/util/deepClone';
import { set } from 'es-toolkit/compat';
import type { InvocationTemplate } from 'features/nodes/types/invocation';
import { describe, expect, it } from 'vitest';

import {
  CONNECTOR_INPUT_HANDLE,
  CONNECTOR_OUTPUT_HANDLE,
  getConnectorDeletionSpliceConnections,
} from './connectorTopology';
import { add, buildEdge, buildNode, collect, img_resize, main_model_loader, sub, templates } from './testUtils';
import { validateConnection } from './validateConnection';

const ifTemplate: InvocationTemplate = {
  title: 'If',
  type: 'if',
  version: '1.0.0',
  tags: [],
  category: 'math',
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
  category: 'primitives',
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

const integerCollectionOutputTemplate: InvocationTemplate = {
  title: 'Integer Collection Output',
  type: 'integer_collection_output',
  version: '1.0.0',
  tags: [],
  category: 'primitives',
  description: 'Outputs an integer collection',
  outputType: 'integer_collection_output',
  inputs: {},
  outputs: {
    value: {
      fieldKind: 'output',
      name: 'value',
      title: 'Value',
      description: 'Integer collection value',
      type: {
        name: 'IntegerField',
        cardinality: 'COLLECTION',
        batch: false,
      },
      ui_hidden: false,
      ui_type: 'IntegerField',
    },
  },
  useCache: true,
  nodePack: 'invokeai',
  classification: 'stable',
};

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

  it('should accept chaining collect collection output to collect collection input', () => {
    const n1 = buildNode(collect);
    const n2 = buildNode(collect);
    const nodes = [n1, n2];
    const c = { source: n1.id, sourceHandle: 'collection', target: n2.id, targetHandle: 'collection' };
    const r = validateConnection(c, nodes, [], templates, null);
    expect(r).toEqual(null);
  });

  it('should reject multiple connections to collect collection input', () => {
    const n1 = buildNode(collect);
    const n2 = buildNode(collect);
    const n3 = buildNode(collect);
    const nodes = [n1, n2, n3];
    const e1 = buildEdge(n1.id, 'collection', n2.id, 'collection');
    const c = { source: n3.id, sourceHandle: 'collection', target: n2.id, targetHandle: 'collection' };
    const r = validateConnection(c, nodes, [e1], templates, null);
    expect(r).toEqual('nodes.inputMayOnlyHaveOneConnection');
  });

  it('should reject mismatched item connection when collect is typed via chained collection', () => {
    const n1 = buildNode(add);
    const n2 = buildNode(collect);
    const n3 = buildNode(collect);
    const n4 = buildNode(main_model_loader);
    const nodes = [n1, n2, n3, n4];
    const e1 = buildEdge(n1.id, 'value', n2.id, 'item');
    const e2 = buildEdge(n2.id, 'collection', n3.id, 'collection');
    const c = { source: n4.id, sourceHandle: 'vae', target: n3.id, targetHandle: 'item' };
    const r = validateConnection(c, nodes, [e1, e2], templates, null);
    expect(r).toEqual('nodes.cannotMixAndMatchCollectionItemTypes');
  });

  it('should reject chaining collection-to-collection for differently typed collects', () => {
    const n1 = buildNode(add);
    const n2 = buildNode(img_resize);
    const n3 = buildNode(collect);
    const n4 = buildNode(collect);
    const nodes = [n1, n2, n3, n4];
    const e1 = buildEdge(n1.id, 'value', n3.id, 'item');
    const e2 = buildEdge(n2.id, 'image', n4.id, 'item');
    const c = { source: n3.id, sourceHandle: 'collection', target: n4.id, targetHandle: 'collection' };
    const r = validateConnection(c, nodes, [e1, e2], templates, null);
    expect(r).toEqual('nodes.cannotMixAndMatchCollectionItemTypes');
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

  it('should accept SINGLE and COLLECTION of the same type between if node branch inputs', () => {
    const n1 = buildNode(add);
    const n2 = buildNode(integerCollectionOutputTemplate);
    const n3 = buildNode(ifTemplate);
    const nodes = [n1, n2, n3];
    const e1 = buildEdge(n1.id, 'value', n3.id, 'true_input');
    const edges = [e1];
    const c = { source: n2.id, sourceHandle: 'value', target: n3.id, targetHandle: 'false_input' };
    const r = validateConnection(
      c,
      nodes,
      edges,
      { ...templates, if: ifTemplate, integer_collection_output: integerCollectionOutputTemplate },
      null
    );
    expect(r).toEqual(null);
  });

  it('should accept if output to collection input when both if branch inputs are collections of matching type', () => {
    const n1 = buildNode(integerCollectionOutputTemplate);
    const n2 = buildNode(integerCollectionOutputTemplate);
    const n3 = buildNode(ifTemplate);
    const n4 = buildNode(templates.iterate!);
    const nodes = [n1, n2, n3, n4];
    const e1 = buildEdge(n1.id, 'value', n3.id, 'true_input');
    const e2 = buildEdge(n2.id, 'value', n3.id, 'false_input');
    const edges = [e1, e2];
    const c = { source: n3.id, sourceHandle: 'value', target: n4.id, targetHandle: 'collection' };
    const r = validateConnection(
      c,
      nodes,
      edges,
      { ...templates, if: ifTemplate, integer_collection_output: integerCollectionOutputTemplate },
      null
    );
    expect(r).toEqual(null);
  });

  it('should reject if output to collection input when if branch inputs are not both collection-compatible', () => {
    const n1 = buildNode(add);
    const n2 = buildNode(integerCollectionOutputTemplate);
    const n3 = buildNode(ifTemplate);
    const n4 = buildNode(templates.iterate!);
    const nodes = [n1, n2, n3, n4];
    const e1 = buildEdge(n1.id, 'value', n3.id, 'true_input');
    const e2 = buildEdge(n2.id, 'value', n3.id, 'false_input');
    const edges = [e1, e2];
    const c = { source: n3.id, sourceHandle: 'value', target: n4.id, targetHandle: 'collection' };
    const r = validateConnection(
      c,
      nodes,
      edges,
      { ...templates, if: ifTemplate, integer_collection_output: integerCollectionOutputTemplate },
      null
    );
    expect(r).toEqual('nodes.fieldTypesMustMatch');
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

  describe('connectors', () => {
    it('should accept invocation output to connector input', () => {
      const n1 = buildNode(add);
      const connector = buildConnectorNode('connector-1');
      const r = validateConnection(
        { source: n1.id, sourceHandle: 'value', target: connector.id, targetHandle: CONNECTOR_INPUT_HANDLE },
        [n1, connector],
        [],
        templates,
        null
      );
      expect(r).toEqual(null);
    });

    it('should reject a second input into a connector', () => {
      const n1 = buildNode(add);
      const n2 = buildNode(sub);
      const connector = buildConnectorNode('connector-1');
      const edges = [buildEdge(n1.id, 'value', connector.id, CONNECTOR_INPUT_HANDLE)];
      const r = validateConnection(
        { source: n2.id, sourceHandle: 'value', target: connector.id, targetHandle: CONNECTOR_INPUT_HANDLE },
        [n1, n2, connector],
        edges,
        templates,
        null
      );
      expect(r).toEqual('nodes.inputMayOnlyHaveOneConnection');
    });

    it('should accept connector output to invocation input when the upstream type matches', () => {
      const n1 = buildNode(add);
      const connector = buildConnectorNode('connector-1');
      const n2 = buildNode(sub);
      const edges = [buildEdge(n1.id, 'value', connector.id, CONNECTOR_INPUT_HANDLE)];
      const r = validateConnection(
        { source: connector.id, sourceHandle: CONNECTOR_OUTPUT_HANDLE, target: n2.id, targetHandle: 'a' },
        [n1, connector, n2],
        edges,
        templates,
        null
      );
      expect(r).toEqual(null);
    });

    it('should reject connector output to invocation input when the upstream type mismatches', () => {
      const n1 = buildNode(add);
      const connector = buildConnectorNode('connector-1');
      const n2 = buildNode(img_resize);
      const edges = [buildEdge(n1.id, 'value', connector.id, CONNECTOR_INPUT_HANDLE)];
      const r = validateConnection(
        { source: connector.id, sourceHandle: CONNECTOR_OUTPUT_HANDLE, target: n2.id, targetHandle: 'image' },
        [n1, connector, n2],
        edges,
        templates,
        null
      );
      expect(r).toEqual('nodes.fieldTypesMustMatch');
    });

    it('should accept unresolved connector output to a typed invocation input as the first downstream constraint', () => {
      const connector = buildConnectorNode('connector-1');
      const n2 = buildNode(sub);
      const r = validateConnection(
        { source: connector.id, sourceHandle: CONNECTOR_OUTPUT_HANDLE, target: n2.id, targetHandle: 'a' },
        [connector, n2],
        [],
        templates,
        null
      );
      expect(r).toEqual(null);
    });

    it('should reject unresolved connector output when it conflicts with an existing downstream typed constraint', () => {
      const connector = buildConnectorNode('connector-1');
      const n1 = buildNode(sub);
      const n2 = buildNode(img_resize);
      const edges = [buildEdge(connector.id, CONNECTOR_OUTPUT_HANDLE, n1.id, 'a')];
      const r = validateConnection(
        { source: connector.id, sourceHandle: CONNECTOR_OUTPUT_HANDLE, target: n2.id, targetHandle: 'image' },
        [connector, n1, n2],
        edges,
        templates,
        null
      );
      expect(r).toEqual('nodes.fieldTypesMustMatch');
    });

    it('should reject connecting an incompatible upstream source into a connector with downstream typed constraints', () => {
      const source = buildNode(main_model_loader);
      const connector = buildConnectorNode('connector-1');
      const target = buildNode(sub);
      const edges = [buildEdge(connector.id, CONNECTOR_OUTPUT_HANDLE, target.id, 'a')];
      const r = validateConnection(
        { source: source.id, sourceHandle: 'vae', target: connector.id, targetHandle: CONNECTOR_INPUT_HANDLE },
        [source, connector, target],
        edges,
        templates,
        null
      );
      expect(r).toEqual('nodes.fieldTypesMustMatch');
    });

    it('should preserve type information through chained connectors', () => {
      const n1 = buildNode(add);
      const connectorA = buildConnectorNode('connector-a');
      const connectorB = buildConnectorNode('connector-b');
      const n2 = buildNode(sub);
      const edges = [
        buildEdge(n1.id, 'value', connectorA.id, CONNECTOR_INPUT_HANDLE),
        buildEdge(connectorA.id, CONNECTOR_OUTPUT_HANDLE, connectorB.id, CONNECTOR_INPUT_HANDLE),
      ];
      const r = validateConnection(
        { source: connectorB.id, sourceHandle: CONNECTOR_OUTPUT_HANDLE, target: n2.id, targetHandle: 'a' },
        [n1, connectorA, connectorB, n2],
        edges,
        templates,
        null
      );
      expect(r).toEqual(null);
    });

    it('should reject cycles routed through connectors', () => {
      const n1 = buildNode(add);
      const n2 = buildNode(sub);
      const connector = buildConnectorNode('connector-1');
      const edges = [
        buildEdge(n1.id, 'value', connector.id, CONNECTOR_INPUT_HANDLE),
        buildEdge(connector.id, CONNECTOR_OUTPUT_HANDLE, n2.id, 'a'),
      ];
      const r = validateConnection(
        { source: n2.id, sourceHandle: 'value', target: n1.id, targetHandle: 'a' },
        [n1, n2, connector],
        edges,
        templates,
        null
      );
      expect(r).toEqual('nodes.connectionWouldCreateCycle');
    });

    it('should preserve collect item validation through connectors', () => {
      const n1 = buildNode(add);
      const n2 = buildNode(collect);
      const n3 = buildNode(main_model_loader);
      const connector = buildConnectorNode('connector-1');
      const edges = [
        buildEdge(n1.id, 'value', n2.id, 'item'),
        buildEdge(n3.id, 'vae', connector.id, CONNECTOR_INPUT_HANDLE),
      ];
      const r = validateConnection(
        { source: connector.id, sourceHandle: CONNECTOR_OUTPUT_HANDLE, target: n2.id, targetHandle: 'item' },
        [n1, n2, n3, connector],
        edges,
        templates,
        null
      );
      expect(r).toEqual('nodes.cannotMixAndMatchCollectionItemTypes');
    });

    it('should preserve if branch validation through connectors', () => {
      const n1 = buildNode(add);
      const n2 = buildNode(img_resize);
      const n3 = buildNode(ifTemplate);
      const connector = buildConnectorNode('connector-1');
      const edges = [
        buildEdge(n1.id, 'value', connector.id, CONNECTOR_INPUT_HANDLE),
        buildEdge(connector.id, CONNECTOR_OUTPUT_HANDLE, n3.id, 'true_input'),
      ];
      const r = validateConnection(
        { source: n2.id, sourceHandle: 'image', target: n3.id, targetHandle: 'false_input' },
        [n1, n2, n3, connector],
        edges,
        { ...templates, if: ifTemplate },
        null
      );
      expect(r).toEqual('nodes.fieldTypesMustMatch');
    });

    it('should reject connector deletion splice-through when it would duplicate an existing direct edge', () => {
      const n1 = buildNode(add);
      const n2 = buildNode(sub);
      const connector = buildConnectorNode('connector-1');
      const edges = [
        buildEdge(n1.id, 'value', connector.id, CONNECTOR_INPUT_HANDLE),
        buildEdge(connector.id, CONNECTOR_OUTPUT_HANDLE, n2.id, 'a'),
        buildEdge(n1.id, 'value', n2.id, 'a'),
      ];

      expect(getConnectorDeletionSpliceConnections(connector.id, [n1, n2, connector], edges, templates)).toBe(null);
    });

    it('should reject connector deletion splice-through when fan-out would violate a single-input target', () => {
      const n1 = buildNode(add);
      const connectorA = buildConnectorNode('connector-a');
      const connectorB = buildConnectorNode('connector-b');
      const n2 = buildNode(sub);
      const edges = [
        buildEdge(n1.id, 'value', connectorA.id, CONNECTOR_INPUT_HANDLE),
        buildEdge(connectorA.id, CONNECTOR_OUTPUT_HANDLE, connectorB.id, CONNECTOR_INPUT_HANDLE),
        buildEdge(connectorA.id, CONNECTOR_OUTPUT_HANDLE, n2.id, 'a'),
        buildEdge(connectorB.id, CONNECTOR_OUTPUT_HANDLE, n2.id, 'a'),
      ];

      expect(
        getConnectorDeletionSpliceConnections(connectorA.id, [n1, connectorA, connectorB, n2], edges, templates)
      ).toBe(null);
    });
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
