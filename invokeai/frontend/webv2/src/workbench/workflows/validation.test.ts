import { describe, expect, it } from 'vitest';

import type {
  FieldType,
  InvocationTemplates,
  ProjectGraphState,
  WorkflowConnectorNode,
  WorkflowEdge,
  WorkflowInvocationNode,
} from './types';

import {
  getCompatibleInputTemplate,
  getCompatibleOutputTemplate,
  getWorkflowSourceFieldType,
  getWorkflowTargetFieldType,
  hasAnyCycle,
  validateConnection,
  validateConnectionTypes,
  wouldCreateCycle,
} from './validation';

const single = (name: string): FieldType => ({ batch: false, cardinality: 'SINGLE', name });
const collection = (name: string): FieldType => ({ batch: false, cardinality: 'COLLECTION', name });
const singleOrCollection = (name: string): FieldType => ({ batch: false, cardinality: 'SINGLE_OR_COLLECTION', name });

describe('validateConnectionTypes', () => {
  it('accepts equal types and rejects mismatched names', () => {
    expect(validateConnectionTypes(single('ImageField'), single('ImageField'))).toBe(true);
    expect(validateConnectionTypes(single('ImageField'), single('LatentsField'))).toBe(false);
  });

  it('applies numeric/string subtype promotions with matching cardinality', () => {
    expect(validateConnectionTypes(single('IntegerField'), single('FloatField'))).toBe(true);
    expect(validateConnectionTypes(single('FloatField'), single('StringField'))).toBe(true);
    expect(validateConnectionTypes(single('FloatField'), single('IntegerField'))).toBe(false);
    expect(validateConnectionTypes(collection('IntegerField'), single('FloatField'))).toBe(false);
  });

  it('handles SINGLE_OR_COLLECTION targets and generic collections', () => {
    expect(validateConnectionTypes(single('StringField'), singleOrCollection('StringField'))).toBe(true);
    expect(validateConnectionTypes(collection('StringField'), singleOrCollection('StringField'))).toBe(true);
    expect(validateConnectionTypes(collection('StringField'), collection('CollectionField'))).toBe(true);
    expect(validateConnectionTypes(single('CollectionField'), collection('ImageField'))).toBe(true);
  });

  it('handles CollectionItemField and AnyField wildcards', () => {
    expect(validateConnectionTypes(single('CollectionItemField'), single('ImageField'))).toBe(true);
    expect(validateConnectionTypes(single('ImageField'), single('CollectionItemField'))).toBe(true);
    expect(validateConnectionTypes(single('ImageField'), single('AnyField'))).toBe(true);
    expect(validateConnectionTypes(single('AnyField'), single('ImageField'))).toBe(true);
  });

  it('rejects batch/non-batch mixes', () => {
    expect(validateConnectionTypes({ ...single('IntegerField'), batch: true }, single('IntegerField'))).toBe(false);
  });
});

describe('cycle detection', () => {
  const edges: WorkflowEdge[] = [
    { id: 'e1', source: 'a', sourceHandle: 'out', target: 'b', targetHandle: 'in', type: 'default' },
    { id: 'e2', source: 'b', sourceHandle: 'out', target: 'c', targetHandle: 'in', type: 'default' },
  ];

  it('detects when a new connection would close a loop', () => {
    expect(wouldCreateCycle('c', 'a', edges)).toBe(true);
    expect(wouldCreateCycle('a', 'c', edges)).toBe(false);
  });

  it('detects existing cycles in a whole graph', () => {
    const nodes = ['a', 'b', 'c'].map((id) => ({ id }) as { id: string });
    const cyclic = [
      ...edges,
      { id: 'e3', source: 'c', sourceHandle: 'out', target: 'a', targetHandle: 'in', type: 'default' as const },
    ];

    expect(hasAnyCycle(nodes as never, edges)).toBe(false);
    expect(hasAnyCycle(nodes as never, cyclic)).toBe(true);
  });
});

const makeNode = (id: string, type: string): WorkflowInvocationNode => ({
  data: {
    inputs: {},
    isIntermediate: true,
    isOpen: true,
    label: '',
    nodePack: 'invokeai',
    notes: '',
    type,
    useCache: true,
    version: '1.0.0',
  },
  id,
  position: { x: 0, y: 0 },
  type: 'invocation',
});

const makeConnector = (id: string): WorkflowConnectorNode => ({
  data: { label: 'Connector' },
  id,
  position: { x: 0, y: 0 },
  type: 'connector',
});

const templates: InvocationTemplates = {
  collect: {
    category: 'util',
    classification: 'stable',
    description: '',
    inputs: {
      item: {
        default: undefined,
        description: '',
        exclusiveMaximum: null,
        exclusiveMinimum: null,
        input: 'connection',
        maximum: null,
        minimum: null,
        multipleOf: null,
        name: 'item',
        options: null,
        required: true,
        title: 'Item',
        type: { batch: false, cardinality: 'SINGLE', name: 'CollectionItemField' },
        uiChoiceLabels: null,
        uiComponent: null,
        uiHidden: false,
        uiModelBase: null,
        uiModelType: null,
        uiOrder: null,
      },
    },
    nodePack: 'invokeai',
    outputs: {},
    outputType: 'collect_output',
    tags: [],
    title: 'Collect',
    type: 'collect',
    useCache: true,
    version: '1.0.0',
  },
  number: {
    category: 'math',
    classification: 'stable',
    description: '',
    inputs: {
      value: {
        default: 0,
        description: '',
        exclusiveMaximum: null,
        exclusiveMinimum: null,
        input: 'any',
        maximum: null,
        minimum: null,
        multipleOf: null,
        name: 'value',
        options: null,
        required: false,
        title: 'Value',
        type: { batch: false, cardinality: 'SINGLE', name: 'IntegerField' },
        uiChoiceLabels: null,
        uiComponent: null,
        uiHidden: false,
        uiModelBase: null,
        uiModelType: null,
        uiOrder: null,
      },
    },
    nodePack: 'invokeai',
    outputs: {
      value: {
        description: '',
        name: 'value',
        title: 'Value',
        type: { batch: false, cardinality: 'SINGLE', name: 'IntegerField' },
      },
    },
    outputType: 'integer_output',
    tags: [],
    title: 'Number',
    type: 'number',
    useCache: true,
    version: '1.0.0',
  },
};

describe('getCompatibleInputTemplate', () => {
  it('returns the first visible connectable input by UI order', () => {
    const baseInput = templates.number.inputs.value;
    const template = {
      ...templates.number,
      inputs: {
        direct: { ...baseInput, input: 'direct' as const, name: 'direct', title: 'Direct', uiOrder: 0 },
        hidden: { ...baseInput, name: 'hidden', title: 'Hidden', uiHidden: true, uiOrder: 1 },
        late: { ...baseInput, name: 'late', title: 'Late', uiOrder: 3 },
        early: { ...baseInput, name: 'early', title: 'Early', uiOrder: 2 },
      },
    };

    expect(getCompatibleInputTemplate(template, single('IntegerField'))?.name).toBe('early');
    expect(getCompatibleInputTemplate(template, single('ImageField'))).toBeNull();
    expect(getCompatibleInputTemplate(template, null)?.name).toBe('early');
  });
});

describe('getWorkflowSourceFieldType', () => {
  it('resolves typed and untyped connector outputs', () => {
    const document: Pick<ProjectGraphState, 'edges' | 'nodes'> = {
      edges: [],
      nodes: [makeConnector('connector')],
    };

    expect(getWorkflowSourceFieldType(document, templates, 'connector', 'out')).toBeNull();

    const upstreamDocument: Pick<ProjectGraphState, 'edges' | 'nodes'> = {
      edges: [
        {
          id: 'e1',
          source: 'n1',
          sourceHandle: 'value',
          target: 'connector',
          targetHandle: 'in',
          type: 'default',
        },
      ],
      nodes: [makeNode('n1', 'number'), makeConnector('connector')],
    };

    expect(getWorkflowSourceFieldType(upstreamDocument, templates, 'connector', 'out')).toEqual(single('IntegerField'));
  });

  it('resolves connector outputs from downstream targets when no input is connected', () => {
    const document: Pick<ProjectGraphState, 'edges' | 'nodes'> = {
      edges: [
        {
          id: 'connector-out',
          source: 'connector',
          sourceHandle: 'out',
          target: 'n1',
          targetHandle: 'value',
          type: 'default',
        },
      ],
      nodes: [makeConnector('connector'), makeNode('n1', 'number')],
    };

    expect(getWorkflowSourceFieldType(document, templates, 'connector', 'out')).toEqual(single('IntegerField'));
  });

  it('leaves fan-out connector outputs untyped when downstream targets disagree', () => {
    const mixedTemplates: InvocationTemplates = {
      ...templates,
      text: {
        ...templates.number,
        inputs: {
          value: { ...templates.number.inputs.value, type: single('StringField') },
        },
        outputs: {
          value: { ...templates.number.outputs.value, type: single('StringField') },
        },
        type: 'text',
      },
    };
    const document: Pick<ProjectGraphState, 'edges' | 'nodes'> = {
      edges: [
        {
          id: 'connector-out-string',
          source: 'connector',
          sourceHandle: 'out',
          target: 'text-1',
          targetHandle: 'value',
          type: 'default',
        },
        {
          id: 'connector-out-number',
          source: 'connector',
          sourceHandle: 'out',
          target: 'number-1',
          targetHandle: 'value',
          type: 'default',
        },
      ],
      nodes: [makeConnector('connector'), makeNode('text-1', 'text'), makeNode('number-1', 'number')],
    };

    expect(getWorkflowSourceFieldType(document, mixedTemplates, 'connector', 'out')).toBeNull();
  });
});

describe('getCompatibleOutputTemplate', () => {
  it('returns the first output compatible with the target type', () => {
    expect(getCompatibleOutputTemplate(templates.number, single('IntegerField'))?.name).toBe('value');
    expect(getCompatibleOutputTemplate(templates.number, single('ImageField'))).toBeNull();
  });

  it('accepts any output for untyped connector targets', () => {
    expect(getCompatibleOutputTemplate(templates.number, null)?.name).toBe('value');
    expect(getCompatibleOutputTemplate(templates.collect, null)).toBeNull();
  });
});

describe('getWorkflowTargetFieldType', () => {
  it('resolves connectable invocation inputs and connector inputs', () => {
    const document: Pick<ProjectGraphState, 'edges' | 'nodes'> = {
      edges: [],
      nodes: [makeNode('n1', 'number'), makeConnector('connector')],
    };

    expect(getWorkflowTargetFieldType(document, templates, 'n1', 'value')).toEqual(single('IntegerField'));
    expect(getWorkflowTargetFieldType(document, templates, 'connector', 'in')).toBeNull();
  });

  it('rejects direct-only inputs', () => {
    const directTemplates: InvocationTemplates = {
      number: {
        ...templates.number,
        inputs: {
          value: { ...templates.number.inputs.value, input: 'direct' },
        },
      },
    };

    expect(
      getWorkflowTargetFieldType({ edges: [], nodes: [makeNode('n1', 'number')] }, directTemplates, 'n1', 'value')
    ).toBe(undefined);
  });

  it('resolves connector inputs from downstream targets', () => {
    const document: Pick<ProjectGraphState, 'edges' | 'nodes'> = {
      edges: [
        {
          id: 'connector-out',
          source: 'connector',
          sourceHandle: 'out',
          target: 'n1',
          targetHandle: 'value',
          type: 'default',
        },
      ],
      nodes: [makeConnector('connector'), makeNode('n1', 'number')],
    };

    expect(getWorkflowTargetFieldType(document, templates, 'connector', 'in')).toEqual(single('IntegerField'));
  });
});

describe('validateConnection', () => {
  const baseDocument: Pick<ProjectGraphState, 'edges' | 'nodes'> = {
    edges: [],
    nodes: [makeNode('n1', 'number'), makeNode('n2', 'number'), makeNode('c1', 'collect')],
  };

  it('accepts a valid connection', () => {
    expect(
      validateConnection(
        { sourceHandle: 'value', sourceNodeId: 'n1', targetHandle: 'value', targetNodeId: 'n2' },
        baseDocument,
        templates
      )
    ).toBeNull();
  });

  it('rejects self-connections and unknown fields', () => {
    expect(
      validateConnection(
        { sourceHandle: 'value', sourceNodeId: 'n1', targetHandle: 'value', targetNodeId: 'n1' },
        baseDocument,
        templates
      )
    ).toMatch(/itself/);
    expect(
      validateConnection(
        { sourceHandle: 'nope', sourceNodeId: 'n1', targetHandle: 'value', targetNodeId: 'n2' },
        baseDocument,
        templates
      )
    ).toMatch(/no known definition/);
  });

  it('rejects a second edge into an occupied input, except for collect item', () => {
    const document = {
      ...baseDocument,
      edges: [
        {
          id: 'e1',
          source: 'n1',
          sourceHandle: 'value',
          target: 'n2',
          targetHandle: 'value',
          type: 'default' as const,
        },
        { id: 'e2', source: 'n1', sourceHandle: 'value', target: 'c1', targetHandle: 'item', type: 'default' as const },
      ],
    };

    expect(
      validateConnection(
        { sourceHandle: 'value', sourceNodeId: 'n2', targetHandle: 'value', targetNodeId: 'n2' },
        document,
        templates
      )
    ).not.toBeNull();
    expect(
      validateConnection(
        { sourceHandle: 'value', sourceNodeId: 'n2', targetHandle: 'item', targetNodeId: 'c1' },
        document,
        templates
      )
    ).toBeNull();
  });

  it('rejects connections that would create a cycle', () => {
    const document = {
      ...baseDocument,
      edges: [
        {
          id: 'e1',
          source: 'n1',
          sourceHandle: 'value',
          target: 'n2',
          targetHandle: 'value',
          type: 'default' as const,
        },
      ],
    };

    expect(
      validateConnection(
        { sourceHandle: 'value', sourceNodeId: 'n2', targetHandle: 'value', targetNodeId: 'n1' },
        document,
        templates
      )
    ).toMatch(/cycle/);
  });

  it('accepts connectors as pass-through routing nodes', () => {
    const connectorDocument: Pick<ProjectGraphState, 'edges' | 'nodes'> = {
      edges: [
        {
          id: 'connector-in',
          source: 'n1',
          sourceHandle: 'value',
          target: 'connector-1',
          targetHandle: 'in',
          type: 'default',
        },
      ],
      nodes: [makeNode('n1', 'number'), makeNode('n2', 'number'), makeConnector('connector-1')],
    };

    expect(
      validateConnection(
        { sourceHandle: 'out', sourceNodeId: 'connector-1', targetHandle: 'value', targetNodeId: 'n2' },
        connectorDocument,
        templates
      )
    ).toBeNull();
  });

  it('rejects a second connector input', () => {
    const connectorDocument: Pick<ProjectGraphState, 'edges' | 'nodes'> = {
      edges: [
        {
          id: 'connector-in',
          source: 'n1',
          sourceHandle: 'value',
          target: 'connector-1',
          targetHandle: 'in',
          type: 'default',
        },
      ],
      nodes: [makeNode('n1', 'number'), makeNode('n2', 'number'), makeConnector('connector-1')],
    };

    expect(
      validateConnection(
        { sourceHandle: 'value', sourceNodeId: 'n2', targetHandle: 'in', targetNodeId: 'connector-1' },
        connectorDocument,
        templates
      )
    ).toMatch(/already has an input/);
  });

  it('rejects connector input sources incompatible with an existing downstream target', () => {
    const stringTemplates: InvocationTemplates = {
      ...templates,
      text: {
        ...templates.number,
        inputs: {
          value: { ...templates.number.inputs.value, type: single('StringField') },
        },
        outputs: {
          value: { ...templates.number.outputs.value, type: single('StringField') },
        },
        type: 'text',
      },
    };
    const connectorDocument: Pick<ProjectGraphState, 'edges' | 'nodes'> = {
      edges: [
        {
          id: 'connector-out',
          source: 'connector-1',
          sourceHandle: 'out',
          target: 'n1',
          targetHandle: 'value',
          type: 'default',
        },
      ],
      nodes: [makeNode('n1', 'number'), makeNode('text-1', 'text'), makeConnector('connector-1')],
    };

    expect(
      validateConnection(
        { sourceHandle: 'value', sourceNodeId: 'text-1', targetHandle: 'in', targetNodeId: 'connector-1' },
        connectorDocument,
        stringTemplates
      )
    ).toMatch(/StringField cannot connect to IntegerField/);
  });

  it('rejects connector input sources incompatible with any downstream fan-out target', () => {
    const stringTemplates: InvocationTemplates = {
      ...templates,
      text: {
        ...templates.number,
        inputs: {
          value: { ...templates.number.inputs.value, type: single('StringField') },
        },
        outputs: {
          value: { ...templates.number.outputs.value, type: single('StringField') },
        },
        type: 'text',
      },
    };
    const connectorDocument: Pick<ProjectGraphState, 'edges' | 'nodes'> = {
      edges: [
        {
          id: 'connector-out-string',
          source: 'connector-1',
          sourceHandle: 'out',
          target: 'text-target',
          targetHandle: 'value',
          type: 'default',
        },
        {
          id: 'connector-out-number',
          source: 'connector-1',
          sourceHandle: 'out',
          target: 'number-target',
          targetHandle: 'value',
          type: 'default',
        },
      ],
      nodes: [
        makeNode('number-target', 'number'),
        makeNode('text-source', 'text'),
        makeNode('text-target', 'text'),
        makeConnector('connector-1'),
      ],
    };

    expect(
      validateConnection(
        { sourceHandle: 'value', sourceNodeId: 'text-source', targetHandle: 'in', targetNodeId: 'connector-1' },
        connectorDocument,
        stringTemplates
      )
    ).toMatch(/StringField cannot connect to IntegerField/);
  });
});
