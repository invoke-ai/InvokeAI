import { describe, expect, it } from 'vitest';

import type { FieldInputTemplate, InvocationTemplate, ProjectGraphState } from './types';

import { compileProjectGraph } from './buildGraph';
import {
  buildCurrentImageNode,
  buildConnectorNode,
  buildInvocationNode,
  buildNotesNode,
  createProjectGraph,
  getFormChildren,
  projectGraphReducer,
} from './document';
import validForLoop from './fixtures/for-loop-valid.json';
import { validateForLoopGraph } from './forLoops';
import { parseWorkflowJson, serializeWorkflowJson, serializeWorkflowJsonForSubmission } from './workflowJson';

const template: InvocationTemplate = {
  category: 'test',
  classification: 'stable',
  description: '',
  inputs: {
    prompt: {
      default: 'a cat',
      description: '',
      exclusiveMaximum: null,
      exclusiveMinimum: null,
      fieldKind: 'input',
      input: 'any',
      maximum: null,
      minimum: null,
      multipleOf: null,
      name: 'prompt',
      options: null,
      required: true,
      title: 'Prompt',
      type: { batch: false, cardinality: 'SINGLE', name: 'StringField' },
      uiChoiceLabels: null,
      uiComponent: 'textarea',
      uiHidden: false,
      uiModelBase: null,
      uiModelFormat: null,
      uiModelType: null,
      uiOrder: null,
    },
  },
  nodePack: 'invokeai',
  outputs: {},
  outputType: 'string_output',
  tags: [],
  title: 'Prompt',
  type: 'prompt',
  useCache: true,
  version: '1.2.0',
};

describe('workflow JSON round-trip', () => {
  it('serializes and re-parses a document, preserving nodes, edges, form, and metadata', () => {
    const node = buildInvocationNode(template, { x: 10, y: 20 });
    let doc = createProjectGraph('roundtrip');

    doc = projectGraphReducer(doc, { node, type: 'addNode' });
    doc = projectGraphReducer(doc, {
      description: 'Custom description',
      fieldName: 'prompt',
      nodeId: node.id,
      type: 'setFieldDescription',
    });
    doc = projectGraphReducer(doc, { fieldIdentifier: { fieldName: 'prompt', nodeId: node.id }, type: 'exposeField' });
    doc = projectGraphReducer(doc, {
      patch: { author: 'josh', description: 'desc', name: 'Round Trip', tags: 'a,b' },
      type: 'setMetadata',
    });

    const serialized = serializeWorkflowJson(doc);
    const { document: parsed, warnings } = parseWorkflowJson(serialized);

    expect(warnings).toEqual([]);
    expect(parsed.name).toBe('Round Trip');
    expect(parsed.author).toBe('josh');
    expect(parsed.nodes).toHaveLength(1);

    const parsedNode = parsed.nodes[0];

    expect(parsedNode?.id).toBe(node.id);
    expect(parsedNode?.position).toEqual({ x: 10, y: 20 });
    expect(parsedNode?.type === 'invocation' && parsedNode.data.inputs.prompt?.value).toBe('a cat');
    expect(parsedNode?.type === 'invocation' && parsedNode.data.inputs.prompt?.description).toBe('Custom description');

    const formChildren = getFormChildren(parsed.form);

    expect(formChildren).toHaveLength(1);
    expect(formChildren[0]?.type).toBe('node-field');
  });

  it('keeps the serialized shape legacy-compatible', () => {
    const doc = createProjectGraph('legacy-shape');
    const serialized = serializeWorkflowJson(doc);

    expect(serialized.meta).toEqual({ category: 'user', version: '3.0.0' });
    expect(serialized.exposedFields).toEqual([]);
    expect(serialized).toHaveProperty('form');
  });

  it('normalizes legacy cleared descriptions without changing other overrides during JSON reload', () => {
    const cases = [
      { description: undefined, descriptionOverride: true, expectedOverride: false },
      { description: '', descriptionOverride: true, expectedOverride: false },
      { description: 'Custom description', descriptionOverride: true, expectedOverride: true },
      { description: 'Generated description', descriptionOverride: false, expectedOverride: false },
    ] as const;

    for (const { description, descriptionOverride, expectedOverride } of cases) {
      const node = buildInvocationNode(template, { x: 0, y: 0 });
      node.data.inputs.prompt = {
        ...node.data.inputs.prompt!,
        description,
        descriptionOverride,
      };
      const document = projectGraphReducer(createProjectGraph('legacy-cleared-description'), {
        node,
        type: 'addNode',
      });

      const parsed = parseWorkflowJson(serializeWorkflowJson(document)).document;
      const parsedNode = parsed.nodes[0];

      expect(parsedNode?.type === 'invocation' && parsedNode.data.inputs.prompt).toMatchObject({
        description,
        descriptionOverride: expectedOverride,
      });
    }
  });

  it('persists dynamic input templates needed to preserve Call Saved Workflow values', () => {
    const node = buildInvocationNode(template, { x: 0, y: 0 });
    node.data.dynamicInputTemplates = { runtime: template.inputs.prompt! };
    let doc = createProjectGraph('runtime-fields');
    doc = projectGraphReducer(doc, { node, type: 'addNode' });

    const serialized = serializeWorkflowJson(doc) as {
      nodes: Array<{ data: Record<string, unknown> }>;
    };

    expect(serialized.nodes[0]?.data).toHaveProperty('dynamicInputTemplates');
    expect(parseWorkflowJson(serialized).document.nodes[0]).toMatchObject({
      data: { dynamicInputTemplates: { runtime: template.inputs.prompt } },
    });
  });

  it('keeps Call Saved Workflow templates for image recall but strips other runtime templates', () => {
    const node = buildInvocationNode(template, { x: 0, y: 0 });
    node.data.dynamicInputTemplates = { runtime: template.inputs.prompt! };
    const callNode = buildInvocationNode(template, { x: 1, y: 1 });
    callNode.data = {
      ...callNode.data,
      dynamicInputTemplates: { runtime: template.inputs.prompt! },
      type: 'call_saved_workflow',
    };
    let doc = createProjectGraph('submission-runtime-fields');
    doc = projectGraphReducer(doc, { node, type: 'addNode' });
    doc = projectGraphReducer(doc, { node: callNode, type: 'addNode' });

    const serialized = serializeWorkflowJsonForSubmission(doc) as {
      nodes: Array<{ data: Record<string, unknown> }>;
    };

    expect(
      serialized.nodes.find((candidate) => candidate.data.type !== 'call_saved_workflow')?.data
    ).not.toHaveProperty('dynamicInputTemplates');
    expect(serialized.nodes.find((candidate) => candidate.data.type === 'call_saved_workflow')?.data).toHaveProperty(
      'dynamicInputTemplates'
    );
  });

  it('omits current-image nodes and their edges from embedded workflows', () => {
    const invocation = buildInvocationNode(template, { x: 0, y: 0 });
    const currentImage = buildCurrentImageNode({ x: 1, y: 1 });
    let doc = createProjectGraph('submission-legacy-shape');
    doc = projectGraphReducer(doc, { node: invocation, type: 'addNode' });
    doc = projectGraphReducer(doc, { node: currentImage, type: 'addNode' });
    doc.edges = [
      {
        id: 'current-image-edge',
        source: currentImage.id,
        sourceHandle: 'image',
        target: invocation.id,
        targetHandle: 'prompt',
        type: 'loop_linkage',
      },
    ];

    const serialized = serializeWorkflowJsonForSubmission(doc) as {
      edges: Array<Record<string, unknown>>;
      nodes: Array<Record<string, unknown>>;
    };

    expect(serialized.nodes.some((node) => node.type === 'current_image')).toBe(false);
    expect(serialized.edges).toEqual([]);
  });

  it('preserves direct loop linkage when embedding a workflow for image recall', () => {
    const doc: ProjectGraphState = {
      ...createProjectGraph('embedded-loop'),
      nodes: validForLoop.nodes as ProjectGraphState['nodes'],
      edges: validForLoop.edges as ProjectGraphState['edges'],
    };
    expect(validateForLoopGraph(doc)).toBeNull();

    const recalled = parseWorkflowJson(serializeWorkflowJsonForSubmission(doc)).document;

    expect(recalled.edges.find((edge) => edge.id === 'linkage')?.type).toBe('loop_linkage');
    expect(validateForLoopGraph(recalled)).toBeNull();
    const compiled = compileProjectGraph(recalled, {
      for: { ...template, type: 'for' },
      for_return: { ...template, type: 'for_return' },
      integer: { ...template, type: 'integer' },
      workflow_return: { ...template, type: 'workflow_return' },
    });
    expect(compiled.backendGraph.edges).toContainEqual({
      destination: { field: 'loop_linkage', node_id: 'return' },
      source: { field: 'loop_linkage', node_id: 'for' },
      type: 'loop_linkage',
    });
  });

  it('round-trips internal BoardField dynamic templates for legacy readers', () => {
    const node = buildInvocationNode(template, { x: 0, y: 0 });
    const boardTemplate: FieldInputTemplate = {
      ...template.inputs.prompt!,
      default: undefined,
      fieldKind: 'internal',
      name: 'board',
      title: 'Board',
      type: { batch: false, cardinality: 'SINGLE', name: 'BoardField' },
    };
    node.data.dynamicInputTemplates = { board: boardTemplate };
    const document = projectGraphReducer(createProjectGraph('legacy-dynamic-board'), {
      node,
      type: 'addNode',
    });

    const serialized = serializeWorkflowJson(document) as {
      nodes: Array<{ data: { dynamicInputTemplates: Record<string, Record<string, unknown>> } }>;
    };
    const persisted = serialized.nodes[0]?.data.dynamicInputTemplates.board;

    expect(persisted).toMatchObject({
      fieldKind: 'internal',
      type: { name: 'BoardField' },
      uiHidden: false,
    });
    expect(parseWorkflowJson(serialized).document.nodes[0]).toMatchObject({
      data: { dynamicInputTemplates: { board: boardTemplate } },
    });
  });

  it('round-trips notes, current_image, and connector UI nodes', () => {
    let doc = createProjectGraph('ui-nodes');

    doc = projectGraphReducer(doc, { node: buildNotesNode({ x: 1, y: 2 }), type: 'addNode' });
    doc = projectGraphReducer(doc, { node: buildCurrentImageNode({ x: 3, y: 4 }), type: 'addNode' });
    doc = projectGraphReducer(doc, { node: buildConnectorNode({ x: 5, y: 6 }), type: 'addNode' });

    const { document: parsed, warnings } = parseWorkflowJson(serializeWorkflowJson(doc));

    expect(warnings).toEqual([]);
    expect(parsed.nodes.map((node) => node.type).sort()).toEqual(['connector', 'current_image', 'notes']);

    const currentImage = parsed.nodes.find((node) => node.type === 'current_image');

    expect(currentImage?.position).toEqual({ x: 3, y: 4 });
    expect(currentImage?.type === 'current_image' && currentImage.data.label).toBe('Current Image');

    const connector = parsed.nodes.find((node) => node.type === 'connector');

    expect(connector?.position).toEqual({ x: 5, y: 6 });
    expect(connector?.type === 'connector' && connector.data.label).toBe('');
  });
});

describe('parseWorkflowJson tolerance', () => {
  it('reads the legacy per-element shuffle setting and keeps the settings bag for the legacy editor', () => {
    const { document, warnings } = parseWorkflowJson({
      edges: [],
      form: {
        elements: {
          root: { data: { children: ['f1'], layout: 'column' }, id: 'root', type: 'container' },
          f1: {
            data: {
              fieldIdentifier: { fieldName: 'steps', nodeId: 'n1' },
              settings: { component: 'number-input', showShuffle: true },
            },
            id: 'f1',
            parentId: 'root',
            type: 'node-field',
          },
        },
        rootElementId: 'root',
      },
      name: 'Legacy Shuffle',
      nodes: [
        {
          data: { id: 'n1', inputs: { steps: { label: '', name: 'steps', value: 20 } }, type: 'noise' },
          id: 'n1',
          position: { x: 0, y: 0 },
          type: 'invocation',
        },
      ],
      version: '1.0.0',
    });

    expect(warnings).toEqual([]);

    const field = document.form.elements.f1;

    expect(field?.type === 'node-field' && field.data).toEqual({
      fieldIdentifier: { fieldName: 'steps', nodeId: 'n1' },
      settings: { component: 'number-input', showShuffle: true },
      showDescription: false,
      showShuffle: true,
    });

    const toggled = projectGraphReducer(document, {
      elementId: 'f1',
      showShuffle: false,
      type: 'setNodeFieldShowShuffle',
    });
    const toggledField = toggled.form.elements.f1;
    const serialized = serializeWorkflowJson(toggled) as { form: { elements: Record<string, { data: unknown }> } };

    expect(toggledField?.type === 'node-field' && toggledField.data.settings).toEqual({
      component: 'number-input',
      showShuffle: false,
    });
    expect(serialized.form.elements.f1?.data).toMatchObject({
      settings: { component: 'number-input', showShuffle: false },
    });
  });

  it('migrates pre-form exposedFields into form elements', () => {
    const { document, warnings } = parseWorkflowJson({
      edges: [],
      exposedFields: [{ fieldName: 'prompt', nodeId: 'n1' }],
      name: 'Old Workflow',
      nodes: [
        {
          data: { id: 'n1', inputs: { prompt: { label: '', name: 'prompt', value: 'hi' } }, type: 'prompt' },
          id: 'n1',
          position: { x: 0, y: 0 },
          type: 'invocation',
        },
      ],
      version: '1.0.0',
    });

    expect(warnings).toEqual([]);

    const children = getFormChildren(document.form);

    expect(children).toHaveLength(1);
    expect(children[0]?.type === 'node-field' && children[0].data.fieldIdentifier).toEqual({
      fieldName: 'prompt',
      nodeId: 'n1',
    });
  });

  it('does not merge stale exposed fields into an existing stored form', () => {
    const { document, warnings } = parseWorkflowJson({
      edges: [],
      exposedFields: [
        { fieldName: 'prompt', nodeId: 'n1' },
        { fieldName: 'other', nodeId: 'n1' },
      ],
      form: {
        elements: {
          root: { data: { children: ['f1'], layout: 'column' }, id: 'root', type: 'container' },
          f1: {
            data: { fieldIdentifier: { fieldName: 'prompt', nodeId: 'n1' } },
            id: 'f1',
            parentId: 'root',
            type: 'node-field',
          },
        },
        rootElementId: 'root',
      },
      nodes: [
        {
          data: {
            id: 'n1',
            inputs: {
              other: { label: '', name: 'other', value: 'world' },
              prompt: { label: '', name: 'prompt', value: 'hi' },
            },
            type: 'prompt',
          },
          id: 'n1',
          position: { x: 0, y: 0 },
          type: 'invocation',
        },
      ],
    });

    expect(warnings).toEqual([]);
    expect(
      getFormChildren(document.form).map(
        (element) => element.type === 'node-field' && element.data.fieldIdentifier.fieldName
      )
    ).toEqual(['prompt']);
  });

  it('preserves connector nodes and edges', () => {
    const { document, warnings } = parseWorkflowJson({
      edges: [
        { id: 'e1', source: 'n1', sourceHandle: 'out', target: 'conn1', targetHandle: 'in', type: 'default' },
        { id: 'e2', source: 'conn1', sourceHandle: 'out', target: 'n2', targetHandle: 'text', type: 'default' },
      ],
      name: 'Connectors',
      nodes: [
        { data: { id: 'n1', inputs: {}, type: 'a' }, id: 'n1', position: { x: 0, y: 0 }, type: 'invocation' },
        { data: { id: 'n2', inputs: {}, type: 'b' }, id: 'n2', position: { x: 0, y: 0 }, type: 'invocation' },
        { data: { label: 'Connector' }, id: 'conn1', position: { x: 4, y: 5 }, type: 'connector' },
      ],
    });

    expect(warnings).toEqual([]);
    expect(document.nodes.map((node) => node.type).sort()).toEqual(['connector', 'invocation', 'invocation']);
    expect(document.nodes.find((node) => node.type === 'connector')?.position).toEqual({ x: 4, y: 5 });
    expect(document.edges).toEqual([
      { id: 'e1', source: 'n1', sourceHandle: 'out', target: 'conn1', targetHandle: 'in', type: 'default' },
      { id: 'e2', source: 'conn1', sourceHandle: 'out', target: 'n2', targetHandle: 'text', type: 'default' },
    ]);
  });

  it('round-trips a direct loop_linkage edge without inferring it from handles', () => {
    const { document, warnings } = parseWorkflowJson({
      edges: [
        {
          id: 'link',
          source: 'for',
          sourceHandle: 'loop_linkage',
          target: 'return',
          targetHandle: 'loop_linkage',
          type: 'loop_linkage',
        },
      ],
      nodes: [
        { data: { id: 'for', inputs: {}, type: 'for' }, id: 'for', position: { x: 0, y: 0 }, type: 'invocation' },
        {
          data: { id: 'return', inputs: {}, type: 'for_return' },
          id: 'return',
          position: { x: 0, y: 0 },
          type: 'invocation',
        },
      ],
    });

    expect(warnings).toEqual([]);
    expect(document.edges[0]?.type).toBe('loop_linkage');
    expect(serializeWorkflowJson(document).edges).toEqual([
      {
        id: 'link',
        source: 'for',
        sourceHandle: 'loop_linkage',
        target: 'return',
        targetHandle: 'loop_linkage',
        type: 'loop_linkage',
      },
    ]);
  });

  it('drops dangling edges and unknown form elements with warnings', () => {
    const { document, warnings } = parseWorkflowJson({
      edges: [{ id: 'e1', source: 'missing', sourceHandle: 'out', target: 'n1', targetHandle: 'in', type: 'default' }],
      form: { elements: { weird: { id: 'weird', type: 'mystery' } }, rootElementId: 'missing-root' },
      name: 'Broken',
      nodes: [{ data: { id: 'n1', inputs: {}, type: 'a' }, id: 'n1', position: { x: 0, y: 0 }, type: 'invocation' }],
    });

    expect(document.edges).toEqual([]);
    expect(document.form.elements[document.form.rootElementId]?.type).toBe('container');
    expect(warnings.length).toBeGreaterThan(0);
  });

  it('rejects non-workflow payloads', () => {
    expect(() => parseWorkflowJson('not a workflow')).toThrow(/not a recognizable/);
  });

  it('accepts form: null, as stored by pre-form-builder library workflows', () => {
    const { document, warnings } = parseWorkflowJson({
      edges: [],
      form: null,
      name: 'No Form',
      nodes: [{ data: { id: 'n1', inputs: {}, type: 'a' }, id: 'n1', position: { x: 0, y: 0 }, type: 'invocation' }],
    });

    expect(warnings).toEqual([]);
    expect(document.form.elements[document.form.rootElementId]?.type).toBe('container');
  });
});

describe('seed modes in workflow JSON', () => {
  it('round-trips a stepping mode and reads its absence as unset', () => {
    const node = buildInvocationNode(template, { x: 0, y: 0 });
    let doc = createProjectGraph('seed-json');

    doc = projectGraphReducer(doc, { node, type: 'addNode' });
    doc = projectGraphReducer(doc, {
      fieldName: 'prompt',
      nodeId: node.id,
      seedMode: 'increment',
      type: 'setFieldSeedMode',
    });

    const serialized = serializeWorkflowJson(doc);
    const parsed = parseWorkflowJson(serialized).document.nodes[0];

    expect(parsed?.type === 'invocation' && parsed.data.inputs.prompt).toMatchObject({ seedMode: 'increment' });

    // Fixed is the absent key, which is also what a legacy reader hands back after stripping it.
    const fixed = serializeWorkflowJson(
      projectGraphReducer(doc, { fieldName: 'prompt', nodeId: node.id, seedMode: 'fixed', type: 'setFieldSeedMode' })
    ).nodes as Array<{ data: { inputs: { prompt: { seedMode?: unknown } } } }>;

    expect(fixed[0]?.data.inputs.prompt).not.toHaveProperty('seedMode');

    const nodes = serialized.nodes as Array<{ data: { inputs: { prompt: { seedMode?: unknown } } } }>;

    (nodes[0] as NonNullable<(typeof nodes)[number]>).data.inputs.prompt.seedMode = 'shuffle';

    const degraded = parseWorkflowJson(serialized).document.nodes[0];

    expect(degraded?.type === 'invocation' && degraded.data.inputs.prompt).not.toHaveProperty('seedMode');
  });
});

describe('field label overrides', () => {
  // Round-trip labelOverride through explicit parsing as well as cloned serialization.
  it('round-trips an explicit label override', () => {
    const node = buildInvocationNode(template, { x: 0, y: 0 });
    let doc = createProjectGraph('label-override');

    doc = projectGraphReducer(doc, { node, type: 'addNode' });
    doc = projectGraphReducer(doc, {
      fieldName: 'prompt',
      label: 'My label',
      nodeId: node.id,
      type: 'setFieldLabel',
    });

    const reloaded = parseWorkflowJson(serializeWorkflowJson(doc)).document.nodes[0];

    expect(reloaded?.type === 'invocation' ? reloaded.data.inputs.prompt : undefined).toMatchObject({
      label: 'My label',
      labelOverride: true,
    });
  });
});

describe('batch and generator nodes', () => {
  it('keeps legacy generator values and batch group ids verbatim through a load and save', () => {
    const generator = {
      count: 3,
      max: 1,
      min: 0,
      seed: null,
      type: 'float_generator_random_distribution_uniform',
      values: [0.2, 0.4, 0.6],
    };
    const { document, warnings } = parseWorkflowJson({
      edges: [],
      name: 'Batch',
      nodes: [
        {
          data: {
            id: 'g',
            inputs: { generator: { label: '', name: 'generator', value: generator } },
            type: 'float_generator',
          },
          id: 'g',
          position: { x: 0, y: 0 },
          type: 'invocation',
        },
        {
          data: {
            id: 'b',
            inputs: {
              batch_group_id: { label: '', name: 'batch_group_id', value: 'Group 2' },
              floats: { label: '', name: 'floats', value: [1, 2] },
            },
            type: 'float_batch',
          },
          id: 'b',
          position: { x: 0, y: 0 },
          type: 'invocation',
        },
      ],
      version: '1.0.0',
    });
    const serialized = serializeWorkflowJson(document) as {
      nodes: Array<{ data: { inputs: Record<string, { value: unknown }> } }>;
    };

    expect(warnings).toEqual([]);
    expect(serialized.nodes[0]?.data.inputs.generator?.value).toEqual(generator);
    expect(serialized.nodes[1]?.data.inputs.batch_group_id?.value).toBe('Group 2');
    expect(serialized.nodes[1]?.data.inputs.floats?.value).toEqual([1, 2]);
  });
});
