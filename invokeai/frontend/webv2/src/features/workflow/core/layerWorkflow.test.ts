import { describe, expect, it } from 'vitest';

import type {
  FieldInputTemplate,
  FieldOutputTemplate,
  InvocationTemplate,
  InvocationTemplatesSnapshot,
  InvocationTemplates,
  ProjectGraphState,
  WorkflowEdge,
  WorkflowInvocationNode,
} from './types';

import { buildConnectorNode, buildInvocationNode, createProjectGraph } from './document';
import {
  buildLayerWorkflowGraph,
  getDefaultLayerWorkflowSelection,
  getLayerWorkflowInputs,
  getLayerWorkflowOutputs,
  getRunnableLayerWorkflowInputs,
  reconcileLayerWorkflowSelection,
  type WorkflowImageBinding,
} from './layerWorkflow';

const imageType = { batch: false, cardinality: 'SINGLE' as const, name: 'ImageField' };

const input = (name: string, overrides: Partial<FieldInputTemplate> = {}): FieldInputTemplate => ({
  default: undefined,
  description: '',
  exclusiveMaximum: null,
  exclusiveMinimum: null,
  fieldKind: 'input',
  input: 'any',
  maximum: null,
  minimum: null,
  multipleOf: null,
  name,
  options: null,
  required: false,
  title: name,
  type: imageType,
  uiChoiceLabels: null,
  uiComponent: null,
  uiHidden: false,
  uiModelBase: null,
  uiModelFormat: null,
  uiModelType: null,
  uiOrder: null,
  ...overrides,
});

const output = (name: string, overrides: Partial<FieldOutputTemplate> = {}): FieldOutputTemplate => ({
  description: '',
  name,
  title: name,
  type: imageType,
  ...overrides,
});

const template = (
  type: string,
  inputs: Record<string, FieldInputTemplate> = {},
  outputs: Record<string, FieldOutputTemplate> = {}
): InvocationTemplate => ({
  category: 'test',
  classification: 'stable',
  description: '',
  inputs,
  nodePack: 'invokeai',
  outputs,
  outputType: `${type}_output`,
  tags: [],
  title: type,
  type,
  useCache: true,
  version: '1.0.0',
});

const node = (
  id: string,
  invocationTemplate: InvocationTemplate,
  options: { fieldLabels?: Record<string, string>; label?: string } = {}
): WorkflowInvocationNode => {
  const built = buildInvocationNode(invocationTemplate, { x: 0, y: 0 });

  for (const [fieldName, label] of Object.entries(options.fieldLabels ?? {})) {
    const instance = built.data.inputs[fieldName];

    if (instance) {
      instance.label = label;
    }
  }

  return { ...built, data: { ...built.data, label: options.label ?? '' }, id };
};

const document = (nodes: WorkflowInvocationNode[], edges: WorkflowEdge[] = []): ProjectGraphState => ({
  ...createProjectGraph('layer-workflow-test', 'Layer Workflow'),
  edges,
  nodes,
});

const loaded = (templates: InvocationTemplates): InvocationTemplatesSnapshot => ({
  error: null,
  status: 'loaded',
  templates,
});

const binding = (nodeId: string, fieldName: string, label = ''): WorkflowImageBinding => ({
  fieldName,
  label,
  nodeId,
});

describe('layer workflow binding discovery', () => {
  it('finds unconnected image inputs across all input modes in document and UI order', () => {
    const sourceTemplate = template('source', {}, { image: output('image', { title: 'Image' }) });
    const sinkTemplate = template('sink', {
      directEdgeTarget: input('directEdgeTarget', { input: 'any', title: 'Connected directly', uiOrder: 0 }),
      directOptional: input('directOptional', { input: 'direct', title: 'Optional image', uiOrder: 1 }),
      directRequired: input('directRequired', { input: 'direct', required: true, title: 'Required image', uiOrder: 2 }),
      anyAvailable: input('anyAvailable', { input: 'any', title: 'Any image', uiOrder: 3 }),
      connectionOnly: input('connectionOnly', {
        input: 'connection',
        required: true,
        title: 'Connection image',
        uiOrder: 4,
      }),
      connectorTarget: input('connectorTarget', {
        input: 'connection',
        title: 'Connected through connector',
        uiOrder: 5,
      }),
      collection: input('collection', {
        title: 'Image collection',
        type: { batch: false, cardinality: 'COLLECTION', name: 'ImageField' },
        uiOrder: 6,
      }),
      batch: input('batch', {
        title: 'Image batch',
        type: { batch: true, cardinality: 'SINGLE', name: 'ImageField' },
        uiOrder: 7,
      }),
      text: input('text', {
        title: 'Text',
        type: { batch: false, cardinality: 'SINGLE', name: 'StringField' },
        uiOrder: 8,
      }),
    });
    const otherTemplate = template('other', {
      image: input('image', { input: 'direct', title: 'Other image', uiOrder: 0 }),
    });
    const unknownTemplate = template('unknown', { image: input('image', { title: 'Unknown image' }) });
    const source = node('source', sourceTemplate);
    const sink = node('sink', sinkTemplate, {
      fieldLabels: { directOptional: 'Custom image field' },
      label: 'Custom sink',
    });
    const unknown = node('unknown', unknownTemplate);
    const other = node('other', otherTemplate);
    const connector = { ...buildConnectorNode({ x: 0, y: 0 }), id: 'connector' };
    const doc = {
      ...document([sink, unknown, other, source]),
      nodes: [sink, connector, unknown, other, source],
      edges: [
        {
          id: 'direct',
          source: 'source',
          sourceHandle: 'image',
          target: 'sink',
          targetHandle: 'directEdgeTarget',
          type: 'default',
        },
        {
          id: 'connector-in',
          source: 'source',
          sourceHandle: 'image',
          target: 'connector',
          targetHandle: 'in',
          type: 'default',
        },
        {
          id: 'connector-out',
          source: 'connector',
          sourceHandle: 'out',
          target: 'sink',
          targetHandle: 'connectorTarget',
          type: 'default',
        },
      ],
    } satisfies ProjectGraphState;
    const templates = { other: otherTemplate, sink: sinkTemplate, source: sourceTemplate };

    expect(getLayerWorkflowInputs(doc, templates)).toEqual([
      binding('sink', 'directOptional', 'Custom sink → Custom image field'),
      binding('sink', 'directRequired', 'Custom sink → Required image'),
      binding('sink', 'anyAvailable', 'Custom sink → Any image'),
      binding('sink', 'connectionOnly', 'Custom sink → Connection image'),
      binding('other', 'image', 'other → Other image'),
    ]);
  });

  it('finds persisted dynamic image inputs on Call Saved Workflow nodes', () => {
    const callTemplate = template('call_saved_workflow');
    const callNode = node('call', callTemplate);
    callNode.data.dynamicInputTemplates = { image: input('image', { title: 'Child image' }) };

    expect(getLayerWorkflowInputs(document([callNode]), { call_saved_workflow: callTemplate })).toEqual([
      binding('call', 'image', 'call_saved_workflow → Child image'),
    ]);
  });

  it('finds only single non-batch image outputs and skips image primitive nodes', () => {
    const processorTemplate = template(
      'processor',
      {},
      {
        first: output('first', { title: 'First image' }),
        collection: output('collection', {
          title: 'Image collection',
          type: { batch: false, cardinality: 'COLLECTION', name: 'ImageField' },
        }),
        second: output('second', { title: 'Second image' }),
        batch: output('batch', {
          title: 'Image batch',
          type: { batch: true, cardinality: 'SINGLE', name: 'ImageField' },
        }),
        text: output('text', {
          title: 'Text',
          type: { batch: false, cardinality: 'SINGLE', name: 'StringField' },
        }),
      }
    );
    const otherTemplate = template('other', {}, { image: output('image', { title: 'Other image' }) });
    const primitiveTemplate = template('image', {}, { image: output('image', { title: 'Primitive image' }) });
    const unknownTemplate = template('unknown', {}, { image: output('image') });
    const doc = document([
      node('processor', processorTemplate, { label: 'Custom processor' }),
      node('primitive', primitiveTemplate),
      node('unknown', unknownTemplate),
      node('other', otherTemplate),
    ]);
    const templates = { image: primitiveTemplate, other: otherTemplate, processor: processorTemplate };

    expect(getLayerWorkflowOutputs(doc, templates)).toEqual([
      binding('processor', 'first', 'Custom processor → First image'),
      binding('processor', 'second', 'Custom processor → Second image'),
      binding('other', 'image', 'other → Other image'),
    ]);
  });

  it('probes every candidate after sentinel injection and keeps a valid candidate after an invalid first one', () => {
    const processorTemplate = template(
      'processor',
      {
        optionalFirst: input('optionalFirst', { input: 'direct', title: 'Optional first', uiOrder: 0 }),
        requiredSecond: input('requiredSecond', {
          input: 'direct',
          required: true,
          title: 'Required second',
          uiOrder: 1,
        }),
      },
      { image: output('image', { title: 'Result' }) }
    );
    const doc = document([node('processor', processorTemplate)]);
    const outputBinding = getLayerWorkflowOutputs(doc, { processor: processorTemplate })[0]!;

    expect(getRunnableLayerWorkflowInputs(doc, loaded({ processor: processorTemplate }), outputBinding)).toEqual([
      binding('processor', 'requiredSecond', 'processor → Required second'),
    ]);
  });
});

describe('layer workflow batch guard', () => {
  it('never runs a document with batch nodes, which a single layer graph cannot expand', () => {
    const floatType = { batch: false, cardinality: 'SINGLE' as const, name: 'FloatField' };
    const processorTemplate = template(
      'processor',
      {
        cfg: input('cfg', { default: 7, input: 'direct', title: 'CFG', type: floatType }),
        source: input('source', { input: 'direct', title: 'Source', uiOrder: 0 }),
      },
      { image: output('image', { title: 'Result' }) }
    );
    const batchTemplate = template('float_batch', {
      batch_group_id: input('batch_group_id', {
        default: 'None',
        input: 'direct',
        options: ['None'],
        type: { batch: false, cardinality: 'SINGLE', name: 'EnumField' },
      }),
      floats: input('floats', {
        default: [],
        required: true,
        type: { batch: true, cardinality: 'COLLECTION', name: 'FloatField' },
      }),
    });
    const templates = { float_batch: batchTemplate, processor: processorTemplate };
    const processor = node('processor', processorTemplate);
    const batch = node('batch', batchTemplate);

    batch.data.inputs.floats = { label: '', name: 'floats', value: [1, 2] };

    const plain = document([processor]);
    const batched = document(
      [processor, batch],
      [{ id: 'e', source: 'batch', sourceHandle: 'value', target: 'processor', targetHandle: 'cfg', type: 'default' }]
    );
    const outputBinding = getLayerWorkflowOutputs(plain, templates)[0]!;

    expect(getRunnableLayerWorkflowInputs(plain, loaded(templates), outputBinding)).toEqual([
      binding('processor', 'source', 'processor → Source'),
    ]);
    expect(getRunnableLayerWorkflowInputs(batched, loaded(templates), outputBinding)).toEqual([]);
  });
});

describe('layer workflow dialog selection', () => {
  it('defaults to Gallery and the first output that has a runnable input', () => {
    const firstOutput = binding('first-output', 'image', 'First output');
    const secondOutput = binding('second-output', 'image', 'Second output');
    const secondInput = binding('second-input', 'image', 'Second input');

    expect(
      getDefaultLayerWorkflowSelection([firstOutput, secondOutput], (candidate) =>
        candidate.nodeId === secondOutput.nodeId ? [secondInput] : []
      )
    ).toEqual({ destination: 'gallery', input: secondInput, output: secondOutput });
  });

  it('returns an empty Gallery selection when no output has a runnable input', () => {
    expect(getDefaultLayerWorkflowSelection([binding('output', 'image')], () => [])).toEqual({
      destination: 'gallery',
      input: null,
      output: null,
    });
  });

  it('preserves the current input by identity when it remains runnable for a changed output', () => {
    const currentInput = binding('input', 'image', 'Old label');
    const changedOutput = binding('changed-output', 'image', 'Changed output');
    const refreshedInput = binding('input', 'image', 'Refreshed label');

    expect(
      reconcileLayerWorkflowSelection(
        { destination: 'staging', input: currentInput, output: binding('old-output', 'image') },
        changedOutput,
        [refreshedInput, binding('fallback', 'image')]
      )
    ).toEqual({ destination: 'staging', input: refreshedInput, output: changedOutput });
  });

  it('falls back to the first runnable input for a changed output', () => {
    const changedOutput = binding('changed-output', 'image', 'Changed output');
    const firstRunnable = binding('first-input', 'image', 'First input');

    expect(
      reconcileLayerWorkflowSelection(
        { destination: 'copy-raster', input: binding('old-input', 'image'), output: binding('old-output', 'image') },
        changedOutput,
        [firstRunnable, binding('second-input', 'image')]
      )
    ).toEqual({ destination: 'copy-raster', input: firstRunnable, output: changedOutput });
  });
});

describe('buildLayerWorkflowGraph', () => {
  it('injects a direct-capable input before readiness without mutating the document', () => {
    const processorTemplate = template(
      'processor',
      {
        image: input('image', { input: 'direct', required: true, title: 'Layer image' }),
      },
      { result: output('result', { title: 'Result image' }) }
    );
    const processor = node('processor', processorTemplate);
    delete processor.data.inputs.image;
    const doc = document([processor]);
    const before = structuredClone(doc);
    const built = buildLayerWorkflowGraph({
      document: doc,
      imageName: 'layer.png',
      input: binding('processor', 'image'),
      output: binding('processor', 'result'),
      templatesSnapshot: loaded({ processor: processorTemplate }),
    });

    expect(doc).toEqual(before);
    expect(built.graph.nodes.processor).toMatchObject({
      image: { image_name: 'layer.png' },
      is_intermediate: true,
      type: 'processor',
    });
    expect(built.graph.nodes[built.outputNodeId]).toMatchObject({
      id: built.outputNodeId,
      is_intermediate: true,
      type: 'image',
    });
    expect(built.graph.edges).toContainEqual({
      destination: { field: 'image', node_id: built.outputNodeId },
      source: { field: 'result', node_id: 'processor' },
      type: 'default',
    });
    expect(Object.values(built.graph.nodes).every((graphNode) => graphNode.is_intermediate === true)).toBe(true);
  });

  it('writes reserved seeds into the built graph without touching the document', () => {
    const seededTemplate = template(
      'seeded',
      {
        image: input('image', { input: 'direct', required: true, title: 'Layer image' }),
        seed: input('seed', {
          default: 0,
          maximum: 4_294_967_295,
          minimum: 0,
          type: { batch: false, cardinality: 'SINGLE', name: 'IntegerField' },
        }),
      },
      { result: output('result', { title: 'Result image' }) }
    );
    const seeded = node('seeded', seededTemplate);
    seeded.data.inputs.seed = { label: '', name: 'seed', seedMode: 'increment', value: 42 };
    const doc = document([seeded]);
    const before = structuredClone(doc);
    const built = buildLayerWorkflowGraph({
      document: doc,
      imageName: 'layer.png',
      input: binding('seeded', 'image'),
      output: binding('seeded', 'result'),
      seeds: [{ fieldName: 'seed', nodeId: 'seeded', seed: 900, seedStep: 1 }],
      templatesSnapshot: loaded({ seeded: seededTemplate }),
    });

    expect(built.graph.nodes.seeded).toMatchObject({ seed: 900 });
    expect(doc).toEqual(before);
  });

  it('externally satisfies only the selected connection input and appends collision-safe source and capture nodes', () => {
    const sinkTemplate = template('sink', {
      image: input('image', { input: 'connection', required: true, title: 'Layer image' }),
    });
    const resultTemplate = template('result', {}, { image: output('image', { title: 'Result image' }) });
    const doc = document([node('layer-workflow-source', sinkTemplate), node('layer-workflow-output', resultTemplate)]);
    const built = buildLayerWorkflowGraph({
      document: doc,
      imageName: 'layer.png',
      input: binding('layer-workflow-source', 'image'),
      output: binding('layer-workflow-output', 'image'),
      templatesSnapshot: loaded({ result: resultTemplate, sink: sinkTemplate }),
    });

    expect(built.outputNodeId).toBe('layer-workflow-output-1');
    expect(built.graph.nodes['layer-workflow-source-1']).toMatchObject({
      id: 'layer-workflow-source-1',
      image: { image_name: 'layer.png' },
      is_intermediate: true,
      type: 'image',
    });
    expect(built.graph.edges).toContainEqual({
      destination: { field: 'image', node_id: 'layer-workflow-source' },
      source: { field: 'image', node_id: 'layer-workflow-source-1' },
      type: 'default',
    });
    expect(built.graph.edges).toContainEqual({
      destination: { field: 'image', node_id: 'layer-workflow-output-1' },
      source: { field: 'image', node_id: 'layer-workflow-output' },
      type: 'default',
    });
  });

  it('reports readiness errors that remain after image injection and preserves the document', () => {
    const processorTemplate = template(
      'processor',
      {
        image: input('image', { input: 'direct', required: true, title: 'Layer image' }),
        prompt: input('prompt', {
          input: 'direct',
          required: true,
          title: 'Prompt',
          type: { batch: false, cardinality: 'SINGLE', name: 'StringField' },
        }),
      },
      { result: output('result') }
    );
    const doc = document([node('processor', processorTemplate)]);
    const before = structuredClone(doc);

    expect(() =>
      buildLayerWorkflowGraph({
        document: doc,
        imageName: 'layer.png',
        input: binding('processor', 'image'),
        output: binding('processor', 'result'),
        templatesSnapshot: loaded({ processor: processorTemplate }),
      })
    ).toThrow(/missing required input "Prompt"/);
    expect(doc).toEqual(before);
  });

  it('rejects stale input and output bindings', () => {
    const processorTemplate = template(
      'processor',
      { image: input('image', { input: 'direct', required: true }) },
      { result: output('result') }
    );
    const doc = document([node('processor', processorTemplate)]);
    const options = {
      document: doc,
      imageName: 'layer.png',
      input: binding('processor', 'image'),
      output: binding('processor', 'result'),
      templatesSnapshot: loaded({ processor: processorTemplate }),
    };

    expect(() => buildLayerWorkflowGraph({ ...options, input: binding('processor', 'removed') })).toThrow(
      /input binding is no longer available/i
    );
    expect(() => buildLayerWorkflowGraph({ ...options, output: binding('processor', 'removed') })).toThrow(
      /output binding is no longer available/i
    );
  });
});
