import { parseOpenApiToTemplates } from '@features/workflow/data/templates';
import { describe, expect, it, vi } from 'vitest';

import type { FieldInputTemplate, InvocationTemplate, InvocationTemplatesSnapshot, ProjectGraphState } from './types';

import { compileProjectGraph, getProjectGraphReadiness, isSeedInputField, planWorkflowSubmission } from './buildGraph';
import { CALL_SAVED_WORKFLOW_DYNAMIC_FIELD_PREFIX } from './callSavedWorkflow';
import {
  buildConnectorNode,
  buildInvocationNode,
  buildNotesNode,
  createProjectGraph,
  projectGraphReducer,
} from './document';
import { formatWorkflowReadinessReason } from './readinessIssues';

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
  type: { batch: false, cardinality: 'SINGLE', name: 'StringField' },
  uiChoiceLabels: null,
  uiComponent: null,
  uiHidden: false,
  uiModelBase: null,
  uiModelFormat: null,
  uiModelType: null,
  uiOrder: null,
  ...overrides,
});

const template = (type: string, inputs: Record<string, FieldInputTemplate>): InvocationTemplate => ({
  category: 'test',
  classification: 'stable',
  description: '',
  inputs,
  nodePack: 'invokeai',
  outputs: {
    out: {
      description: '',
      name: 'out',
      title: 'Out',
      type: { batch: false, cardinality: 'SINGLE', name: 'StringField' },
    },
  },
  outputType: `${type}_output`,
  tags: [],
  title: type,
  type,
  useCache: true,
  version: '1.0.0',
});

const templates = {
  sink: template('sink', {
    board: input('board', { type: { batch: false, cardinality: 'SINGLE', name: 'BoardField' } }),
    text: input('text', { required: true }),
  }),
  source: template('source', { value: input('value', { default: 'hello', required: true }) }),
};

const loadedSnapshot: InvocationTemplatesSnapshot = { error: null, status: 'loaded', templates };

const buildDocument = (): { doc: ProjectGraphState; sinkId: string; sourceId: string } => {
  const sourceNode = buildInvocationNode(templates.source, { x: 0, y: 0 });
  const sinkNode = buildInvocationNode(templates.sink, { x: 10, y: 0 });
  let doc = createProjectGraph('compile-test');

  doc = projectGraphReducer(doc, { node: sourceNode, type: 'addNode' });
  doc = projectGraphReducer(doc, { node: sinkNode, type: 'addNode' });
  doc = projectGraphReducer(doc, { node: buildNotesNode({ x: 0, y: 100 }), type: 'addNode' });
  doc = projectGraphReducer(doc, {
    edge: {
      id: 'e1',
      source: sourceNode.id,
      sourceHandle: 'out',
      target: sinkNode.id,
      targetHandle: 'text',
      type: 'default',
    },
    type: 'addEdge',
  });

  return { doc, sinkId: sinkNode.id, sourceId: sourceNode.id };
};

/** Readiness with each reason in its English wording, so expectations read as the user would. */
const readinessInEnglish = (...args: Parameters<typeof getProjectGraphReadiness>) => {
  const readiness = getProjectGraphReadiness(...args);

  return { ...readiness, reasons: readiness.reasons.map(formatWorkflowReadinessReason) };
};

describe('getProjectGraphReadiness', () => {
  it('requires loaded templates and at least one node', () => {
    const empty = createProjectGraph('empty');

    expect(readinessInEnglish(empty, { error: null, status: 'loading', templates: {} }).reasons[0]).toMatch(
      /still loading/
    );
    expect(readinessInEnglish(empty, loadedSnapshot).reasons[0]).toMatch(/no nodes/);
  });

  it('treats resolved connector output edges as connected required inputs', () => {
    const sourceNode = buildInvocationNode(templates.source, { x: 0, y: 0 });
    const sinkNode = buildInvocationNode(templates.sink, { x: 10, y: 0 });
    const connector = buildConnectorNode({ x: 5, y: 0 });
    let doc = createProjectGraph('connector-readiness');

    doc = projectGraphReducer(doc, { node: sourceNode, type: 'addNode' });
    doc = projectGraphReducer(doc, { node: sinkNode, type: 'addNode' });
    doc = projectGraphReducer(doc, { node: connector, type: 'addNode' });
    doc = projectGraphReducer(doc, {
      edge: {
        id: 'connector-in',
        source: sourceNode.id,
        sourceHandle: 'out',
        target: connector.id,
        targetHandle: 'in',
        type: 'default',
      },
      type: 'addEdge',
    });
    doc = projectGraphReducer(doc, {
      edge: {
        id: 'connector-out',
        source: connector.id,
        sourceHandle: 'out',
        target: sinkNode.id,
        targetHandle: 'text',
        type: 'default',
      },
      type: 'addEdge',
    });

    expect(readinessInEnglish(doc, loadedSnapshot)).toEqual({ batch: null, canInvoke: true, reasons: [] });
  });

  it('reports missing required inputs and unknown node types', () => {
    const { doc, sourceId } = buildDocument();
    const withEmptyString = projectGraphReducer(doc, {
      fieldName: 'value',
      nodeId: sourceId,
      type: 'setFieldValue',
      value: '',
    });

    expect(readinessInEnglish(withEmptyString, loadedSnapshot)).toEqual({
      batch: null,
      canInvoke: true,
      reasons: [],
    });

    const withMissingValue = projectGraphReducer(doc, {
      fieldName: 'value',
      nodeId: sourceId,
      type: 'setFieldValue',
      value: undefined,
    });

    expect(readinessInEnglish(withMissingValue, loadedSnapshot).reasons[0]).toMatch(/missing required input/);

    const unknownTemplates: InvocationTemplatesSnapshot = {
      error: null,
      status: 'loaded',
      templates: { source: templates.source },
    };

    expect(readinessInEnglish(doc, unknownTemplates).reasons[0]).toMatch(/Unknown node type "sink"/);
  });

  it('treats only explicitly externally satisfied connection inputs as ready', () => {
    const connectionTemplate = template('connection-sink', {
      image: input('image', {
        input: 'connection',
        required: true,
        title: 'Layer image',
        type: { batch: false, cardinality: 'SINGLE', name: 'ImageField' },
      }),
      mask: input('mask', {
        input: 'connection',
        required: true,
        title: 'Mask image',
        type: { batch: false, cardinality: 'SINGLE', name: 'ImageField' },
      }),
    });
    const node = buildInvocationNode(connectionTemplate, { x: 0, y: 0 });
    const doc = { ...createProjectGraph('external-input'), nodes: [{ ...node, id: 'sink' }] };
    const snapshot: InvocationTemplatesSnapshot = {
      error: null,
      status: 'loaded',
      templates: { 'connection-sink': connectionTemplate },
    };

    expect(readinessInEnglish(doc, snapshot, { externallySatisfiedInputs: new Set(['sink:image']) })).toEqual({
      batch: null,
      canInvoke: false,
      reasons: ['"connection-sink" is missing a connection for "Mask image".'],
    });
    expect(
      readinessInEnglish(doc, snapshot, {
        externallySatisfiedInputs: new Set(['sink:image', 'sink:mask']),
      })
    ).toEqual({ batch: null, canInvoke: true, reasons: [] });
  });
});

describe('compileProjectGraph', () => {
  it('compiles nodes and edges, omitting notes nodes and connected direct values', () => {
    const { doc, sinkId, sourceId } = buildDocument();
    const withStaleValue = projectGraphReducer(doc, {
      fieldName: 'text',
      nodeId: sinkId,
      type: 'setFieldValue',
      value: 'stale direct value',
    });
    const compiled = compileProjectGraph(withStaleValue, templates);
    const backendGraph = compiled.backendGraph;

    expect(backendGraph).toBeDefined();
    expect(Object.keys(backendGraph?.nodes ?? {}).sort()).toEqual([sinkId, sourceId].sort());
    expect(backendGraph?.nodes[sourceId]).toMatchObject({ type: 'source', use_cache: true, value: 'hello' });
    // The connected input's direct value must not be sent alongside the edge.
    expect(backendGraph?.nodes[sinkId]).not.toHaveProperty('text');
    expect(backendGraph?.edges).toEqual([
      { destination: { field: 'text', node_id: sinkId }, source: { field: 'out', node_id: sourceId }, type: 'default' },
    ]);
  });

  it('resolves Auto and an unset board to the auto-add board and leaves None Uncategorized', () => {
    const { doc, sinkId } = buildDocument();
    const withBoard = (value: unknown) =>
      compileProjectGraph(
        projectGraphReducer(doc, { fieldName: 'board', nodeId: sinkId, type: 'setFieldValue', value }),
        templates,
        {
          autoBoardId: 'auto-board',
        }
      ).backendGraph?.nodes[sinkId];

    expect(withBoard('auto')).toMatchObject({ board: { board_id: 'auto-board' } });
    expect(withBoard(undefined)).toMatchObject({ board: { board_id: 'auto-board' } });
    expect(withBoard('none')).not.toHaveProperty('board');
    expect(withBoard({ board_id: 'board-1' })).toMatchObject({ board: { board_id: 'board-1' } });

    // An auto-add board of Uncategorized resolves Auto to no board at all.
    const toUncategorized = projectGraphReducer(doc, {
      fieldName: 'board',
      nodeId: sinkId,
      type: 'setFieldValue',
      value: 'auto',
    });

    expect(
      compileProjectGraph(toUncategorized, templates, { autoBoardId: 'none' }).backendGraph?.nodes[sinkId]
    ).not.toHaveProperty('board');
  });

  it('preserves auto/none board sentinels only in Call Saved Workflow inputs', () => {
    const callTemplate = template('call_saved_workflow', {
      workflow_id: input('workflow_id', { type: { batch: false, cardinality: 'SINGLE', name: 'StringField' } }),
    });
    const callNode = buildInvocationNode(callTemplate, { x: 0, y: 0 });
    const dynamicBoardName = `${CALL_SAVED_WORKFLOW_DYNAMIC_FIELD_PREFIX}${callNode.id}::board`;
    callNode.data.dynamicInputTemplates = {
      [dynamicBoardName]: input(dynamicBoardName, {
        type: { batch: false, cardinality: 'SINGLE', name: 'BoardField' },
      }),
    };
    callNode.data.inputs[dynamicBoardName] = { label: 'Board', name: dynamicBoardName, value: 'auto' };
    const document = { ...createProjectGraph('call-saved-workflow-board'), nodes: [callNode] };
    const graph = compileProjectGraph(document, { call_saved_workflow: callTemplate }).backendGraph;

    expect(graph.nodes[callNode.id]).not.toHaveProperty(dynamicBoardName);
    expect(graph.nodes[callNode.id]).toMatchObject({
      workflow_inputs: { [dynamicBoardName]: 'auto' },
    });
    // The child workflow's Auto resolves like this workflow's own; None stays a sentinel for its builder.
    const resolved = compileProjectGraph(
      document,
      { call_saved_workflow: callTemplate },
      { autoBoardId: 'auto-board' }
    );

    expect(resolved.backendGraph.nodes[callNode.id]).toMatchObject({
      workflow_inputs: { [dynamicBoardName]: { board_id: 'auto-board' } },
    });

    callNode.data.inputs[dynamicBoardName] = { label: 'Board', name: dynamicBoardName, value: 'none' };

    expect(
      compileProjectGraph(document, { call_saved_workflow: callTemplate }, { autoBoardId: 'auto-board' }).backendGraph
        .nodes[callNode.id]
    ).toMatchObject({ workflow_inputs: { [dynamicBoardName]: 'none' } });
  });

  it('resolves connector chains into executable backend edges', () => {
    const sourceNode = buildInvocationNode(templates.source, { x: 0, y: 0 });
    const sinkNode = buildInvocationNode(templates.sink, { x: 10, y: 0 });
    const connector = buildConnectorNode({ x: 5, y: 0 });
    let doc = createProjectGraph('connector-compile');

    doc = projectGraphReducer(doc, { node: sourceNode, type: 'addNode' });
    doc = projectGraphReducer(doc, { node: sinkNode, type: 'addNode' });
    doc = projectGraphReducer(doc, { node: connector, type: 'addNode' });
    doc = projectGraphReducer(doc, {
      edge: {
        id: 'connector-in',
        source: sourceNode.id,
        sourceHandle: 'out',
        target: connector.id,
        targetHandle: 'in',
        type: 'default',
      },
      type: 'addEdge',
    });
    doc = projectGraphReducer(doc, {
      edge: {
        id: 'connector-out',
        source: connector.id,
        sourceHandle: 'out',
        target: sinkNode.id,
        targetHandle: 'text',
        type: 'default',
      },
      type: 'addEdge',
    });

    expect(compileProjectGraph(doc, templates).backendGraph?.edges).toEqual([
      {
        destination: { field: 'text', node_id: sinkNode.id },
        source: { field: 'out', node_id: sourceNode.id },
        type: 'default',
      },
    ]);
  });

  it('preserves the direct loop_linkage edge type in the queue graph', () => {
    const forTemplate: InvocationTemplate = {
      ...template('for', {
        collection: input('collection', {
          default: [],
          type: { batch: false, cardinality: 'COLLECTION', name: 'CollectionField' },
        }),
      }),
      outputs: {
        item: {
          description: '',
          name: 'item',
          outputScope: 'iteration',
          title: 'Item',
          type: { batch: false, cardinality: 'SINGLE', name: 'CollectionItemField' },
        },
        loop_linkage: {
          description: '',
          name: 'loop_linkage',
          title: 'Loop linkage',
          type: { batch: false, cardinality: 'SINGLE', name: 'AnyField' },
        },
        output_collection: {
          description: '',
          name: 'output_collection',
          outputScope: 'final',
          title: 'Output collection',
          type: { batch: false, cardinality: 'COLLECTION', name: 'CollectionField' },
        },
      },
    };
    const returnTemplate: InvocationTemplate = {
      ...template('for_return', {
        loop_linkage: input('loop_linkage', {
          input: 'connection',
          type: { batch: false, cardinality: 'SINGLE', name: 'AnyField' },
        }),
        output: input('output', {
          input: 'any',
          type: { batch: false, cardinality: 'SINGLE', name: 'CollectionItemField' },
        }),
      }),
      outputs: {},
    };
    const forNode = buildInvocationNode(forTemplate, { x: 0, y: 0 });
    const returnNode = buildInvocationNode(returnTemplate, { x: 100, y: 0 });
    let doc = createProjectGraph('loop-compile');

    doc = projectGraphReducer(doc, { node: forNode, type: 'addNode' });
    doc = projectGraphReducer(doc, { node: returnNode, type: 'addNode' });
    doc = projectGraphReducer(doc, {
      edge: {
        id: 'item',
        source: forNode.id,
        sourceHandle: 'item',
        target: returnNode.id,
        targetHandle: 'output',
        type: 'default',
      },
      type: 'addEdge',
    });
    doc = projectGraphReducer(doc, {
      edge: {
        id: 'linkage',
        source: forNode.id,
        sourceHandle: 'loop_linkage',
        target: returnNode.id,
        targetHandle: 'loop_linkage',
        type: 'loop_linkage',
      },
      type: 'addEdge',
    });

    const graph = compileProjectGraph(doc, { for: forTemplate, for_return: returnTemplate }).backendGraph;

    expect(graph.edges).toEqual([
      {
        destination: { field: 'output', node_id: returnNode.id },
        source: { field: 'item', node_id: forNode.id },
        type: 'default',
      },
      {
        destination: { field: 'loop_linkage', node_id: returnNode.id },
        source: { field: 'loop_linkage', node_id: forNode.id },
        type: 'loop_linkage',
      },
    ]);
  });
});

describe('planWorkflowSubmission', () => {
  /** These graphs have no batch nodes, so a plan is always produced. */
  const planSubmission = (...args: Parameters<typeof planWorkflowSubmission>) => {
    const result = planWorkflowSubmission(...args);

    if (!result) {
      throw new Error('Expected a submission plan');
    }

    return result;
  };

  const SEED_MAX = 4_294_967_295;
  const seedInput = input('seed', {
    default: 0,
    maximum: SEED_MAX,
    minimum: 0,
    type: { batch: false, cardinality: 'SINGLE', name: 'IntegerField' },
  });
  const seededTemplates = {
    integer: template('integer', {
      value: input('value', { default: 0, type: { batch: false, cardinality: 'SINGLE', name: 'IntegerField' } }),
    }),
    noise: template('noise', { seed: seedInput, width: input('width', { default: 512 }) }),
  };
  const seededTemplate = seededTemplates.noise;

  const buildSeeded = (nodes: Array<{ seed?: number; seedMode?: 'random' | 'fixed' | 'increment' | 'decrement' }>) => {
    let doc = createProjectGraph('seed-plan');
    const ids: string[] = [];

    for (const [index, config] of nodes.entries()) {
      const node = buildInvocationNode(seededTemplate, { x: index * 100, y: 0 });

      ids.push(node.id);
      doc = projectGraphReducer(doc, { node, type: 'addNode' });

      if (config.seed !== undefined) {
        doc = projectGraphReducer(doc, {
          fieldName: 'seed',
          nodeId: node.id,
          type: 'setFieldValue',
          value: config.seed,
        });
      }

      if (config.seedMode) {
        doc = projectGraphReducer(doc, {
          fieldName: 'seed',
          nodeId: node.id,
          seedMode: config.seedMode,
          type: 'setFieldSeedMode',
        });
      }
    }

    return { doc, ids };
  };

  it('records one start per stepping seed and where each field goes after the batch', () => {
    const { doc, ids } = buildSeeded([
      { seed: 42, seedMode: 'increment' },
      { seed: 100, seedMode: 'decrement' },
      { seed: 5 },
    ]);
    const plan = planSubmission(doc, seededTemplates, { batchCount: 3 });

    expect(plan.seeds).toEqual([
      { fieldName: 'seed', nodeId: ids[0], seed: 42, seedStep: 1 },
      { fieldName: 'seed', nodeId: ids[1], seed: 100, seedStep: -1 },
    ]);
    expect(plan.seedAdvances).toEqual([
      { fieldName: 'seed', fromSeed: 42, nodeId: ids[0], seedMode: 'increment', toSeed: 45 },
      { fieldName: 'seed', fromSeed: 100, nodeId: ids[1], seedMode: 'decrement', toSeed: 97 },
    ]);
    // Every node carries its first run's seed as a constant; the fixed one stays as authored.
    expect(plan.graph.backendGraph.nodes[ids[0] as string]?.seed).toBe(42);
    expect(plan.graph.backendGraph.nodes[ids[2] as string]?.seed).toBe(5);
    expect(plan.graph.nodes.find((node) => node.id === ids[1])?.inputs.seed).toBe(100);
  });

  it('draws a random start into the graph and never advances the entered seed', () => {
    const { doc, ids } = buildSeeded([
      { seed: 42, seedMode: 'random' },
      { seed: 42, seedMode: 'random' },
    ]);
    // A constant draw: the graph id generator shares the random source, so a queue of draws would skew.
    const random = vi.spyOn(Math, 'random').mockReturnValue(0.25);
    const start = Math.floor(0.25 * SEED_MAX);

    try {
      const plan = planSubmission(doc, seededTemplates, { batchCount: 2 });

      expect(plan.seeds).toEqual([
        { fieldName: 'seed', nodeId: ids[0], seed: start, seedStep: 1 },
        { fieldName: 'seed', nodeId: ids[1], seed: start, seedStep: 1 },
      ]);
      expect(plan.seedAdvances).toEqual([]);
      expect(plan.graph.backendGraph.nodes[ids[0] as string]?.seed).toBe(start);
      // The document keeps the entered seed in reserve.
      expect(doc.nodes[0]).toMatchObject({ data: { inputs: { seed: { value: 42 } } } });
    } finally {
      random.mockRestore();
    }
  });
});

describe('seed inputs', () => {
  const seed = (overrides: Partial<FieldInputTemplate> = {}) =>
    input('seed', {
      maximum: 4_294_967_295,
      minimum: 0,
      type: { batch: false, cardinality: 'SINGLE', name: 'IntegerField' },
      ...overrides,
    });

  it('recognises the scalar integer a node declares as seed over the seed range', () => {
    expect(isSeedInputField(seed())).toBe(true);
    // The title is what a user edits; the name and range are what the backend declared.
    expect(isSeedInputField(seed({ title: 'Noise seed' }))).toBe(true);
  });
});

describe('integer Literal enum values in the compiled graph', () => {
  const literalEnumSchema = {
    components: {
      schemas: {
        IntegerOutput: {
          class: 'output',
          properties: { type: { const: 'integer_output' }, value: { field_kind: 'output', type: 'integer' } },
          type: 'object',
        },
        MaxSeqLenInvocation: {
          class: 'invocation',
          output: { $ref: '#/components/schemas/IntegerOutput' },
          properties: {
            max_seq_len: {
              default: 512,
              enum: [256, 512],
              field_kind: 'input',
              orig_required: false,
              title: 'Max Seq Length',
              type: 'integer',
            },
            type: { default: 'max_seq_len_invocation' },
          },
          title: 'MaxSeqLen',
          type: 'object',
        },
      },
    },
  };

  it('sends a numeric Literal enum value', () => {
    const parsedTemplates = parseOpenApiToTemplates(literalEnumSchema);
    const parsedTemplate = parsedTemplates.max_seq_len_invocation;

    if (!parsedTemplate) {
      throw new Error('fixture template was not parsed');
    }

    const node = buildInvocationNode(parsedTemplate, { x: 0, y: 0 });
    const doc = projectGraphReducer(createProjectGraph('literal-enum'), { node, type: 'addNode' });

    expect(compileProjectGraph(doc, parsedTemplates).backendGraph?.nodes[node.id]).toMatchObject({
      max_seq_len: 512,
    });
  });
});

describe('batch nodes', () => {
  const list = (name: string, typeName: string): FieldInputTemplate =>
    input(name, { required: true, type: { batch: true, cardinality: 'COLLECTION', name: typeName } });
  const batchTemplates = {
    ...templates,
    float_batch: template('float_batch', {
      batch_group_id: input('batch_group_id', {
        default: 'None',
        options: ['None', 'Group 1'],
        type: { batch: false, cardinality: 'SINGLE', name: 'EnumField' },
      }),
      floats: list('floats', 'FloatField'),
    }),
    float_generator: template('float_generator', {
      generator: input('generator', {
        default: { count: 10, start: 0, step: 0.1, type: 'float_generator_arithmetic_sequence' },
        input: 'direct',
        required: true,
        type: { batch: false, cardinality: 'SINGLE', name: 'FloatGeneratorField' },
      }),
    }),
    noise: template('noise', {
      cfg: input('cfg', { default: 7, type: { batch: false, cardinality: 'SINGLE', name: 'FloatField' } }),
      seed: input('seed', {
        default: 0,
        maximum: 4_294_967_295,
        minimum: 0,
        type: { batch: false, cardinality: 'SINGLE', name: 'IntegerField' },
      }),
    }),
  };
  const snapshot: InvocationTemplatesSnapshot = { error: null, status: 'loaded', templates: batchTemplates };
  const build = (floats: number[], generator?: Record<string, unknown>) => {
    const noise = buildInvocationNode(batchTemplates.noise, { x: 0, y: 0 });
    const batch = buildInvocationNode(batchTemplates.float_batch, { x: 0, y: 100 });
    let doc = createProjectGraph('batch-plan');

    doc = projectGraphReducer(doc, { node: noise, type: 'addNode' });
    doc = projectGraphReducer(doc, { node: batch, type: 'addNode' });
    doc = projectGraphReducer(doc, { fieldName: 'floats', nodeId: batch.id, type: 'setFieldValue', value: floats });
    doc = projectGraphReducer(doc, {
      edge: {
        id: 'b->n',
        source: batch.id,
        sourceHandle: 'value',
        target: noise.id,
        targetHandle: 'cfg',
        type: 'default',
      },
      type: 'addEdge',
    });

    if (generator) {
      const gen = buildInvocationNode(batchTemplates.float_generator, { x: 0, y: 200 });

      doc = projectGraphReducer(doc, { node: gen, type: 'addNode' });
      doc = projectGraphReducer(doc, {
        fieldName: 'generator',
        nodeId: gen.id,
        type: 'setFieldValue',
        value: generator,
      });
      doc = projectGraphReducer(doc, {
        edge: {
          id: 'g->b',
          source: gen.id,
          sourceHandle: 'floats',
          target: batch.id,
          targetHandle: 'floats',
          type: 'default',
        },
        type: 'addEdge',
      });
    }

    return { batchId: batch.id, doc, noiseId: noise.id };
  };

  it('refuses a batch whose runs would overflow the queue before anything is submitted', () => {
    const { doc } = build([1.5, 2.5]);

    expect(readinessInEnglish(doc, snapshot, { batchCount: 5_000 })).toEqual({
      batch: { size: 2 },
      canInvoke: true,
      reasons: [],
    });
    expect(readinessInEnglish(doc, snapshot, { batchCount: 5_001 })).toMatchObject({
      canInvoke: false,
      reasons: ['This batch would queue 10,002 sessions; the queue accepts at most 10,000.'],
    });
  });

  it("compiles the executable graph without batch or generator nodes and keeps the fed input's static value", () => {
    const { batchId, doc, noiseId } = build([1.5, 2.5], {
      count: 3,
      start: 1,
      step: 1,
      type: 'float_generator_arithmetic_sequence',
    });
    const compiled = compileProjectGraph(doc, batchTemplates);

    expect(Object.keys(compiled.backendGraph.nodes)).toEqual([noiseId]);
    expect(compiled.backendGraph.edges).toEqual([]);
    expect(compiled.backendGraph.nodes[noiseId]?.cfg).toBe(7);
    expect(compiled.nodes.map((node) => node.id)).not.toContain(batchId);
  });

  it('plans batch data beside the seed walk, sized for every session, and nothing while the batch is blocked', () => {
    const { doc, noiseId } = build([1.5, 2.5]);
    const stepping = projectGraphReducer(doc, {
      fieldName: 'seed',
      nodeId: noiseId,
      seedMode: 'increment',
      type: 'setFieldSeedMode',
    });
    const plan = planWorkflowSubmission(stepping, batchTemplates, { batchCount: 3 });

    expect(plan).toMatchObject({
      batchCount: 3,
      batchData: [[{ fieldName: 'cfg', items: [1.5, 2.5], nodeId: noiseId }]],
      batchSize: 2,
      seeds: [{ fieldName: 'seed', nodeId: noiseId, seed: 0, seedStep: 1 }],
    });
    // Three runs of two sessions: the next authored seed sits past all six.
    expect(plan?.seedAdvances).toEqual([
      { fieldName: 'seed', fromSeed: 0, nodeId: noiseId, seedMode: 'increment', toSeed: 6 },
    ]);

    expect(planWorkflowSubmission(build([]).doc, batchTemplates, { batchCount: 1 })).toBeNull();
    expect(planWorkflowSubmission(doc, batchTemplates, { batchCount: 6_000 })).toBeNull();
  });
});
