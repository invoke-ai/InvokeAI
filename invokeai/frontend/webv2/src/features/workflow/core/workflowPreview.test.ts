import { describe, expect, it } from 'vitest';

import type {
  FieldInputTemplate,
  FieldType,
  InvocationTemplate,
  InvocationTemplates,
  ProjectGraphState,
  WorkflowEdge,
  WorkflowFormElement,
  WorkflowInvocationNode,
} from './types';

import extendVideo from './fixtures/extend-video-ltx2.json';
import { parseWorkflowJson } from './workflowJson';
import {
  buildWorkflowPreviewModel,
  chooseWorkflowPreviewChipWidth,
  compiledGraphToPreviewDocument,
  getWorkflowPreviewMetrics,
  findWorkflowPreviewNodes,
  WORKFLOW_PREVIEW_METRICS,
  type WorkflowPreviewMetrics,
  type WorkflowPreviewModel,
} from './workflowPreview';

const METRICS: WorkflowPreviewMetrics = {
  authorScale: 0.5,
  layerGap: 40,
  nodeHeight: 50,
  nodeWidth: 100,
  pillHeight: 20,
  rowGap: 10,
};

const type = (name: string, cardinality: FieldType['cardinality'] = 'SINGLE'): FieldType => ({
  batch: false,
  cardinality,
  name,
});

const input = (
  name: string,
  fieldType: FieldType,
  overrides: Partial<FieldInputTemplate> = {}
): FieldInputTemplate => ({
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
  title: name.replace(/^./, (letter) => letter.toUpperCase()),
  type: fieldType,
  uiChoiceLabels: null,
  uiComponent: null,
  uiHidden: false,
  uiModelBase: null,
  uiModelFormat: null,
  uiModelType: null,
  uiOrder: null,
  ...overrides,
});

const template = (name: string, title: string, inputs: FieldInputTemplate[]): InvocationTemplate => ({
  category: 'test',
  classification: 'stable',
  description: '',
  inputs: Object.fromEntries(inputs.map((field) => [field.name, field])),
  nodePack: 'invokeai',
  outputs: {},
  outputType: `${name}_output`,
  tags: [],
  title,
  type: name,
  useCache: true,
  version: '1.0.0',
});

const TEMPLATES: InvocationTemplates = {
  collect: template('collect', 'Collect', [input('item', type('AnyField'))]),
  core_metadata: template('core_metadata', 'Metadata', [input('seed', type('IntegerField'))]),
  denoise: template('denoise', 'Denoise Latents', [
    input('steps', type('IntegerField')),
    input('cfg_scale', type('FloatField')),
    input('noise', type('LatentsField')),
    input('positive', type('ConditioningField')),
    input('seed', type('IntegerField')),
  ]),
  image_batch: template('image_batch', 'Image Batch', [input('images', type('ImageField', 'COLLECTION'))]),
  integer: template('integer', 'Integer Primitive', [input('value', type('IntegerField'))]),
  l2i: template('l2i', 'Latents to Image', [
    input('latents', type('LatentsField')),
    input('metadata', type('MetadataField')),
  ]),
  model_loader: template('model_loader', 'Main Model', [
    input('model', type('ModelIdentifierField'), { required: true, uiModelBase: ['sdxl'], uiModelType: ['main'] }),
  ]),
  prompt: template('prompt', 'Prompt', [input('prompt', type('StringField'))]),
  string: template('string', 'String Primitive', [input('value', type('StringField'))]),
};

const node = (
  id: string,
  nodeType: string,
  values: Record<string, unknown> = {},
  extras: { label?: string; notes?: string; fieldLabels?: Record<string, string> } = {}
): WorkflowInvocationNode => ({
  data: {
    inputs: Object.fromEntries(
      Object.entries(values).map(([name, value]) => [name, { label: extras.fieldLabels?.[name] ?? '', name, value }])
    ),
    isIntermediate: true,
    isOpen: true,
    label: extras.label ?? '',
    nodePack: 'invokeai',
    notes: extras.notes ?? '',
    type: nodeType,
    useCache: true,
    version: '1.0.0',
  },
  id,
  position: { x: id.length * 10, y: 0 },
  type: 'invocation',
});

const edge = (source: string, target: string, targetHandle: string, sourceHandle = 'value'): WorkflowEdge => ({
  id: `${source}-${target}-${targetHandle}`,
  source,
  sourceHandle,
  target,
  targetHandle,
  type: 'default',
});

const exposed = (...fields: [nodeId: string, fieldName: string][]): ProjectGraphState['form'] => {
  const elements: Record<string, WorkflowFormElement> = {
    root: {
      data: { children: fields.map((_, index) => `field-${index}`), layout: 'column' },
      id: 'root',
      type: 'container',
    },
  };

  fields.forEach(([nodeId, fieldName], index) => {
    elements[`field-${index}`] = {
      data: { fieldIdentifier: { fieldName, nodeId }, showDescription: false, showShuffle: false },
      id: `field-${index}`,
      parentId: 'root',
      type: 'node-field',
    };
  });

  return { elements, rootElementId: 'root' };
};

const graph = (
  nodes: WorkflowInvocationNode[],
  edges: WorkflowEdge[],
  form: ProjectGraphState['form'] = exposed()
): ProjectGraphState => ({
  author: '',
  contact: '',
  description: '',
  edges,
  form,
  id: 'preview-test',
  name: 'Preview test',
  nodes,
  notes: '',
  tags: '',
  updatedAt: '2026-01-01T00:00:00.000Z',
  version: 2,
  workflowVersion: '1.0.0',
});

/**
 * A text-to-image graph: prompt and seed primitives, a model loader, a collect feeding the denoiser, and metadata
 * recording the seed on the decoder.
 */
const TEXT_TO_IMAGE = graph(
  [
    node('positive', 'string', { value: 'a lighthouse at dusk, long exposure, film grain and soft haze' }),
    node('seed', 'integer', { value: 7 }),
    node('model', 'model_loader', { model: undefined }),
    node('prompt', 'prompt', { prompt: '' }, { label: 'Prompt - SDXL', notes: 'Encodes the positive prompt.' }),
    node('collect', 'collect'),
    node('denoise', 'denoise', { cfg_scale: 4.5, seed: 0, steps: 30 }, { fieldLabels: { steps: 'Sampling steps' } }),
    node('metadata', 'core_metadata'),
    node('decode', 'l2i'),
  ],
  [
    edge('positive', 'prompt', 'prompt'),
    edge('model', 'prompt', 'clip', 'clip'),
    edge('prompt', 'collect', 'item', 'conditioning'),
    edge('collect', 'denoise', 'positive', 'collection'),
    edge('seed', 'denoise', 'seed'),
    edge('seed', 'metadata', 'seed'),
    edge('model', 'metadata', 'model', 'unet'),
    edge('denoise', 'decode', 'latents', 'latents'),
    edge('metadata', 'decode', 'metadata', 'metadata'),
  ],
  exposed(['positive', 'value'], ['denoise', 'steps'])
);

const build = (document: ProjectGraphState) => buildWorkflowPreviewModel(document, TEMPLATES, METRICS);
const titles = (model: WorkflowPreviewModel, ids: readonly string[]) => ids.map((id) => model.nodes.get(id)!.title);

describe('buildWorkflowPreviewModel', () => {
  it('names nodes by their own label, then their template, then their type id, and marks an unknown type', () => {
    const model = build(
      graph(
        [node('prompt', 'prompt', {}, { label: 'Prompt - SDXL' }), node('seed', 'integer'), node('x', 'face_mask')],
        []
      )
    );

    expect(model.nodes.get('prompt')!.title).toBe('Prompt - SDXL');
    expect(model.nodes.get('seed')!.title).toBe('Integer Primitive');
    expect(model.nodes.get('x')).toMatchObject({ isUnknown: true, title: 'face_mask' });
    expect(model.nodes.get('seed')!.isUnknown).toBe(false);
  });

  it('lists a node’s own values with field labels, leaving out connected and empty inputs', () => {
    const model = build(TEXT_TO_IMAGE);

    // `seed` is fed by the seed primitive, so its stale 0 is not the node's own value.
    expect(model.nodes.get('denoise')!.fields).toEqual([
      { isExposed: true, label: 'Sampling steps', name: 'steps', value: { kind: 'number', value: 30 } },
      { isExposed: false, label: 'Cfg_scale', name: 'cfg_scale', value: { kind: 'number', value: 4.5 } },
    ]);
    expect(model.nodes.get('prompt')!.fields).toEqual([]);
    expect(model.nodes.get('prompt')!.notes).toBe('Encodes the positive prompt.');
  });

  it('marks exposed nodes, results and the model fields a node still needs', () => {
    const model = build(TEXT_TO_IMAGE);
    const marked = (key: 'isExposed' | 'isResult') =>
      [...model.nodes.values()].filter((candidate) => candidate[key]).map((candidate) => candidate.id);

    expect(marked('isExposed')).toEqual(['positive', 'denoise']);
    // The decoder feeds nothing; metadata and the primitives are helpers, never results.
    expect(marked('isResult')).toEqual(['decode']);
    expect(model.nodes.get('model')!.modelRequirements).toEqual([
      { base: 'sdxl', kind: 'slot', label: 'SDXL checkpoint', modelType: 'main' },
    ]);
    expect(model.nodes.get('denoise')!.modelRequirements).toEqual([]);
  });

  it('orders nodes and their neighbours topologically', () => {
    const model = build(TEXT_TO_IMAGE);

    expect(model.all.nodeIds).toEqual([
      'positive',
      'seed',
      'model',
      'prompt',
      'metadata',
      'collect',
      'denoise',
      'decode',
    ]);
    expect(model.all.neighbours.denoise).toEqual({ sources: ['seed', 'collect'], targets: ['decode'] });
    expect(model.simplified!.neighbours.denoise).toEqual({ sources: ['prompt'], targets: ['decode'] });
  });

  it('draws every node in the author’s arrangement in the full view, scaled to the chips, one edge per pair', () => {
    const a = { ...node('a', 'integer', { value: 1 }), position: { x: -400, y: 300 } };
    const b = { ...node('b', 'denoise'), position: { x: 200, y: 100 } };
    const model = build(graph([a, b], [edge('a', 'b', 'steps'), edge('a', 'b', 'seed')]));

    // Moved to the origin and halved: editor nodes are about twice a chip's size.
    expect(model.all.positions).toEqual({ a: { x: 0, y: 100 }, b: { x: 300, y: 0 } });
    expect(model.all.edges).toHaveLength(1);
    expect(model.all.pills).toEqual({});
  });

  it('folds value-only primitives into pills on their consumers and hides plumbing, rerouting through a collect', () => {
    const { nodes, simplified } = build(TEXT_TO_IMAGE);

    expect(simplified).not.toBeNull();
    expect(titles({ nodes } as WorkflowPreviewModel, simplified!.nodeIds)).toEqual([
      'Main Model',
      'Prompt - SDXL',
      'Denoise Latents',
      'Latents to Image',
    ]);
    // The exposed prompt is labelled as the form labels it, its long value left off; the seed shows its value.
    expect(simplified!.pills).toEqual({
      denoise: [{ label: 'Integer Primitive', nodeId: 'seed', value: { kind: 'number', value: 7 } }],
      prompt: [{ label: 'Value', nodeId: 'positive', value: null }],
    });
    // The collect is rerouted; metadata only records what reaches it, so the seed and model get no line to the decoder.
    expect(simplified!.pills.decode).toBeUndefined();
    expect(
      simplified!.edges.map(
        ({ isRerouted, source, target }) => `${source}->${target}${isRerouted ? ' (rerouted)' : ''}`
      )
    ).toEqual(['model->prompt', 'denoise->decode', 'prompt->denoise (rerouted)']);
  });

  it('lays the simplified view out in layers, leaving room above a node for its pills', () => {
    const { simplified } = build(TEXT_TO_IMAGE);
    const { positions } = simplified!;

    expect(positions.model!.x).toBeLessThan(positions.prompt!.x);
    expect(positions.prompt!.x).toBeLessThan(positions.denoise!.x);
    expect(positions.denoise!.x).toBeLessThan(positions.decode!.x);
    // Layer pitch is the node width plus the gap.
    expect(positions.prompt!.x - positions.model!.x).toBe(METRICS.nodeWidth + METRICS.layerGap);

    // In one layer, a node with a pill keeps the row gap between the chip above and its pill, not its chip.
    const stacked = build(
      graph(
        [
          node('model', 'model_loader'),
          node('seed', 'integer', { value: 1 }),
          node('a', 'denoise'),
          node('b', 'denoise'),
        ],
        [edge('model', 'a', 'noise', 'unet'), edge('model', 'b', 'noise', 'unet'), edge('seed', 'b', 'seed')]
      )
    ).simplified!;

    expect(stacked.pills.b).toHaveLength(1);
    expect(stacked.positions.b!.y - stacked.positions.a!.y).toBe(
      METRICS.nodeHeight + METRICS.rowGap + METRICS.pillHeight
    );
  });

  it('levels a chain’s chips whatever pills they carry, starting the topmost pill at 0', () => {
    // model -> prompt (two pills) -> denoise (none) -> decode (one pill).
    const { simplified } = build(
      graph(
        [
          node('model', 'model_loader'),
          node('positive', 'string', { value: 'a lighthouse' }),
          node('negative', 'string', { value: 'blurry' }),
          node('prompt', 'prompt'),
          node('denoise', 'denoise'),
          node('strength', 'integer', { value: 2 }),
          node('decode', 'l2i'),
        ],
        [
          edge('model', 'prompt', 'clip', 'clip'),
          edge('positive', 'prompt', 'prompt'),
          edge('negative', 'prompt', 'prompt'),
          edge('prompt', 'denoise', 'positive', 'conditioning'),
          edge('denoise', 'decode', 'latents', 'latents'),
          edge('strength', 'decode', 'metadata'),
        ]
      )
    );
    const { pills, positions } = simplified!;

    expect([pills.model?.length, pills.prompt?.length, pills.denoise?.length, pills.decode?.length]).toEqual([
      undefined,
      2,
      undefined,
      1,
    ]);
    // Every chip on one row, so every edge runs level, and the tallest pill stack fits above it from 0.
    expect(new Set(['model', 'prompt', 'denoise', 'decode'].map((id) => positions[id]!.y))).toEqual(
      new Set([2 * METRICS.pillHeight])
    );
  });

  it('keeps a primitive that feeds nothing, a collection primitive and a primitive fed by another node', () => {
    const model = build(
      graph(
        [
          node('lonely', 'integer', { value: 3 }),
          node('images', 'image_batch', { images: [{ image_name: 'a.png' }] }),
          node('source', 'integer', { value: 1 }),
          node('fed', 'integer'),
          node('denoise', 'denoise'),
        ],
        [edge('images', 'denoise', 'noise', 'images'), edge('source', 'fed', 'value'), edge('fed', 'denoise', 'steps')]
      )
    );

    expect(model.simplified!.nodeIds).toEqual(['lonely', 'images', 'fed', 'denoise']);
    expect(model.simplified!.pills).toEqual({
      fed: [{ label: 'Integer Primitive', nodeId: 'source', value: { kind: 'number', value: 1 } }],
    });
    expect(model.nodes.get('images')!.fields[0]!.value).toEqual({ count: 1, kind: 'list' });
  });

  it('has no simplified view when nothing folds or hides', () => {
    const model = build(
      graph([node('denoise', 'denoise'), node('decode', 'l2i')], [edge('denoise', 'decode', 'latents')])
    );

    expect(model.simplified).toBeNull();
  });

  it('previews a workflow that would not compile: a connection to a node type the server lacks', () => {
    const model = build(graph([node('mask', 'face_mask'), node('decode', 'l2i')], [edge('mask', 'decode', 'latents')]));

    expect(model.all.edges).toHaveLength(1);
    expect(model.all.neighbours.mask!.targets).toEqual(['decode']);
  });

  it('reads a choice by its template’s label', () => {
    const sampler: InvocationTemplates = {
      sampler: template('sampler', 'Sampler', [
        input('scheduler', type('EnumField'), { uiChoiceLabels: { dpmpp_2m_sde_k: 'DPM++ 2M SDE Karras' } }),
        input('mode', type('EnumField')),
      ]),
    };
    const model = buildWorkflowPreviewModel(
      graph([node('s', 'sampler', { mode: 'fast', scheduler: 'dpmpp_2m_sde_k' })], []),
      sampler,
      METRICS
    );

    expect(model.nodes.get('s')!.fields.map((field) => field.value)).toEqual([
      { kind: 'text', text: 'DPM++ 2M SDE Karras' },
      { kind: 'text', text: 'fast' },
    ]);
  });

  it('reads the form as authored, with each input’s saved value and the model it still needs', () => {
    const document = graph(
      [
        node('positive', 'string', { value: '' }, { fieldLabels: { value: 'Prompt' } }),
        node('model', 'model_loader'),
        node('denoise', 'denoise', { steps: 30 }),
      ],
      [],
      {
        elements: {
          heading: { data: { content: 'Prompt' }, id: 'heading', parentId: 'root', type: 'heading' },
          intro: { data: { content: 'Describe the scene.' }, id: 'intro', parentId: 'root', type: 'text' },
          'field-positive': {
            data: {
              fieldIdentifier: { fieldName: 'value', nodeId: 'positive' },
              showDescription: false,
              showShuffle: false,
            },
            id: 'field-positive',
            parentId: 'row',
            type: 'node-field',
          },
          'field-steps': {
            data: {
              fieldIdentifier: { fieldName: 'steps', nodeId: 'denoise' },
              showDescription: false,
              showShuffle: false,
            },
            id: 'field-steps',
            parentId: 'row',
            type: 'node-field',
          },
          'field-gone': {
            data: {
              fieldIdentifier: { fieldName: 'value', nodeId: 'deleted' },
              showDescription: false,
              showShuffle: false,
            },
            id: 'field-gone',
            parentId: 'row',
            type: 'node-field',
          },
          row: {
            data: { children: ['field-positive', 'field-steps', 'field-gone'], layout: 'row' },
            id: 'row',
            type: 'container',
          },
          rule: { id: 'rule', parentId: 'root', type: 'divider' },
          'heading-model': { data: { content: 'Model' }, id: 'heading-model', parentId: 'root', type: 'heading' },
          'field-model': {
            data: {
              fieldIdentifier: { fieldName: 'model', nodeId: 'model' },
              showDescription: false,
              showShuffle: false,
            },
            id: 'field-model',
            parentId: 'root',
            type: 'node-field',
          },
          root: {
            data: { children: ['heading', 'intro', 'row', 'rule', 'heading-model', 'field-model'], layout: 'column' },
            id: 'root',
            type: 'container',
          },
        },
        rootElementId: 'root',
      }
    );
    const model = build(document);

    expect(model.form.map((item) => item.kind)).toEqual([
      'heading',
      'text',
      'container',
      'divider',
      'heading',
      'input',
    ]);
    expect(model.inputs).toEqual([
      {
        fieldName: 'value',
        id: 'field-positive',
        isModelField: false,
        label: 'Prompt',
        modelRequirements: [],
        nodeId: 'positive',
        nodeTitle: 'String Primitive',
        value: null,
      },
      {
        fieldName: 'steps',
        id: 'field-steps',
        isModelField: false,
        label: 'Steps',
        modelRequirements: [],
        nodeId: 'denoise',
        nodeTitle: 'Denoise Latents',
        value: { kind: 'number', value: 30 },
      },
      {
        fieldName: 'model',
        id: 'field-model',
        isModelField: true,
        label: 'Model',
        modelRequirements: [{ base: 'sdxl', kind: 'slot', label: 'SDXL checkpoint', modelType: 'main' }],
        nodeId: 'model',
        nodeTitle: 'Main Model',
        value: null,
      },
    ]);
    expect(model.inputGroups).toEqual([
      { heading: 'Prompt', labels: ['Prompt', 'Steps'] },
      { heading: 'Model', labels: ['Model'] },
    ]);
  });

  it('builds once per document, templates and metrics', () => {
    expect(build(TEXT_TO_IMAGE)).toBe(build(TEXT_TO_IMAGE));
    expect(buildWorkflowPreviewModel(TEXT_TO_IMAGE, { ...TEMPLATES }, METRICS)).not.toBe(build(TEXT_TO_IMAGE));
  });
});

describe('the Extend Video - LTX-2 template', () => {
  const document = parseWorkflowJson(extendVideo.workflow).document;
  const model = buildWorkflowPreviewModel(document, extendVideo.templates as unknown as InvocationTemplates);
  const { positions } = model.simplified!;

  it('labels a folded primitive by the form even while its value is empty', () => {
    expect(model.simplified!.pills.pos_cond!.map((pill) => pill.label)).toEqual(['Negative Prompt', 'Prompt']);
  });

  it('lays its eight drawn nodes out six columns wide, over rows kept well apart', () => {
    const { layerGap, nodeHeight, nodeWidth, pillHeight, rowGap } = WORKFLOW_PREVIEW_METRICS;
    const right = Math.max(...Object.values(positions).map((position) => position.x)) + nodeWidth;
    const pillsTop = (id: string) => positions[id]!.y - (model.simplified!.pills[id]?.length ?? 0) * pillHeight;

    expect(model.simplified!.nodeIds).toHaveLength(8);
    expect(right).toBe(6 * nodeWidth + 5 * layerGap);
    expect(new Set(Object.values(positions).map((position) => position.y)).size).toBeGreaterThan(4);

    // Within a column, each node's pills start below the chip above them by at least the row gap, and the topmost
    // pill, not the topmost chip, starts at 0.
    const columns = new Map<number, string[]>();
    for (const id of model.simplified!.nodeIds) {
      columns.set(positions[id]!.x, [...(columns.get(positions[id]!.x) ?? []), id]);
    }
    for (const ids of columns.values()) {
      const sorted = [...ids].sort((a, b) => positions[a]!.y - positions[b]!.y);
      for (const [index, id] of sorted.slice(1).entries()) {
        const above = sorted[index]!;
        expect(pillsTop(id) - (positions[above]!.y + nodeHeight), id).toBeGreaterThanOrEqual(rowGap);
      }
    }
    expect(Math.min(...model.simplified!.nodeIds.map(pillsTop))).toBe(0);
    // The denoiser's chip sits level with the middle of the chips feeding it, its two pills above it.
    const feeding = ['model_loader', 'pos_cond', 'extend_conditioning', 'source_video', 'ideal_dimensions'];
    expect(model.simplified!.pills.denoise_latents).toHaveLength(2);
    expect(positions.denoise_latents!.y).toBeCloseTo(
      feeding.reduce((sum, id) => sum + positions[id]!.y, 0) / feeding.length
    );
  });

  it('sums up a node by its first exposed input with a short value', () => {
    const summary = (id: string) => {
      const field = model.nodes.get(id)!.exposedSummary;
      return field ? [field.label, field.value] : null;
    };

    expect(summary('extend_conditioning')).toEqual(['Context Frames', { kind: 'number', value: 17 }]);
    expect(summary('ideal_dimensions')).toEqual(['Target Resolution', { kind: 'text', text: '704p (short edge 704)' }]);
    expect(summary('denoise_latents')).toEqual(['Number of Frames', { kind: 'number', value: 121 }]);
    expect(summary('pos_cond')).toBeNull();
  });

  it('groups the form’s inputs under its headings', () => {
    expect(model.inputs).toHaveLength(13);
    expect(model.inputGroups.map((group) => [group.heading, group.labels.length])).toEqual([
      ['Initial Video', 4],
      ['Extension', 6],
      ['Model', 3],
    ]);
  });
});

describe('findWorkflowPreviewNodes', () => {
  it('matches drawn nodes by title or type id, ignoring case, in keyboard order', () => {
    const model = build(TEXT_TO_IMAGE);

    expect(findWorkflowPreviewNodes(model, model.all, 'PRIM')).toEqual(['positive', 'seed']);
    expect(findWorkflowPreviewNodes(model, model.simplified!, 'l2i')).toEqual(['decode']);
    expect(findWorkflowPreviewNodes(model, model.all, '  ')).toEqual([]);
  });
});

describe('compiled graphs', () => {
  const COMPILED: Parameters<typeof compiledGraphToPreviewDocument>[0] = {
    edges: [
      { id: 'e1', sourceField: 'value', sourceNodeId: 'seed', targetField: 'seed', targetNodeId: 'denoise' },
      { id: 'e2', sourceField: 'latents', sourceNodeId: 'denoise', targetField: 'latents', targetNodeId: 'upscale' },
    ],
    id: 'generate-preview',
    label: 'Generate',
    nodes: [
      { id: 'seed', inputs: { id: 'seed', is_intermediate: true, type: 'integer', value: 42 }, type: 'integer' },
      { id: 'denoise', inputs: { cfg_scale: 4.5, steps: 30, use_cache: true }, type: 'denoise' },
      { id: 'upscale', inputs: { factor: 2 }, type: 'pack_upscale' },
    ],
  };

  it('draws every node, unknown types included, marking what was set and showing what runs', () => {
    const document = compiledGraphToPreviewDocument(COMPILED, {
      getSetLabel: (nodeId, fieldName) =>
        `${nodeId}.${fieldName}` === 'denoise.steps' ? 'Steps' : nodeId === 'seed' ? 'Seed' : null,
      metrics: METRICS,
      overrides: { seed: { value: 'regenerated each run' } },
    });
    const model = build(document);

    expect(model.all.nodeIds).toEqual(['seed', 'denoise', 'upscale']);
    expect(model.nodes.get('upscale')).toMatchObject({ isResult: true, isUnknown: true, title: 'pack_upscale' });
    expect(model.nodes.get('upscale')!.fields).toEqual([
      { isExposed: false, label: 'factor', name: 'factor', value: { kind: 'number', value: 2 } },
    ]);
    // The backend's bookkeeping is not a value; the steps were set, the scale was not.
    expect(model.nodes.get('denoise')!.fields.map((field) => [field.name, field.isExposed])).toEqual([
      ['steps', true],
      ['cfg_scale', false],
    ]);
    expect(model.nodes.get('denoise')!.isExposed).toBe(true);
    // The seed shows what runs rather than the placeholder it was compiled with, folded onto the denoiser.
    expect(model.simplified!.pills.denoise).toEqual([
      { label: 'Seed', nodeId: 'seed', value: { kind: 'text', text: 'regenerated each run' } },
    ]);
    // A layered arrangement in editor units, which the full view scales like an author's.
    const step = (METRICS.nodeWidth + METRICS.layerGap) / METRICS.authorScale;

    expect(document.nodes.map((node) => node.position)).toEqual([
      { x: 0, y: 0 },
      { x: step, y: 0 },
      { x: step * 2, y: 0 },
    ]);
    expect(model.all.positions.upscale).toEqual({ x: (METRICS.nodeWidth + METRICS.layerGap) * 2, y: 0 });
  });

  it('places nodes where the source hints', () => {
    const document = compiledGraphToPreviewDocument(COMPILED, { positionHints: { denoise: { x: 5, y: 6 } } });

    expect(document.nodes.find((node) => node.id === 'denoise')!.position).toEqual({ x: 5, y: 6 });
  });

  it('keys the structure on node ids, types and connections, in any order, not on values', () => {
    const key = (graph: Parameters<typeof compiledGraphToPreviewDocument>[0]) =>
      build(compiledGraphToPreviewDocument(graph)).structureKey;
    const recompiled = {
      ...COMPILED,
      nodes: COMPILED.nodes.map((node) => (node.id === 'denoise' ? { ...node, inputs: { steps: 12 } } : node)),
    };
    const reordered = { ...COMPILED, edges: [...COMPILED.edges].reverse(), nodes: [...COMPILED.nodes].reverse() };
    const retyped = {
      ...COMPILED,
      nodes: COMPILED.nodes.map((node) => (node.id === 'upscale' ? { ...node, type: 'l2i' } : node)),
    };
    const rewired = {
      ...COMPILED,
      edges: [
        ...COMPILED.edges,
        { id: 'e3', sourceField: 'value', sourceNodeId: 'seed', targetField: 'seed', targetNodeId: 'upscale' },
      ],
    };

    expect(key(recompiled)).toBe(key(COMPILED));
    expect(key(reordered)).toBe(key(COMPILED));
    expect(key(retyped)).not.toBe(key(COMPILED));
    expect(key(rewired)).not.toBe(key(COMPILED));
  });

  it('keeps what a recompile did not change, so a view redraws only what did', () => {
    const compile = (steps: number, seed: string) =>
      compiledGraphToPreviewDocument(
        {
          ...COMPILED,
          nodes: COMPILED.nodes.map((node) =>
            node.id === 'denoise' ? { ...node, inputs: { ...node.inputs, steps } } : node
          ),
        },
        { metrics: METRICS, overrides: { seed: { value: seed } } }
      );
    const first = build(compile(30, 'random'));
    const stepsChanged = buildWorkflowPreviewModel(compile(12, 'random'), TEMPLATES, METRICS, first);

    expect(stepsChanged.nodes.get('denoise')).not.toBe(first.nodes.get('denoise'));
    expect(stepsChanged.nodes.get('denoise')!.values.steps).toBe(12);
    expect(stepsChanged.nodes.get('seed')).toBe(first.nodes.get('seed'));
    expect(stepsChanged.nodes.get('upscale')).toBe(first.nodes.get('upscale'));
    expect(stepsChanged.all).toBe(first.all);
    expect(stepsChanged.simplified).toBe(first.simplified);

    // A pill's value changes only its row; the layout stays.
    const seedChanged = buildWorkflowPreviewModel(compile(12, 'fixed'), TEMPLATES, METRICS, stepsChanged);

    expect(seedChanged.simplified).not.toBe(stepsChanged.simplified);
    expect(seedChanged.simplified!.positions).toBe(stepsChanged.simplified!.positions);
    expect(seedChanged.simplified!.pills.denoise![0]!.value).toEqual({ kind: 'text', text: 'fixed' });
  });
});

describe('chip width', () => {
  it('is wide enough for the longest title, within 136 to 200 px, as wide as still fits', () => {
    const tried: number[] = [];
    const fitsUpTo = (limit: number) => (width: number) => {
      tried.push(width);
      return width <= limit;
    };

    expect(chooseWorkflowPreviewChipWidth(170.2, fitsUpTo(400))).toBe(171);
    expect(chooseWorkflowPreviewChipWidth(260, fitsUpTo(400))).toBe(200);
    expect(chooseWorkflowPreviewChipWidth(90, fitsUpTo(400))).toBe(136);
    // Where the widest does not fit, narrower steps are tried, down to the minimum.
    tried.length = 0;
    expect(chooseWorkflowPreviewChipWidth(200, fitsUpTo(170))).toBe(168);
    expect(tried).toEqual([200, 184, 168]);
    expect(chooseWorkflowPreviewChipWidth(200, () => false)).toBe(136);
  });

  it('lays a drawing out at its chip width, columns and the author’s arrangement widening with it', () => {
    const narrow = buildWorkflowPreviewModel(TEXT_TO_IMAGE, TEMPLATES, getWorkflowPreviewMetrics(136));
    const wide = buildWorkflowPreviewModel(TEXT_TO_IMAGE, TEMPLATES, getWorkflowPreviewMetrics(200));
    const { layerGap } = WORKFLOW_PREVIEW_METRICS;

    expect(getWorkflowPreviewMetrics(200)).toBe(getWorkflowPreviewMetrics(200));
    expect([narrow.all.nodeWidth, wide.all.nodeWidth, wide.simplified!.nodeWidth]).toEqual([136, 200, 200]);
    // Layered columns are a chip and a gap apart.
    const columns = (model: WorkflowPreviewModel) =>
      [...new Set(model.simplified!.nodeIds.map((id) => model.simplified!.positions[id]!.x))].sort((a, b) => a - b);
    expect(columns(wide).slice(0, 2)).toEqual([0, 200 + layerGap]);
    expect(columns(narrow).slice(0, 2)).toEqual([0, 136 + layerGap]);
    // The author's positions scale with the chips.
    expect(wide.all.positions.denoise!.x / narrow.all.positions.denoise!.x).toBeCloseTo(200 / 136);
  });
});
