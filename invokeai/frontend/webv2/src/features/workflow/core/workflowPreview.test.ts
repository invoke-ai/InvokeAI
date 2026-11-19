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

import { parseWorkflowJson } from './workflowJson';
import {
  buildWorkflowPreviewModel,
  compiledGraphToPreviewDocument,
  findWorkflowPreviewNodes,
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
  it.each(['toString', 'constructor', '__proto__'])('previews a parsed unknown node type named %s', (name) => {
    const parsed = parseWorkflowJson(graph([node('unknown', name, { value: 'saved' })], []));
    const model = build(parsed.document);

    expect(model.nodes.get('unknown')).toMatchObject({ isUnknown: true, title: name });
    expect(model.nodes.get('unknown')!.fields).toEqual([
      { isExposed: false, label: 'value', name: 'value', value: { kind: 'text', text: 'saved' } },
    ]);
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
});
