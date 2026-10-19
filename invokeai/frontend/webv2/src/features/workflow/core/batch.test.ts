import { describe, expect, it } from 'vitest';

import type {
  FieldInputTemplate,
  InvocationTemplate,
  ProjectGraphState,
  WorkflowEdge,
  WorkflowInvocationNode,
} from './types';

import {
  createSeededRandom,
  getDefaultWorkflowGeneratorValue,
  getWorkflowBatchGroupId,
  getWorkflowGeneratorInvalidReason,
  getWorkflowGeneratorPropertyInvalidReason,
  getWorkflowGeneratorKey,
  getWorkflowGeneratorVariantDefaults,
  isRandomWorkflowGeneratorValue,
  isWorkflowBatchNodeType,
  isWorkflowGeneratorNodeType,
  parseWorkflowGeneratorValue,
  planWorkflowBatch,
  resolveWorkflowGeneratorValue,
  type WorkflowGeneratorValue,
} from './batch';
import { createProjectGraph } from './document';

const items = (value: WorkflowGeneratorValue, random?: () => number) => {
  const resolution = resolveWorkflowGeneratorValue(value, random);

  if (resolution.kind !== 'items') {
    throw new Error('Expected synchronous items');
  }

  return resolution.items;
};

describe('generator values', () => {
  it('classifies the eight client-resolved node types', () => {
    expect(isWorkflowBatchNodeType('image_batch')).toBe(true);
    expect(isWorkflowGeneratorNodeType('float_generator')).toBe(true);
    expect(isWorkflowBatchNodeType('float_generator')).toBe(false);
    expect(isWorkflowGeneratorNodeType('noise')).toBe(false);
  });

  it('starts each field on its legacy default variant', () => {
    expect(getDefaultWorkflowGeneratorValue('FloatGeneratorField')).toEqual({
      count: 10,
      start: 0,
      step: 0.1,
      type: 'float_generator_arithmetic_sequence',
    });
    expect(getDefaultWorkflowGeneratorValue('StringGeneratorField')?.type).toBe('string_generator_parse_string');
    expect(getDefaultWorkflowGeneratorValue('ImageGeneratorField')).toEqual({
      category: 'images',
      type: 'image_generator_images_from_board',
    });
    expect(getDefaultWorkflowGeneratorValue('StringField')).toBeUndefined();
    // Defaults are copies: an edit to one node's value cannot leak into the next node's default.
    expect(getWorkflowGeneratorVariantDefaults('integer_generator_parse_string')).not.toBe(
      getWorkflowGeneratorVariantDefaults('integer_generator_parse_string')
    );
  });

  it('reads stored values tolerantly, as the legacy schemas did', () => {
    // A legacy float value with its resolved override and a null seed.
    expect(
      parseWorkflowGeneratorValue('FloatGeneratorField', {
        count: 3,
        max: 1,
        min: 0,
        seed: null,
        type: 'float_generator_random_distribution_uniform',
        values: [0.1, 0.2, 0.3],
      })
    ).toEqual({
      count: 3,
      max: 1,
      min: 0,
      seed: null,
      type: 'float_generator_random_distribution_uniform',
      values: [0.1, 0.2, 0.3],
    });
    // Missing properties take their defaults; a missing variant is the field's first one.
    expect(parseWorkflowGeneratorValue('IntegerGeneratorField', {})).toEqual({
      count: 10,
      start: 0,
      step: 1,
      type: 'integer_generator_arithmetic_sequence',
    });
    expect(parseWorkflowGeneratorValue('ImageGeneratorField', { board_id: ' ', category: 'bogus' })).toEqual({
      category: 'images',
      type: 'image_generator_images_from_board',
    });
    // A wrong-typed required property, a variant from another field, or a non-object is unusable.
    expect(
      parseWorkflowGeneratorValue('IntegerGeneratorField', {
        start: '1',
        type: 'integer_generator_arithmetic_sequence',
      })
    ).toBeNull();
    // Numbers are kept as entered, even off their rule; the reason check says whether they can run.
    expect(
      parseWorkflowGeneratorValue('IntegerGeneratorField', {
        count: 0,
        start: null,
        step: 1.5,
        type: 'integer_generator_arithmetic_sequence',
      })
    ).toEqual({ count: 0, start: null, step: 1.5, type: 'integer_generator_arithmetic_sequence' });
    expect(parseWorkflowGeneratorValue('IntegerGeneratorField', { type: 'float_generator_parse_string' })).toBeNull();
    expect(parseWorkflowGeneratorValue('FloatGeneratorField', 'nope')).toBeNull();
    // Optional properties recover from bad input instead.
    expect(
      parseWorkflowGeneratorValue('FloatGeneratorField', {
        seed: 'x',
        type: 'float_generator_random_distribution_uniform',
        values: 'x',
      })
    ).toMatchObject({ seed: null });
  });

  it('names the first setting that keeps a readable generator from running, and yields nothing for it', () => {
    const arithmetic = (overrides: Record<string, unknown>) =>
      parseWorkflowGeneratorValue('IntegerGeneratorField', {
        count: 3,
        start: 0,
        step: 1,
        type: 'integer_generator_arithmetic_sequence',
        ...overrides,
      })!;
    const reason = (value: WorkflowGeneratorValue) => getWorkflowGeneratorInvalidReason(value);

    expect(reason(arithmetic({}))).toBeNull();
    expect(reason(arithmetic({ start: null }))).toBe('Start is empty.');
    expect(reason(arithmetic({ step: 1.5 }))).toBe('Step must be a whole number.');
    expect(reason(arithmetic({ count: 0 }))).toBe('Count must be at least 1.');
    expect(reason(arithmetic({ count: 2.5 }))).toBe('Count must be a whole number.');
    expect(getWorkflowGeneratorPropertyInvalidReason(arithmetic({ count: 0, start: null }), 'count')).toBe(
      'Count must be at least 1.'
    );
    // Floats may be fractional; a seed is optional but must be a whole number of at least 0 when pinned.
    expect(
      reason(
        parseWorkflowGeneratorValue('FloatGeneratorField', {
          count: 2,
          max: 1,
          min: 0.5,
          seed: 2.5,
          type: 'float_generator_random_distribution_uniform',
        })!
      )
    ).toBe('Seed must be a whole number.');
    expect(
      reason(
        parseWorkflowGeneratorValue('FloatGeneratorField', {
          count: 2,
          max: 1,
          min: 0.5,
          seed: -1,
          type: 'float_generator_random_distribution_uniform',
        })!
      )
    ).toBe('Seed must be at least 0.');
    expect(
      reason(
        parseWorkflowGeneratorValue('StringGeneratorField', {
          count: 1001,
          input: 'a',
          seed: null,
          type: 'string_generator_dynamic_prompts_random',
        })!
      )
    ).toBe('Count can be at most 1000.');

    // A legacy resolved override stands in for its settings, whatever they say.
    expect(
      reason(
        parseWorkflowGeneratorValue('FloatGeneratorField', {
          count: 0,
          start: 0,
          step: 1,
          type: 'float_generator_arithmetic_sequence',
          values: [4, 5],
        })!
      )
    ).toBeNull();

    // An off-rule generator resolves to no items rather than a broken sequence.
    expect(resolveWorkflowGeneratorValue(arithmetic({ count: 0 }))).toEqual({ items: [], kind: 'items' });
    expect(resolveWorkflowGeneratorValue(arithmetic({ start: null }))).toEqual({ items: [], kind: 'items' });
  });

  it('fingerprints a value independently of key order and spots unseeded random variants', () => {
    expect(
      getWorkflowGeneratorKey({ count: 2, start: 0, step: 1, type: 'integer_generator_arithmetic_sequence' })
    ).toBe(getWorkflowGeneratorKey({ type: 'integer_generator_arithmetic_sequence', step: 1, start: 0, count: 2 }));
    expect(
      isRandomWorkflowGeneratorValue({
        count: 1,
        max: 1,
        min: 0,
        seed: null,
        type: 'float_generator_random_distribution_uniform',
      })
    ).toBe(true);
    expect(
      isRandomWorkflowGeneratorValue({
        count: 1,
        max: 1,
        min: 0,
        seed: 4,
        type: 'float_generator_random_distribution_uniform',
      })
    ).toBe(false);
  });
});

describe('generator resolution', () => {
  it('produces sequences and distributions with the legacy edge cases', () => {
    expect(items({ count: 3, start: 1, step: 0.5, type: 'float_generator_arithmetic_sequence' })).toEqual([1, 1.5, 2]);
    expect(items({ count: 5, start: 4, step: 0, type: 'integer_generator_arithmetic_sequence' })).toEqual([4]);
    expect(items({ count: 3, end: 1, start: 0, type: 'float_generator_linear_distribution' })).toEqual([0, 0.5, 1]);
    expect(items({ count: 1, end: 9, start: 2, type: 'float_generator_linear_distribution' })).toEqual([2]);
    expect(items({ count: 4, end: 10, start: 0, type: 'integer_generator_linear_distribution' })).toEqual([
      0, 3, 7, 10,
    ]);
    // A resolved override wins over the settings it was computed from.
    expect(items({ count: 3, start: 0, step: 1, type: 'float_generator_arithmetic_sequence', values: [9] })).toEqual([
      9,
    ]);
  });

  it('draws random values inside the bounds, repeatably for a seed and from the injected source without one', () => {
    const seeded = items({
      count: 5,
      max: 10,
      min: 5,
      seed: 42,
      type: 'integer_generator_random_distribution_uniform',
    });

    expect(seeded).toEqual(
      items({ count: 5, max: 10, min: 5, seed: 42, type: 'integer_generator_random_distribution_uniform' })
    );
    expect(seeded.every((value) => Number.isInteger(value) && (value as number) >= 5 && (value as number) <= 10)).toBe(
      true
    );
    expect(
      items({ count: 2, max: 1, min: 0, seed: null, type: 'float_generator_random_distribution_uniform' }, () => 0.25)
    ).toEqual([0.25, 0.25]);

    const random = createSeededRandom(7);
    const draw = random();

    expect(draw).toBeGreaterThanOrEqual(0);
    expect(draw).toBeLessThan(1);
    expect(createSeededRandom(7)()).toBe(draw);
  });

  it('splits text the legacy way: escapes honoured, numbers trimmed and filtered, strings kept verbatim', () => {
    expect(items({ input: ' 1, 2 ,x, 3.5', splitOn: ',', type: 'float_generator_parse_string' })).toEqual([1, 2, 3.5]);
    expect(items({ input: '1\n2\n\n3', splitOn: '\\n', type: 'integer_generator_parse_string' })).toEqual([1, 2, 3]);
    expect(items({ input: 'a b', splitOn: '', type: 'string_generator_parse_string' })).toEqual(['a b']);
    expect(items({ input: ' foo|bar||baz ', splitOn: '|', type: 'string_generator_parse_string' })).toEqual([
      ' foo',
      'bar',
      'baz ',
    ]);
    // An unparseable escape falls back to the raw separator.
    expect(items({ input: 'a\\qb', splitOn: '\\q', type: 'string_generator_parse_string' })).toEqual(['a', 'b']);
  });

  it('hands dynamic prompts and board listings to the async resolver, and an unset board to nobody', () => {
    expect(
      resolveWorkflowGeneratorValue({
        count: 4,
        input: '{a|b}',
        seed: 3,
        type: 'string_generator_dynamic_prompts_random',
      })
    ).toEqual({
      kind: 'async',
      request: { combinatorial: false, kind: 'dynamicPrompts', maxPrompts: 4, prompt: '{a|b}', seed: 3 },
    });
    expect(
      resolveWorkflowGeneratorValue({
        input: '{a|b}',
        maxPrompts: 9,
        type: 'string_generator_dynamic_prompts_combinatorial',
      })
    ).toEqual({
      kind: 'async',
      request: { combinatorial: true, kind: 'dynamicPrompts', maxPrompts: 9, prompt: '{a|b}', seed: null },
    });
    expect(
      resolveWorkflowGeneratorValue({ board_id: 'b1', category: 'assets', type: 'image_generator_images_from_board' })
    ).toEqual({
      kind: 'async',
      request: { boardId: 'b1', category: 'assets', kind: 'boardImages' },
    });
    expect(resolveWorkflowGeneratorValue({ category: 'images', type: 'image_generator_images_from_board' })).toEqual({
      items: [],
      kind: 'items',
    });
  });
});

describe('planWorkflowBatch', () => {
  const field = (name: string, typeName: string, overrides: Partial<FieldInputTemplate> = {}): FieldInputTemplate => ({
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
    required: true,
    title: name,
    type: { batch: false, cardinality: 'SINGLE', name: typeName },
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
    outputs: {},
    outputType: `${type}_output`,
    tags: [],
    title: type,
    type,
    useCache: true,
    version: '1.0.0',
  });
  const templates = {
    float_batch: template('float_batch', {
      batch_group_id: field('batch_group_id', 'EnumField'),
      floats: field('floats', 'FloatField'),
    }),
    float_generator: template('float_generator', { generator: field('generator', 'FloatGeneratorField') }),
    image_batch: template('image_batch', {
      batch_group_id: field('batch_group_id', 'EnumField'),
      images: field('images', 'ImageField'),
    }),
    prompt: template('prompt', { text: field('text', 'StringField') }),
    sink: template('sink', {
      cfg: field('cfg', 'FloatField'),
      image: field('image', 'ImageField'),
      steps: field('steps', 'IntegerField'),
    }),
    string_batch: template('string_batch', {
      batch_group_id: field('batch_group_id', 'EnumField'),
      strings: field('strings', 'StringField'),
    }),
  };
  const node = (id: string, type: string, inputs: Record<string, unknown>, label = ''): WorkflowInvocationNode => ({
    data: {
      inputs: Object.fromEntries(Object.entries(inputs).map(([name, value]) => [name, { label: '', name, value }])),
      isIntermediate: true,
      isOpen: true,
      label,
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
  const edge = (source: string, sourceHandle: string, target: string, targetHandle: string): WorkflowEdge => ({
    id: `${source}.${sourceHandle}->${target}.${targetHandle}`,
    source,
    sourceHandle,
    target,
    targetHandle,
    type: 'default',
  });
  const doc = (nodes: WorkflowInvocationNode[], edges: WorkflowEdge[]): ProjectGraphState => ({
    ...createProjectGraph('batch-test'),
    edges,
    nodes,
  });

  it('reports no batch for a plain graph', () => {
    expect(planWorkflowBatch(doc([node('s', 'sink', {})], []), templates)).toEqual({
      batchSize: 1,
      groups: [],
      hasBatchNodes: false,
      pendingGenerators: [],
      reasons: [],
    });
  });

  it('multiplies ungrouped nodes and zips a named group, one datum per fed input', () => {
    const graph = doc(
      [
        node('a', 'float_batch', { batch_group_id: 'None', floats: [1, 2] }),
        node('b', 'string_batch', { batch_group_id: 'Group 1', strings: ['x', 'y', 'z'] }),
        node('c', 'float_batch', { batch_group_id: 'Group 1', floats: [7, 8, 9] }),
        node('s', 'sink', {}),
        node('p', 'prompt', {}),
      ],
      [
        edge('a', 'value', 's', 'cfg'),
        edge('a', 'value', 's', 'steps'),
        edge('b', 'value', 'p', 'text'),
        edge('c', 'value', 's', 'cfg'),
      ]
    );
    const plan = planWorkflowBatch(graph, templates);

    // 'cfg' is fed twice: once from the ungrouped node and once from the group.
    expect(plan.reasons).toEqual(['"cfg" on "sink" receives values from more than one batch node.']);
    expect(plan.batchSize).toBe(6);
    expect(plan.groups).toEqual([
      [
        { fieldName: 'cfg', items: [1, 2], nodeId: 's' },
        { fieldName: 'steps', items: [1, 2], nodeId: 's' },
      ],
      [{ fieldName: 'text', items: ['x', 'y', 'z'], nodeId: 'p' }],
    ]);
    expect(getWorkflowBatchGroupId(node('g', 'float_batch', { batch_group_id: 'Group 9' }))).toBe('None');
  });

  it('feeds a batch from a generator, through connectors, and coerces numbers into string inputs', () => {
    const graph = doc(
      [
        node('g', 'float_generator', {
          generator: { count: 3, start: 1, step: 1, type: 'float_generator_arithmetic_sequence' },
        }),
        node('b', 'string_batch', { batch_group_id: 'None' }),
        node('p', 'prompt', {}),
      ],
      [edge('g', 'floats', 'b', 'strings'), edge('b', 'value', 'p', 'text')]
    );
    // The batch's own (missing) list is ignored once a generator feeds it; a string batch stringifies its items.
    const plan = planWorkflowBatch(graph, templates);

    expect(plan.reasons).toEqual([]);
    expect(plan.groups).toEqual([[{ fieldName: 'text', items: ['1', '2', '3'], nodeId: 'p' }]]);
    expect(plan.batchSize).toBe(3);
  });

  it('leaves an async generator pending until its resolution matches the value it was computed from', () => {
    const generator = {
      board_id: 'board-1',
      category: 'images' as const,
      type: 'image_generator_images_from_board' as const,
    };
    const graph = doc(
      [
        node('g', 'image_generator', { generator }),
        node('b', 'image_batch', { batch_group_id: 'None' }),
        node('s', 'sink', {}),
      ],
      [edge('g', 'images', 'b', 'images'), edge('b', 'image', 's', 'image')]
    );
    const pending = planWorkflowBatch(graph, templates);
    const key = getWorkflowGeneratorKey(generator);

    expect(pending.batchSize).toBeNull();
    expect(pending.reasons).toEqual([]);
    expect(pending.pendingGenerators).toEqual([
      {
        key,
        nodeId: 'g',
        nodeLabel: 'image_generator',
        request: { boardId: 'board-1', category: 'images', kind: 'boardImages' },
      },
    ]);

    const stale = planWorkflowBatch(graph, templates, {
      generators: { g: { items: [{ image_name: 'old.png' }], key: 'other' } },
    });

    expect(stale.pendingGenerators).toHaveLength(1);

    const resolved = planWorkflowBatch(graph, templates, {
      generators: { g: { items: [{ image_name: 'a.png' }, { image_name: 'b.png' }], key } },
    });

    expect(resolved.pendingGenerators).toEqual([]);
    expect(resolved.batchSize).toBe(2);
    expect(resolved.groups).toEqual([
      [{ fieldName: 'image', items: [{ image_name: 'a.png' }, { image_name: 'b.png' }], nodeId: 's' }],
    ]);

    const empty = planWorkflowBatch(graph, templates, { generators: { g: { items: [], key } } });

    expect(empty.reasons).toEqual(['Batch node "image_batch" has an empty collection.']);

    // Without a board there is nothing to list; the fix is on the generator, so the reason names it.
    const boardless = doc(
      [
        node('g', 'image_generator', { generator: { category: 'images', type: 'image_generator_images_from_board' } }),
        node('b', 'image_batch', { batch_group_id: 'None' }),
        node('s', 'sink', {}),
      ],
      [edge('g', 'images', 'b', 'images'), edge('b', 'image', 's', 'image')]
    );

    expect(planWorkflowBatch(boardless, templates).reasons).toEqual([
      'Image generator "image_generator" has no board selected.',
    ]);
  });

  it('explains every way a batch node can be wired wrong', () => {
    const graph = doc(
      [
        node('lonely', 'float_batch', { batch_group_id: 'None', floats: [1] }, 'Lonely'),
        node('empty', 'float_batch', { batch_group_id: 'Group 2', floats: [] }, 'Empty'),
        node('short', 'float_batch', { batch_group_id: 'Group 2', floats: [1, 2] }, 'Short'),
        node('fed', 'float_batch', { batch_group_id: 'None', floats: [1] }, 'Fed'),
        node('pics', 'image_batch', { batch_group_id: 'None', images: [{ image_name: 'a.png' }] }, 'Pics'),
        node('p', 'prompt', {}),
        node('s', 'sink', {}),
      ],
      [
        edge('empty', 'value', 's', 'cfg'),
        edge('short', 'value', 's', 'steps'),
        edge('p', 'text', 'fed', 'floats'),
        edge('fed', 'value', 'pics', 'images'),
        edge('pics', 'width', 's', 'cfg'),
        edge('pics', 'image', 's', 'image'),
      ]
    );
    const plan = planWorkflowBatch(graph, templates);

    expect(plan.reasons).toEqual([
      'Batch node "Lonely" is not connected to any node.',
      'Batch node "Empty" has an empty collection.',
      'Batch node "Fed" can only receive its collection from a generator.',
      '"Fed" cannot supply batch values to "images" on "Pics".',
      'Batch node "Pics" can only receive its collection from a generator.',
      'Batch node "Pics" can only batch its image output; width and height are not supported.',
      'Batch group "Group 2" mixes collection sizes (Empty: 0, Short: 2).',
    ]);
  });

  it('refuses an oversized generator by its count alone, without building the list', () => {
    const graph = doc(
      [
        node('g', 'float_generator', {
          generator: { count: 2_000_000_000, start: 0, step: 1, type: 'float_generator_arithmetic_sequence' },
        }),
        node('b', 'float_batch', { batch_group_id: 'None' }),
        node('s', 'sink', {}),
      ],
      [edge('g', 'floats', 'b', 'floats'), edge('b', 'value', 's', 'cfg')]
    );

    expect(planWorkflowBatch(graph, templates).reasons).toEqual([
      'Batch node "float_batch" has more than 10,000 items.',
    ]);
  });

  it('refuses more items than the queue accepts', () => {
    const graph = doc(
      [
        node('big', 'float_batch', { batch_group_id: 'None', floats: Array.from({ length: 10_001 }, (_, i) => i) }),
        node('s', 'sink', {}),
      ],
      [edge('big', 'value', 's', 'cfg')]
    );

    expect(planWorkflowBatch(graph, templates).reasons).toEqual([
      'Batch node "float_batch" has more than 10,000 items.',
    ]);
  });
});
