/**
 * Batch and generator nodes never run on the backend: the editor resolves them into queue batch groups at submit
 * time. Value shapes and resolution rules mirror the legacy editor so saved workflows keep their meaning, with one
 * exception: seeded random draws use a different generator, so a legacy seed yields different values here.
 */

import type { InvocationTemplates, ProjectGraphState, WorkflowInvocationNode } from './types';

import { getCanonicalWorkflowEdges } from './forLoops';
import { isInvocationNode } from './types';

// #region Node classification

export const WORKFLOW_BATCH_GROUP_IDS = ['None', 'Group 1', 'Group 2', 'Group 3', 'Group 4', 'Group 5'] as const;
export type WorkflowBatchGroupId = (typeof WORKFLOW_BATCH_GROUP_IDS)[number];

/** The backend truncates a batch to its remaining queue capacity, whose default ceiling this mirrors. */
export const WORKFLOW_BATCH_MAX_ITEMS = 10_000;

const BATCH_COLLECTION_FIELDS = {
  float_batch: 'floats',
  image_batch: 'images',
  integer_batch: 'integers',
  string_batch: 'strings',
} as const;

const GENERATOR_OUTPUT_FIELDS = {
  float_generator: 'floats',
  image_generator: 'images',
  integer_generator: 'integers',
  string_generator: 'strings',
} as const;

const GENERATOR_FIELD_TYPE_NAMES = {
  float_generator: 'FloatGeneratorField',
  image_generator: 'ImageGeneratorField',
  integer_generator: 'IntegerGeneratorField',
  string_generator: 'StringGeneratorField',
} as const;

export type WorkflowBatchNodeType = keyof typeof BATCH_COLLECTION_FIELDS;
export type WorkflowGeneratorNodeType = keyof typeof GENERATOR_OUTPUT_FIELDS;

export const isWorkflowBatchNodeType = (type: string): type is WorkflowBatchNodeType =>
  Object.hasOwn(BATCH_COLLECTION_FIELDS, type);

export const isWorkflowGeneratorNodeType = (type: string): type is WorkflowGeneratorNodeType =>
  Object.hasOwn(GENERATOR_OUTPUT_FIELDS, type);

/** The list input a batch node iterates, or the list output a generator produces. */
export const getWorkflowBatchCollectionField = (type: string): string | null =>
  isWorkflowBatchNodeType(type) ? BATCH_COLLECTION_FIELDS[type] : null;

export const getWorkflowGeneratorOutputField = (type: string): string | null =>
  isWorkflowGeneratorNodeType(type) ? GENERATOR_OUTPUT_FIELDS[type] : null;

/** The one output a batch node substitutes per session; Image Batch also declares width/height, which cannot batch. */
export const getWorkflowBatchOutputField = (type: WorkflowBatchNodeType): 'image' | 'value' =>
  type === 'image_batch' ? 'image' : 'value';

export const getWorkflowBatchGroupId = (node: WorkflowInvocationNode): WorkflowBatchGroupId => {
  const value = node.data.inputs.batch_group_id?.value;

  return WORKFLOW_BATCH_GROUP_IDS.includes(value as WorkflowBatchGroupId) ? (value as WorkflowBatchGroupId) : 'None';
};

// #endregion

// #region Generator values

export interface FloatGeneratorArithmeticSequence {
  type: 'float_generator_arithmetic_sequence';
  start: number | null;
  step: number | null;
  count: number | null;
  values?: number[];
}

export interface FloatGeneratorLinearDistribution {
  type: 'float_generator_linear_distribution';
  start: number | null;
  end: number | null;
  count: number | null;
  values?: number[];
}

export interface FloatGeneratorUniformRandom {
  type: 'float_generator_random_distribution_uniform';
  min: number | null;
  max: number | null;
  count: number | null;
  seed: number | null;
  values?: number[];
}

export interface FloatGeneratorParseString {
  type: 'float_generator_parse_string';
  input: string;
  splitOn: string;
  values?: number[];
}

export type FloatGeneratorValue =
  | FloatGeneratorArithmeticSequence
  | FloatGeneratorLinearDistribution
  | FloatGeneratorUniformRandom
  | FloatGeneratorParseString;

export interface IntegerGeneratorArithmeticSequence {
  type: 'integer_generator_arithmetic_sequence';
  start: number | null;
  step: number | null;
  count: number | null;
}

export interface IntegerGeneratorLinearDistribution {
  type: 'integer_generator_linear_distribution';
  start: number | null;
  end: number | null;
  count: number | null;
}

export interface IntegerGeneratorUniformRandom {
  type: 'integer_generator_random_distribution_uniform';
  min: number | null;
  max: number | null;
  count: number | null;
  seed: number | null;
}

export interface IntegerGeneratorParseString {
  type: 'integer_generator_parse_string';
  input: string;
  splitOn: string;
}

export type IntegerGeneratorValue =
  | IntegerGeneratorArithmeticSequence
  | IntegerGeneratorLinearDistribution
  | IntegerGeneratorUniformRandom
  | IntegerGeneratorParseString;

export interface StringGeneratorParseString {
  type: 'string_generator_parse_string';
  input: string;
  splitOn: string;
}

export interface StringGeneratorDynamicPromptsRandom {
  type: 'string_generator_dynamic_prompts_random';
  input: string;
  count: number | null;
  seed: number | null;
}

export interface StringGeneratorDynamicPromptsCombinatorial {
  type: 'string_generator_dynamic_prompts_combinatorial';
  input: string;
  maxPrompts: number | null;
}

export type StringGeneratorValue =
  | StringGeneratorParseString
  | StringGeneratorDynamicPromptsRandom
  | StringGeneratorDynamicPromptsCombinatorial;

export interface ImageGeneratorImagesFromBoard {
  type: 'image_generator_images_from_board';
  board_id?: string;
  category: 'images' | 'assets';
}

export type ImageGeneratorValue = ImageGeneratorImagesFromBoard;

export type WorkflowGeneratorValue =
  | FloatGeneratorValue
  | IntegerGeneratorValue
  | StringGeneratorValue
  | ImageGeneratorValue;
export type WorkflowGeneratorVariant = WorkflowGeneratorValue['type'];

/** Variants per generator field type, first entry being the default a fresh node starts with (as in legacy). */
export const WORKFLOW_GENERATOR_VARIANTS = {
  FloatGeneratorField: [
    'float_generator_arithmetic_sequence',
    'float_generator_linear_distribution',
    'float_generator_random_distribution_uniform',
    'float_generator_parse_string',
  ],
  ImageGeneratorField: ['image_generator_images_from_board'],
  IntegerGeneratorField: [
    'integer_generator_arithmetic_sequence',
    'integer_generator_linear_distribution',
    'integer_generator_random_distribution_uniform',
    'integer_generator_parse_string',
  ],
  StringGeneratorField: [
    'string_generator_parse_string',
    'string_generator_dynamic_prompts_random',
    'string_generator_dynamic_prompts_combinatorial',
  ],
} as const satisfies Record<string, readonly WorkflowGeneratorVariant[]>;

export type WorkflowGeneratorFieldTypeName = keyof typeof WORKFLOW_GENERATOR_VARIANTS;

export const isWorkflowGeneratorFieldTypeName = (name: string): name is WorkflowGeneratorFieldTypeName =>
  Object.hasOwn(WORKFLOW_GENERATOR_VARIANTS, name);

const VARIANT_DEFAULTS: { [V in WorkflowGeneratorVariant]: Extract<WorkflowGeneratorValue, { type: V }> } = {
  float_generator_arithmetic_sequence: { count: 10, start: 0, step: 0.1, type: 'float_generator_arithmetic_sequence' },
  float_generator_linear_distribution: { count: 10, end: 1, start: 0, type: 'float_generator_linear_distribution' },
  float_generator_parse_string: {
    input: '0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1',
    splitOn: ',',
    type: 'float_generator_parse_string',
  },
  float_generator_random_distribution_uniform: {
    count: 10,
    max: 1,
    min: 0,
    seed: null,
    type: 'float_generator_random_distribution_uniform',
  },
  image_generator_images_from_board: { category: 'images', type: 'image_generator_images_from_board' },
  integer_generator_arithmetic_sequence: {
    count: 10,
    start: 0,
    step: 1,
    type: 'integer_generator_arithmetic_sequence',
  },
  integer_generator_linear_distribution: {
    count: 10,
    end: 10,
    start: 0,
    type: 'integer_generator_linear_distribution',
  },
  integer_generator_parse_string: {
    input: '1,2,3,4,5,6,7,8,9,10',
    splitOn: ',',
    type: 'integer_generator_parse_string',
  },
  integer_generator_random_distribution_uniform: {
    count: 10,
    max: 10,
    min: 0,
    seed: null,
    type: 'integer_generator_random_distribution_uniform',
  },
  string_generator_dynamic_prompts_combinatorial: {
    input: 'a super {cute|ferocious} {dog|cat}',
    maxPrompts: 10,
    type: 'string_generator_dynamic_prompts_combinatorial',
  },
  string_generator_dynamic_prompts_random: {
    count: 10,
    input: 'a super {cute|ferocious} {dog|cat}',
    seed: null,
    type: 'string_generator_dynamic_prompts_random',
  },
  string_generator_parse_string: { input: 'foo,bar,baz,qux', splitOn: ',', type: 'string_generator_parse_string' },
};

/** How each stored property is read: what it must be, and what an absent one falls back to. */
type PropertySpec = 'boardId' | 'category' | 'count' | 'integer' | 'number' | 'seed' | 'string' | 'values';

const VARIANT_PROPERTIES: Record<WorkflowGeneratorVariant, Record<string, PropertySpec>> = {
  float_generator_arithmetic_sequence: { count: 'count', start: 'number', step: 'number', values: 'values' },
  float_generator_linear_distribution: { count: 'count', end: 'number', start: 'number', values: 'values' },
  float_generator_parse_string: { input: 'string', splitOn: 'string', values: 'values' },
  float_generator_random_distribution_uniform: {
    count: 'count',
    max: 'number',
    min: 'number',
    seed: 'seed',
    values: 'values',
  },
  image_generator_images_from_board: { board_id: 'boardId', category: 'category' },
  integer_generator_arithmetic_sequence: { count: 'count', start: 'integer', step: 'integer' },
  integer_generator_linear_distribution: { count: 'count', end: 'integer', start: 'integer' },
  integer_generator_parse_string: { input: 'string', splitOn: 'string' },
  integer_generator_random_distribution_uniform: { count: 'count', max: 'integer', min: 'integer', seed: 'seed' },
  string_generator_dynamic_prompts_combinatorial: { input: 'string', maxPrompts: 'count' },
  string_generator_dynamic_prompts_random: { count: 'count', input: 'string', seed: 'seed' },
  string_generator_parse_string: { input: 'string', splitOn: 'string' },
};

export const getWorkflowGeneratorVariantDefaults = (variant: WorkflowGeneratorVariant): WorkflowGeneratorValue =>
  structuredClone(VARIANT_DEFAULTS[variant]);

export const getDefaultWorkflowGeneratorValue = (fieldTypeName: string): WorkflowGeneratorValue | undefined =>
  isWorkflowGeneratorFieldTypeName(fieldTypeName)
    ? getWorkflowGeneratorVariantDefaults(WORKFLOW_GENERATOR_VARIANTS[fieldTypeName][0])
    : undefined;

const isFiniteNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

const UNUSABLE = Symbol('unusable');

/** `undefined` means the property is omitted; `UNUSABLE` means the stored value cannot be read at all. */
const readProperty = (spec: PropertySpec, raw: unknown, fallback: unknown): unknown => {
  if (raw === undefined) {
    return fallback;
  }

  switch (spec) {
    case 'number':
    case 'integer':
    case 'count':
      // Kept as entered: `null` is a cleared setting, an off-rule number a readable one that cannot run yet.
      return raw === null || isFiniteNumber(raw) ? raw : UNUSABLE;
    case 'string':
      return typeof raw === 'string' ? raw : UNUSABLE;
    case 'seed':
      return raw === null || isFiniteNumber(raw) ? raw : undefined;
    case 'category':
      return raw === 'images' || raw === 'assets' ? raw : undefined;
    case 'boardId':
      return raw === null ? undefined : typeof raw === 'string' && raw.trim() !== '' ? raw.trim() : undefined;
    case 'values':
      return raw === null ? undefined : Array.isArray(raw) && raw.every(isFiniteNumber) ? raw : undefined;
  }
};

/**
 * Reads a stored generator value the way the legacy schemas did: missing properties take their defaults, a missing
 * variant is the field's first one, and a wrong-typed property makes the whole value unusable (null). Numbers are
 * kept as entered; `getWorkflowGeneratorInvalidReason` says whether they can run.
 */
export const parseWorkflowGeneratorValue = (fieldTypeName: string, raw: unknown): WorkflowGeneratorValue | null => {
  if (!isWorkflowGeneratorFieldTypeName(fieldTypeName) || typeof raw !== 'object' || raw === null) {
    return null;
  }

  const variants = WORKFLOW_GENERATOR_VARIANTS[fieldTypeName] as readonly WorkflowGeneratorVariant[];
  const source = raw as Record<string, unknown>;
  const variant = source.type === undefined ? variants[0] : (source.type as WorkflowGeneratorVariant);

  if (variant === undefined || !variants.includes(variant)) {
    return null;
  }

  const defaults = VARIANT_DEFAULTS[variant] as unknown as Record<string, unknown>;
  const value: Record<string, unknown> = { type: variant };

  for (const [key, spec] of Object.entries(VARIANT_PROPERTIES[variant])) {
    const read = readProperty(spec, source[key], defaults[key]);

    // Optional properties (seed, board, override values, category) recover from bad input; required ones do not.
    if (read === UNUSABLE) {
      return null;
    }

    if (read === undefined) {
      if (spec === 'seed') {
        value[key] = null;
      } else if (spec === 'category') {
        value[key] = 'images';
      }

      continue;
    }

    value[key] = read;
  }

  return value as unknown as WorkflowGeneratorValue;
};

const sortKeys = (value: unknown): unknown =>
  Array.isArray(value)
    ? value.map(sortKeys)
    : typeof value === 'object' && value !== null
      ? Object.fromEntries(
          Object.keys(value as Record<string, unknown>)
            .sort()
            .map((key) => [key, sortKeys((value as Record<string, unknown>)[key])])
        )
      : value;

/** A stable fingerprint of a generator value, so an async result can be matched to the value it came from. */
export const getWorkflowGeneratorKey = (value: WorkflowGeneratorValue): string => JSON.stringify(sortKeys(value));

export const isWorkflowGeneratorVariant = (type: string): type is WorkflowGeneratorVariant =>
  Object.hasOwn(VARIANT_PROPERTIES, type);

/** Dynamic prompt expansion is bounded server-side; the legacy editor capped a request at this. */
export const WORKFLOW_DYNAMIC_PROMPTS_MAX = 1000;

const PROPERTY_LABELS: Record<string, string> = {
  count: 'Count',
  end: 'End',
  max: 'Max',
  maxPrompts: 'Max prompts',
  min: 'Min',
  seed: 'Seed',
  start: 'Start',
  step: 'Step',
};

/** Why one numeric setting cannot run, or null. Text, board, and category settings always can. */
export const getWorkflowGeneratorPropertyInvalidReason = (
  value: WorkflowGeneratorValue,
  key: string
): string | null => {
  const spec = VARIANT_PROPERTIES[value.type][key];

  if (spec !== 'number' && spec !== 'integer' && spec !== 'count' && spec !== 'seed') {
    return null;
  }

  const raw = (value as unknown as Record<string, unknown>)[key];
  const label = PROPERTY_LABELS[key] ?? key;

  if (raw === null) {
    return spec === 'seed' ? null : `${label} is empty.`;
  }

  if (!isFiniteNumber(raw)) {
    return `${label} is invalid.`;
  }

  if (spec === 'number') {
    return null;
  }

  if (!Number.isInteger(raw)) {
    return `${label} must be a whole number.`;
  }

  if (spec === 'count' && raw < 1) {
    return `${label} must be at least 1.`;
  }

  if (spec === 'seed' && raw < 0) {
    return `${label} must be at least 0.`;
  }

  if (
    spec === 'count' &&
    value.type.startsWith('string_generator_dynamic_prompts') &&
    raw > WORKFLOW_DYNAMIC_PROMPTS_MAX
  ) {
    return `${label} can be at most ${WORKFLOW_DYNAMIC_PROMPTS_MAX}.`;
  }

  return null;
};

/** The first setting that keeps a readable generator from running, or null when it can. */
export const getWorkflowGeneratorInvalidReason = (value: WorkflowGeneratorValue): string | null => {
  // A legacy resolved override stands in for the settings, as it does when resolving.
  if ('values' in value && value.values) {
    return null;
  }

  for (const key of Object.keys(VARIANT_PROPERTIES[value.type])) {
    const reason = getWorkflowGeneratorPropertyInvalidReason(value, key);

    if (reason !== null) {
      return reason;
    }
  }

  return null;
};

/** A generator whose settings all pass `getWorkflowGeneratorInvalidReason`: numbers present, seeds still optional. */
type Runnable<T> = T extends unknown ? { [K in keyof T]: K extends 'seed' ? T[K] : Exclude<T[K], null> } : never;
export type RunnableWorkflowGeneratorValue = Runnable<WorkflowGeneratorValue>;

/**
 * How many items a generator asks for, before any are built: a stored count can be arbitrarily large, and the
 * queue accepts at most `WORKFLOW_BATCH_MAX_ITEMS`, so callers refuse oversized requests without materializing them.
 */
export const getWorkflowGeneratorRequestedCount = (value: WorkflowGeneratorValue): number | null =>
  'values' in value && value.values
    ? value.values.length
    : 'step' in value && value.step === 0
      ? 1
      : 'count' in value
        ? value.count
        : 'maxPrompts' in value
          ? value.maxPrompts
          : null;

/** Random variants without a seed draw fresh values on every resolve, so no preview can show them. */
export const isRandomWorkflowGeneratorValue = (value: WorkflowGeneratorValue): boolean =>
  (value.type === 'float_generator_random_distribution_uniform' ||
    value.type === 'integer_generator_random_distribution_uniform' ||
    value.type === 'string_generator_dynamic_prompts_random') &&
  value.seed === null;

// #endregion

// #region Resolution

export type WorkflowBatchItem = number | string | { image_name: string };

export type WorkflowAsyncGeneratorRequest =
  | { kind: 'dynamicPrompts'; prompt: string; maxPrompts: number; combinatorial: boolean; seed: number | null }
  | { kind: 'boardImages'; boardId: string; category: 'images' | 'assets' };

export type WorkflowGeneratorResolution =
  | { kind: 'items'; items: WorkflowBatchItem[] }
  | { kind: 'async'; request: WorkflowAsyncGeneratorRequest };

/** mulberry32: small, seedable, and good enough for spreading sample values; not bit-compatible with legacy. */
export const createSeededRandom = (seed: number): (() => number) => {
  let state = seed >>> 0;

  return () => {
    state = (state + 0x6d2b79f5) >>> 0;

    let t = state;

    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);

    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

/** An empty separator keeps the whole input; escapes such as `\n` are honoured, a lone backslash is literal. */
const splitInput = (input: string, splitOn: string): string[] => {
  if (splitOn === '') {
    return [input];
  }

  let separator = splitOn;

  try {
    const parsed: unknown = JSON.parse(`"${splitOn}"`);

    if (typeof parsed === 'string') {
      separator = parsed;
    }
  } catch {
    // Not a valid escape sequence: split on the text as typed.
  }

  return input.split(separator);
};

const parseNumbers = (input: string, splitOn: string, parse: (text: string) => number): number[] =>
  splitInput(input, splitOn)
    .map((text) => text.trim())
    .filter((text) => text !== '')
    .map(parse)
    .filter((number) => !Number.isNaN(number));

const sequence = (count: number, at: (index: number) => number): number[] =>
  Array.from({ length: count }, (_, index) => at(index));

/** Turns a stored generator into its items, or into the request an async source needs. */
export const resolveWorkflowGeneratorValue = (
  stored: WorkflowGeneratorValue,
  random: () => number = Math.random
): WorkflowGeneratorResolution => {
  const items = (list: WorkflowBatchItem[]): WorkflowGeneratorResolution => ({ items: list, kind: 'items' });

  if ('values' in stored && stored.values) {
    return items(stored.values);
  }

  // A setting that cannot run yields nothing; the field check names it.
  if (getWorkflowGeneratorInvalidReason(stored) !== null) {
    return items([]);
  }

  const value = stored as RunnableWorkflowGeneratorValue;

  switch (value.type) {
    case 'float_generator_arithmetic_sequence':
    case 'integer_generator_arithmetic_sequence':
      return items(value.step === 0 ? [value.start] : sequence(value.count, (i) => value.start + i * value.step));
    case 'float_generator_linear_distribution':
      return items(
        value.count === 1
          ? [value.start]
          : sequence(value.count, (i) => value.start + (value.end - value.start) * (i / (value.count - 1)))
      );
    case 'integer_generator_linear_distribution':
      return items(
        value.count === 1
          ? [value.start]
          : sequence(value.count, (i) => value.start + Math.round((value.end - value.start) * (i / (value.count - 1))))
      );
    case 'float_generator_random_distribution_uniform': {
      const draw = value.seed === null ? random : createSeededRandom(value.seed);

      return items(sequence(value.count, () => draw() * (value.max - value.min) + value.min));
    }
    case 'integer_generator_random_distribution_uniform': {
      const draw = value.seed === null ? random : createSeededRandom(value.seed);

      return items(sequence(value.count, () => Math.floor(draw() * (value.max - value.min + 1)) + value.min));
    }
    case 'float_generator_parse_string':
      return items(parseNumbers(value.input, value.splitOn, parseFloat));
    case 'integer_generator_parse_string':
      return items(parseNumbers(value.input, value.splitOn, (text) => parseInt(text, 10)));
    case 'string_generator_parse_string':
      return items(splitInput(value.input, value.splitOn).filter((text) => text !== ''));
    case 'string_generator_dynamic_prompts_random':
      return {
        kind: 'async',
        request: {
          combinatorial: false,
          kind: 'dynamicPrompts',
          maxPrompts: value.count,
          prompt: value.input,
          seed: value.seed,
        },
      };
    case 'string_generator_dynamic_prompts_combinatorial':
      return {
        kind: 'async',
        request: {
          combinatorial: true,
          kind: 'dynamicPrompts',
          maxPrompts: value.maxPrompts,
          prompt: value.input,
          seed: null,
        },
      };
    case 'image_generator_images_from_board':
      return value.board_id === undefined
        ? items([])
        : { kind: 'async', request: { boardId: value.board_id, category: value.category, kind: 'boardImages' } };
  }
};

// #endregion

// #region Planning

export interface WorkflowBatchDatum {
  nodeId: string;
  fieldName: string;
  items: WorkflowBatchItem[];
}

/** Async generator outputs resolved before submission, keyed by generator node id and fingerprinted by value. */
export type WorkflowGeneratorResolutions = Record<string, { key: string; items: WorkflowBatchItem[] }>;

export interface WorkflowPendingGenerator {
  nodeId: string;
  /** The generator's display name, for the error a failed resolution shows. */
  nodeLabel: string;
  key: string;
  request: WorkflowAsyncGeneratorRequest;
}

export interface WorkflowBatchPlan {
  /** Outer list is a cartesian product; each inner list is zipped, the backend's `Batch.data` shape. */
  groups: WorkflowBatchDatum[][];
  /** Sessions one run of the graph produces, or null while an async generator is still unresolved. */
  batchSize: number | null;
  hasBatchNodes: boolean;
  pendingGenerators: WorkflowPendingGenerator[];
  reasons: string[];
}

export interface WorkflowBatchPlanOptions {
  generators?: WorkflowGeneratorResolutions;
  random?: () => number;
}

const getNodeDisplayName = (node: WorkflowInvocationNode, templates: InvocationTemplates): string =>
  node.data.label || templates[node.data.type]?.title || node.data.type;

const toBatchKind = (type: WorkflowBatchNodeType, items: unknown[]): WorkflowBatchItem[] =>
  items.map((item) =>
    type === 'string_batch' && typeof item === 'number' ? String(item) : item
  ) as WorkflowBatchItem[];

/**
 * Resolves every batch node into the datums its outgoing connections need, grouped the way the backend zips and
 * multiplies them: ungrouped nodes each form a cartesian dimension, named groups zip their members together.
 */
export const planWorkflowBatch = (
  document: Pick<ProjectGraphState, 'nodes' | 'edges'>,
  templates: InvocationTemplates,
  { generators = {}, random = Math.random }: WorkflowBatchPlanOptions = {}
): WorkflowBatchPlan => {
  const batchNodes = document.nodes.filter(
    (node): node is WorkflowInvocationNode => isInvocationNode(node) && isWorkflowBatchNodeType(node.data.type)
  );

  if (batchNodes.length === 0) {
    return { batchSize: 1, groups: [], hasBatchNodes: false, pendingGenerators: [], reasons: [] };
  }

  const nodesById = new Map(document.nodes.map((node) => [node.id, node]));
  const edges = getCanonicalWorkflowEdges(document);
  const reasons: string[] = [];
  const pendingGenerators: WorkflowPendingGenerator[] = [];
  const ungrouped: Array<{ datums: WorkflowBatchDatum[]; size: number | null }> = [];
  const grouped = new Map<
    WorkflowBatchGroupId,
    Array<{ node: WorkflowInvocationNode; datums: WorkflowBatchDatum[]; size: number | null }>
  >();
  const claimedTargets = new Map<string, WorkflowInvocationNode>();
  const pendingNodeIds = new Set<string>();

  for (const batchNode of batchNodes) {
    const type = batchNode.data.type as WorkflowBatchNodeType;
    const name = getNodeDisplayName(batchNode, templates);
    const collectionField = BATCH_COLLECTION_FIELDS[type];
    const incoming = edges.find(
      (edge) => edge.destination.node_id === batchNode.id && edge.destination.field === collectionField
    );
    let items: WorkflowBatchItem[] | null = null;
    let oversizedGenerator = false;

    if (incoming) {
      const source = nodesById.get(incoming.source.node_id);
      const generatorType = source && isInvocationNode(source) ? source.data.type : '';

      if (!source || !isInvocationNode(source) || !isWorkflowGeneratorNodeType(generatorType)) {
        reasons.push(`Batch node "${name}" can only receive its collection from a generator.`);
      } else {
        const value = parseWorkflowGeneratorValue(
          GENERATOR_FIELD_TYPE_NAMES[generatorType],
          source.data.inputs.generator?.value
        );

        // An unreadable or off-rule generator is reported by the field check; here it only leaves the size unknown.
        const runnable = value !== null && getWorkflowGeneratorInvalidReason(value) === null ? value : null;
        const requested = runnable ? getWorkflowGeneratorRequestedCount(runnable) : null;

        if (requested !== null && requested > WORKFLOW_BATCH_MAX_ITEMS) {
          // Refused by its count alone: building the list would cost memory in every readiness pass.
          oversizedGenerator = true;
        } else if (runnable?.type === 'image_generator_images_from_board' && runnable.board_id === undefined) {
          reasons.push(`Image generator "${getNodeDisplayName(source, templates)}" has no board selected.`);
        } else if (runnable) {
          const resolution = resolveWorkflowGeneratorValue(runnable, random);

          if (resolution.kind === 'items') {
            items = resolution.items;
          } else {
            const key = getWorkflowGeneratorKey(runnable);
            const resolved = generators[source.id];

            if (resolved && resolved.key === key) {
              items = resolved.items;
            } else if (!pendingNodeIds.has(source.id)) {
              // One generator may feed several batch nodes; it is resolved once and shared.
              pendingNodeIds.add(source.id);
              pendingGenerators.push({
                key,
                nodeId: source.id,
                nodeLabel: getNodeDisplayName(source, templates),
                request: resolution.request,
              });
            }
          }
        }
      }
    } else {
      const own = batchNode.data.inputs[collectionField]?.value;

      items = Array.isArray(own) ? (own as WorkflowBatchItem[]) : [];
    }

    if (items) {
      items = toBatchKind(type, items);
    }

    const outputField = getWorkflowBatchOutputField(type);
    const datums: WorkflowBatchDatum[] = [];
    let hasOutgoing = false;

    for (const edge of edges) {
      if (edge.source.node_id !== batchNode.id) {
        continue;
      }

      hasOutgoing = true;

      if (edge.source.field !== outputField) {
        reasons.push(`Batch node "${name}" can only batch its image output; width and height are not supported.`);
        continue;
      }

      const target = nodesById.get(edge.destination.node_id);

      if (!target || !isInvocationNode(target)) {
        continue;
      }

      const targetName = getNodeDisplayName(target, templates);
      const targetTemplate = templates[target.data.type];
      const targetInput = targetTemplate?.inputs[edge.destination.field];

      if (
        isWorkflowBatchNodeType(target.data.type) ||
        isWorkflowGeneratorNodeType(target.data.type) ||
        (targetTemplate && !targetInput)
      ) {
        reasons.push(
          `"${name}" cannot supply batch values to "${targetInput?.title ?? edge.destination.field}" on "${targetName}".`
        );
        continue;
      }

      const targetKey = `${target.id}:${edge.destination.field}`;
      const claimedBy = claimedTargets.get(targetKey);

      if (claimedBy && claimedBy !== batchNode) {
        reasons.push(
          `"${targetInput?.title ?? edge.destination.field}" on "${targetName}" receives values from more than one batch node.`
        );
        continue;
      }

      claimedTargets.set(targetKey, batchNode);

      if (items) {
        const coerced =
          targetInput?.type.name === 'StringField'
            ? items.map((item) => (typeof item === 'number' ? String(item) : item))
            : items;

        datums.push({ fieldName: edge.destination.field, items: coerced, nodeId: target.id });
      }
    }

    if (!hasOutgoing) {
      reasons.push(`Batch node "${name}" is not connected to any node.`);
    }

    const size = items ? items.length : null;

    if (size === 0) {
      reasons.push(`Batch node "${name}" has an empty collection.`);
    } else if (oversizedGenerator || (size !== null && size > WORKFLOW_BATCH_MAX_ITEMS)) {
      reasons.push(`Batch node "${name}" has more than ${WORKFLOW_BATCH_MAX_ITEMS.toLocaleString('en-US')} items.`);
    }

    const groupId = getWorkflowBatchGroupId(batchNode);

    if (groupId === 'None') {
      ungrouped.push({ datums, size });
    } else {
      const members = grouped.get(groupId) ?? [];

      members.push({ datums, node: batchNode, size });
      grouped.set(groupId, members);
    }
  }

  const groups: WorkflowBatchDatum[][] = ungrouped
    .filter((entry) => entry.datums.length > 0)
    .map((entry) => entry.datums);
  let batchSize: number | null = ungrouped.reduce<number | null>(
    (product, entry) => (product === null || entry.size === null ? null : product * entry.size),
    1
  );

  for (const [groupId, members] of grouped) {
    const knownSizes = members.flatMap((member) => (member.size === null ? [] : [member.size]));

    if (new Set(knownSizes).size > 1) {
      const sizes = members
        .filter((member) => member.size !== null)
        .map((member) => `${getNodeDisplayName(member.node, templates)}: ${member.size}`)
        .join(', ');

      reasons.push(`Batch group "${groupId}" mixes collection sizes (${sizes}).`);
    }

    const groupDatums = members.flatMap((member) => member.datums);

    if (groupDatums.length > 0) {
      groups.push(groupDatums);
    }

    const size = members.some((member) => member.size === null) ? null : (knownSizes[0] ?? 1);

    batchSize = batchSize === null || size === null ? null : batchSize * size;
  }

  return { batchSize, groups, hasBatchNodes: true, pendingGenerators, reasons };
};

// #endregion
