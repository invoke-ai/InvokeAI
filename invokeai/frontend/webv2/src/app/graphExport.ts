/**
 * Composition root of `pnpm graphs:export` (scripts/export-graphs.mjs): installed model configs, the capability table
 * and test cases become the enqueue batches the UI would send, offline, so end-to-end runs exercise the graphs users
 * submit. Each batch is what Generate or Canvas puts into `enqueue_batch` minus `runs`, `project_id` and
 * `idempotency_key`, with the case seed held fixed. Generation and Queue own every rule; this module maps the wire
 * format and supplies the Workbench defaults the widgets would.
 */

import type { GeneratePromptBatchPlan } from '@features/queue/contracts';

import {
  compileGraphExportCase,
  withArchitectureCapabilities,
  type ArchitectureCapabilitiesRow,
  type CompiledGraphExportCase,
  type GraphExportCase,
  type GraphExportEnvironment,
  type GraphExportMode,
} from '@features/generation/graphExport';
import { planGenerateSubmissionBatch } from '@features/queue/contracts';
import { resolveRandDeviceMetadata } from '@features/queue/devices';
import { SEED_MAX } from '@platform/core/seed';
import { DEFAULT_CANVAS_COMPOSITING } from '@workbench/widgets/canvas/invoke/canvasCompositing';
import { DEFAULT_CANVAS_SCALING } from '@workbench/widgets/canvas/invoke/canvasScaling';
import { DEFAULT_CANVAS_DENOISING_STRENGTH } from '@workbench/widgets/canvas/invoke/canvasStrength';

const GRAPH_EXPORT_VERSION = 1;

/** Durable outputs: Canvas staging would mark them intermediate. */
const DESTINATION = 'gallery';
/** Not a webv2 origin, so no open editor adopts the items as its own submissions. */
const ORIGIN_PREFIX = 'graphs-export:';
/** `DEFAULT_PROJECT_SETTINGS` (workbench/settings/store.ts); importing it would load the whole settings runtime. */
const PROJECT_SETTINGS = { useCpuNoise: true };
/** ROCm builds of torch report their GPUs as CUDA devices as well. */
const GPU_DEVICE_OPTIONS = [{ device: 'cuda:0', name: 'GPU' }];
const MODES: readonly GraphExportMode[] = ['txt2img', 'img2img', 'controlnet'];
const CASE_FIELDS = [
  'components',
  'control',
  'height',
  'id',
  'init_image',
  'loras',
  'mode',
  'model_key',
  'negative_prompt',
  'positive_prompt',
  'reference_images',
  'seed',
  'settings',
  'steps',
  'width',
];

/** The request itself is unusable; no response is written. */
export class GraphExportRequestError extends Error {
  override name = 'GraphExportRequestError';
}

export interface GraphExportBatch {
  data: GeneratePromptBatchPlan['data'];
  destination: typeof DESTINATION;
  graph: CompiledGraphExportCase['graph'];
  origin: string;
}

export type GraphExportCaseResult =
  | { id: string; batch: GraphExportBatch; components: Record<string, string> }
  | { id: string; error: string };

export interface GraphExportResponse {
  version: typeof GRAPH_EXPORT_VERSION;
  cases: GraphExportCaseResult[];
}

type JsonRecord = Record<string, unknown>;

const isRecord = (value: unknown): value is JsonRecord =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const fail = (message: string): never => {
  throw new Error(message);
};

const readString = (record: JsonRecord, field: string, label = field): string =>
  typeof record[field] === 'string' ? record[field] : fail(`${label} must be a string.`);

const readNumber = (record: JsonRecord, field: string, label = field): number =>
  typeof record[field] === 'number' && Number.isFinite(record[field])
    ? record[field]
    : fail(`${label} must be a number.`);

const readInteger = (record: JsonRecord, field: string, min: number, max = Number.MAX_SAFE_INTEGER): number => {
  const value = record[field];

  return typeof value === 'number' && Number.isInteger(value) && value >= min && value <= max
    ? value
    : fail(`${field} must be an integer from ${min} to ${max}.`);
};

/** Absent and null both read as empty. */
const readList = <T>(record: JsonRecord, field: string, read: (item: unknown, label: string) => T): T[] => {
  const value = record[field] ?? [];

  return Array.isArray(value)
    ? value.map((item, index) => read(item, `${field}[${index}]`))
    : fail(`${field} must be a list.`);
};

const readObject = (value: unknown, label: string): JsonRecord =>
  isRecord(value) ? value : fail(`${label} must be an object.`);

const readCase = (record: JsonRecord): GraphExportCase => {
  const unknownField = Object.keys(record).find((field) => !CASE_FIELDS.includes(field));

  if (unknownField !== undefined) {
    fail(`Unknown case field "${unknownField}". Valid fields: ${CASE_FIELDS.join(', ')}.`);
  }

  const mode = record.mode;
  const components = readObject(record.components ?? {}, 'components');
  const initImage =
    record.init_image === undefined || record.init_image === null ? null : readObject(record.init_image, 'init_image');

  return {
    components: Object.fromEntries(
      Object.keys(components).map((slot) => [slot, readString(components, slot, `components.${slot}`)])
    ),
    control: readList(record, 'control', (item, label) => {
      const layer = readObject(item, label);

      return {
        begin: readNumber(layer, 'begin', `${label}.begin`),
        end: readNumber(layer, 'end', `${label}.end`),
        imageName: readString(layer, 'image_name', `${label}.image_name`),
        modelKey: readString(layer, 'model_key', `${label}.model_key`),
        weight: readNumber(layer, 'weight', `${label}.weight`),
      };
    }),
    height: readInteger(record, 'height', 1),
    id: readString(record, 'id'),
    initImage: initImage && {
      imageName: readString(initImage, 'image_name', 'init_image.image_name'),
      strength: readNumber(initImage, 'strength', 'init_image.strength'),
    },
    loras: readList(record, 'loras', (item, label) => {
      const lora = readObject(item, label);

      return { key: readString(lora, 'key', `${label}.key`), weight: readNumber(lora, 'weight', `${label}.weight`) };
    }),
    mode: MODES.includes(mode as GraphExportMode)
      ? (mode as GraphExportMode)
      : fail(`mode must be one of ${MODES.join(', ')}.`),
    modelKey: readString(record, 'model_key'),
    negativePrompt:
      record.negative_prompt === null || record.negative_prompt === undefined
        ? ''
        : readString(record, 'negative_prompt'),
    positivePrompt: readString(record, 'positive_prompt'),
    referenceImages: readList(record, 'reference_images', (item, label) =>
      typeof item === 'string' ? item : fail(`${label} must be an image name.`)
    ),
    seed: readInteger(record, 'seed', 0, SEED_MAX),
    settings: readObject(record.settings ?? {}, 'settings'),
    steps: readInteger(record, 'steps', 1),
    width: readInteger(record, 'width', 1),
  };
};

const exportCase = (record: JsonRecord, env: GraphExportEnvironment): GraphExportCaseResult => {
  const id = record.id as string;

  try {
    const { components, graph, submission } = compileGraphExportCase(readCase(record), env);
    const batch: GraphExportBatch = {
      data: planGenerateSubmissionBatch(submission).data,
      destination: DESTINATION,
      graph,
      origin: `${ORIGIN_PREFIX}${id}`,
    };

    return { batch, components, id };
  } catch (error) {
    return { error: error instanceof Error ? error.message : String(error), id };
  }
};

const isModelConfig = (value: unknown): value is GraphExportEnvironment['models'][number] =>
  isRecord(value) && ['key', 'name', 'base', 'type'].every((field) => typeof value[field] === 'string');

const isCapabilitiesRow = (value: unknown): value is ArchitectureCapabilitiesRow =>
  isRecord(value) && typeof value.base === 'string' && isRecord(value.features);

const readRequest = (request: unknown) => {
  if (!isRecord(request)) {
    throw new GraphExportRequestError('The request must be a JSON object.');
  }

  if (request.version !== GRAPH_EXPORT_VERSION) {
    throw new GraphExportRequestError(`Unsupported request version ${JSON.stringify(request.version)}; expected 1.`);
  }

  const { capabilities, cases, models } = request;

  if (!Array.isArray(models) || !models.every(isModelConfig)) {
    throw new GraphExportRequestError('models must be the `models` list of GET /api/v2/models/.');
  }

  if (!Array.isArray(capabilities) || !capabilities.every(isCapabilitiesRow)) {
    throw new GraphExportRequestError('capabilities must be the body of GET /api/v2/models/capabilities.');
  }

  if (!Array.isArray(cases) || !cases.every((item) => isRecord(item) && typeof item.id === 'string' && item.id)) {
    throw new GraphExportRequestError('cases must be a list of objects with a non-empty string id.');
  }

  const ids = cases.map((item) => item.id as string);
  const duplicate = ids.find((id, index) => ids.indexOf(id) !== index);

  if (duplicate !== undefined) {
    throw new GraphExportRequestError(`Case id "${duplicate}" is not unique.`);
  }

  return { capabilities, cases: cases as JsonRecord[], models };
};

/** Throws GraphExportRequestError for a malformed request; a case that cannot compile carries its error instead. */
export const exportGraphs = (request: unknown): GraphExportResponse => {
  const { capabilities, cases, models } = readRequest(request);
  const env: GraphExportEnvironment = {
    canvas: {
      compositing: DEFAULT_CANVAS_COMPOSITING,
      scaling: DEFAULT_CANVAS_SCALING,
      strength: DEFAULT_CANVAS_DENOISING_STRENGTH,
    },
    destination: DESTINATION,
    models,
    projectSettings: PROJECT_SETTINGS,
    randDevice: resolveRandDeviceMetadata(PROJECT_SETTINGS.useCpuNoise, GPU_DEVICE_OPTIONS),
  };

  return withArchitectureCapabilities(capabilities, () => ({
    cases: cases.map((record) => exportCase(record, env)),
    version: GRAPH_EXPORT_VERSION,
  }));
};
