import type {
  QueueBackendGraph,
  QueueCompiledSubmission,
  QueueSourceId,
  QueueSubmissionPresentation,
} from '@features/queue/contracts';
import type { CanvasStateContractV2 } from '@workbench/canvas-engine/api';
import type { GraphContract } from '@workbench/graphContracts';
import type { WidgetInstanceContract, WidgetInstanceId, WidgetStateMap } from '@workbench/widgetContracts';

import { normalizeGenerateSettings, sanitizeBatchCount } from '@features/generation/settings';
import { getUpscaleOutputDimensions, normalizeUpscaleWidgetValues } from '@features/upscale';

import type { WorkbenchQueueItem, WorkbenchQueueState } from './queueHistoryContracts';

import { migrateCanvasStateToV2 } from './canvasMigration';

type UnknownRecord = Record<string, unknown>;
type PresentationSource = { batchCount: number; height?: number; positivePrompt?: string; width?: number } | null;

export interface QueueHistoryNormalizationContext {
  canvas: CanvasStateContractV2;
  widgetInstances: Record<WidgetInstanceId, WidgetInstanceContract>;
}

const isRecord = (value: unknown): value is UnknownRecord => value !== null && typeof value === 'object';

const normalizeSourceId = (value: unknown): QueueSourceId | null => {
  if (value === 'project-graph') {
    return 'workflow';
  }
  if (value === 'canvas-fill') {
    return 'canvas';
  }
  return value === 'canvas' || value === 'generate' || value === 'upscale' || value === 'workflow' ? value : null;
};

const isBackendGraph = (value: unknown): value is QueueBackendGraph =>
  isRecord(value) && typeof value.id === 'string' && isRecord(value.nodes) && Array.isArray(value.edges);

const isCurrentBackendSubmission = (value: unknown): value is QueueCompiledSubmission => {
  if (!isRecord(value) || typeof value.kind !== 'string') {
    return false;
  }
  if (value.kind === 'invalid') {
    return typeof value.error === 'string';
  }
  if (
    (value.kind !== 'generate' && value.kind !== 'workflow') ||
    !isBackendGraph(value.graph) ||
    typeof value.batchCount !== 'number' ||
    !Number.isFinite(value.batchCount) ||
    value.batchCount < 1
  ) {
    return false;
  }
  if (value.kind === 'workflow') {
    return true;
  }
  return (
    typeof value.negativePrompt === 'string' &&
    typeof value.negativePromptNodeId === 'string' &&
    typeof value.positivePrompt === 'string' &&
    typeof value.positivePromptNodeId === 'string' &&
    typeof value.seed === 'number' &&
    Number.isFinite(value.seed) &&
    typeof value.seedNodeId === 'string' &&
    typeof value.shouldRandomizeSeed === 'boolean'
  );
};

const isCurrentPresentation = (value: unknown): value is QueueSubmissionPresentation =>
  isRecord(value) &&
  typeof value.batchCount === 'number' &&
  Number.isFinite(value.batchCount) &&
  value.batchCount > 0 &&
  typeof value.height === 'number' &&
  Number.isFinite(value.height) &&
  value.height > 0 &&
  typeof value.width === 'number' &&
  Number.isFinite(value.width) &&
  value.width > 0;

const isCurrentSnapshot = (value: UnknownRecord): boolean =>
  isCurrentBackendSubmission(value.backendSubmission) &&
  isCurrentPresentation(value.presentation) &&
  normalizeSourceId(value.sourceId) === value.sourceId &&
  (value.destination === 'canvas' || value.destination === 'gallery') &&
  isRecord(value.graph) &&
  (value.galleryBoardId === null || typeof value.galleryBoardId === 'string') &&
  typeof value.filterIntermediateResults === 'boolean' &&
  typeof value.submittedAt === 'string';

const getWidgetStates = (snapshot: UnknownRecord): WidgetStateMap =>
  isRecord(snapshot.widgetStates) ? (snapshot.widgetStates as WidgetStateMap) : {};

const getWidgetValues = (widgetStates: WidgetStateMap, widgetId: string): unknown => {
  const state = widgetStates[widgetId];
  return isRecord(state) && isRecord(state.values) ? state.values : undefined;
};

const getGenerateCapture = (snapshot: UnknownRecord): UnknownRecord | null =>
  isRecord(snapshot.generate) ? snapshot.generate : null;

const getFiniteDimension = (value: unknown, fallback: number): number =>
  typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : fallback;

const createInvalidSubmission = (message: string): QueueCompiledSubmission => ({ error: message, kind: 'invalid' });

const getBackendSubmission = (
  sourceId: QueueSourceId | null,
  snapshot: UnknownRecord,
  widgetStates: WidgetStateMap
): { presentationSource: PresentationSource; submission: QueueCompiledSubmission } => {
  const graph = isRecord(snapshot.graph) ? snapshot.graph : null;
  const backendGraph = graph?.backendGraph;
  const generateCapture = getGenerateCapture(snapshot);
  const generateValues = getWidgetValues(widgetStates, 'generate');
  const presentationSource = normalizeGenerateSettings(
    sourceId === 'canvas' ? (generateCapture?.values ?? generateValues) : generateValues
  );

  if (!sourceId) {
    return {
      presentationSource,
      submission: createInvalidSubmission('Legacy queue item has an unsupported or missing source.'),
    };
  }
  if (!isBackendGraph(backendGraph)) {
    return {
      presentationSource,
      submission: createInvalidSubmission(`Legacy ${sourceId} queue item is missing a compiled backend graph.`),
    };
  }
  if (sourceId === 'workflow') {
    return {
      presentationSource,
      submission: {
        batchCount: sanitizeBatchCount(isRecord(generateValues) ? generateValues.batchCount : undefined),
        graph: backendGraph,
        kind: 'workflow',
      },
    };
  }

  const sourceSettings =
    sourceId === 'upscale'
      ? normalizeUpscaleWidgetValues(getWidgetValues(widgetStates, 'upscale'))
      : sourceId === 'canvas'
        ? normalizeGenerateSettings(generateCapture?.values ?? generateValues)
        : normalizeGenerateSettings(generateValues);

  if (!sourceSettings) {
    return {
      presentationSource,
      submission: createInvalidSubmission(`Legacy ${sourceId} queue item is missing source submission metadata.`),
    };
  }

  return {
    presentationSource: sourceId === 'upscale' ? null : sourceSettings,
    submission: {
      batchCount: sourceSettings.batchCount,
      graph: backendGraph,
      kind: 'generate',
      negativePrompt: sourceSettings.negativePromptEnabled ? sourceSettings.negativePrompt : '',
      negativePromptNodeId:
        typeof generateCapture?.negativePromptNodeId === 'string'
          ? generateCapture.negativePromptNodeId
          : 'negative_prompt',
      positivePrompt: sourceSettings.positivePrompt,
      positivePromptNodeId:
        typeof generateCapture?.positivePromptNodeId === 'string'
          ? generateCapture.positivePromptNodeId
          : 'positive_prompt',
      seed: sourceSettings.seed,
      seedNodeId: typeof generateCapture?.seedNodeId === 'string' ? generateCapture.seedNodeId : 'seed',
      shouldRandomizeSeed: sourceSettings.shouldRandomizeSeed,
    },
  };
};

const normalizeLegacyQueueItem = (
  value: unknown,
  index: number,
  context: QueueHistoryNormalizationContext
): WorkbenchQueueItem => {
  const item = isRecord(value) ? value : {};
  const snapshot = isRecord(item.snapshot) ? item.snapshot : {};
  const sourceId = normalizeSourceId(snapshot.sourceId);
  const widgetStates = getWidgetStates(snapshot);
  const { presentationSource, submission } = getBackendSubmission(sourceId, snapshot, widgetStates);
  const upscaleValues =
    sourceId === 'upscale' ? normalizeUpscaleWidgetValues(getWidgetValues(widgetStates, 'upscale')) : null;
  const canvas = migrateCanvasStateToV2(snapshot.canvas ?? context.canvas);
  const canvasDocument = canvas.document;
  const dimensions =
    upscaleValues?.inputImage && Number.isFinite(upscaleValues.scale)
      ? getUpscaleOutputDimensions(upscaleValues.inputImage, upscaleValues.scale)
      : {
          height: getFiniteDimension(presentationSource?.height, canvasDocument.height),
          width: getFiniteDimension(presentationSource?.width, canvasDocument.width),
        };
  const galleryValues = getWidgetValues(widgetStates, 'gallery');
  const selectedBoardId = isRecord(galleryValues) ? galleryValues.selectedBoardId : undefined;
  const graph = isRecord(snapshot.graph)
    ? (snapshot.graph as unknown as GraphContract)
    : {
        edges: [],
        id: `invalid-legacy-queue-graph-${index}`,
        label: 'Unavailable legacy queue graph',
        nodes: [],
        updatedAt: new Date(0).toISOString(),
        version: 1,
      };
  const safeSourceId = sourceId ?? 'workflow';

  return {
    ...item,
    cancellable: typeof item.cancellable === 'boolean' ? item.cancellable : false,
    id: typeof item.id === 'string' ? item.id : `invalid-legacy-queue-item-${index}`,
    snapshot: {
      ...snapshot,
      backendSubmission: submission,
      canvas,
      destination: snapshot.destination === 'gallery' ? 'gallery' : 'canvas',
      filterIntermediateResults: safeSourceId === 'workflow',
      galleryBoardId: typeof selectedBoardId === 'string' ? selectedBoardId : null,
      graph,
      presentation: {
        batchCount: submission.kind === 'invalid' ? 1 : submission.batchCount,
        height: dimensions.height,
        ...(presentationSource?.positivePrompt ? { positivePrompt: presentationSource.positivePrompt } : {}),
        width: dimensions.width,
      },
      ...(safeSourceId === 'generate' || safeSourceId === 'canvas'
        ? { resultNodeIds: ['canvas_output'] }
        : safeSourceId === 'upscale'
          ? { resultNodeIds: ['upscale_output'] }
          : {}),
      sourceId: safeSourceId,
      submittedAt: typeof snapshot.submittedAt === 'string' ? snapshot.submittedAt : new Date(0).toISOString(),
      widgetInstances: isRecord(snapshot.widgetInstances)
        ? (snapshot.widgetInstances as Record<WidgetInstanceId, WidgetInstanceContract>)
        : context.widgetInstances,
      widgetStates,
    },
    status:
      item.status === 'pending' ||
      item.status === 'running' ||
      item.status === 'completed' ||
      item.status === 'failed' ||
      item.status === 'cancelled'
        ? item.status
        : 'failed',
  } as WorkbenchQueueItem;
};

/** Upgrades legacy nested snapshots at Workbench's project-ingestion boundary. */
export const normalizeWorkbenchQueueHistory = (
  value: unknown,
  context: QueueHistoryNormalizationContext
): WorkbenchQueueState => {
  if (!isRecord(value) || !Array.isArray(value.items)) {
    return { items: [] };
  }

  let didChange = false;
  const items = value.items.map((item, index) => {
    const snapshot = isRecord(item) && isRecord(item.snapshot) ? item.snapshot : null;
    if (snapshot && isCurrentSnapshot(snapshot)) {
      return item as unknown as WorkbenchQueueItem;
    }
    didChange = true;
    return normalizeLegacyQueueItem(item, index, context);
  });

  return didChange ? ({ ...value, items } as WorkbenchQueueState) : (value as unknown as WorkbenchQueueState);
};
