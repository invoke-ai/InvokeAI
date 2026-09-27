import type { ModelConfig } from '@features/models';
import type { ProjectGraphState } from '@features/workflow/contracts';
import type { ForLoopValidationReason } from '@features/workflow/utility';
import type { CanvasLayerContract } from '@workbench/canvas-engine/api';
import type {
  InvocationMode,
  InvocationRoute,
  InvocationSourceId,
  ResolvedInvocationRoute,
  ResultDestination,
} from '@workbench/invocationContracts';
import type { Project } from '@workbench/projectContracts';
import type { WidgetId } from '@workbench/widgetContracts';

import {
  getGenerationModelAvailabilityReasons,
  getGenerationValidationReasons,
  isSupportedGenerateModel,
  normalizeGenerateWidgetValues,
  sanitizeBatchCount,
} from '@features/generation/settings';
import { getUpscaleValidationReasons, normalizeUpscaleWidgetValues } from '@features/upscale';
import { getVideoWidgetValidationReasons, normalizeVideoWidgetValues } from '@features/video';
import { getProjectGraphReadiness } from '@features/workflow/graph';
import { getInvocationTemplatesSnapshot } from '@features/workflow/react';
import { areArraysEqual, createStableSelector } from '@platform/state/selectors';
import { compileContributingLayers } from '@workbench/canvas-engine/api';

import { getBlockingControlLayerIssues } from './controlLayerChecks';
import { getProjectWidgetValues } from './widgetState';

export interface InvocationSourceMeta {
  id: InvocationSourceId;
  label: string;
  /** Whether the source is selectable yet, or a forward-looking placeholder. */
  available: boolean;
}

export interface ResultDestinationMeta {
  id: ResultDestination;
  label: string;
}

export const invocationSources: InvocationSourceMeta[] = [
  { id: 'generate', label: 'Generate', available: true },
  { id: 'workflow', label: 'Workflow', available: true },
  { id: 'upscale', label: 'Upscale', available: true },
  { id: 'video', label: 'Video', available: true },
  { id: 'canvas', label: 'Canvas', available: true },
];

export const resultDestinations: ResultDestinationMeta[] = [
  { id: 'canvas', label: 'Canvas' },
  { id: 'gallery', label: 'Gallery' },
];

const sourceLabels = new Map(invocationSources.map((source) => [source.id, source.label]));
const destinationLabels = new Map(resultDestinations.map((destination) => [destination.id, destination.label]));

export const getSourceLabel = (id: InvocationSourceId): string => sourceLabels.get(id) ?? 'Generate';

export const isInvocationSourceAvailable = (id: InvocationSourceId): boolean =>
  invocationSources.some((source) => source.id === id && source.available);

export const getDestinationLabel = (id: ResultDestination): string => destinationLabels.get(id) ?? 'Canvas';

export const formatRoute = (route: InvocationRoute): string =>
  `${getSourceLabel(route.sourceId)} → ${getDestinationLabel(route.destination)}`;

export const defaultInvocationRoute: InvocationRoute = {
  sourceId: 'generate',
  destination: 'canvas',
  sourceLocked: false,
  destinationLocked: false,
};

const validDestinationIds = new Set(resultDestinations.map((destination) => destination.id));

const sourceWidgetIds: Partial<Record<InvocationSourceId, WidgetId>> = {
  canvas: 'canvas',
  generate: 'generate',
  upscale: 'upscale',
  video: 'video',
  workflow: 'workflow',
};

export interface InvocationRouteInput {
  generateValues: Record<string, unknown>;
  upscaleValues: Record<string, unknown>;
  videoValues: Record<string, unknown>;
  workflowValues: Record<string, unknown>;
  invocation: InvocationRoute;
  mountedWidgetIds: readonly WidgetId[];
  projectGraph: ProjectGraphState;
  projectId: string;
  /** The canvas generation frame (document space) — its area gates a canvas invoke. */
  canvasBbox: { width: number; height: number };
  /** The canvas layer stack (identity-stable) — control layers gate a canvas invoke. */
  canvasLayers: readonly CanvasLayerContract[];
}

const getMountedWidgetIds = (project: Project): WidgetId[] => {
  const mountedWidgetIds = new Set<WidgetId>();

  for (const region of Object.values(project.widgetRegions)) {
    for (const instanceId of region.instanceIds) {
      const widgetId = project.widgetInstances[instanceId]?.typeId;

      if (widgetId) {
        mountedWidgetIds.add(widgetId);
      }
    }
  }

  // A floated instance is mounted in its window, not in a region.
  for (const instanceId of Object.keys(project.floatingWidgets ?? {})) {
    const widgetId = project.widgetInstances[instanceId]?.typeId;

    if (widgetId) {
      mountedWidgetIds.add(widgetId);
    }
  }

  return Array.from(mountedWidgetIds).sort();
};

export const getInvocationRouteInput = (project: Project): InvocationRouteInput => ({
  canvasBbox: {
    height: project.canvas.document.bbox.height,
    width: project.canvas.document.bbox.width,
  },
  canvasLayers: compileContributingLayers(project.canvas.document),
  generateValues: getProjectWidgetValues(project, 'generate'),
  upscaleValues: getProjectWidgetValues(project, 'upscale'),
  videoValues: getProjectWidgetValues(project, 'video'),
  workflowValues: getProjectWidgetValues(project, 'workflow'),
  invocation: project.invocation,
  mountedWidgetIds: getMountedWidgetIds(project),
  projectGraph: project.projectGraph,
  projectId: project.id,
});

export const areInvocationRouteInputsEqual = (left: InvocationRouteInput, right: InvocationRouteInput): boolean =>
  left.projectId === right.projectId &&
  left.invocation === right.invocation &&
  left.projectGraph === right.projectGraph &&
  left.generateValues === right.generateValues &&
  left.upscaleValues === right.upscaleValues &&
  left.videoValues === right.videoValues &&
  left.workflowValues === right.workflowValues &&
  left.canvasBbox.width === right.canvasBbox.width &&
  left.canvasBbox.height === right.canvasBbox.height &&
  left.canvasLayers === right.canvasLayers &&
  areArraysEqual(left.mountedWidgetIds, right.mountedWidgetIds);

export const createInvocationRouteInputSelector = () =>
  createStableSelector(getInvocationRouteInput, areInvocationRouteInputsEqual);

const isWidgetMounted = (input: InvocationRouteInput, widgetId: WidgetId): boolean =>
  input.mountedWidgetIds.includes(widgetId);

const getGenerateSnapshotValidationReasons = (
  generateValues: Record<string, unknown>,
  models?: readonly ModelConfig[]
): string[] => {
  const values = normalizeGenerateWidgetValues(generateValues);

  if (!values || !isSupportedGenerateModel(values.model)) {
    return ['Generate needs a supported model before it can be invoked.'];
  }

  return [
    ...getGenerationValidationReasons(values.model, values),
    ...(models ? getGenerationModelAvailabilityReasons(values.model, values, models) : []),
  ];
};

export const isResultDestinationAvailable = (destination: ResultDestination): boolean =>
  validDestinationIds.has(destination);

export const resolveInvocationRoute = (
  project: Project,
  mode: InvocationMode = 'global',
  route: InvocationRoute = project.invocation,
  models?: readonly ModelConfig[]
): ResolvedInvocationRoute => resolveInvocationRouteInput(getInvocationRouteInput(project), mode, route, models);

export const resolveInvocationRouteInput = (
  input: InvocationRouteInput,
  mode: InvocationMode = 'global',
  route: InvocationRoute = input.invocation,
  models?: readonly ModelConfig[]
): ResolvedInvocationRoute => {
  const sourceId = route.sourceId;
  const destination = route.destination;
  const sourceWidgetId = sourceWidgetIds[sourceId];
  // Route-rendering surfaces subscribe to templates because readiness reads them imperatively.
  const projectGraphReadiness =
    sourceId === 'workflow'
      ? getProjectGraphReadiness(input.projectGraph, getInvocationTemplatesSnapshot(), {
          batchCount: sanitizeBatchCount(input.workflowValues.batchCount),
        })
      : null;
  const validationReasons: Array<string | ForLoopValidationReason> = [];

  if (!isInvocationSourceAvailable(sourceId)) {
    validationReasons.push(`${getSourceLabel(sourceId)} is not an available invocation source.`);
  } else if (sourceWidgetId && !isWidgetMounted(input, sourceWidgetId)) {
    validationReasons.push(`The ${getSourceLabel(sourceId)} widget is not mounted in this project.`);
  }

  if (sourceId === 'generate') {
    validationReasons.push(...getGenerateSnapshotValidationReasons(input.generateValues, models));
  }

  if (sourceId === 'upscale') {
    const values = normalizeUpscaleWidgetValues(input.upscaleValues);

    validationReasons.push(
      ...(values ? getUpscaleValidationReasons(values, models) : ['Upscale settings are incomplete.'])
    );
  }

  if (sourceId === 'video') {
    const values = normalizeVideoWidgetValues(input.videoValues);

    validationReasons.push(
      ...(values ? getVideoWidgetValidationReasons(values, models) : ['Video settings are incomplete.'])
    );
  }

  if (sourceId === 'canvas') {
    // Canvas shares Generate readiness and additionally requires a nonzero generation frame.
    validationReasons.push(...getGenerateSnapshotValidationReasons(input.generateValues, models));

    if (input.canvasBbox.width <= 0 || input.canvasBbox.height <= 0) {
      validationReasons.push('Canvas generation frame must have a positive area.');
    }

    // Wait for models before validating control layers to avoid false missing-model errors.
    if (models) {
      const values = normalizeGenerateWidgetValues(input.generateValues);
      if (values && isSupportedGenerateModel(values.model)) {
        validationReasons.push(
          ...getBlockingControlLayerIssues({
            layers: input.canvasLayers,
            mainModel: values.model,
            models,
          }).map((issue) => issue.message)
        );
      }
    }
  }

  if (projectGraphReadiness && !projectGraphReadiness.canInvoke) {
    validationReasons.push(...projectGraphReadiness.reasons);
  }

  const sourceValid = validationReasons.length === 0;
  // Canvas staging accepts images only; video routes must target Gallery or their outputs would be unreachable.
  const destinationCompatible = !(sourceId === 'video' && destination === 'canvas');
  const destinationAvailable = isResultDestinationAvailable(destination);
  const destinationValid = destinationAvailable && destinationCompatible;

  if (!destinationAvailable) {
    validationReasons.push(`${getDestinationLabel(destination)} is not an available result destination.`);
  } else if (!destinationCompatible) {
    validationReasons.push('Video results can only be sent to the Gallery. Switch the destination to Gallery.');
  }

  return {
    ...route,
    destination,
    destinationValid,
    mode,
    sourceId,
    sourceValid,
    validationMessage: validationReasons[0],
    validationReasons,
    ...(projectGraphReadiness?.batch ? { workflowBatch: projectGraphReadiness.batch } : {}),
  };
};

export const isInvocationRouteValid = (route: ResolvedInvocationRoute): boolean =>
  route.sourceValid && route.destinationValid;
