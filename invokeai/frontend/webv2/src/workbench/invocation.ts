import type { ModelConfig } from './models/types';
import type {
  InvocationMode,
  InvocationRoute,
  InvocationSourceId,
  Project,
  ResolvedInvocationRoute,
  ResultDestination,
  WidgetId,
} from './types';
import type { ProjectGraphState } from './workflows/types';

import {
  getGenerationModelAvailabilityReasons,
  getGenerationValidationReasons,
  isSupportedGenerateModel,
} from './generation/baseGenerationPolicies';
import { normalizeGenerateWidgetValues } from './generation/settings';
import { getUpscaleValidationReasons, normalizeUpscaleWidgetValues } from './upscale/settings';
import { getProjectWidgetValues } from './widgetState';
import { areArraysEqual, createStableSelector } from './workbenchSelectors';
import { getProjectGraphReadiness } from './workflows/buildGraph';
import { getInvocationTemplatesSnapshot } from './workflows/templates';

/**
 * Static metadata for the Invocation Controller surfaces.
 *
 * MVP destinations are limited to Canvas and Gallery per the spec. Sources are
 * the first-party graph-bearing surfaces; only `generate` is wired in Phase 1,
 * the rest are declared so the source menu reads as a real (if partly inert)
 * placeholder for later phases.
 */
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
  workflow: 'workflow',
};

export interface InvocationRouteInput {
  generateValues: Record<string, unknown>;
  upscaleValues: Record<string, unknown>;
  invocation: InvocationRoute;
  mountedWidgetIds: readonly WidgetId[];
  projectGraph: ProjectGraphState;
  projectId: string;
  /** The canvas generation frame (document space) — its area gates a canvas invoke. */
  canvasBbox: { width: number; height: number };
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

  return Array.from(mountedWidgetIds).sort();
};

export const getInvocationRouteInput = (project: Project): InvocationRouteInput => ({
  canvasBbox: {
    height: project.canvas.document.bbox.height,
    width: project.canvas.document.bbox.width,
  },
  generateValues: getProjectWidgetValues(project, 'generate'),
  upscaleValues: getProjectWidgetValues(project, 'upscale'),
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
  left.canvasBbox.width === right.canvasBbox.width &&
  left.canvasBbox.height === right.canvasBbox.height &&
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
  // The project graph validates against its compiled readiness; templates are
  // read imperatively, and surfaces that render the route subscribe to the
  // templates store so the result stays live.
  const projectGraphReadiness =
    sourceId === 'workflow' ? getProjectGraphReadiness(input.projectGraph, getInvocationTemplatesSnapshot()) : null;
  const validationReasons: string[] = [];

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

  if (sourceId === 'canvas') {
    // Canvas shares the generate model/prompt/steps, so reuse those reasons, then
    // require a non-degenerate generation frame (a zero-area bbox has nothing to
    // generate into and the graph compiler would reject it anyway).
    validationReasons.push(...getGenerateSnapshotValidationReasons(input.generateValues, models));

    if (input.canvasBbox.width <= 0 || input.canvasBbox.height <= 0) {
      validationReasons.push('Canvas generation frame must have a positive area.');
    }
  }

  if (projectGraphReadiness && !projectGraphReadiness.canInvoke) {
    validationReasons.push(...projectGraphReadiness.reasons);
  }

  const sourceValid = validationReasons.length === 0;
  const destinationValid = isResultDestinationAvailable(destination);

  if (!destinationValid) {
    validationReasons.push(`${getDestinationLabel(destination)} is not an available result destination.`);
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
  };
};

export const isInvocationRouteValid = (route: ResolvedInvocationRoute): boolean =>
  route.sourceValid && route.destinationValid;
