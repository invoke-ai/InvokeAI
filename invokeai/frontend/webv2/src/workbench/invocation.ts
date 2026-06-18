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

import {
  getGenerationModelAvailabilityReasons,
  getGenerationValidationReasons,
  isSupportedGenerateModel,
} from './generation/baseGenerationPolicies';
import { normalizeGenerateWidgetValues } from './generation/settings';
import { getProjectWidgetValues } from './widgetState';
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
  { id: 'project-graph', label: 'Workflow', available: true },
  { id: 'upscale', label: 'Upscale', available: false },
  { id: 'canvas-fill', label: 'Canvas Fill', available: false },
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
  'canvas-fill': 'canvas',
  generate: 'generate',
  'project-graph': 'workflow',
};

const isWidgetMounted = (project: Project, widgetId: WidgetId): boolean =>
  Object.values(project.widgetRegions).some((region) =>
    region.instanceIds.some((instanceId) => project.widgetInstances[instanceId]?.typeId === widgetId)
  );

const getGenerateSnapshotValidationReasons = (project: Project, models?: readonly ModelConfig[]): string[] => {
  const values = normalizeGenerateWidgetValues(getProjectWidgetValues(project, 'generate'));

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
): ResolvedInvocationRoute => {
  const sourceId = route.sourceId;
  const destination = route.destination;
  const sourceWidgetId = sourceWidgetIds[sourceId];
  // The project graph validates against its compiled readiness; templates are
  // read imperatively, and surfaces that render the route subscribe to the
  // templates store so the result stays live.
  const projectGraphReadiness =
    sourceId === 'project-graph'
      ? getProjectGraphReadiness(project.projectGraph, getInvocationTemplatesSnapshot())
      : null;
  const validationReasons: string[] = [];

  if (!isInvocationSourceAvailable(sourceId)) {
    validationReasons.push(`${getSourceLabel(sourceId)} is not an available invocation source.`);
  } else if (sourceWidgetId && !isWidgetMounted(project, sourceWidgetId)) {
    validationReasons.push(`The ${getSourceLabel(sourceId)} widget is not mounted in this project.`);
  }

  if (sourceId === 'generate') {
    validationReasons.push(...getGenerateSnapshotValidationReasons(project, models));
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
