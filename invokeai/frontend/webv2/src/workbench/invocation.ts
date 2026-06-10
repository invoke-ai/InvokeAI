import type {
  InvocationMode,
  InvocationRoute,
  InvocationSourceId,
  Project,
  ResolvedInvocationRoute,
  ResultDestination,
  WidgetId,
} from './types';

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
  { id: 'project-graph', label: 'Workflow', available: false },
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
  Object.values(project.widgetRegions).some((region) => region.enabledWidgetIds.includes(widgetId));

const hasGenerateSnapshotInputs = (project: Project): boolean => {
  const values = project.widgetStates.generate.values;
  const model = values.model;
  const hasFiniteNumber = (key: string) => typeof values[key] === 'number' && Number.isFinite(values[key]);

  return (
    typeof values.modelKey === 'string' &&
    typeof values.positivePrompt === 'string' &&
    typeof values.negativePrompt === 'string' &&
    hasFiniteNumber('width') &&
    hasFiniteNumber('height') &&
    hasFiniteNumber('steps') &&
    hasFiniteNumber('cfgScale') &&
    hasFiniteNumber('cfgRescaleMultiplier') &&
    typeof values.scheduler === 'string' &&
    hasFiniteNumber('seed') &&
    typeof values.shouldRandomizeSeed === 'boolean' &&
    Boolean(model && typeof model === 'object' && (model as Record<string, unknown>).type === 'main')
  );
};

export const isResultDestinationAvailable = (destination: ResultDestination): boolean =>
  validDestinationIds.has(destination);

export const resolveInvocationRoute = (
  project: Project,
  mode: InvocationMode = 'global',
  route: InvocationRoute = project.invocation
): ResolvedInvocationRoute => {
  const sourceId = route.sourceId;
  const destination = route.destination;
  const sourceWidgetId = sourceWidgetIds[sourceId];
  const sourceValid =
    isInvocationSourceAvailable(sourceId) &&
    (!sourceWidgetId || isWidgetMounted(project, sourceWidgetId)) &&
    (sourceId !== 'generate' || hasGenerateSnapshotInputs(project));
  const destinationValid = isResultDestinationAvailable(destination);
  const validationMessage = !sourceValid
    ? sourceId === 'generate'
      ? 'Generate needs a supported model before it can be invoked.'
      : `${getSourceLabel(sourceId)} is not an available invocation source.`
    : !destinationValid
      ? `${getDestinationLabel(destination)} is not an available result destination.`
      : undefined;

  return {
    ...route,
    destination,
    destinationValid,
    mode,
    sourceId,
    sourceValid,
    validationMessage,
  };
};

export const isInvocationRouteValid = (route: ResolvedInvocationRoute): boolean =>
  route.sourceValid && route.destinationValid;
