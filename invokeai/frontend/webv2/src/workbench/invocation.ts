import type { ModelConfig } from '@features/models';
import type { InvocationTemplatesSnapshot, ProjectGraphState } from '@features/workflow/contracts';
import type { CanvasLayerContract, CanvasRegionalGuidanceLayerContract } from '@workbench/canvas-engine/api';
import type {
  DynamicPromptsExpansionSource,
  InvocationMode,
  InvocationRoute,
  InvocationValidationReason,
  InvocationSourceId,
  RouteIssue,
  ResolvedInvocationRoute,
  ResultDestination,
} from '@workbench/invocationContracts';
import type { Project } from '@workbench/projectContracts';
import type { WidgetId } from '@workbench/widgetContracts';
import type { WorkspacePlacement } from '@workbench/workspace/workspaceDerivations';

import { getLayoutDraftRejectionReason } from '@features/generation/graph';
import {
  describeGenerationIssue,
  expandsDynamicPrompts,
  getEffectivePrompts,
  getGenerateInvocationIssues,
  getPromptDraftFromValues,
  isExternalTextOnlyCanvasModel,
  isGenerateModelConfig,
  isSupportedGenerateModel,
  NO_GENERATE_MODEL_ISSUE,
  normalizeGenerateSettings,
  normalizeGenerateWidgetValues,
  type ProjectPromptDraft,
  sanitizeBatchCount,
} from '@features/generation/settings';
import { getUpscaleValidationReasons, normalizeUpscaleWidgetValues, readsUpscalePrompt } from '@features/upscale';
import { getVideoWidgetValidationReasons, normalizeVideoWidgetValues } from '@features/video';
import { getProjectGraphReadiness } from '@features/workflow/graph';
import { getInvocationTemplatesSnapshot } from '@features/workflow/react';
import { localizeWorkflowReadinessReason } from '@features/workflow/utility';
import { areArraysEqual } from '@platform/state/selectors';
import {
  compileContributingLayers,
  compileDocumentLeaves,
  getSourceBounds,
  isBaseRasterContentLayer,
  lookupDocumentNode,
} from '@workbench/canvas-engine/api';
import { isCompositableRegionalGuidanceLayer } from '@workbench/canvasLayerContent';

import { describeCanvasInputIssue, getCanvasInputIssues } from './canvasGenerationInputs';
import { getWidgetTypeIdForSourceId, invocationSources, isInvocationSourceAvailable } from './invocationSources';
import { getActiveProjectGraph } from './projectWorkflows';
import { getWorkbenchPreferences } from './settings/store';
import { getProjectWidgetValues } from './widgetState';

/** Upscale's Refine prompt is the draft it shares with Generate; Model only reads no prompt. */
const getUpscalePromptDraft = (project: Project): ProjectPromptDraft | null =>
  readsUpscalePrompt(getProjectWidgetValues(project, 'upscale'))
    ? getPromptDraftFromValues(getProjectWidgetValues(project, 'generate'))
    : null;

/** Null when the route sends its prompt as typed; Canvas and Upscale expand Generate's prompt. */
export const getDynamicPromptsExpansionSource = (
  project: Project,
  sourceId: InvocationSourceId
): (DynamicPromptsExpansionSource & { sampleSeed: number }) | null => {
  if (sourceId === 'workflow') {
    return null;
  }

  const settings =
    sourceId === 'video'
      ? normalizeVideoWidgetValues(getProjectWidgetValues(project, 'video'))
      : sourceId === 'upscale'
        ? getUpscalePromptDraft(project)
        : normalizeGenerateSettings(getProjectWidgetValues(project, 'generate'));

  if (!settings) {
    return null;
  }

  const positivePrompt =
    'promptTemplate' in settings ? getEffectivePrompts(settings).positivePrompt : settings.positivePrompt;

  return expandsDynamicPrompts(positivePrompt, settings.dynamicPromptsEnabled)
    ? {
        combinatorial: settings.dynamicPromptsCombinatorial,
        maxPrompts: settings.dynamicPromptsMaxPrompts,
        positivePrompt,
        sampleSeed: settings.dynamicPromptsSampleSeed,
      }
    : null;
};

export const isSameDynamicPromptsExpansionSource = (
  prepared: DynamicPromptsExpansionSource | null,
  current: DynamicPromptsExpansionSource | null
): boolean =>
  prepared === null || current === null
    ? prepared === current
    : prepared.positivePrompt === current.positivePrompt &&
      prepared.combinatorial === current.combinatorial &&
      prepared.maxPrompts === current.maxPrompts;

export interface ResultDestinationMeta {
  id: ResultDestination;
  label: string;
}

export const resultDestinations: ResultDestinationMeta[] = [
  { id: 'canvas', label: 'Canvas' },
  { id: 'gallery', label: 'Gallery' },
];

const sourceLabels = new Map(invocationSources.map((source) => [source.id, source.label]));
const destinationLabels = new Map(resultDestinations.map((destination) => [destination.id, destination.label]));

export const getSourceLabel = (id: InvocationSourceId): string => sourceLabels.get(id) ?? 'Generate';

export const getDestinationLabel = (id: ResultDestination): string => destinationLabels.get(id) ?? 'Canvas';

/** Locking routing pins both halves; either one locked reads as a locked route. */
export const isRoutingLocked = (route: InvocationRoute): boolean => route.sourceLocked || route.destinationLocked;

export const formatRoute = (route: InvocationRoute): string =>
  `${getSourceLabel(route.sourceId)} → ${getDestinationLabel(route.destination)}`;

export const defaultInvocationRoute: InvocationRoute = {
  sourceId: 'generate',
  destination: 'canvas',
  sourceLocked: false,
  destinationLocked: false,
};

const validDestinationIds = new Set(resultDestinations.map((destination) => destination.id));

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
  canvasLayers: readonly CanvasLayerContract[];
  /** Measured only with a text-only hosted model, which a raster in frame blocks; false otherwise. */
  canvasRasterInFrame: boolean;
  /** Measured whatever the model: which bases read regions is capability data that can arrive later. */
  canvasRegionalLayersInFrame: readonly CanvasRegionalGuidanceLayerContract[];
}

/** The widget types the layout places, windows included: a source runs only while its widget is placed. */
const getMountedWidgetIds = (layout: WorkspacePlacement): WidgetId[] => {
  const mountedWidgetIds = new Set<WidgetId>();

  for (const instanceId of [
    ...Object.values(layout.widgetRegions).flatMap((region) => region.instanceIds),
    ...Object.keys(layout.floatingWidgets ?? {}),
  ]) {
    const widgetId = layout.widgetInstances[instanceId]?.typeId;

    if (widgetId) {
      mountedWidgetIds.add(widgetId);
    }
  }

  return Array.from(mountedWidgetIds).sort();
};

type FrameRect = Project['canvas']['document']['bbox'];

const overlapsFrame = (bounds: FrameRect, frame: FrameRect): boolean =>
  bounds.x < frame.x + frame.width &&
  frame.x < bounds.x + bounds.width &&
  bounds.y < frame.y + frame.height &&
  frame.y < bounds.y + bounds.height;

/** Errs toward not blocking: text and zero-opacity layers are skipped; transparent pixels count as content. */
const hasRasterContentInFrame = (document: Project['canvas']['document']): boolean => {
  const { bbox } = document;

  if (bbox.width <= 0 || bbox.height <= 0) {
    return false;
  }

  return compileDocumentLeaves(document).some(({ contributionEnabled, layer, parentIds }) => {
    if (
      !contributionEnabled ||
      !isBaseRasterContentLayer(layer) ||
      layer.source.type === 'text' ||
      layer.opacity <= 0 ||
      parentIds.some((id) => {
        const group = lookupDocumentNode(document, id);
        return group?.type === 'group' && (group.opacity ?? 1) <= 0;
      })
    ) {
      return false;
    }

    return overlapsFrame(getSourceBounds(layer, document), bbox);
  });
};

const NO_REGIONAL_LAYERS: readonly CanvasRegionalGuidanceLayerContract[] = [];

/** A mask reaching the frame with transparent pixels only still counts here, though the compiler leaves it out. */
const getRegionalLayersInFrame = (
  document: Project['canvas']['document'],
  layers: readonly CanvasLayerContract[]
): readonly CanvasRegionalGuidanceLayerContract[] => {
  const { bbox } = document;

  if (bbox.width <= 0 || bbox.height <= 0) {
    return NO_REGIONAL_LAYERS;
  }

  const inFrame = layers.filter(
    (layer): layer is CanvasRegionalGuidanceLayerContract =>
      isCompositableRegionalGuidanceLayer(layer) && overlapsFrame(getSourceBounds(layer, document), bbox)
  );

  return inFrame.length > 0 ? inFrame : NO_REGIONAL_LAYERS;
};

/** `layout` is what the project shows: its workspace's structure seen through its view. */
export const getInvocationRouteInput = (project: Project, layout: WorkspacePlacement): InvocationRouteInput => {
  const generateValues = getProjectWidgetValues(project, 'generate');
  // Measured whatever source is selected: a graph preview resolves the Canvas route from this input.
  const measuresRaster =
    isGenerateModelConfig(generateValues.model) && isExternalTextOnlyCanvasModel(generateValues.model);
  const canvasLayers = compileContributingLayers(project.canvas.document);

  return {
    canvasBbox: {
      height: project.canvas.document.bbox.height,
      width: project.canvas.document.bbox.width,
    },
    canvasLayers,
    canvasRegionalLayersInFrame: getRegionalLayersInFrame(project.canvas.document, canvasLayers),
    canvasRasterInFrame: measuresRaster && hasRasterContentInFrame(project.canvas.document),
    generateValues,
    upscaleValues: getProjectWidgetValues(project, 'upscale'),
    videoValues: getProjectWidgetValues(project, 'video'),
    workflowValues: getProjectWidgetValues(project, 'workflow'),
    invocation: project.invocation,
    mountedWidgetIds: getMountedWidgetIds(layout),
    projectGraph: getActiveProjectGraph(project),
    projectId: project.id,
  };
};

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
  left.canvasRasterInFrame === right.canvasRasterInFrame &&
  areArraysEqual(left.canvasRegionalLayersInFrame, right.canvasRegionalLayersInFrame) &&
  areArraysEqual(left.mountedWidgetIds, right.mountedWidgetIds);

export const createInvocationRouteInputSelector = (): ((
  project: Project,
  layout: WorkspacePlacement
) => InvocationRouteInput) => {
  let last: { input: InvocationRouteInput; layout: WorkspacePlacement; project: Project } | undefined;

  return (project, layout) => {
    if (last?.project !== project || last.layout !== layout) {
      const next = getInvocationRouteInput(project, layout);

      last = { input: last && areInvocationRouteInputsEqual(last.input, next) ? last.input : next, layout, project };
    }

    return last.input;
  };
};

const isWidgetMounted = (input: InvocationRouteInput, widgetId: WidgetId): boolean =>
  input.mountedWidgetIds.includes(widgetId);

const getGenerateSnapshotValidationReasons = (
  generateValues: Record<string, unknown>,
  models?: readonly ModelConfig[],
  canvasBbox?: InvocationRouteInput['canvasBbox']
): InvocationValidationReason[] => {
  const values = normalizeGenerateWidgetValues(generateValues);

  if (!values || !isSupportedGenerateModel(values.model)) {
    return [{ generationIssue: NO_GENERATE_MODEL_ISSUE }];
  }

  const layoutDraftReason = getLayoutDraftRejectionReason({
    base: values.model.base,
    positivePrompt: values.positivePrompt,
    promptTemplate: values.promptTemplate,
  });

  // Generate itself stays text to image; the Canvas source checks the provider at the size its frame asks for.
  return [
    ...getGenerateInvocationIssues(values, {
      ...(canvasBbox ? { canvasFrame: canvasBbox } : {}),
      ...(models ? { models } : {}),
    }).map((generationIssue) => ({ generationIssue })),
    ...(layoutDraftReason === null ? [] : [layoutDraftReason]),
  ];
};

export const isResultDestinationAvailable = (destination: ResultDestination): boolean =>
  validDestinationIds.has(destination);

export const resolveInvocationRoute = (
  project: Project,
  layout: WorkspacePlacement,
  mode: InvocationMode = 'global',
  route: InvocationRoute = project.invocation,
  models?: readonly ModelConfig[],
  workflowDocument?: ProjectGraphState,
  templates?: Readonly<InvocationTemplatesSnapshot>
): ResolvedInvocationRoute =>
  resolveInvocationRouteInput(
    workflowDocument
      ? { ...getInvocationRouteInput(project, layout), projectGraph: workflowDocument }
      : getInvocationRouteInput(project, layout),
    mode,
    route,
    models,
    templates
  );

export const resolveInvocationRouteInput = (
  input: InvocationRouteInput,
  mode: InvocationMode = 'global',
  route: InvocationRoute = input.invocation,
  models?: readonly ModelConfig[],
  templates?: Readonly<InvocationTemplatesSnapshot>
): ResolvedInvocationRoute => {
  const sourceId = route.sourceId;
  const destination = route.destination;
  const sourceWidgetId = getWidgetTypeIdForSourceId(sourceId);
  // Route-rendering surfaces subscribe to templates and preferences because readiness reads them imperatively.
  const projectGraphReadiness =
    sourceId === 'workflow'
      ? getProjectGraphReadiness(input.projectGraph, templates ?? getInvocationTemplatesSnapshot(), {
          batchCount: sanitizeBatchCount(input.workflowValues.batchCount),
        })
      : null;
  const validationReasons: InvocationValidationReason[] = [];

  if (!isInvocationSourceAvailable(sourceId)) {
    validationReasons.push(`${getSourceLabel(sourceId)} is not an available invocation source.`);
  } else if (!isWidgetMounted(input, sourceWidgetId)) {
    validationReasons.push({ routeIssue: { kind: 'sourceNotMounted', sourceId } });
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
    validationReasons.push(...getGenerateSnapshotValidationReasons(input.generateValues, models, input.canvasBbox));

    if (input.canvasBbox.width <= 0 || input.canvasBbox.height <= 0) {
      validationReasons.push({ routeIssue: { kind: 'emptyCanvasFrame' } });
    }

    const values = normalizeGenerateWidgetValues(input.generateValues);
    if (values && (values.model.type === 'external_image_generator' || isSupportedGenerateModel(values.model))) {
      validationReasons.push(
        ...getCanvasInputIssues({
          cfgScale: values.cfgScale,
          controlModelMemory: getWorkbenchPreferences().controlModelMemory,
          layers: input.canvasLayers,
          model: values.model,
          models,
          prompt: getEffectivePrompts(values).positivePrompt,
          rasterInFrame: input.canvasRasterInFrame,
          regionalLayersInFrame: input.canvasRegionalLayersInFrame,
        }).map((canvasInputIssue) => ({ canvasInputIssue }))
      );
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
    validationReasons.push({ routeIssue: { kind: 'videoNeedsGallery' } });
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

export const ROUTE_LABEL_KEYS: Record<InvocationSourceId | ResultDestination, string> = {
  canvas: 'widgets.labels.canvas',
  gallery: 'widgets.labels.gallery',
  generate: 'widgets.labels.generate',
  upscale: 'widgets.labels.upscale',
  video: 'widgets.labels.video',
  workflow: 'widgets.labels.workflow',
};

const describeRouteIssue = (issue: RouteIssue, t: (key: string, options?: Record<string, unknown>) => string): string =>
  t(
    `topbar.invoke.routeIssues.${issue.kind}`,
    issue.kind === 'sourceNotMounted' ? { source: t(ROUTE_LABEL_KEYS[issue.sourceId]) } : undefined
  );

export const localizeInvocationValidationReason = (
  reason: InvocationValidationReason,
  t: (key: string, options?: Record<string, unknown>) => string
): string =>
  typeof reason === 'object' && 'canvasInputIssue' in reason
    ? describeCanvasInputIssue(t, reason.canvasInputIssue)
    : typeof reason === 'object' && 'generationIssue' in reason
      ? describeGenerationIssue(t, reason.generationIssue)
      : typeof reason === 'object' && 'routeIssue' in reason
        ? describeRouteIssue(reason.routeIssue, t)
        : localizeWorkflowReadinessReason(reason, t);
