import type { ModelConfig } from '@features/models';
import type { ProjectGraphState } from '@features/workflow/contracts';
import type { CanvasLayerContract, CanvasRegionalGuidanceLayerContract } from '@workbench/canvas-engine/api';
import type {
  InvocationMode,
  InvocationRoute,
  InvocationValidationReason,
  InvocationSourceId,
  ResolvedInvocationRoute,
  ResultDestination,
} from '@workbench/invocationContracts';
import type { Project } from '@workbench/projectContracts';
import type { WidgetId } from '@workbench/widgetContracts';

import { getLayoutPromptRejectionReason } from '@features/generation/graph';
import {
  describeExternalCanvasRefusal,
  getEffectivePrompts,
  getGenerationModelAvailabilityReasons,
  getGenerationValidationReasons,
  isExternalTextOnlyCanvasModel,
  isGenerateModelConfig,
  isSupportedGenerateModel,
  normalizeGenerateWidgetValues,
  resolveExternalCanvasRequestSize,
  sanitizeBatchCount,
} from '@features/generation/settings';
import { getUpscaleValidationReasons, normalizeUpscaleWidgetValues } from '@features/upscale';
import { getVideoWidgetValidationReasons, normalizeVideoWidgetValues } from '@features/video';
import { getProjectGraphReadiness } from '@features/workflow/graph';
import { getInvocationTemplatesSnapshot } from '@features/workflow/react';
import { localizeForLoopValidationReason } from '@features/workflow/utility';
import { areArraysEqual, createStableSelector } from '@platform/state/selectors';
import {
  compileContributingLayers,
  compileDocumentLeaves,
  getSourceBounds,
  isBaseRasterContentLayer,
  lookupDocumentNode,
} from '@workbench/canvas-engine/api';
import { isCompositableControlLayer, isCompositableRegionalGuidanceLayer } from '@workbench/canvasLayerContent';

import {
  createControlLayerCheckContext,
  describeControlLayerIssue,
  getBlockingControlLayerIssues,
} from './controlLayerChecks';
import { getActiveProjectGraph } from './projectWorkflows';
import { getWorkbenchPreferences } from './settings/store';
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
  /**
   * Whether a visible raster layer's bounds overlap the frame, so a text-only hosted model is blocked before invoke
   * without reading pixels. Measured only with such a model, whatever the selected source; false otherwise.
   */
  canvasRasterInFrame: boolean;
  /**
   * The enabled, painted regional guidance layers whose mask bounds overlap the frame (identity-stable), which a
   * layout base places around the global prompt. Measured whatever the model: which bases read them is capability
   * data that can arrive after the project, so readiness decides that when it runs.
   */
  canvasRegionalLayersInFrame: readonly CanvasRegionalGuidanceLayerContract[];
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

type FrameRect = Project['canvas']['document']['bbox'];

/** Whether document-space bounds overlap the frame's interior; touching its edge does not. */
const overlapsFrame = (bounds: FrameRect, frame: FrameRect): boolean =>
  bounds.x < frame.x + frame.width &&
  frame.x < bounds.x + bounds.width &&
  bounds.y < frame.y + frame.height &&
  frame.y < bounds.y + bounds.height;

/**
 * Whether a raster layer the generation composite draws overlaps the frame's interior with its own bounds. Invoke
 * decides from the composite's pixels, so this errs toward not blocking where geometry is uncertain: text is skipped
 * (readiness only has an estimate of its extent) and so is anything at zero opacity, its own or a group's. The one
 * remaining approximation is transparent pixels inside a layer's bounds, which this treats as content.
 */
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

/**
 * The compositable regional guidance layers whose mask bounds, offset and transform included, overlap the frame's
 * interior: the regions a layout base can place. The one remaining approximation is a mask bitmap that reaches into
 * the frame with transparent pixels only (a legacy document-sized mask, say), which still counts here although the
 * canvas compiler, reading the uploaded mask's pixels, leaves it out.
 */
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

export const getInvocationRouteInput = (project: Project): InvocationRouteInput => {
  const generateValues = getProjectWidgetValues(project, 'generate');
  // Canvas facts are measured whatever source is selected: a graph preview resolves the Canvas route from this same
  // input while another source is. This runs on every project change, over leaves the document index caches; raster
  // layers are measured only for the model they gate.
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
    mountedWidgetIds: getMountedWidgetIds(project),
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

export const createInvocationRouteInputSelector = () =>
  createStableSelector(getInvocationRouteInput, areInvocationRouteInputsEqual);

const isWidgetMounted = (input: InvocationRouteInput, widgetId: WidgetId): boolean =>
  input.mountedWidgetIds.includes(widgetId);

const getGenerateSnapshotValidationReasons = (
  generateValues: Record<string, unknown>,
  models?: readonly ModelConfig[],
  canvasBbox?: InvocationRouteInput['canvasBbox']
): string[] => {
  const values = normalizeGenerateWidgetValues(generateValues);

  if (!values || !isSupportedGenerateModel(values.model)) {
    return ['Generate needs a supported model before it can be invoked.'];
  }

  // Canvas discovers the mode from pixel coverage during preparation and asks a provider for its size nearest the
  // bbox, so readiness validates that size and permits an advertised whole-image mode; the compiler validates the
  // captured one. Generate itself remains strictly text-to-image.
  const providerModes =
    canvasBbox !== undefined && values.model.type === 'external_image_generator'
      ? (values.model.capabilities?.modes ?? ['txt2img'])
      : null;
  const mode = providerModes !== null && !providerModes.includes('txt2img') ? 'img2img' : 'txt2img';

  return [
    ...getGenerationValidationReasons(
      values.model,
      providerModes !== null && canvasBbox
        ? { ...values, ...resolveExternalCanvasRequestSize(values.model, canvasBbox) }
        : values,
      mode
    ),
    ...(models ? getGenerationModelAvailabilityReasons(values.model, values, models) : []),
  ];
};

export const isResultDestinationAvailable = (destination: ResultDestination): boolean =>
  validDestinationIds.has(destination);

export const resolveInvocationRoute = (
  project: Project,
  mode: InvocationMode = 'global',
  route: InvocationRoute = project.invocation,
  models?: readonly ModelConfig[],
  /** Validate a specific project workflow (a captured submission target) instead of the active one. */
  workflowDocument?: ProjectGraphState
): ResolvedInvocationRoute =>
  resolveInvocationRouteInput(
    workflowDocument
      ? { ...getInvocationRouteInput(project), projectGraph: workflowDocument }
      : getInvocationRouteInput(project),
    mode,
    route,
    models
  );

export const resolveInvocationRouteInput = (
  input: InvocationRouteInput,
  mode: InvocationMode = 'global',
  route: InvocationRoute = input.invocation,
  models?: readonly ModelConfig[]
): ResolvedInvocationRoute => {
  const sourceId = route.sourceId;
  const destination = route.destination;
  const sourceWidgetId = sourceWidgetIds[sourceId];
  // Route-rendering surfaces subscribe to templates and preferences because readiness reads them imperatively.
  const projectGraphReadiness =
    sourceId === 'workflow'
      ? getProjectGraphReadiness(input.projectGraph, getInvocationTemplatesSnapshot(), {
          batchCount: sanitizeBatchCount(input.workflowValues.batchCount),
        })
      : null;
  const validationReasons: InvocationValidationReason[] = [];

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
    validationReasons.push(...getGenerateSnapshotValidationReasons(input.generateValues, models, input.canvasBbox));

    if (input.canvasBbox.width <= 0 || input.canvasBbox.height <= 0) {
      validationReasons.push('Canvas generation frame must have a positive area.');
    }

    const values = normalizeGenerateWidgetValues(input.generateValues);
    if (values?.model.type === 'external_image_generator') {
      const { model } = values;

      if (input.canvasRasterInFrame && isExternalTextOnlyCanvasModel(model)) {
        validationReasons.push(describeExternalCanvasRefusal(model, { kind: 'raster_content' }));
      }
      for (const layer of input.canvasLayers) {
        if (isCompositableControlLayer(layer)) {
          validationReasons.push(
            describeExternalCanvasRefusal(model, { kind: 'control_layer', layerName: layer.name })
          );
        } else if (isCompositableRegionalGuidanceLayer(layer)) {
          validationReasons.push(
            describeExternalCanvasRefusal(model, { kind: 'regional_guidance', layerName: layer.name })
          );
        }
      }
    } else if (values && isSupportedGenerateModel(values.model)) {
      // Wait for models before validating local adapters to avoid false missing-model errors.
      if (models) {
        validationReasons.push(
          ...getBlockingControlLayerIssues({
            context: createControlLayerCheckContext(values.model.base, getWorkbenchPreferences().controlModelMemory),
            layers: input.canvasLayers,
            mainModel: values.model,
            models,
          }).map((controlLayerIssue) => ({ controlLayerIssue }))
        );
      }
      // Before anything uploads, so only the regions whose mask bounds reach into the frame count.
      const layoutPromptReason = getLayoutPromptRejectionReason({
        base: values.model.base,
        prompt: getEffectivePrompts(values).positivePrompt,
        regions: input.canvasRegionalLayersInFrame,
      });
      if (layoutPromptReason) {
        validationReasons.push(layoutPromptReason);
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

/** Words any route validation reason for display; control-layer issues come from the locale. */
export const localizeInvocationValidationReason = (
  reason: InvocationValidationReason,
  t: (key: string, options?: Record<string, unknown>) => string
): string =>
  typeof reason === 'object' && 'controlLayerIssue' in reason
    ? describeControlLayerIssue(t, reason.controlLayerIssue)
    : localizeForLoopValidationReason(reason, t);
