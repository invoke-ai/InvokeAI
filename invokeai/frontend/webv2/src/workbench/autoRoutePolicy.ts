import type { ProjectGraphAction } from '@features/workflow/utility';
import type { InvocationRoute, InvocationSourceId, ResultDestination } from '@workbench/invocationContracts';

import type { CanvasProjectMutation } from './canvasProjectMutations';

import { isInvocationSourceAvailable } from './invocation';

/** Programmatic dispatches never auto-switch the Invoke route; absent means user. */
export type WorkbenchActionOrigin = 'user' | 'system';

/** Destination each surface maps to when the source auto-switches. */
export const autoSwitchDestinations: Record<InvocationSourceId, ResultDestination> = {
  canvas: 'canvas',
  generate: 'gallery',
  upscale: 'gallery',
  workflow: 'gallery',
};

/**
 * Route after a high-confidence edit on `sourceId`; identity-preserving when
 * nothing changes. The destination remaps only on an actual source transition,
 * so repeated edits on the same surface never re-force a manually chosen
 * destination, and locks always win.
 */
export const getRouteAfterHighConfidenceEdit = (
  invocation: InvocationRoute,
  sourceId: InvocationSourceId
): InvocationRoute => {
  if (invocation.sourceLocked || invocation.sourceId === sourceId || !isInvocationSourceAvailable(sourceId)) {
    return invocation;
  }

  const destination = invocation.destinationLocked ? invocation.destination : autoSwitchDestinations[sourceId];

  return { ...invocation, destination, sourceId };
};

export const getChangedValueKeys = (previous: Record<string, unknown>, patch: Record<string, unknown>): string[] =>
  Object.keys(patch).filter((key) => !Object.is(previous[key], patch[key]));

const GENERATE_UI_NOISE_KEYS: ReadonlySet<string> = new Set([
  'aspectRatioIsLocked',
  'batchCount',
  'negativePromptHeightPx',
  'positivePromptHeightPx',
]);

const UPSCALE_UI_NOISE_KEYS: ReadonlySet<string> = new Set([
  'batchCount',
  'negativePromptHeightPx',
  'positivePromptHeightPx',
]);

const hasIntentBearingKey = (changedKeys: readonly string[], noiseKeys: ReadonlySet<string>): boolean =>
  changedKeys.some((key) => !noiseKeys.has(key));

export const isHighConfidenceGenerateEdit = (changedKeys: readonly string[]): boolean =>
  hasIntentBearingKey(changedKeys, GENERATE_UI_NOISE_KEYS);

export const isHighConfidenceUpscaleEdit = (changedKeys: readonly string[]): boolean =>
  hasIntentBearingKey(changedKeys, UPSCALE_UI_NOISE_KEYS);

const GRAPH_EDIT_CONFIDENCE = {
  addEdge: true,
  addFormElement: false,
  addGraphElements: true,
  addNode: true,
  addNodeAndEdge: true,
  exposeField: false,
  moveFormElement: false,
  moveFormElementTo: false,
  removeEdges: true,
  removeFormElement: false,
  removeNodes: true,
  setContainerLayout: false,
  setFieldDescription: false,
  setFieldLabel: false,
  setFieldValue: true,
  setFormElementContent: false,
  setMetadata: false,
  setNodeFieldShowDescription: false,
  setNodeIsIntermediate: false,
  setNodeIsOpen: false,
  setNodeLabel: false,
  setNodeNotes: false,
  setNodePosition: false,
  setNodeUseCache: false,
  unexposeField: false,
} satisfies Record<ProjectGraphAction['type'], boolean>;

export const isHighConfidenceGraphEdit = (action: ProjectGraphAction): boolean => GRAPH_EDIT_CONFIDENCE[action.type];

type CanvasEditConfidence = 'conditional' | 'high' | 'none';

const CANVAS_EDIT_CONFIDENCE = {
  addCanvasLayer: 'high',
  applyCanvasLayerStackMutation: 'conditional',
  clearCanvasStaging: 'none',
  commitStagedImage: 'none',
  convertCanvasLayer: 'high',
  cycleStagedImage: 'none',
  deleteCanvasSnapshot: 'none',
  discardAllStagedImages: 'none',
  discardSelectedStagedImage: 'none',
  duplicateCanvasLayer: 'high',
  mergeCanvasLayersDown: 'high',
  removeCanvasLayers: 'high',
  reorderCanvasLayers: 'high',
  replaceCanvasDocument: 'none',
  replaceCanvasLayer: 'high',
  resizeCanvasDocument: 'high',
  restoreCanvasSnapshot: 'none',
  rollbackStagedImageCommit: 'none',
  saveCanvasSnapshot: 'none',
  setCanvasBbox: 'high',
  setCanvasLayersEnabled: 'conditional',
  setCanvasSelectedLayer: 'none',
  setCanvasStagingAutoSwitch: 'none',
  setStagedImageIndex: 'none',
  toggleCanvasStagingThumbnailsVisibility: 'none',
  toggleCanvasStagingVisibility: 'none',
  updateCanvasLayer: 'conditional',
  updateCanvasLayerConfig: 'high',
  updateCanvasLayerSource: 'high',
} satisfies Record<CanvasProjectMutation['type'], CanvasEditConfidence>;

const CONTENT_BEARING_LAYER_PATCH_KEYS = ['blendMode', 'isEnabled', 'opacity', 'transform'] as const;

export const isHighConfidenceCanvasEdit = (mutation: CanvasProjectMutation): boolean => {
  const confidence = CANVAS_EDIT_CONFIDENCE[mutation.type];

  if (confidence === 'high') {
    return true;
  }
  if (confidence === 'none') {
    return false;
  }

  if (mutation.type === 'applyCanvasLayerStackMutation') {
    return (
      (mutation.add?.layers.length ?? 0) > 0 ||
      (mutation.removeIds?.length ?? 0) > 0 ||
      mutation.enabledUpdates.length > 0
    );
  }
  if (mutation.type === 'setCanvasLayersEnabled') {
    return mutation.updates.length > 0;
  }
  if (mutation.type === 'updateCanvasLayer') {
    return CONTENT_BEARING_LAYER_PATCH_KEYS.some((key) => key in mutation.patch);
  }

  return false;
};

export type CanvasEditIntent = { kind: 'paint' } | { kind: 'mutation'; mutation: CanvasProjectMutation };

export const isHighConfidenceCanvasEditIntent = (intent: CanvasEditIntent): boolean =>
  intent.kind === 'paint' || isHighConfidenceCanvasEdit(intent.mutation);
