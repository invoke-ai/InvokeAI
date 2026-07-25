import type {
  CanvasAdjustmentsContract,
  CanvasControlAdapterContract,
  CanvasControlLayerContract,
  CanvasDocumentContractV2,
  CanvasLayerBaseContract,
  CanvasLayerContract,
  CanvasLayerSourceContract,
  CanvasMaskContract,
  CanvasRasterLayerContractV2,
  CanvasRegionalGuidanceLayerContract,
  CanvasStagingAreaContractV2,
  CanvasStateContractV2,
} from '@workbench/canvas-engine/api';
import type { ProjectEvent, Project } from '@workbench/projectContracts';

import { normalizeCanvasDocumentControlAdapters } from './canvasMigration';
import {
  getCanvasStagingCandidateFingerprint,
  getCanvasStagingSlotCount,
  getCanvasStagingSlots,
} from './canvasStagingView';

export type CanvasLayerBasePatch = Partial<
  Pick<CanvasLayerBaseContract, 'name' | 'isEnabled' | 'isLocked' | 'opacity' | 'blendMode'>
> & { transform?: Partial<CanvasLayerBaseContract['transform']> };

export type CanvasLayerConfigPatch =
  | {
      layerType: 'raster';
      adjustments?: CanvasAdjustmentsContract;
      isTransparencyLocked?: boolean;
      filter?: CanvasRasterLayerContractV2['filter'];
    }
  | {
      layerType: 'control';
      adapter?: Partial<CanvasControlAdapterContract>;
      withTransparencyEffect?: boolean;
      filter?: CanvasControlLayerContract['filter'];
    }
  | {
      layerType: 'regional_guidance';
      mask?: Partial<CanvasMaskContract>;
      positivePrompt?: string | null;
      negativePrompt?: string | null;
      autoNegative?: boolean;
      referenceImages?: CanvasRegionalGuidanceLayerContract['referenceImages'];
    }
  | { layerType: 'inpaint_mask'; mask?: Partial<CanvasMaskContract>; noiseLevel?: number; denoiseLimit?: number };

export type CanvasProjectMutation =
  | {
      type: 'commitStagedImage';
      candidateFingerprint: string;
      event: ProjectEvent;
      layer: CanvasRasterLayerContractV2;
      selectedImageIndex: number;
    }
  | {
      type: 'rollbackStagedImageCommit';
      event: ProjectEvent;
      layer: CanvasRasterLayerContractV2;
      selectedLayerId: string | null;
      stagingArea: CanvasStagingAreaContractV2;
    }
  | { type: 'setStagedImageIndex'; imageIndex: number }
  | { type: 'cycleStagedImage'; direction: -1 | 1 }
  | { type: 'discardSelectedStagedImage' }
  | { type: 'discardAllStagedImages' }
  | { type: 'toggleCanvasStagingVisibility' }
  | { type: 'toggleCanvasStagingThumbnailsVisibility' }
  | { type: 'clearCanvasStaging' }
  | { type: 'addCanvasLayer'; layer: CanvasLayerContract; index?: number }
  | {
      type: 'applyCanvasLayerStackMutation';
      add?: { index: number; layers: readonly CanvasLayerContract[] };
      removeIds?: readonly string[];
      enabledUpdates: readonly { id: string; isEnabled: boolean }[];
      selectedLayerId: string | null;
    }
  | { type: 'removeCanvasLayers'; ids: string[] }
  | { type: 'duplicateCanvasLayer'; sourceId: string; newId: string }
  | { type: 'reorderCanvasLayers'; orderedIds: string[] }
  | { type: 'updateCanvasLayer'; id: string; patch: CanvasLayerBasePatch }
  | { type: 'replaceCanvasLayer'; layerId: string; layer: CanvasLayerContract }
  | { type: 'setCanvasLayersEnabled'; updates: readonly { id: string; isEnabled: boolean }[] }
  | { type: 'updateCanvasLayerSource'; id: string; source: CanvasLayerSourceContract }
  | { type: 'updateCanvasLayerConfig'; id: string; config: CanvasLayerConfigPatch }
  | { type: 'convertCanvasLayer'; id: string; targetType: CanvasLayerContract['type']; layer: CanvasLayerContract }
  | {
      type: 'mergeCanvasLayersDown';
      upperLayerId: string;
      source: Extract<CanvasLayerSourceContract, { type: 'paint' }>;
    }
  | { type: 'setCanvasBbox'; bbox: CanvasDocumentContractV2['bbox'] }
  | { type: 'setCanvasSelectedLayer'; id: string | null }
  | { type: 'resizeCanvasDocument'; width: number; height: number; offsetX?: number; offsetY?: number }
  | { type: 'replaceCanvasDocument'; document: CanvasDocumentContractV2 }
  | { type: 'saveCanvasSnapshot'; id: string; name: string; createdAt: string }
  | { type: 'restoreCanvasSnapshot'; snapshotId: string }
  | { type: 'deleteCanvasSnapshot'; snapshotId: string }
  | { type: 'setCanvasStagingAutoSwitch'; mode: CanvasStagingAreaContractV2['autoSwitchMode'] };

const CANVAS_PROJECT_MUTATION_TYPES: ReadonlySet<string> = new Set<CanvasProjectMutation['type']>([
  'commitStagedImage',
  'rollbackStagedImageCommit',
  'addCanvasLayer',
  'applyCanvasLayerStackMutation',
  'clearCanvasStaging',
  'convertCanvasLayer',
  'cycleStagedImage',
  'deleteCanvasSnapshot',
  'discardAllStagedImages',
  'discardSelectedStagedImage',
  'duplicateCanvasLayer',
  'mergeCanvasLayersDown',
  'removeCanvasLayers',
  'reorderCanvasLayers',
  'replaceCanvasDocument',
  'replaceCanvasLayer',
  'resizeCanvasDocument',
  'restoreCanvasSnapshot',
  'saveCanvasSnapshot',
  'setCanvasBbox',
  'setCanvasLayersEnabled',
  'setCanvasSelectedLayer',
  'setCanvasStagingAutoSwitch',
  'setStagedImageIndex',
  'toggleCanvasStagingThumbnailsVisibility',
  'toggleCanvasStagingVisibility',
  'updateCanvasLayer',
  'updateCanvasLayerConfig',
  'updateCanvasLayerSource',
]);

export const isCanvasProjectMutation = (value: { type: string }): value is CanvasProjectMutation =>
  CANVAS_PROJECT_MUTATION_TYPES.has(value.type);

type CanvasLayers = CanvasDocumentContractV2['layers'];

const layerExists = (layers: CanvasLayers, id: string): boolean => layers.some((layer) => layer.id === id);

const AUTO_LAYER_NAME_PATTERN = /^Layer (\d+)$/;

export const nextLayerName = (existingNames: readonly string[]): string => {
  const used = new Set<number>();
  for (const name of existingNames) {
    const match = AUTO_LAYER_NAME_PATTERN.exec(name.trim());
    if (match) {
      const n = Number(match[1]);
      if (Number.isInteger(n) && n > 0) {
        used.add(n);
      }
    }
  }
  let n = 1;
  while (used.has(n)) {
    n += 1;
  }
  return `Layer ${n}`;
};

const repairSelectedLayerId = (document: CanvasDocumentContractV2): CanvasDocumentContractV2 =>
  document.selectedLayerId === null || layerExists(document.layers, document.selectedLayerId)
    ? document
    : { ...document, selectedLayerId: document.layers[0]?.id ?? null };

const setCanvasDocument = (project: Project, document: CanvasDocumentContractV2): Project =>
  document === project.canvas.document ? project : { ...project, canvas: { ...project.canvas, document } };

const updateCanvasDocument = (
  project: Project,
  update: (document: CanvasDocumentContractV2) => CanvasDocumentContractV2
): Project => setCanvasDocument(project, update(project.canvas.document));

const setCanvasState = (project: Project, canvas: CanvasStateContractV2): Project =>
  canvas === project.canvas ? project : { ...project, canvas };

const mapCanvasLayer = (
  document: CanvasDocumentContractV2,
  id: string,
  update: (layer: CanvasLayerContract) => CanvasLayerContract
): CanvasDocumentContractV2 => {
  let changed = false;
  const layers = document.layers.map((layer) => {
    if (layer.id !== id) {
      return layer;
    }
    const next = update(layer);
    changed ||= next !== layer;
    return next;
  });
  return changed ? { ...document, layers } : document;
};

const setCanvasLayersEnabled = (
  document: CanvasDocumentContractV2,
  updates: readonly { id: string; isEnabled: boolean }[]
): CanvasDocumentContractV2 => {
  const targets = new Map(updates.map((update) => [update.id, update.isEnabled]));
  let changed = false;
  const layers = document.layers.map((layer) => {
    const isEnabled = targets.get(layer.id);
    if (isEnabled === undefined || isEnabled === layer.isEnabled) {
      return layer;
    }
    changed = true;
    return { ...layer, isEnabled };
  });
  return changed ? { ...document, layers } : document;
};

const applyLayerStackMutation = (
  document: CanvasDocumentContractV2,
  mutation: Extract<CanvasProjectMutation, { type: 'applyCanvasLayerStackMutation' }>
): CanvasDocumentContractV2 => {
  const currentIds = new Set(document.layers.map((layer) => layer.id));
  if (mutation.add && (mutation.removeIds?.length ?? 0) > 0) {
    return document;
  }
  const removeIds = new Set(mutation.removeIds ?? []);
  if ([...removeIds].some((id) => !currentIds.has(id))) {
    return document;
  }
  const projectedIds = new Set(currentIds);
  for (const id of removeIds) {
    projectedIds.delete(id);
  }
  if (mutation.add) {
    for (const layer of mutation.add.layers) {
      if (projectedIds.has(layer.id)) {
        return document;
      }
      projectedIds.add(layer.id);
    }
  }
  if (
    mutation.enabledUpdates.some((update) => !projectedIds.has(update.id)) ||
    (mutation.selectedLayerId !== null && !projectedIds.has(mutation.selectedLayerId))
  ) {
    return document;
  }
  let layers = document.layers;
  let changed = false;
  if (mutation.add?.layers.length) {
    const index = Math.min(Math.max(0, Math.round(mutation.add.index)), layers.length);
    layers = [...layers.slice(0, index), ...mutation.add.layers, ...layers.slice(index)];
    changed = true;
  }
  if (removeIds.size > 0) {
    layers = layers.filter((layer) => !removeIds.has(layer.id));
    changed = true;
  }
  const enabledById = new Map(mutation.enabledUpdates.map((update) => [update.id, update.isEnabled]));
  let enabledChanged = false;
  const nextLayers = layers.map((layer) => {
    const isEnabled = enabledById.get(layer.id);
    if (isEnabled === undefined || isEnabled === layer.isEnabled) {
      return layer;
    }
    enabledChanged = true;
    changed = true;
    return { ...layer, isEnabled };
  });
  if (enabledChanged) {
    layers = nextLayers;
  }
  changed ||= document.selectedLayerId !== mutation.selectedLayerId;
  return changed ? { ...document, layers, selectedLayerId: mutation.selectedLayerId } : document;
};

const addLayer = (document: CanvasDocumentContractV2, layer: CanvasLayerContract, index = 0) => {
  const insertIndex = Math.min(Math.max(0, Math.round(index)), document.layers.length);
  return {
    ...document,
    layers: [...document.layers.slice(0, insertIndex), layer, ...document.layers.slice(insertIndex)],
    selectedLayerId: layer.id,
  };
};

const removeLayers = (document: CanvasDocumentContractV2, ids: readonly string[]): CanvasDocumentContractV2 => {
  const removed = new Set(ids);
  const layers = document.layers.filter((layer) => !removed.has(layer.id));
  if (layers.length === document.layers.length) {
    return document;
  }
  let selectedLayerId = document.selectedLayerId;
  if (selectedLayerId !== null && removed.has(selectedLayerId)) {
    const removedIndex = document.layers.findIndex((layer) => layer.id === selectedLayerId);
    const remainingIds = new Set(layers.map((layer) => layer.id));
    selectedLayerId = null;
    for (let index = removedIndex + 1; index < document.layers.length; index += 1) {
      const id = document.layers[index]?.id;
      if (id && remainingIds.has(id)) {
        selectedLayerId = id;
        break;
      }
    }
    if (selectedLayerId === null) {
      for (let index = removedIndex - 1; index >= 0; index -= 1) {
        const id = document.layers[index]?.id;
        if (id && remainingIds.has(id)) {
          selectedLayerId = id;
          break;
        }
      }
    }
  }
  return { ...document, layers, selectedLayerId };
};

const duplicateLayer = (document: CanvasDocumentContractV2, sourceId: string, newId: string) => {
  const index = document.layers.findIndex((layer) => layer.id === sourceId);
  if (index === -1) {
    return document;
  }
  const source = document.layers[index] as CanvasLayerContract;
  const duplicate = structuredClone(source);
  duplicate.id = newId;
  duplicate.name = `${source.name} copy`;
  return {
    ...document,
    layers: [...document.layers.slice(0, index), duplicate, ...document.layers.slice(index)],
    selectedLayerId: newId,
  };
};

const reorderLayers = (document: CanvasDocumentContractV2, orderedIds: readonly string[]) => {
  if (orderedIds.length !== document.layers.length || new Set(orderedIds).size !== orderedIds.length) {
    return document;
  }
  const byId = new Map(document.layers.map((layer) => [layer.id, layer]));
  const layers: CanvasLayers = [];
  for (const id of orderedIds) {
    const layer = byId.get(id);
    if (!layer) {
      return document;
    }
    layers.push(layer);
  }
  return { ...document, layers };
};

const patchLayer = (layer: CanvasLayerContract, patch: CanvasLayerBasePatch): CanvasLayerContract => {
  const { transform, ...rest } = patch;
  return { ...layer, ...rest, transform: transform ? { ...layer.transform, ...transform } : layer.transform };
};

const patchLayerConfig = (layer: CanvasLayerContract, config: CanvasLayerConfigPatch): CanvasLayerContract => {
  if (layer.type !== config.layerType) {
    return layer;
  }
  if (layer.type === 'raster' && config.layerType === 'raster') {
    return {
      ...layer,
      ...(Object.hasOwn(config, 'adjustments') ? { adjustments: config.adjustments } : {}),
      ...(Object.hasOwn(config, 'isTransparencyLocked') ? { isTransparencyLocked: config.isTransparencyLocked } : {}),
      ...(Object.hasOwn(config, 'filter') ? { filter: config.filter } : {}),
    };
  }
  if (layer.type === 'control' && config.layerType === 'control') {
    return {
      ...layer,
      ...(config.adapter ? { adapter: { ...layer.adapter, ...config.adapter } } : {}),
      ...(Object.hasOwn(config, 'withTransparencyEffect')
        ? { withTransparencyEffect: config.withTransparencyEffect }
        : {}),
      ...(Object.hasOwn(config, 'filter') ? { filter: config.filter } : {}),
    };
  }
  if (layer.type === 'regional_guidance' && config.layerType === 'regional_guidance') {
    return {
      ...layer,
      ...(config.mask ? { mask: { ...layer.mask, ...config.mask } } : {}),
      ...(Object.hasOwn(config, 'positivePrompt') ? { positivePrompt: config.positivePrompt } : {}),
      ...(Object.hasOwn(config, 'negativePrompt') ? { negativePrompt: config.negativePrompt } : {}),
      ...(Object.hasOwn(config, 'autoNegative') ? { autoNegative: config.autoNegative } : {}),
      ...(Object.hasOwn(config, 'referenceImages') ? { referenceImages: config.referenceImages } : {}),
    };
  }
  if (layer.type === 'inpaint_mask' && config.layerType === 'inpaint_mask') {
    return {
      ...layer,
      ...(config.mask ? { mask: { ...layer.mask, ...config.mask } } : {}),
      ...(Object.hasOwn(config, 'noiseLevel') ? { noiseLevel: config.noiseLevel } : {}),
      ...(Object.hasOwn(config, 'denoiseLimit') ? { denoiseLimit: config.denoiseLimit } : {}),
    };
  }
  return layer;
};

const clampBbox = (bbox: CanvasDocumentContractV2['bbox'], width: number, height: number) => {
  const clampedWidth = Math.min(Math.max(1, Math.round(bbox.width)), width);
  const clampedHeight = Math.min(Math.max(1, Math.round(bbox.height)), height);
  return {
    height: clampedHeight,
    width: clampedWidth,
    x: Math.min(Math.max(0, Math.round(bbox.x)), width - clampedWidth),
    y: Math.min(Math.max(0, Math.round(bbox.y)), height - clampedHeight),
  };
};

const clearStagingArea = (stagingArea: CanvasStateContractV2['stagingArea']) => ({
  ...stagingArea,
  isVisible: false,
  pendingImageIds: [],
  pendingImages: [],
  selectedImageIndex: 0,
  sourceQueueItemId: undefined,
});

const clampStagedImageIndex = (imageIndex: number, slotCount: number): number =>
  Math.min(Math.max(0, slotCount - 1), Math.max(0, imageIndex));

const selectedCandidate = (project: Project) => {
  const slot = getCanvasStagingSlots(project.canvas, project.queue.items)[
    project.canvas.stagingArea.selectedImageIndex
  ];
  return slot?.kind === 'candidate' ? slot.candidate : undefined;
};

export const applyCanvasProjectMutation = (project: Project, mutation: CanvasProjectMutation): Project => {
  switch (mutation.type) {
    case 'commitStagedImage': {
      const stagedImage = selectedCandidate(project);
      if (
        project.canvas.stagingArea.selectedImageIndex !== mutation.selectedImageIndex ||
        !stagedImage ||
        getCanvasStagingCandidateFingerprint(stagedImage) !== mutation.candidateFingerprint
      ) {
        return project;
      }
      const { layer } = mutation;
      if (project.canvas.document.layers.some((candidate) => candidate.id === layer.id)) {
        return project;
      }
      return {
        ...project,
        canvas: {
          ...project.canvas,
          document: {
            ...project.canvas.document,
            layers: [layer, ...project.canvas.document.layers],
            selectedLayerId: layer.id,
          },
          stagingArea: clearStagingArea(project.canvas.stagingArea),
        },
        events: [mutation.event, ...project.events],
      };
    }
    case 'rollbackStagedImageCommit': {
      if (
        project.canvas.document.selectedLayerId !== mutation.layer.id ||
        project.canvas.document.layers[0] !== mutation.layer ||
        project.events[0] !== mutation.event ||
        project.canvas.stagingArea.pendingImages.length !== 0
      ) {
        return project;
      }
      return {
        ...project,
        canvas: {
          ...project.canvas,
          document: {
            ...project.canvas.document,
            layers: project.canvas.document.layers.slice(1),
            selectedLayerId: mutation.selectedLayerId,
          },
          stagingArea: mutation.stagingArea,
        },
        events: project.events.slice(1),
      };
    }
    case 'setStagedImageIndex': {
      const selectedImageIndex = clampStagedImageIndex(
        mutation.imageIndex,
        getCanvasStagingSlotCount(project.canvas, project.queue.items)
      );
      return selectedImageIndex === project.canvas.stagingArea.selectedImageIndex
        ? project
        : {
            ...project,
            canvas: {
              ...project.canvas,
              stagingArea: { ...project.canvas.stagingArea, selectedImageIndex },
            },
          };
    }
    case 'cycleStagedImage': {
      const count = getCanvasStagingSlotCount(project.canvas, project.queue.items);
      const current = project.canvas.stagingArea.selectedImageIndex;
      const selectedImageIndex = count < 2 ? 0 : (current + mutation.direction + count) % count;
      return selectedImageIndex === current
        ? project
        : {
            ...project,
            canvas: {
              ...project.canvas,
              stagingArea: { ...project.canvas.stagingArea, selectedImageIndex },
            },
          };
    }
    case 'discardSelectedStagedImage': {
      const selected = selectedCandidate(project);
      if (!selected) {
        return project;
      }
      const pendingImages = project.canvas.stagingArea.pendingImages.filter(
        (image) => image.sourceQueueItemId !== selected.sourceQueueItemId || image.imageName !== selected.imageName
      );
      const canvas = {
        ...project.canvas,
        stagingArea: {
          ...project.canvas.stagingArea,
          pendingImageIds: pendingImages.map((image) => image.imageName),
          pendingImages,
        },
      };
      const slotCount = getCanvasStagingSlotCount(canvas, project.queue.items);
      return {
        ...project,
        canvas: {
          ...canvas,
          stagingArea: {
            ...canvas.stagingArea,
            isVisible: slotCount > 0 && canvas.stagingArea.isVisible,
            selectedImageIndex: clampStagedImageIndex(canvas.stagingArea.selectedImageIndex, slotCount),
            sourceQueueItemId: pendingImages.length > 0 ? canvas.stagingArea.sourceQueueItemId : undefined,
          },
        },
      };
    }
    case 'discardAllStagedImages': {
      const canvas = { ...project.canvas, stagingArea: clearStagingArea(project.canvas.stagingArea) };
      return {
        ...project,
        canvas: {
          ...canvas,
          stagingArea: {
            ...canvas.stagingArea,
            isVisible: getCanvasStagingSlotCount(canvas, project.queue.items) > 0,
          },
        },
      };
    }
    case 'toggleCanvasStagingVisibility':
      return getCanvasStagingSlotCount(project.canvas, project.queue.items) === 0
        ? project
        : {
            ...project,
            canvas: {
              ...project.canvas,
              stagingArea: {
                ...project.canvas.stagingArea,
                isVisible: !project.canvas.stagingArea.isVisible,
              },
            },
          };
    case 'toggleCanvasStagingThumbnailsVisibility':
      return getCanvasStagingSlotCount(project.canvas, project.queue.items) === 0
        ? project
        : {
            ...project,
            canvas: {
              ...project.canvas,
              stagingArea: {
                ...project.canvas.stagingArea,
                areThumbnailsVisible: !project.canvas.stagingArea.areThumbnailsVisible,
              },
            },
          };
    case 'clearCanvasStaging':
      return { ...project, canvas: { ...project.canvas, stagingArea: clearStagingArea(project.canvas.stagingArea) } };
    case 'addCanvasLayer':
      return updateCanvasDocument(project, (document) => addLayer(document, mutation.layer, mutation.index));
    case 'applyCanvasLayerStackMutation':
      return updateCanvasDocument(project, (document) => applyLayerStackMutation(document, mutation));
    case 'removeCanvasLayers':
      return updateCanvasDocument(project, (document) => removeLayers(document, mutation.ids));
    case 'duplicateCanvasLayer':
      return updateCanvasDocument(project, (document) => duplicateLayer(document, mutation.sourceId, mutation.newId));
    case 'reorderCanvasLayers':
      return updateCanvasDocument(project, (document) => reorderLayers(document, mutation.orderedIds));
    case 'updateCanvasLayer':
      return updateCanvasDocument(project, (document) =>
        mapCanvasLayer(document, mutation.id, (layer) => patchLayer(layer, mutation.patch))
      );
    case 'replaceCanvasLayer':
      return updateCanvasDocument(project, (document) =>
        mapCanvasLayer(document, mutation.layerId, () => mutation.layer)
      );
    case 'setCanvasLayersEnabled':
      return updateCanvasDocument(project, (document) => setCanvasLayersEnabled(document, mutation.updates));
    case 'updateCanvasLayerSource':
      return updateCanvasDocument(project, (document) =>
        mapCanvasLayer(document, mutation.id, (layer) =>
          layer.type === 'raster' || layer.type === 'control' ? { ...layer, source: mutation.source } : layer
        )
      );
    case 'updateCanvasLayerConfig':
      return updateCanvasDocument(project, (document) =>
        mapCanvasLayer(document, mutation.id, (layer) => patchLayerConfig(layer, mutation.config))
      );
    case 'convertCanvasLayer': {
      if (mutation.layer.type !== mutation.targetType || !layerExists(project.canvas.document.layers, mutation.id)) {
        return project;
      }
      const converted = structuredClone(mutation.layer);
      converted.id = mutation.id;
      return updateCanvasDocument(project, (document) => mapCanvasLayer(document, mutation.id, () => converted));
    }
    case 'mergeCanvasLayersDown': {
      const document = project.canvas.document;
      const upperIndex = document.layers.findIndex((layer) => layer.id === mutation.upperLayerId);
      const below = upperIndex === -1 ? undefined : document.layers[upperIndex + 1];
      if (!below) {
        return project;
      }
      const merged: CanvasRasterLayerContractV2 = {
        blendMode: below.blendMode,
        id: below.id,
        isEnabled: below.isEnabled,
        isLocked: below.isLocked,
        name: below.name,
        opacity: below.opacity,
        source: mutation.source,
        transform: below.transform,
        type: 'raster',
      };
      return setCanvasDocument(project, {
        ...document,
        layers: document.layers
          .filter((_, index) => index !== upperIndex)
          .map((layer) => (layer.id === below.id ? merged : layer)),
        selectedLayerId: document.selectedLayerId === mutation.upperLayerId ? below.id : document.selectedLayerId,
      });
    }
    case 'setCanvasBbox':
      return updateCanvasDocument(project, (document) => ({
        ...document,
        bbox: {
          height: Math.max(1, Math.round(mutation.bbox.height)),
          width: Math.max(1, Math.round(mutation.bbox.width)),
          x: Math.round(mutation.bbox.x),
          y: Math.round(mutation.bbox.y),
        },
      }));
    case 'setCanvasSelectedLayer':
      return updateCanvasDocument(project, (document) =>
        mutation.id !== null && !layerExists(document.layers, mutation.id)
          ? document
          : document.selectedLayerId === mutation.id
            ? document
            : { ...document, selectedLayerId: mutation.id }
      );
    case 'resizeCanvasDocument': {
      const width = Math.max(1, Math.round(mutation.width));
      const height = Math.max(1, Math.round(mutation.height));
      const offsetX = mutation.offsetX ?? 0;
      const offsetY = mutation.offsetY ?? 0;
      return updateCanvasDocument(project, (document) => ({
        ...document,
        bbox: clampBbox(
          { ...document.bbox, x: document.bbox.x + offsetX, y: document.bbox.y + offsetY },
          width,
          height
        ),
        height,
        layers:
          offsetX === 0 && offsetY === 0
            ? document.layers
            : document.layers.map((layer) => ({
                ...layer,
                transform: { ...layer.transform, x: layer.transform.x + offsetX, y: layer.transform.y + offsetY },
              })),
        width,
      }));
    }
    case 'replaceCanvasDocument':
      return setCanvasState(project, {
        ...project.canvas,
        document: repairSelectedLayerId(normalizeCanvasDocumentControlAdapters(structuredClone(mutation.document))),
        documentRevision: project.canvas.documentRevision + 1,
        stagingArea: clearStagingArea(project.canvas.stagingArea),
      });
    case 'saveCanvasSnapshot':
      return setCanvasState(project, {
        ...project.canvas,
        snapshots: [
          ...project.canvas.snapshots,
          {
            createdAt: mutation.createdAt,
            document: normalizeCanvasDocumentControlAdapters(structuredClone(project.canvas.document)),
            id: mutation.id,
            name: mutation.name,
          },
        ],
      });
    case 'restoreCanvasSnapshot': {
      const snapshot = project.canvas.snapshots.find((entry) => entry.id === mutation.snapshotId);
      return snapshot
        ? setCanvasState(project, {
            ...project.canvas,
            document: repairSelectedLayerId(normalizeCanvasDocumentControlAdapters(structuredClone(snapshot.document))),
            documentRevision: project.canvas.documentRevision + 1,
          })
        : project;
    }
    case 'deleteCanvasSnapshot': {
      const snapshots = project.canvas.snapshots.filter((entry) => entry.id !== mutation.snapshotId);
      return snapshots.length === project.canvas.snapshots.length
        ? project
        : setCanvasState(project, { ...project.canvas, snapshots });
    }
    case 'setCanvasStagingAutoSwitch':
      return project.canvas.stagingArea.autoSwitchMode === mutation.mode
        ? project
        : {
            ...project,
            canvas: {
              ...project.canvas,
              stagingArea: { ...project.canvas.stagingArea, autoSwitchMode: mutation.mode },
            },
          };
  }
};
