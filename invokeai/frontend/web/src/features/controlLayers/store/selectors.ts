import { createSelector } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type {
  CanvasControlLayerState,
  CanvasEntity,
  CanvasEntityIdentifier,
  CanvasEntityState,
  CanvasEntityType,
  CanvasInpaintMaskState,
  CanvasMetadata,
  CanvasRasterLayerState,
  CanvasRegionalGuidanceState,
} from 'features/controlLayers/store/types';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

/**
 * Selects the active tab
 */
export const selectActiveTab = (state: RootState) => state.tab.activeTab;

/**
 * Selects the canvas slice from the root state
 */
const selectCanvasSlice = (state: RootState) => state.canvas;

/**
 * Selects the canvases
 */
export const selectCanvases = createSelector(selectCanvasSlice, (state) =>
  Object.values(state.canvases).map((instance) => ({
    id: instance.id,
    name: instance.name,
    ...instance.canvas.present,
    isActive: instance.id === state.activeCanvasId,
    canDelete: Object.keys(state.canvases).length > 1,
  }))
);

/**
 * Selects the active canvas with history
 */
const selectActiveCanvasWithHistory = createSelector(
  selectCanvasSlice,
  (state) => state.canvases[state.activeCanvasId]!.canvas
);

export const selectActiveCanvas = createSelector(selectActiveCanvasWithHistory, (canvas) => canvas.present);
export const selectActiveCanvasId = createSelector(selectCanvasSlice, (state) => state.activeCanvasId);

export const selectCanvasByCanvasId = (state: RootState, canvasId: string) => {
  const instance = selectCanvasSlice(state).canvases[canvasId];
  assert(instance, 'Canvas does not exist');
  return instance.canvas.present;
};

/**
 * Selects the total canvas entity count:
 * - Regions
 * - IP adapters
 * - Raster layers
 * - Control layers
 * - Inpaint masks
 *
 * All entities are counted, regardless of their state.
 */
const selectEntityCountAll = createSelector(selectActiveCanvas, (canvas) => {
  return (
    canvas.regionalGuidance.entities.length +
    canvas.rasterLayers.entities.length +
    canvas.controlLayers.entities.length +
    canvas.inpaintMasks.entities.length
  );
});

const isVisibleEntity = (entity: CanvasEntityState) => entity.isEnabled && entity.objects.length > 0;

export const selectRasterLayerEntities = createSelector(selectActiveCanvas, (canvas) => canvas.rasterLayers.entities);
export const selectActiveRasterLayerEntities = createSelector(selectRasterLayerEntities, (entities) =>
  entities.filter(isVisibleEntity)
);

export const selectControlLayerEntities = createSelector(selectActiveCanvas, (canvas) => canvas.controlLayers.entities);
export const selectActiveControlLayerEntities = createSelector(selectControlLayerEntities, (entities) =>
  entities.filter(isVisibleEntity)
);

export const selectInpaintMaskEntities = createSelector(selectActiveCanvas, (canvas) => canvas.inpaintMasks.entities);
export const selectActiveInpaintMaskEntities = createSelector(selectInpaintMaskEntities, (entities) =>
  entities.filter(isVisibleEntity)
);

export const selectRegionalGuidanceEntities = createSelector(
  selectActiveCanvas,
  (canvas) => canvas.regionalGuidance.entities
);
export const selectActiveRegionalGuidanceEntities = createSelector(selectRegionalGuidanceEntities, (entities) =>
  entities.filter(isVisibleEntity)
);

/**
 * Selects if the canvas has any entities.
 */
export const selectHasEntities = createSelector(selectEntityCountAll, (count) => count > 0);

/**
 * Selects a single entity from the canvas slice. If the entity identifier is narrowed to a specific type, the
 * return type will be narrowed as well.
 */
export function selectEntity<T extends CanvasEntityIdentifier>(
  state: CanvasEntity,
  entityIdentifier: T
): Extract<CanvasEntityState, T> | undefined {
  const { id, type } = entityIdentifier;

  let entity: CanvasEntityState | undefined = undefined;

  switch (type) {
    case 'raster_layer':
      entity = state.rasterLayers.entities.find((entity) => entity.id === id);
      break;
    case 'control_layer':
      entity = state.controlLayers.entities.find((entity) => entity.id === id);
      break;
    case 'inpaint_mask':
      entity = state.inpaintMasks.entities.find((entity) => entity.id === id);
      break;
    case 'regional_guidance':
      entity = state.regionalGuidance.entities.find((entity) => entity.id === id);
      break;
  }

  // This cast is safe, but TS seems to be unable to infer the type
  return entity as Extract<CanvasEntityState, T> | undefined;
}

/**
 * Selects the entity identifier for the entity that is below the given entity in terms of draw order.
 */
export function selectEntityIdentifierBelowThisOne<T extends CanvasEntityIdentifier>(
  state: CanvasEntity,
  entityIdentifier: T
): Extract<CanvasEntityState, T> | undefined {
  const { id, type } = entityIdentifier;

  let entities: CanvasEntityState[];

  switch (type) {
    case 'raster_layer': {
      entities = state.rasterLayers.entities;
      break;
    }
    case 'control_layer': {
      entities = state.controlLayers.entities;
      break;
    }
    case 'inpaint_mask': {
      entities = state.inpaintMasks.entities;
      break;
    }
    case 'regional_guidance': {
      entities = state.regionalGuidance.entities;
      break;
    }
  }

  // Must reverse to get the draw order
  const reversedEntities = entities.toReversed();
  const idx = reversedEntities.findIndex((entity) => entity.id === id);
  const entity = reversedEntities.at(idx + 1);

  // This cast is safe, but TS seems to be unable to infer the type
  return entity as Extract<CanvasEntityState, T> | undefined;
}

/**
 * Selected an entity from the canvas slice. If the entity is not found, an error is thrown.
 *
 * Provide a `caller` string to help identify the caller in the error message.
 *
 * Wrapper around {@link selectEntity}.
 */
export function selectEntityOrThrow<T extends CanvasEntityIdentifier>(
  state: CanvasEntity,
  entityIdentifier: T,
  caller: string
): Extract<CanvasEntityState, T> {
  const entity = selectEntity(state, entityIdentifier);
  assert(entity, `Entity with id ${entityIdentifier.id} not found in ${caller}`);
  return entity;
}

export const selectEntityExists = <T extends CanvasEntityIdentifier>(entityIdentifier: T) => {
  return createSelector(selectActiveCanvas, (canvas) => Boolean(selectEntity(canvas, entityIdentifier)));
};

/**
 * Selects all entities of the given type.
 */
export function selectAllEntitiesOfType<T extends CanvasEntityState['type']>(
  state: CanvasEntity,
  type: T
): Extract<CanvasEntityState, { type: T }>[] {
  let entities: CanvasEntityState[] = [];

  switch (type) {
    case 'raster_layer':
      entities = state.rasterLayers.entities;
      break;
    case 'control_layer':
      entities = state.controlLayers.entities;
      break;
    case 'inpaint_mask':
      entities = state.inpaintMasks.entities;
      break;
    case 'regional_guidance':
      entities = state.regionalGuidance.entities;
      break;
  }

  // This cast is safe, but TS seems to be unable to infer the type
  return entities as Extract<CanvasEntityState, { type: T }>[];
}

/**
 * Selects all entities, in the order they are displayed in the list.
 */
export function selectAllEntities(state: CanvasEntity): CanvasEntityState[] {
  // These are in the same order as they are displayed in the list!
  return [
    ...state.inpaintMasks.entities.toReversed(),
    ...state.regionalGuidance.entities.toReversed(),
    ...state.controlLayers.entities.toReversed(),
    ...state.rasterLayers.entities.toReversed(),
  ];
}

/**
 * Selects all _renderable_ entities:
 * - Raster layers
 * - Control layers
 * - Inpaint masks
 * - Regional guidance
 */
export function selectAllRenderableEntities(
  state: CanvasEntity
): (CanvasRasterLayerState | CanvasControlLayerState | CanvasInpaintMaskState | CanvasRegionalGuidanceState)[] {
  return [
    ...state.rasterLayers.entities,
    ...state.controlLayers.entities,
    ...state.inpaintMasks.entities,
    ...state.regionalGuidance.entities,
  ];
}

/**
 * Selects the IP adapter for the specific Regional Guidance layer.
 */
export function selectRegionalGuidanceReferenceImage(
  state: CanvasEntity,
  entityIdentifier: CanvasEntityIdentifier<'regional_guidance'>,
  referenceImageId: string
) {
  const entity = selectEntity(state, entityIdentifier);
  if (!entity) {
    return undefined;
  }
  return entity.referenceImages.find(({ id }) => id === referenceImageId);
}

export const selectBbox = createSelector(selectActiveCanvas, (canvas) => canvas.bbox);

export const selectSelectedEntityIdentifier = createSelector(
  selectActiveCanvas,
  (canvas) => canvas.selectedEntityIdentifier
);

export const selectBookmarkedEntityIdentifier = createSelector(
  selectActiveCanvas,
  (canvas) => canvas.bookmarkedEntityIdentifier
);

export const selectCanvasMayUndo = createSelector(selectActiveCanvasWithHistory, (canvas) => canvas.past.length > 0);
export const selectCanvasMayRedo = createSelector(selectActiveCanvasWithHistory, (canvas) => canvas.future.length > 0);
export const selectSelectedEntityFill = createSelector(
  selectActiveCanvas,
  selectSelectedEntityIdentifier,
  (canvas, selectedEntityIdentifier) => {
    if (!selectedEntityIdentifier) {
      return null;
    }
    const entity = selectEntity(canvas, selectedEntityIdentifier);
    if (!entity) {
      return null;
    }
    if (entity.type !== 'inpaint_mask' && entity.type !== 'regional_guidance') {
      return null;
    }
    return entity.fill;
  }
);

const selectRasterLayersIsHidden = createSelector(selectActiveCanvas, (canvas) => canvas.rasterLayers.isHidden);
const selectControlLayersIsHidden = createSelector(selectActiveCanvas, (canvas) => canvas.controlLayers.isHidden);
const selectInpaintMasksIsHidden = createSelector(selectActiveCanvas, (canvas) => canvas.inpaintMasks.isHidden);
const selectRegionalGuidanceIsHidden = createSelector(selectActiveCanvas, (canvas) => canvas.regionalGuidance.isHidden);

/**
 * Returns the hidden selector for the given entity type.
 */
export const getSelectIsTypeHidden = (type: CanvasEntityType) => {
  switch (type) {
    case 'raster_layer':
      return selectRasterLayersIsHidden;
    case 'control_layer':
      return selectControlLayersIsHidden;
    case 'inpaint_mask':
      return selectInpaintMasksIsHidden;
    case 'regional_guidance':
      return selectRegionalGuidanceIsHidden;
    default:
      assert<Equals<typeof type, never>>(false, 'Unhandled entity type');
  }
};

/**
 * Builds a selector taht selects if the entity is selected.
 */
export const buildSelectIsSelected = (entityIdentifier: CanvasEntityIdentifier) => {
  return createSelector(
    selectSelectedEntityIdentifier,
    (selectedEntityIdentifier) => selectedEntityIdentifier?.id === entityIdentifier.id
  );
};

/**
 * Builds a selector that selects if the entity is empty.
 *
 * Reference images are considered empty if the IP adapter is empty.
 *
 * Other entities are considered empty if they have no objects.
 */
export const buildSelectHasObjects = (entityIdentifier: CanvasEntityIdentifier) => {
  return createSelector(selectActiveCanvas, (canvas) => {
    const entity = selectEntity(canvas, entityIdentifier);

    if (!entity) {
      return false;
    }
    return entity.objects.length > 0;
  });
};

export const selectWidth = createSelector(selectActiveCanvas, (canvas) => canvas.bbox.rect.width);
export const selectHeight = createSelector(selectActiveCanvas, (canvas) => canvas.bbox.rect.height);
export const selectAspectRatioID = createSelector(selectActiveCanvas, (canvas) => canvas.bbox.aspectRatio.id);
export const selectAspectRatioValue = createSelector(selectActiveCanvas, (canvas) => canvas.bbox.aspectRatio.value);
export const selectScaledSize = createSelector(selectBbox, (bbox) => bbox.scaledSize);
export const selectScaleMethod = createSelector(selectBbox, (bbox) => bbox.scaleMethod);
export const selectBboxRect = createSelector(selectBbox, (bbox) => bbox.rect);
export const selectBboxModelBase = createSelector(selectBbox, (bbox) => bbox.modelBase);

export const selectCanvasMetadata = createSelector(
  selectActiveCanvas,
  (canvas): { canvas_v2_metadata: CanvasMetadata } => {
    const canvas_v2_metadata: CanvasMetadata = {
      controlLayers: selectAllEntitiesOfType(canvas, 'control_layer'),
      inpaintMasks: selectAllEntitiesOfType(canvas, 'inpaint_mask'),
      rasterLayers: selectAllEntitiesOfType(canvas, 'raster_layer'),
      regionalGuidance: selectAllEntitiesOfType(canvas, 'regional_guidance'),
    };
    return { canvas_v2_metadata };
  }
);

/**
 * Selects whether all non-raster layer categories (control layers, inpaint masks, regional guidance) are hidden.
 * This is used to determine the state of the toggle button that shows/hides all non-raster layers.
 */
export const selectNonRasterLayersIsHidden = createSelector(selectActiveCanvas, (canvas) => {
  return canvas.controlLayers.isHidden && canvas.inpaintMasks.isHidden && canvas.regionalGuidance.isHidden;
});
