import type { Selector } from '@reduxjs/toolkit';
import { createSelector } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import { selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import type {
  CanvasControlLayerState,
  CanvasEntityIdentifier,
  CanvasEntityState,
  CanvasEntityType,
  CanvasInpaintMaskState,
  CanvasMetadata,
  CanvasRasterLayerState,
  CanvasRegionalGuidanceState,
  CanvasState,
} from 'features/controlLayers/store/types';
import { getGridSize, getOptimalDimension } from 'features/parameters/util/optimalDimension';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

/**
 * Selects the canvases slice from the root state
 */
export const selectCanvasesSlice = (state: RootState) => state.canvases;

/**
 * Selects a specific canvas instance from the root state
 */
export const selectCanvasInstance = (state: RootState, canvasId: string) => 
  state.canvases.instances[canvasId]?.present;

/**
 * Selects the active canvas instance from the root state
 */
export const selectActiveCanvas = (state: RootState) => {
  const activeId = state.canvases.activeInstanceId;
  return activeId ? state.canvases.instances[activeId]?.present : null;
};

/**
 * Selects the active canvas ID
 */
export const selectActiveCanvasId = (state: RootState) => state.canvases.activeInstanceId;

/**
 * Selects the total number of canvas instances
 */
export const selectCanvasCount = (state: RootState) => Object.keys(state.canvases.instances).length;

/**
 * Selects all canvas instances
 */
export const selectCanvasInstances = (state: RootState) => state.canvases.instances;

/**
 * Legacy selector for backward compatibility - selects the active canvas
 * @deprecated Use selectActiveCanvas instead
 */
export const selectCanvasSlice = (state: RootState) => {
  const activeCanvas = selectActiveCanvas(state);
  return activeCanvas || null;
};

// Legacy canvas selector factory for backward compatibility - keeping for potential future use
// const createCanvasSelector = <T>(selector: Selector<CanvasState, T>) => createSelector(selectCanvasSlice, selector);

// New parameterized canvas selector factory
export const createCanvasInstanceSelector = <T>(selector: Selector<CanvasState, T>) => 
  createSelector(
    [selectCanvasInstance, (_state: RootState, canvasId: string) => canvasId],
    (canvas, _canvasId) => canvas ? selector(canvas) : null
  );

// Active canvas selector factory
export const createActiveCanvasSelector = <T>(selector: Selector<CanvasState, T>) => 
  createSelector(selectActiveCanvas, (canvas) => canvas ? selector(canvas) : null);

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
const selectEntityCountAll = createActiveCanvasSelector((canvas) => {
  return (
    canvas.regionalGuidance.entities.length +
    canvas.rasterLayers.entities.length +
    canvas.controlLayers.entities.length +
    canvas.inpaintMasks.entities.length
  );
});

const isVisibleEntity = (entity: CanvasEntityState) => entity.isEnabled && entity.objects.length > 0;

export const selectRasterLayerEntities = createActiveCanvasSelector((canvas) => canvas.rasterLayers.entities);
export const selectActiveRasterLayerEntities = createSelector(selectRasterLayerEntities, (entities) =>
  entities ? entities.filter(isVisibleEntity) : []
);

export const selectControlLayerEntities = createActiveCanvasSelector((canvas) => canvas.controlLayers.entities);
export const selectActiveControlLayerEntities = createSelector(selectControlLayerEntities, (entities) =>
  entities ? entities.filter(isVisibleEntity) : []
);

export const selectInpaintMaskEntities = createActiveCanvasSelector((canvas) => canvas.inpaintMasks.entities);
export const selectActiveInpaintMaskEntities = createSelector(selectInpaintMaskEntities, (entities) =>
  entities ? entities.filter(isVisibleEntity) : []
);

export const selectRegionalGuidanceEntities = createActiveCanvasSelector((canvas) => canvas.regionalGuidance.entities);
export const selectActiveRegionalGuidanceEntities = createSelector(selectRegionalGuidanceEntities, (entities) =>
  entities ? entities.filter(isVisibleEntity) : []
);

/**
 * Selects if the canvas has any entities.
 */
export const selectHasEntities = createSelector(selectEntityCountAll, (count) => count ? count > 0 : false);

/**
 * Selects the optimal dimension for the canvas based on the currently-selected model
 */
export const selectOptimalDimension = createSelector(selectParamsSlice, (params): number => {
  const modelBase = params.model?.base;
  return getOptimalDimension(modelBase ?? null);
});

/**
 * Selects the grid size for the canvas based on the currently-selected model
 */
export const selectGridSize = createSelector(selectParamsSlice, (params): number => {
  const modelBase = params.model?.base;
  return getGridSize(modelBase ?? null);
});

/**
 * Selects a single entity from the canvas slice. If the entity identifier is narrowed to a specific type, the
 * return type will be narrowed as well.
 */
export function selectEntity<T extends CanvasEntityIdentifier>(
  state: CanvasState,
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
  state: CanvasState,
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
  state: CanvasState,
  entityIdentifier: T,
  caller: string
): Extract<CanvasEntityState, T> {
  const entity = selectEntity(state, entityIdentifier);
  assert(entity, `Entity with id ${entityIdentifier.id} not found in ${caller}`);
  return entity;
}

export const selectEntityExists = <T extends CanvasEntityIdentifier>(entityIdentifier: T) => {
  return createActiveCanvasSelector((canvas) => Boolean(selectEntity(canvas, entityIdentifier)));
};

/**
 * Selects all entities of the given type.
 */
export function selectAllEntitiesOfType<T extends CanvasEntityState['type']>(
  state: CanvasState,
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
export function selectAllEntities(state: CanvasState): CanvasEntityState[] {
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
  state: CanvasState
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
  state: CanvasState,
  entityIdentifier: CanvasEntityIdentifier<'regional_guidance'>,
  referenceImageId: string
) {
  const entity = selectEntity(state, entityIdentifier);
  if (!entity) {
    return undefined;
  }
  return entity.referenceImages.find(({ id }) => id === referenceImageId);
}

export const selectBbox = createActiveCanvasSelector((canvas) => canvas.bbox);

export const selectSelectedEntityIdentifier = createActiveCanvasSelector(
  (canvas) => canvas.selectedEntityIdentifier
);

export const selectBookmarkedEntityIdentifier = createActiveCanvasSelector(
  (canvas) => canvas.bookmarkedEntityIdentifier
);

export const selectCanvasMayUndo = (state: RootState) => {
  const activeId = state.canvases.activeInstanceId;
  return activeId ? (state.canvases.instances[activeId]?.past?.length ?? 0) > 0 : false;
};
export const selectCanvasMayRedo = (state: RootState) => {
  const activeId = state.canvases.activeInstanceId;
  return activeId ? (state.canvases.instances[activeId]?.future?.length ?? 0) > 0 : false;
};
export const selectSelectedEntityFill = createSelector(
  selectActiveCanvas,
  selectSelectedEntityIdentifier,
  (canvas, selectedEntityIdentifier) => {
    if (!canvas || !selectedEntityIdentifier) {
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

const selectRasterLayersIsHidden = createActiveCanvasSelector((canvas) => canvas.rasterLayers.isHidden);
const selectControlLayersIsHidden = createActiveCanvasSelector((canvas) => canvas.controlLayers.isHidden);
const selectInpaintMasksIsHidden = createActiveCanvasSelector((canvas) => canvas.inpaintMasks.isHidden);
const selectRegionalGuidanceIsHidden = createActiveCanvasSelector((canvas) => canvas.regionalGuidance.isHidden);

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
  return createActiveCanvasSelector((canvas) => {
    const entity = selectEntity(canvas, entityIdentifier);

    if (!entity) {
      return false;
    }
    return entity.objects.length > 0;
  });
};

export const selectWidth = createActiveCanvasSelector((canvas) => canvas.bbox.rect.width);
export const selectHeight = createActiveCanvasSelector((canvas) => canvas.bbox.rect.height);
export const selectAspectRatioID = createActiveCanvasSelector((canvas) => canvas.bbox.aspectRatio.id);
export const selectAspectRatioValue = createActiveCanvasSelector((canvas) => canvas.bbox.aspectRatio.value);
export const selectScaledSize = createSelector(selectBbox, (bbox) => bbox.scaledSize);
export const selectScaleMethod = createSelector(selectBbox, (bbox) => bbox.scaleMethod);
export const selectBboxRect = createSelector(selectBbox, (bbox) => bbox.rect);
export const selectBboxModelBase = createSelector(selectBbox, (bbox) => bbox.modelBase);

export const selectCanvasMetadata = createSelector(
  selectActiveCanvas,
  (canvas): { canvas_v2_metadata: CanvasMetadata } | null => {
    if (!canvas) {
      return null;
    }
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
  return canvas ? canvas.controlLayers.isHidden && canvas.inpaintMasks.isHidden && canvas.regionalGuidance.isHidden : false;
});
