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
 * Selects the canvas slice from the root state
 */
export const selectCanvasSlice = (state: RootState) => state.canvas.present;

const createCanvasSelector = <T>(selector: Selector<CanvasState, T>) => createSelector(selectCanvasSlice, selector);

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
const selectEntityCountAll = createCanvasSelector((canvas) => {
  return (
    canvas.regionalGuidance.entities.length +
    canvas.rasterLayers.entities.length +
    canvas.controlLayers.entities.length +
    canvas.inpaintMasks.entities.length
  );
});

const isVisibleEntity = (entity: CanvasEntityState) => entity.isEnabled && entity.objects.length > 0;

export const selectRasterLayerEntities = createCanvasSelector((canvas) => canvas.rasterLayers.entities);
export const selectActiveRasterLayerEntities = createSelector(selectRasterLayerEntities, (entities) =>
  entities.filter(isVisibleEntity)
);

export const selectControlLayerEntities = createCanvasSelector((canvas) => canvas.controlLayers.entities);
export const selectActiveControlLayerEntities = createSelector(selectControlLayerEntities, (entities) =>
  entities.filter(isVisibleEntity)
);

export const selectInpaintMaskEntities = createCanvasSelector((canvas) => canvas.inpaintMasks.entities);
export const selectActiveInpaintMaskEntities = createSelector(selectInpaintMaskEntities, (entities) =>
  entities.filter(isVisibleEntity)
);

export const selectRegionalGuidanceEntities = createCanvasSelector((canvas) => canvas.regionalGuidance.entities);
export const selectActiveRegionalGuidanceEntities = createSelector(selectRegionalGuidanceEntities, (entities) =>
  entities.filter(isVisibleEntity)
);

/**
 * Selects if the canvas has any entities.
 */
export const selectHasEntities = createSelector(selectEntityCountAll, (count) => count > 0);

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
  return createCanvasSelector((canvas) => Boolean(selectEntity(canvas, entityIdentifier)));
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

export const selectBbox = createCanvasSelector((canvas) => canvas.bbox);

export const selectSelectedEntityIdentifier = createSelector(
  selectCanvasSlice,
  (canvas) => canvas.selectedEntityIdentifier
);

export const selectBookmarkedEntityIdentifier = createSelector(
  selectCanvasSlice,
  (canvas) => canvas.bookmarkedEntityIdentifier
);

export const selectCanvasMayUndo = (state: RootState) => state.canvas.past.length > 0;
export const selectCanvasMayRedo = (state: RootState) => state.canvas.future.length > 0;
export const selectSelectedEntityFill = createSelector(
  selectCanvasSlice,
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

const selectRasterLayersIsHidden = createCanvasSelector((canvas) => canvas.rasterLayers.isHidden);
const selectControlLayersIsHidden = createCanvasSelector((canvas) => canvas.controlLayers.isHidden);
const selectInpaintMasksIsHidden = createCanvasSelector((canvas) => canvas.inpaintMasks.isHidden);
const selectRegionalGuidanceIsHidden = createCanvasSelector((canvas) => canvas.regionalGuidance.isHidden);

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
  return createCanvasSelector((canvas) => {
    const entity = selectEntity(canvas, entityIdentifier);

    if (!entity) {
      return false;
    }
    return entity.objects.length > 0;
  });
};

export const selectWidth = createCanvasSelector((canvas) => canvas.bbox.rect.width);
export const selectHeight = createCanvasSelector((canvas) => canvas.bbox.rect.height);
export const selectAspectRatioID = createCanvasSelector((canvas) => canvas.bbox.aspectRatio.id);
export const selectAspectRatioValue = createCanvasSelector((canvas) => canvas.bbox.aspectRatio.value);
export const selectScaledSize = createSelector(selectBbox, (bbox) => bbox.scaledSize);
export const selectScaleMethod = createSelector(selectBbox, (bbox) => bbox.scaleMethod);
export const selectBboxRect = createSelector(selectBbox, (bbox) => bbox.rect);
export const selectBboxModelBase = createSelector(selectBbox, (bbox) => bbox.modelBase);

export const selectCanvasMetadata = createSelector(
  selectCanvasSlice,
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
export const selectNonRasterLayersIsHidden = createSelector(selectCanvasSlice, (canvas) => {
  return canvas.controlLayers.isHidden && canvas.inpaintMasks.isHidden && canvas.regionalGuidance.isHidden;
});
