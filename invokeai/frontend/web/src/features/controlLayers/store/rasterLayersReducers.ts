import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import { deepClone } from 'common/util/deepClone';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectEntity } from 'features/controlLayers/store/selectors';
import { merge } from 'lodash-es';

import type { CanvasControlLayerState, CanvasRasterLayerState, CanvasState, EntityIdentifierPayload } from './types';
import { getEntityIdentifier, initialControlNet } from './types';

export const rasterLayersReducers = {
  rasterLayerAdded: {
    reducer: (
      state,
      action: PayloadAction<{ id: string; overrides?: Partial<CanvasRasterLayerState>; isSelected?: boolean }>
    ) => {
      const { id, overrides, isSelected } = action.payload;
      const entity: CanvasRasterLayerState = {
        id,
        name: null,
        type: 'raster_layer',
        isEnabled: true,
        objects: [],
        opacity: 1,
        position: { x: 0, y: 0 },
      };
      merge(entity, overrides);
      state.rasterLayers.entities.push(entity);
      if (isSelected) {
        state.selectedEntityIdentifier = getEntityIdentifier(entity);
      }
    },
    prepare: (payload: { overrides?: Partial<CanvasRasterLayerState>; isSelected?: boolean }) => ({
      payload: { ...payload, id: getPrefixedId('raster_layer') },
    }),
  },
  rasterLayerRecalled: (state, action: PayloadAction<{ data: CanvasRasterLayerState }>) => {
    const { data } = action.payload;
    state.rasterLayers.entities.push(data);
    state.selectedEntityIdentifier = getEntityIdentifier(data);
  },
  rasterLayerConvertedToControlLayer: {
    reducer: (state, action: PayloadAction<EntityIdentifierPayload<{ newId: string }, 'raster_layer'>>) => {
      const { entityIdentifier, newId } = action.payload;
      const layer = selectEntity(state, entityIdentifier);
      if (!layer) {
        return;
      }

      // Convert the raster layer to control layer
      const controlLayerState: CanvasControlLayerState = {
        ...deepClone(layer),
        id: newId,
        type: 'control_layer',
        controlAdapter: deepClone(initialControlNet),
        withTransparencyEffect: true,
      };

      // Remove the raster layer
      state.rasterLayers.entities = state.rasterLayers.entities.filter((layer) => layer.id !== entityIdentifier.id);

      // Add the converted control layer
      state.controlLayers.entities.push(controlLayerState);

      state.selectedEntityIdentifier = { type: controlLayerState.type, id: controlLayerState.id };
    },
    prepare: (payload: EntityIdentifierPayload<void, 'raster_layer'>) => ({
      payload: { ...payload, newId: getPrefixedId('control_layer') },
    }),
  },
} satisfies SliceCaseReducers<CanvasState>;
