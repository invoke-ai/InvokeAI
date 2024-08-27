import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import { deepClone } from 'common/util/deepClone';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectEntity } from 'features/controlLayers/store/selectors';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { merge, omit } from 'lodash-es';
import type { ControlNetModelConfig, T2IAdapterModelConfig } from 'services/api/types';

import type {
  CanvasControlLayerState,
  CanvasRasterLayerState,
  CanvasState,
  ControlModeV2,
  ControlNetConfig,
  EntityIdentifierPayload,
  T2IAdapterConfig,
} from './types';
import { getEntityIdentifier, initialControlNet } from './types';

export const controlLayersReducers = {
  controlLayerAdded: {
    reducer: (
      state,
      action: PayloadAction<{ id: string; overrides?: Partial<CanvasControlLayerState>; isSelected?: boolean }>
    ) => {
      const { id, overrides, isSelected } = action.payload;
      const entity: CanvasControlLayerState = {
        id,
        name: null,
        type: 'control_layer',
        isEnabled: true,
        withTransparencyEffect: true,
        objects: [],
        opacity: 1,
        position: { x: 0, y: 0 },
        controlAdapter: deepClone(initialControlNet),
      };
      merge(entity, overrides);
      state.controlLayers.entities.push(entity);
      if (isSelected) {
        state.selectedEntityIdentifier = getEntityIdentifier(entity);
      }
    },
    prepare: (payload: { overrides?: Partial<CanvasControlLayerState>; isSelected?: boolean }) => ({
      payload: { ...payload, id: getPrefixedId('control_layer') },
    }),
  },
  controlLayerRecalled: (state, action: PayloadAction<{ data: CanvasControlLayerState }>) => {
    const { data } = action.payload;
    state.controlLayers.entities.push(data);
    state.selectedEntityIdentifier = { type: 'control_layer', id: data.id };
  },
  controlLayerConvertedToRasterLayer: {
    reducer: (state, action: PayloadAction<EntityIdentifierPayload<{ newId: string }, 'control_layer'>>) => {
      const { entityIdentifier, newId } = action.payload;
      const layer = selectEntity(state, entityIdentifier);
      if (!layer) {
        return;
      }

      // Convert the raster layer to control layer
      const rasterLayerState: CanvasRasterLayerState = {
        ...omit(deepClone(layer), ['type', 'controlAdapter', 'withTransparencyEffect']),
        id: newId,
        type: 'raster_layer',
      };

      // Remove the control layer
      state.controlLayers.entities = state.controlLayers.entities.filter((layer) => layer.id !== entityIdentifier.id);

      // Add the new raster layer
      state.rasterLayers.entities.push(rasterLayerState);

      state.selectedEntityIdentifier = { type: rasterLayerState.type, id: rasterLayerState.id };
    },
    prepare: (payload: EntityIdentifierPayload<void, 'control_layer'>) => ({
      payload: { ...payload, newId: getPrefixedId('raster_layer') },
    }),
  },
  controlLayerModelChanged: (
    state,
    action: PayloadAction<
      EntityIdentifierPayload<
        {
          modelConfig: ControlNetModelConfig | T2IAdapterModelConfig | null;
        },
        'control_layer'
      >
    >
  ) => {
    const { entityIdentifier, modelConfig } = action.payload;
    const layer = selectEntity(state, entityIdentifier);
    if (!layer || !layer.controlAdapter) {
      return;
    }
    if (!modelConfig) {
      layer.controlAdapter.model = null;
      return;
    }
    layer.controlAdapter.model = zModelIdentifierField.parse(modelConfig);

    // We may need to convert the CA to match the model
    if (layer.controlAdapter.type === 't2i_adapter' && layer.controlAdapter.model.type === 'controlnet') {
      // Converting from T2I Adapter to ControlNet - add `controlMode`
      const controlNetConfig: ControlNetConfig = {
        ...layer.controlAdapter,
        type: 'controlnet',
        controlMode: 'balanced',
      };
      layer.controlAdapter = controlNetConfig;
    } else if (layer.controlAdapter.type === 'controlnet' && layer.controlAdapter.model.type === 't2i_adapter') {
      // Converting from ControlNet to T2I Adapter - remove `controlMode`
      const { controlMode: _, ...rest } = layer.controlAdapter;
      const t2iAdapterConfig: T2IAdapterConfig = { ...rest, type: 't2i_adapter' };
      layer.controlAdapter = t2iAdapterConfig;
    }
  },
  controlLayerControlModeChanged: (
    state,
    action: PayloadAction<EntityIdentifierPayload<{ controlMode: ControlModeV2 }, 'control_layer'>>
  ) => {
    const { entityIdentifier, controlMode } = action.payload;
    const layer = selectEntity(state, entityIdentifier);
    if (!layer || !layer.controlAdapter || layer.controlAdapter.type !== 'controlnet') {
      return;
    }
    layer.controlAdapter.controlMode = controlMode;
  },
  controlLayerWeightChanged: (
    state,
    action: PayloadAction<EntityIdentifierPayload<{ weight: number }, 'control_layer'>>
  ) => {
    const { entityIdentifier, weight } = action.payload;
    const layer = selectEntity(state, entityIdentifier);
    if (!layer || !layer.controlAdapter) {
      return;
    }
    layer.controlAdapter.weight = weight;
  },
  controlLayerBeginEndStepPctChanged: (
    state,
    action: PayloadAction<EntityIdentifierPayload<{ beginEndStepPct: [number, number] }, 'control_layer'>>
  ) => {
    const { entityIdentifier, beginEndStepPct } = action.payload;
    const layer = selectEntity(state, entityIdentifier);
    if (!layer || !layer.controlAdapter) {
      return;
    }
    layer.controlAdapter.beginEndStepPct = beginEndStepPct;
  },
  controlLayerWithTransparencyEffectToggled: (
    state,
    action: PayloadAction<EntityIdentifierPayload<void, 'control_layer'>>
  ) => {
    const { entityIdentifier } = action.payload;
    const layer = selectEntity(state, entityIdentifier);
    if (!layer) {
      return;
    }
    layer.withTransparencyEffect = !layer.withTransparencyEffect;
  },
} satisfies SliceCaseReducers<CanvasState>;
