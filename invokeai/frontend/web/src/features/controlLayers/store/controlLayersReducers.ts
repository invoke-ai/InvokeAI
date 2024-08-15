import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import { deepClone } from 'common/util/deepClone';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { merge, omit } from 'lodash-es';
import type { ControlNetModelConfig, T2IAdapterModelConfig } from 'services/api/types';
import { assert } from 'tsafe';

import type {
  CanvasControlLayerState,
  CanvasRasterLayerState,
  CanvasV2State,
  ControlModeV2,
  ControlNetConfig,
  T2IAdapterConfig,
} from './types';
import { initialControlNetV2 } from './types';

export const selectControlLayer = (state: CanvasV2State, id: string) =>
  state.controlLayers.entities.find((layer) => layer.id === id);
export const selectControlLayerOrThrow = (state: CanvasV2State, id: string) => {
  const layer = selectControlLayer(state, id);
  assert(layer, `Layer with id ${id} not found`);
  return layer;
};

export const controlLayersReducers = {
  controlLayerAdded: {
    reducer: (
      state,
      action: PayloadAction<{ id: string; overrides?: Partial<CanvasControlLayerState>; isSelected?: boolean }>
    ) => {
      const { id, overrides, isSelected } = action.payload;
      const layer: CanvasControlLayerState = {
        id,
        type: 'control_layer',
        isEnabled: true,
        objects: [],
        opacity: 1,
        position: { x: 0, y: 0 },
        rasterizationCache: [],
        controlAdapter: deepClone(initialControlNetV2),
      };
      merge(layer, overrides);
      state.controlLayers.entities.push(layer);
      if (isSelected) {
        state.selectedEntityIdentifier = { type: 'control_layer', id };
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
  controlLayerAllDeleted: (state) => {
    state.controlLayers.entities = [];
  },
  controlLayerConvertedToRasterLayer: {
    reducer: (state, action: PayloadAction<{ id: string; newId: string }>) => {
      const { id, newId } = action.payload;
      const layer = selectControlLayer(state, id);
      if (!layer) {
        return;
      }

      // Convert the raster layer to control layer
      const rasterLayerState: CanvasRasterLayerState = {
        ...omit(deepClone(layer), ['type', 'controlAdapter']),
        id: newId,
        type: 'raster_layer',
      };

      // Remove the control layer
      state.controlLayers.entities = state.controlLayers.entities.filter((layer) => layer.id !== id);

      // Add the new raster layer
      state.rasterLayers.entities.push(rasterLayerState);

      // The composite layer's image data will change when the control layer is converted to raster layer.
      state.rasterLayers.compositeRasterizationCache = [];

      state.selectedEntityIdentifier = { type: rasterLayerState.type, id: rasterLayerState.id };
    },
    prepare: (payload: { id: string }) => ({
      payload: { ...payload, newId: getPrefixedId('raster_layer') },
    }),
  },
  controlLayerModelChanged: (
    state,
    action: PayloadAction<{
      id: string;
      modelConfig: ControlNetModelConfig | T2IAdapterModelConfig | null;
    }>
  ) => {
    const { id, modelConfig } = action.payload;
    const layer = selectControlLayer(state, id);
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
  controlLayerControlModeChanged: (state, action: PayloadAction<{ id: string; controlMode: ControlModeV2 }>) => {
    const { id, controlMode } = action.payload;
    const layer = selectControlLayer(state, id);
    if (!layer || !layer.controlAdapter || layer.controlAdapter.type !== 'controlnet') {
      return;
    }
    layer.controlAdapter.controlMode = controlMode;
  },
  controlLayerWeightChanged: (state, action: PayloadAction<{ id: string; weight: number }>) => {
    const { id, weight } = action.payload;
    const layer = selectControlLayer(state, id);
    if (!layer || !layer.controlAdapter) {
      return;
    }
    layer.controlAdapter.weight = weight;
  },
  controlLayerBeginEndStepPctChanged: (
    state,
    action: PayloadAction<{ id: string; beginEndStepPct: [number, number] }>
  ) => {
    const { id, beginEndStepPct } = action.payload;
    const layer = selectControlLayer(state, id);
    if (!layer || !layer.controlAdapter) {
      return;
    }
    layer.controlAdapter.beginEndStepPct = beginEndStepPct;
  },
} satisfies SliceCaseReducers<CanvasV2State>;
