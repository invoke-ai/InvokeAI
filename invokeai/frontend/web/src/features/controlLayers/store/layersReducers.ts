import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { isEqual, merge } from 'lodash-es';
import type { ControlNetModelConfig, T2IAdapterModelConfig } from 'services/api/types';
import { assert } from 'tsafe';

import type { CanvasLayerState, CanvasV2State, ControlModeV2, ControlNetConfig, Rect, T2IAdapterConfig } from './types';

export const selectLayer = (state: CanvasV2State, id: string) => state.layers.entities.find((layer) => layer.id === id);
export const selectLayerOrThrow = (state: CanvasV2State, id: string) => {
  const layer = selectLayer(state, id);
  assert(layer, `Layer with id ${id} not found`);
  return layer;
};

export const layersReducers = {
  layerAdded: {
    reducer: (
      state,
      action: PayloadAction<{ id: string; overrides?: Partial<CanvasLayerState>; isSelected?: boolean }>
    ) => {
      const { id, overrides, isSelected } = action.payload;
      const layer: CanvasLayerState = {
        id,
        type: 'layer',
        isEnabled: true,
        objects: [],
        opacity: 1,
        position: { x: 0, y: 0 },
        rasterizationCache: [],
        controlAdapter: null,
      };
      merge(layer, overrides);
      state.layers.entities.push(layer);
      if (isSelected) {
        state.selectedEntityIdentifier = { type: 'layer', id };
      }

      if (layer.objects.length > 0) {
        // This new layer will change the composite layer's image data. Invalidate the cache.
        state.layers.compositeRasterizationCache = [];
      }
    },
    prepare: (payload: { overrides?: Partial<CanvasLayerState>; isSelected?: boolean }) => ({
      payload: { ...payload, id: getPrefixedId('layer') },
    }),
  },
  layerRecalled: (state, action: PayloadAction<{ data: CanvasLayerState }>) => {
    const { data } = action.payload;
    state.layers.entities.push(data);
    state.selectedEntityIdentifier = { type: 'layer', id: data.id };
    if (data.objects.length > 0) {
      // This new layer will change the composite layer's image data. Invalidate the cache.
      state.layers.compositeRasterizationCache = [];
    }
  },
  layerAllDeleted: (state) => {
    state.layers.entities = [];
    state.layers.compositeRasterizationCache = [];
  },
  layerCompositeRasterized: (state, action: PayloadAction<{ imageName: string; rect: Rect }>) => {
    state.layers.compositeRasterizationCache = state.layers.compositeRasterizationCache.filter(
      (cache) => !isEqual(cache.rect, action.payload.rect)
    );
    state.layers.compositeRasterizationCache.push(action.payload);
  },
  layerUsedAsControlChanged: (
    state,
    action: PayloadAction<{ id: string; controlAdapter: ControlNetConfig | T2IAdapterConfig | null }>
  ) => {
    const { id, controlAdapter } = action.payload;
    const layer = selectLayer(state, id);
    if (!layer) {
      return;
    }
    layer.controlAdapter = controlAdapter;
    // The composite layer's image data will change when the layer is used as control (or not). Invalidate the cache.
    state.layers.compositeRasterizationCache = [];
  },
  layerControlAdapterModelChanged: (
    state,
    action: PayloadAction<{
      id: string;
      modelConfig: ControlNetModelConfig | T2IAdapterModelConfig | null;
    }>
  ) => {
    const { id, modelConfig } = action.payload;
    const layer = selectLayer(state, id);
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
  layerControlAdapterControlModeChanged: (state, action: PayloadAction<{ id: string; controlMode: ControlModeV2 }>) => {
    const { id, controlMode } = action.payload;
    const layer = selectLayer(state, id);
    if (!layer || !layer.controlAdapter || layer.controlAdapter.type !== 'controlnet') {
      return;
    }
    layer.controlAdapter.controlMode = controlMode;
  },
  layerControlAdapterWeightChanged: (state, action: PayloadAction<{ id: string; weight: number }>) => {
    const { id, weight } = action.payload;
    const layer = selectLayer(state, id);
    if (!layer || !layer.controlAdapter) {
      return;
    }
    layer.controlAdapter.weight = weight;
  },
  layerControlAdapterBeginEndStepPctChanged: (
    state,
    action: PayloadAction<{ id: string; beginEndStepPct: [number, number] }>
  ) => {
    const { id, beginEndStepPct } = action.payload;
    const layer = selectLayer(state, id);
    if (!layer || !layer.controlAdapter) {
      return;
    }
    layer.controlAdapter.beginEndStepPct = beginEndStepPct;
  },
} satisfies SliceCaseReducers<CanvasV2State>;
