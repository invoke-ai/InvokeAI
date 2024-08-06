import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { merge } from 'lodash-es';
import type { ImageDTO } from 'services/api/types';
import { assert } from 'tsafe';

import type { CanvasLayerState, CanvasV2State, ImageObjectAddedArg } from './types';
import { imageDTOToImageObject, imageDTOToImageWithDims } from './types';

export const selectLayer = (state: CanvasV2State, id: string) => state.layers.entities.find((layer) => layer.id === id);
export const selectLayerOrThrow = (state: CanvasV2State, id: string) => {
  const layer = selectLayer(state, id);
  assert(layer, `Layer with id ${id} not found`);
  return layer;
};

export const layersReducers = {
  layerAdded: {
    reducer: (state, action: PayloadAction<{ id: string; overrides?: Partial<CanvasLayerState> }>) => {
      const { id } = action.payload;
      const layer: CanvasLayerState = {
        id,
        type: 'layer',
        isEnabled: true,
        objects: [],
        opacity: 1,
        position: { x: 0, y: 0 },
      };
      merge(layer, action.payload.overrides);
      state.layers.entities.push(layer);
      state.selectedEntityIdentifier = { type: 'layer', id };
      state.layers.imageCache = null;
    },
    prepare: (payload: { overrides?: Partial<CanvasLayerState> }) => ({
      payload: { ...payload, id: getPrefixedId('layer') },
    }),
  },
  layerRecalled: (state, action: PayloadAction<{ data: CanvasLayerState }>) => {
    const { data } = action.payload;
    state.layers.entities.push(data);
    state.selectedEntityIdentifier = { type: 'layer', id: data.id };
    state.layers.imageCache = null;
  },
  layerAllDeleted: (state) => {
    state.layers.entities = [];
    state.layers.imageCache = null;
  },
  layerOpacityChanged: (state, action: PayloadAction<{ id: string; opacity: number }>) => {
    const { id, opacity } = action.payload;
    const layer = selectLayer(state, id);
    if (!layer) {
      return;
    }
    layer.opacity = opacity;
    state.layers.imageCache = null;
  },
  layerImageAdded: (
    state,
    action: PayloadAction<ImageObjectAddedArg & { objectId: string; pos?: { x: number; y: number } }>
  ) => {
    const { id, imageDTO, pos } = action.payload;
    const layer = selectLayer(state, id);
    if (!layer) {
      return;
    }
    const imageObject = imageDTOToImageObject(imageDTO);
    if (pos) {
      imageObject.x = pos.x;
      imageObject.y = pos.y;
    }
    layer.objects.push(imageObject);
    state.layers.imageCache = null;
  },
  layerImageCacheChanged: (state, action: PayloadAction<{ imageDTO: ImageDTO | null }>) => {
    const { imageDTO } = action.payload;
    state.layers.imageCache = imageDTO ? imageDTOToImageWithDims(imageDTO) : null;
  },
} satisfies SliceCaseReducers<CanvasV2State>;
