import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import { moveOneToEnd, moveOneToStart, moveToEnd, moveToStart } from 'common/util/arrayUtils';
import { getBrushLineId, getEraserLineId, getRectShapeId } from 'features/controlLayers/konva/naming';
import type { IRect } from 'konva/lib/types';
import type { ImageDTO } from 'services/api/types';
import { assert } from 'tsafe';
import { v4 as uuidv4 } from 'uuid';

import type {
  BrushLineAddedArg,
  CanvasV2State,
  EraserLineAddedArg,
  ImageObjectAddedArg,
  LayerEntity,
  PointAddedToLineArg,
  RectShapeAddedArg,
} from './types';
import { imageDTOToImageObject, imageDTOToImageWithDims, isLine } from './types';

export const selectLayer = (state: CanvasV2State, id: string) => state.layers.entities.find((layer) => layer.id === id);
export const selectLayerOrThrow = (state: CanvasV2State, id: string) => {
  const layer = selectLayer(state, id);
  assert(layer, `Layer with id ${id} not found`);
  return layer;
};

export const layersReducers = {
  layerAdded: {
    reducer: (state, action: PayloadAction<{ id: string }>) => {
      const { id } = action.payload;
      state.layers.entities.push({
        id,
        type: 'layer',
        isEnabled: true,
        bbox: null,
        bboxNeedsUpdate: false,
        objects: [],
        opacity: 1,
        x: 0,
        y: 0,
      });
      state.selectedEntityIdentifier = { type: 'layer', id };
      state.layers.baseLayerImageCache = null;
    },
    prepare: () => ({ payload: { id: uuidv4() } }),
  },
  layerRecalled: (state, action: PayloadAction<{ data: LayerEntity }>) => {
    const { data } = action.payload;
    state.layers.entities.push(data);
    state.selectedEntityIdentifier = { type: 'layer', id: data.id };
    state.layers.baseLayerImageCache = null;
  },
  layerIsEnabledToggled: (state, action: PayloadAction<{ id: string }>) => {
    const { id } = action.payload;
    const layer = selectLayer(state, id);
    if (!layer) {
      return;
    }
    layer.isEnabled = !layer.isEnabled;
    state.layers.baseLayerImageCache = null;
  },
  layerTranslated: (state, action: PayloadAction<{ id: string; x: number; y: number }>) => {
    const { id, x, y } = action.payload;
    const layer = selectLayer(state, id);
    if (!layer) {
      return;
    }
    layer.x = x;
    layer.y = y;
    state.layers.baseLayerImageCache = null;
  },
  layerBboxChanged: (state, action: PayloadAction<{ id: string; bbox: IRect | null }>) => {
    const { id, bbox } = action.payload;
    const layer = selectLayer(state, id);
    if (!layer) {
      return;
    }
    layer.bbox = bbox;
    layer.bboxNeedsUpdate = false;
    if (bbox === null) {
      // TODO(psyche): Clear objects when bbox is cleared - right now this doesn't work bc bbox calculation for layers
      // doesn't work - always returns null
      // layer.objects = [];
    }
  },
  layerReset: (state, action: PayloadAction<{ id: string }>) => {
    const { id } = action.payload;
    const layer = selectLayer(state, id);
    if (!layer) {
      return;
    }
    layer.isEnabled = true;
    layer.objects = [];
    layer.bbox = null;
    layer.bboxNeedsUpdate = false;
    state.layers.baseLayerImageCache = null;
  },
  layerDeleted: (state, action: PayloadAction<{ id: string }>) => {
    const { id } = action.payload;
    state.layers.entities = state.layers.entities.filter((l) => l.id !== id);
    state.layers.baseLayerImageCache = null;
  },
  layerAllDeleted: (state) => {
    state.layers.entities = [];
    state.layers.baseLayerImageCache = null;
  },
  layerOpacityChanged: (state, action: PayloadAction<{ id: string; opacity: number }>) => {
    const { id, opacity } = action.payload;
    const layer = selectLayer(state, id);
    if (!layer) {
      return;
    }
    layer.opacity = opacity;
    state.layers.baseLayerImageCache = null;
  },
  layerMovedForwardOne: (state, action: PayloadAction<{ id: string }>) => {
    const { id } = action.payload;
    const layer = selectLayer(state, id);
    if (!layer) {
      return;
    }
    moveOneToEnd(state.layers.entities, layer);
    state.layers.baseLayerImageCache = null;
  },
  layerMovedToFront: (state, action: PayloadAction<{ id: string }>) => {
    const { id } = action.payload;
    const layer = selectLayer(state, id);
    if (!layer) {
      return;
    }
    moveToEnd(state.layers.entities, layer);
    state.layers.baseLayerImageCache = null;
  },
  layerMovedBackwardOne: (state, action: PayloadAction<{ id: string }>) => {
    const { id } = action.payload;
    const layer = selectLayer(state, id);
    if (!layer) {
      return;
    }
    moveOneToStart(state.layers.entities, layer);
    state.layers.baseLayerImageCache = null;
  },
  layerMovedToBack: (state, action: PayloadAction<{ id: string }>) => {
    const { id } = action.payload;
    const layer = selectLayer(state, id);
    if (!layer) {
      return;
    }
    moveToStart(state.layers.entities, layer);
    state.layers.baseLayerImageCache = null;
  },
  layerBrushLineAdded: {
    reducer: (state, action: PayloadAction<BrushLineAddedArg & { lineId: string }>) => {
      const { id, points, lineId, color, width, clip } = action.payload;
      const layer = selectLayer(state, id);
      if (!layer) {
        return;
      }

      layer.objects.push({
        id: getBrushLineId(id, lineId),
        type: 'brush_line',
        points,
        strokeWidth: width,
        color,
        clip,
      });
      layer.bboxNeedsUpdate = true;
      state.layers.baseLayerImageCache = null;
    },
    prepare: (payload: BrushLineAddedArg) => ({
      payload: { ...payload, lineId: uuidv4() },
    }),
  },
  layerEraserLineAdded: {
    reducer: (state, action: PayloadAction<EraserLineAddedArg & { lineId: string }>) => {
      const { id, points, lineId, width, clip } = action.payload;
      const layer = selectLayer(state, id);
      if (!layer) {
        return;
      }

      layer.objects.push({
        id: getEraserLineId(id, lineId),
        type: 'eraser_line',
        points,
        strokeWidth: width,
        clip,
      });
      layer.bboxNeedsUpdate = true;
      state.layers.baseLayerImageCache = null;
    },
    prepare: (payload: EraserLineAddedArg) => ({
      payload: { ...payload, lineId: uuidv4() },
    }),
  },
  layerLinePointAdded: (state, action: PayloadAction<PointAddedToLineArg>) => {
    const { id, point } = action.payload;
    const layer = selectLayer(state, id);
    if (!layer) {
      return;
    }
    const lastObject = layer.objects[layer.objects.length - 1];
    if (!lastObject || !isLine(lastObject)) {
      return;
    }
    lastObject.points.push(...point);
    layer.bboxNeedsUpdate = true;
    state.layers.baseLayerImageCache = null;
  },
  layerRectAdded: {
    reducer: (state, action: PayloadAction<RectShapeAddedArg & { rectId: string }>) => {
      const { id, rect, rectId, color } = action.payload;
      if (rect.height === 0 || rect.width === 0) {
        // Ignore zero-area rectangles
        return;
      }
      const layer = selectLayer(state, id);
      if (!layer) {
        return;
      }
      layer.objects.push({
        type: 'rect_shape',
        id: getRectShapeId(id, rectId),
        ...rect,
        color,
      });
      layer.bboxNeedsUpdate = true;
      state.layers.baseLayerImageCache = null;
    },
    prepare: (payload: RectShapeAddedArg) => ({ payload: { ...payload, rectId: uuidv4() } }),
  },
  layerImageAdded: {
    reducer: (state, action: PayloadAction<ImageObjectAddedArg & { objectId: string }>) => {
      const { id, objectId, imageDTO } = action.payload;
      const layer = selectLayer(state, id);
      if (!layer) {
        return;
      }
      layer.objects.push(imageDTOToImageObject(id, objectId, imageDTO));
      layer.bboxNeedsUpdate = true;
      state.layers.baseLayerImageCache = null;
    },
    prepare: (payload: ImageObjectAddedArg) => ({ payload: { ...payload, objectId: uuidv4() } }),
  },
  baseLayerImageCacheChanged: (state, action: PayloadAction<ImageDTO | null>) => {
    state.layers.baseLayerImageCache = action.payload ? imageDTOToImageWithDims(action.payload) : null;
  },
} satisfies SliceCaseReducers<CanvasV2State>;
