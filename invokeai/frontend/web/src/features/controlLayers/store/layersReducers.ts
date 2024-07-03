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
      state.layers.imageCache = null;
    },
    prepare: () => ({ payload: { id: uuidv4() } }),
  },
  layerRecalled: (state, action: PayloadAction<{ data: LayerEntity }>) => {
    const { data } = action.payload;
    state.layers.entities.push(data);
    state.selectedEntityIdentifier = { type: 'layer', id: data.id };
    state.layers.imageCache = null;
  },
  layerIsEnabledToggled: (state, action: PayloadAction<{ id: string }>) => {
    const { id } = action.payload;
    const layer = selectLayer(state, id);
    if (!layer) {
      return;
    }
    layer.isEnabled = !layer.isEnabled;
    state.layers.imageCache = null;
  },
  layerTranslated: (state, action: PayloadAction<{ id: string; x: number; y: number }>) => {
    const { id, x, y } = action.payload;
    const layer = selectLayer(state, id);
    if (!layer) {
      return;
    }
    layer.x = x;
    layer.y = y;
    state.layers.imageCache = null;
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
    state.layers.imageCache = null;
    layer.x = 0;
    layer.y = 0;
  },
  layerDeleted: (state, action: PayloadAction<{ id: string }>) => {
    const { id } = action.payload;
    state.layers.entities = state.layers.entities.filter((l) => l.id !== id);
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
  layerMovedForwardOne: (state, action: PayloadAction<{ id: string }>) => {
    const { id } = action.payload;
    const layer = selectLayer(state, id);
    if (!layer) {
      return;
    }
    moveOneToEnd(state.layers.entities, layer);
    state.layers.imageCache = null;
  },
  layerMovedToFront: (state, action: PayloadAction<{ id: string }>) => {
    const { id } = action.payload;
    const layer = selectLayer(state, id);
    if (!layer) {
      return;
    }
    moveToEnd(state.layers.entities, layer);
    state.layers.imageCache = null;
  },
  layerMovedBackwardOne: (state, action: PayloadAction<{ id: string }>) => {
    const { id } = action.payload;
    const layer = selectLayer(state, id);
    if (!layer) {
      return;
    }
    moveOneToStart(state.layers.entities, layer);
    state.layers.imageCache = null;
  },
  layerMovedToBack: (state, action: PayloadAction<{ id: string }>) => {
    const { id } = action.payload;
    const layer = selectLayer(state, id);
    if (!layer) {
      return;
    }
    moveToStart(state.layers.entities, layer);
    state.layers.imageCache = null;
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
      state.layers.imageCache = null;
    },
    prepare: (payload: BrushLineAddedArg) => ({
      payload: { ...payload, lineId: uuidv4() },
    }),
  },
  layerScaled: (state, action: PayloadAction<{ id: string; scale: number; x: number; y: number }>) => {
    const { id, scale, x, y } = action.payload;
    const layer = selectLayer(state, id);
    if (!layer) {
      return;
    }
    for (const obj of layer.objects) {
      if (obj.type === 'brush_line') {
        obj.points = obj.points.map((point) => point * scale);
        obj.strokeWidth *= scale;
      } else if (obj.type === 'eraser_line') {
        obj.points = obj.points.map((point) => point * scale);
        obj.strokeWidth *= scale;
      } else if (obj.type === 'rect_shape') {
        obj.x *= scale;
        obj.y *= scale;
        obj.height *= scale;
        obj.width *= scale;
      } else if (obj.type === 'image') {
        obj.x *= scale;
        obj.y *= scale;
        obj.height *= scale;
        obj.width *= scale;
      }
    }
    layer.x = x;
    layer.y = y;
    layer.bboxNeedsUpdate = true;
    state.layers.imageCache = null;
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
      state.layers.imageCache = null;
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
    state.layers.imageCache = null;
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
      state.layers.imageCache = null;
    },
    prepare: (payload: RectShapeAddedArg) => ({ payload: { ...payload, rectId: uuidv4() } }),
  },
  layerImageAdded: {
    reducer: (
      state,
      action: PayloadAction<ImageObjectAddedArg & { objectId: string; pos?: { x: number; y: number } }>
    ) => {
      const { id, objectId, imageDTO, pos } = action.payload;
      const layer = selectLayer(state, id);
      if (!layer) {
        return;
      }
      const imageObject = imageDTOToImageObject(id, objectId, imageDTO);
      if (pos) {
        imageObject.x = pos.x;
        imageObject.y = pos.y;
      }
      layer.objects.push(imageObject);
      layer.bboxNeedsUpdate = true;
      state.layers.imageCache = null;
    },
    prepare: (payload: ImageObjectAddedArg & { pos?: { x: number; y: number } }) => ({
      payload: { ...payload, objectId: uuidv4() },
    }),
  },
  layerImageCacheChanged: (state, action: PayloadAction<{ imageDTO: ImageDTO | null }>) => {
    const { imageDTO } = action.payload;
    state.layers.imageCache = imageDTO ? imageDTOToImageWithDims(imageDTO) : null;
  },
} satisfies SliceCaseReducers<CanvasV2State>;
