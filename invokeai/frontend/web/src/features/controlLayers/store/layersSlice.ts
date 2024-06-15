import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { moveOneToEnd, moveOneToStart, moveToEnd, moveToStart } from 'common/util/arrayUtils';
import { getBrushLineId, getEraserLineId, getImageObjectId, getRectShapeId } from 'features/controlLayers/konva/naming';
import type { IRect } from 'konva/lib/types';
import { assert } from 'tsafe';
import { v4 as uuidv4 } from 'uuid';

import type {
  BrushLineAddedArg,
  EraserLineAddedArg,
  ImageObjectAddedArg,
  LayerData,
  PointAddedToLineArg,
  RectShapeAddedArg,
} from './types';
import { isLine } from './types';

type LayersState = {
  _version: 1;
  layers: LayerData[];
};

const initialState: LayersState = { _version: 1, layers: [] };
export const selectLayer = (state: LayersState, id: string) => state.layers.find((layer) => layer.id === id);
export const selectLayerOrThrow = (state: LayersState, id: string) => {
  const layer = selectLayer(state, id);
  assert(layer, `Layer with id ${id} not found`);
  return layer;
};

export const layersSlice = createSlice({
  name: 'layers',
  initialState,
  reducers: {
    layerAdded: {
      reducer: (state, action: PayloadAction<{ id: string }>) => {
        const { id } = action.payload;
        state.layers.push({
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
      },
      prepare: () => ({ payload: { id: uuidv4() } }),
    },
    layerRecalled: (state, action: PayloadAction<{ data: LayerData }>) => {
      state.layers.push(action.payload.data);
    },
    layerIsEnabledToggled: (state, action: PayloadAction<{ id: string }>) => {
      const { id } = action.payload;
      const layer = selectLayer(state, id);
      if (!layer) {
        return;
      }
      layer.isEnabled = !layer.isEnabled;
    },
    layerTranslated: (state, action: PayloadAction<{ id: string; x: number; y: number }>) => {
      const { id, x, y } = action.payload;
      const layer = selectLayer(state, id);
      if (!layer) {
        return;
      }
      layer.x = x;
      layer.y = y;
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
        layer.objects = [];
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
    },
    layerDeleted: (state, action: PayloadAction<{ id: string }>) => {
      const { id } = action.payload;
      state.layers = state.layers.filter((l) => l.id !== id);
    },
    layerOpacityChanged: (state, action: PayloadAction<{ id: string; opacity: number }>) => {
      const { id, opacity } = action.payload;
      const layer = selectLayer(state, id);
      if (!layer) {
        return;
      }
      layer.opacity = opacity;
    },
    layerMovedForwardOne: (state, action: PayloadAction<{ id: string }>) => {
      const { id } = action.payload;
      const layer = selectLayer(state, id);
      if (!layer) {
        return;
      }
      moveOneToEnd(state.layers, layer);
    },
    layerMovedToFront: (state, action: PayloadAction<{ id: string }>) => {
      const { id } = action.payload;
      const layer = selectLayer(state, id);
      if (!layer) {
        return;
      }
      moveToEnd(state.layers, layer);
    },
    layerMovedBackwardOne: (state, action: PayloadAction<{ id: string }>) => {
      const { id } = action.payload;
      const layer = selectLayer(state, id);
      if (!layer) {
        return;
      }
      moveOneToStart(state.layers, layer);
    },
    layerMovedToBack: (state, action: PayloadAction<{ id: string }>) => {
      const { id } = action.payload;
      const layer = selectLayer(state, id);
      if (!layer) {
        return;
      }
      moveToStart(state.layers, layer);
    },
    layerBrushLineAdded: {
      reducer: (state, action: PayloadAction<BrushLineAddedArg & { lineId: string }>) => {
        const { id, points, lineId, color, width } = action.payload;
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
        });
        layer.bboxNeedsUpdate = true;
      },
      prepare: (payload: BrushLineAddedArg) => ({
        payload: { ...payload, lineId: uuidv4() },
      }),
    },
    layerEraserLineAdded: {
      reducer: (state, action: PayloadAction<EraserLineAddedArg & { lineId: string }>) => {
        const { id, points, lineId, width } = action.payload;
        const layer = selectLayer(state, id);
        if (!layer) {
          return;
        }

        layer.objects.push({
          id: getEraserLineId(id, lineId),
          type: 'eraser_line',
          points,
          strokeWidth: width,
        });
        layer.bboxNeedsUpdate = true;
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
      },
      prepare: (payload: RectShapeAddedArg) => ({ payload: { ...payload, rectId: uuidv4() } }),
    },
    layerImageAdded: {
      reducer: (state, action: PayloadAction<ImageObjectAddedArg & { imageId: string }>) => {
        const { id, imageId, imageDTO } = action.payload;
        const layer = selectLayer(state, id);
        if (!layer) {
          return;
        }
        const { width, height, image_name: name } = imageDTO;
        layer.objects.push({
          type: 'image',
          id: getImageObjectId(id, imageId),
          x: 0,
          y: 0,
          width,
          height,
          image: { width, height, name },
        });
        layer.bboxNeedsUpdate = true;
      },
      prepare: (payload: ImageObjectAddedArg) => ({ payload: { ...payload, imageId: uuidv4() } }),
    },
  },
});

export const {
  layerAdded,
  layerDeleted,
  layerReset,
  layerMovedForwardOne,
  layerMovedToFront,
  layerMovedBackwardOne,
  layerMovedToBack,
  layerIsEnabledToggled,
  layerOpacityChanged,
  layerTranslated,
  layerBboxChanged,
  layerBrushLineAdded,
  layerEraserLineAdded,
  layerLinePointAdded,
  layerRectAdded,
  layerImageAdded,
} = layersSlice.actions;

export const selectLayersSlice = (state: RootState) => state.layers;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrate = (state: any): any => {
  return state;
};

export const layersPersistConfig: PersistConfig<LayersState> = {
  name: layersSlice.name,
  initialState,
  migrate,
  persistDenylist: [],
};
