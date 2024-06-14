import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { moveOneToEnd, moveOneToStart, moveToEnd, moveToStart } from 'common/util/arrayUtils';
import { getBrushLineId, getEraserLineId, getImageObjectId, getRectShapeId } from 'features/controlLayers/konva/naming';
import type { IRect } from 'konva/lib/types';
import { v4 as uuidv4 } from 'uuid';

import type {
  AddBrushLineArg,
  AddEraserLineArg,
  AddImageObjectArg,
  AddPointToLineArg,
  AddRectShapeArg,
  LayerData,
} from './types';
import { isLine } from './types';

type LayersState = {
  _version: 1;
  layers: LayerData[];
};

const initialState: LayersState = { _version: 1, layers: [] };
const selectLayer = (state: LayersState, id: string) => state.layers.find((layer) => layer.id === id);

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
    layerIsEnabledChanged: (state, action: PayloadAction<{ id: string; isEnabled: boolean }>) => {
      const { id, isEnabled } = action.payload;
      const layer = selectLayer(state, id);
      if (!layer) {
        return;
      }
      layer.isEnabled = isEnabled;
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
      reducer: (state, action: PayloadAction<AddBrushLineArg & { lineId: string }>) => {
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
      prepare: (payload: AddBrushLineArg) => ({
        payload: { ...payload, lineId: uuidv4() },
      }),
    },
    layerEraserLineAdded: {
      reducer: (state, action: PayloadAction<AddEraserLineArg & { lineId: string }>) => {
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
      prepare: (payload: AddEraserLineArg) => ({
        payload: { ...payload, lineId: uuidv4() },
      }),
    },
    layerLinePointAdded: (state, action: PayloadAction<AddPointToLineArg>) => {
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
      reducer: (state, action: PayloadAction<AddRectShapeArg & { rectId: string }>) => {
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
          x: rect.x - layer.x,
          y: rect.y - layer.y,
          width: rect.width,
          height: rect.height,
          color,
        });
        layer.bboxNeedsUpdate = true;
      },
      prepare: (payload: AddRectShapeArg) => ({ payload: { ...payload, rectId: uuidv4() } }),
    },
    layerImageAdded: {
      reducer: (state, action: PayloadAction<AddImageObjectArg & { imageId: string }>) => {
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
      prepare: (payload: AddImageObjectArg) => ({ payload: { ...payload, imageId: uuidv4() } }),
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
  layerIsEnabledChanged,
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
