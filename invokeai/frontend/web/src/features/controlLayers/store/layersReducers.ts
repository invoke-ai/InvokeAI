import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import { moveOneToEnd, moveOneToStart, moveToEnd, moveToStart } from 'common/util/arrayUtils';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { IRect } from 'konva/lib/types';
import { merge } from 'lodash-es';
import type { ImageDTO } from 'services/api/types';
import { assert } from 'tsafe';

import type {
  BrushLine,
  CanvasV2State,
  Coordinate,
  EraserLine,
  ImageObject,
  ImageObjectAddedArg,
  LayerEntity,
  PositionChangedArg,
  RectShape,
  ScaleChangedArg,
} from './types';
import { imageDTOToImageObject, imageDTOToImageWithDims } from './types';

export const selectLayer = (state: CanvasV2State, id: string) => state.layers.entities.find((layer) => layer.id === id);
export const selectLayerOrThrow = (state: CanvasV2State, id: string) => {
  const layer = selectLayer(state, id);
  assert(layer, `Layer with id ${id} not found`);
  return layer;
};

export const layersReducers = {
  layerAdded: {
    reducer: (state, action: PayloadAction<{ id: string; overrides?: Partial<LayerEntity> }>) => {
      const { id } = action.payload;
      const layer: LayerEntity = {
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
    prepare: (payload: { overrides?: Partial<LayerEntity> }) => ({
      payload: { ...payload, id: getPrefixedId('layer') },
    }),
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
  layerTranslated: (state, action: PayloadAction<PositionChangedArg>) => {
    const { id, position } = action.payload;
    const layer = selectLayer(state, id);
    if (!layer) {
      return;
    }
    layer.position = position;
    state.layers.imageCache = null;
  },
  layerBboxChanged: (state, action: PayloadAction<{ id: string; bbox: IRect | null }>) => {
    const { id, bbox } = action.payload;
    const layer = selectLayer(state, id);
    if (!layer) {
      return;
    }
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
    state.layers.imageCache = null;
    layer.position = { x: 0, y: 0 };
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
  layerBrushLineAdded: (state, action: PayloadAction<{ id: string; brushLine: BrushLine }>) => {
    const { id, brushLine } = action.payload;
    const layer = selectLayer(state, id);
    if (!layer) {
      return;
    }

    layer.objects.push(brushLine);
    state.layers.imageCache = null;
  },
  layerEraserLineAdded: (state, action: PayloadAction<{ id: string; eraserLine: EraserLine }>) => {
    const { id, eraserLine } = action.payload;
    const layer = selectLayer(state, id);
    if (!layer) {
      return;
    }

    layer.objects.push(eraserLine);
    state.layers.imageCache = null;
  },
  layerRectShapeAdded: (state, action: PayloadAction<{ id: string; rectShape: RectShape }>) => {
    const { id, rectShape } = action.payload;
    const layer = selectLayer(state, id);
    if (!layer) {
      return;
    }

    layer.objects.push(rectShape);
    state.layers.imageCache = null;
  },
  layerScaled: (state, action: PayloadAction<ScaleChangedArg>) => {
    const { id, scale, position } = action.payload;
    const layer = selectLayer(state, id);
    if (!layer) {
      return;
    }
    for (const obj of layer.objects) {
      if (obj.type === 'brush_line') {
        obj.points = obj.points.map((point) => Math.round(point * scale));
        obj.strokeWidth = Math.round(obj.strokeWidth * scale);
      } else if (obj.type === 'eraser_line') {
        obj.points = obj.points.map((point) => Math.round(point * scale));
        obj.strokeWidth = Math.round(obj.strokeWidth * scale);
      } else if (obj.type === 'rect_shape') {
        obj.x = Math.round(obj.x * scale);
        obj.y = Math.round(obj.y * scale);
        obj.height = Math.round(obj.height * scale);
        obj.width = Math.round(obj.width * scale);
      } else if (obj.type === 'image') {
        obj.x = Math.round(obj.x * scale);
        obj.y = Math.round(obj.y * scale);
        obj.height = Math.round(obj.height * scale);
        obj.width = Math.round(obj.width * scale);
      }
    }
    layer.position.x = Math.round(position.x);
    layer.position.y = Math.round(position.y);
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
  layerRasterized: (state, action: PayloadAction<{ id: string; imageObject: ImageObject; position: Coordinate }>) => {
    const { id, imageObject, position } = action.payload;
    const layer = selectLayer(state, id);
    if (!layer) {
      return;
    }
    layer.objects = [imageObject];
    layer.position = position;
    state.layers.imageCache = null;
  },
} satisfies SliceCaseReducers<CanvasV2State>;

const scalePoints = (points: number[], scaleX: number, scaleY: number) => {
  const newPoints: number[] = [];
  for (let i = 0; i < points.length; i += 2) {
    newPoints.push(points[i]! * scaleX);
    newPoints.push(points[i + 1]! * scaleY);
  }
  return newPoints;
};
