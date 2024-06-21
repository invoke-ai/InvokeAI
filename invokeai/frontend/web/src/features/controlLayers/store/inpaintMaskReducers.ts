import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import { getBrushLineId, getEraserLineId, getRectShapeId } from 'features/controlLayers/konva/naming';
import type { CanvasV2State, InpaintMaskEntity } from 'features/controlLayers/store/types';
import { imageDTOToImageWithDims } from 'features/controlLayers/store/types';
import type { IRect } from 'konva/lib/types';
import type { ImageDTO } from 'services/api/types';
import { v4 as uuidv4 } from 'uuid';

import type { BrushLineAddedArg, EraserLineAddedArg, PointAddedToLineArg, RectShapeAddedArg, RgbColor } from './types';
import { isLine } from './types';

export const inpaintMaskReducers = {
  imReset: (state) => {
    state.inpaintMask.objects = [];
    state.inpaintMask.bbox = null;
    state.inpaintMask.bboxNeedsUpdate = false;
    state.inpaintMask.imageCache = null;
  },
  imRecalled: (state, action: PayloadAction<{ data: InpaintMaskEntity }>) => {
    const { data } = action.payload;
    state.inpaintMask = data;
    state.selectedEntityIdentifier = { type: 'inpaint_mask', id: data.id };
  },
  imIsEnabledToggled: (state) => {
    state.inpaintMask.isEnabled = !state.inpaintMask.isEnabled;
  },
  imTranslated: (state, action: PayloadAction<{ x: number; y: number }>) => {
    const { x, y } = action.payload;
    state.inpaintMask.x = x;
    state.inpaintMask.y = y;
  },
  imBboxChanged: (state, action: PayloadAction<{ bbox: IRect | null }>) => {
    const { bbox } = action.payload;
    state.inpaintMask.bbox = bbox;
    state.inpaintMask.bboxNeedsUpdate = false;
  },
  imFillChanged: (state, action: PayloadAction<{ fill: RgbColor }>) => {
    const { fill } = action.payload;
    state.inpaintMask.fill = fill;
  },
  imImageCacheChanged: (state, action: PayloadAction<{ imageDTO: ImageDTO | null }>) => {
    const { imageDTO } = action.payload;
    state.inpaintMask.imageCache = imageDTO ? imageDTOToImageWithDims(imageDTO) : null;
  },
  imBrushLineAdded: {
    reducer: (state, action: PayloadAction<Omit<BrushLineAddedArg, 'id'> & { lineId: string }>) => {
      const { points, lineId, color, width, clip } = action.payload;
      state.inpaintMask.objects.push({
        id: getBrushLineId(state.inpaintMask.id, lineId),
        type: 'brush_line',
        points,
        strokeWidth: width,
        color,
        clip,
      });
      state.inpaintMask.bboxNeedsUpdate = true;
      state.inpaintMask.imageCache = null;
    },
    prepare: (payload: Omit<BrushLineAddedArg, 'id'>) => ({
      payload: { ...payload, lineId: uuidv4() },
    }),
  },
  imEraserLineAdded: {
    reducer: (state, action: PayloadAction<Omit<EraserLineAddedArg, 'id'> & { lineId: string }>) => {
      const { points, lineId, width, clip } = action.payload;
      state.inpaintMask.objects.push({
        id: getEraserLineId(state.inpaintMask.id, lineId),
        type: 'eraser_line',
        points,
        strokeWidth: width,
        clip,
      });
      state.inpaintMask.bboxNeedsUpdate = true;
      state.inpaintMask.imageCache = null;
    },
    prepare: (payload: Omit<EraserLineAddedArg, 'id'>) => ({
      payload: { ...payload, lineId: uuidv4() },
    }),
  },
  imLinePointAdded: (state, action: PayloadAction<Omit<PointAddedToLineArg, 'id'>>) => {
    const { point } = action.payload;
    const lastObject = state.inpaintMask.objects[state.inpaintMask.objects.length - 1];
    if (!lastObject || !isLine(lastObject)) {
      return;
    }
    lastObject.points.push(...point);
    state.inpaintMask.bboxNeedsUpdate = true;
    state.inpaintMask.imageCache = null;
  },
  imRectAdded: {
    reducer: (state, action: PayloadAction<Omit<RectShapeAddedArg, 'id'> & { rectId: string }>) => {
      const { rect, rectId, color } = action.payload;
      if (rect.height === 0 || rect.width === 0) {
        // Ignore zero-area rectangles
        return;
      }
      state.inpaintMask.objects.push({
        type: 'rect_shape',
        id: getRectShapeId(state.inpaintMask.id, rectId),
        ...rect,
        color,
      });
      state.inpaintMask.bboxNeedsUpdate = true;
      state.inpaintMask.imageCache = null;
    },
    prepare: (payload: Omit<RectShapeAddedArg, 'id'>) => ({ payload: { ...payload, rectId: uuidv4() } }),
  },
} satisfies SliceCaseReducers<CanvasV2State>;
