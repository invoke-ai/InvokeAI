import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import type {
  BrushLine,
  CanvasV2State,
  EraserLine,
  InpaintMaskEntity,
  RectShape,
  ScaleChangedArg,
} from 'features/controlLayers/store/types';
import { imageDTOToImageWithDims } from 'features/controlLayers/store/types';
import type { IRect } from 'konva/lib/types';
import type { ImageDTO } from 'services/api/types';

import type { RgbColor } from './types';

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
  imScaled: (state, action: PayloadAction<ScaleChangedArg>) => {
    const { scale, x, y } = action.payload;
    for (const obj of state.inpaintMask.objects) {
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
      }
    }
    state.inpaintMask.x = x;
    state.inpaintMask.y = y;
    state.inpaintMask.bboxNeedsUpdate = true;
    state.inpaintMask.imageCache = null;
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
  imBrushLineAdded: (state, action: PayloadAction<{ brushLine: BrushLine }>) => {
    const { brushLine } = action.payload;
    state.inpaintMask.objects.push(brushLine);
    state.inpaintMask.bboxNeedsUpdate = true;
    state.layers.imageCache = null;
  },
  imEraserLineAdded: (state, action: PayloadAction<{ eraserLine: EraserLine }>) => {
    const { eraserLine } = action.payload;
    state.inpaintMask.objects.push(eraserLine);
    state.inpaintMask.bboxNeedsUpdate = true;
    state.layers.imageCache = null;
  },
  imRectShapeAdded: (state, action: PayloadAction<{ rectShape: RectShape }>) => {
    const { rectShape } = action.payload;
    state.inpaintMask.objects.push(rectShape);
    state.inpaintMask.bboxNeedsUpdate = true;
    state.layers.imageCache = null;
  },
} satisfies SliceCaseReducers<CanvasV2State>;
