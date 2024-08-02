import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import type {
  CanvasBrushLineState,
  CanvasV2State,
  Coordinate,
  CanvasEraserLineState,
  CanvasInpaintMaskState,
  CanvasRectState,
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
  imRecalled: (state, action: PayloadAction<{ data: CanvasInpaintMaskState }>) => {
    const { data } = action.payload;
    state.inpaintMask = data;
    state.selectedEntityIdentifier = { type: 'inpaint_mask', id: data.id };
  },
  imIsEnabledToggled: (state) => {
    state.inpaintMask.isEnabled = !state.inpaintMask.isEnabled;
  },
  imTranslated: (state, action: PayloadAction<{ position: Coordinate }>) => {
    const { position } = action.payload;
    state.inpaintMask.position = position;
  },
  imScaled: (state, action: PayloadAction<ScaleChangedArg>) => {
    const { scale, position } = action.payload;
    for (const obj of state.inpaintMask.objects) {
      if (obj.type === 'brush_line') {
        obj.points = obj.points.map((point) => point * scale);
        obj.strokeWidth *= scale;
      } else if (obj.type === 'eraser_line') {
        obj.points = obj.points.map((point) => point * scale);
        obj.strokeWidth *= scale;
      } else if (obj.type === 'rect') {
        obj.x *= scale;
        obj.y *= scale;
        obj.height *= scale;
        obj.width *= scale;
      }
    }
    state.inpaintMask.position = position;
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
  imBrushLineAdded: (state, action: PayloadAction<{ brushLine: CanvasBrushLineState }>) => {
    const { brushLine } = action.payload;
    state.inpaintMask.objects.push(brushLine);
    state.inpaintMask.bboxNeedsUpdate = true;
    state.layers.imageCache = null;
  },
  imEraserLineAdded: (state, action: PayloadAction<{ eraserLine: CanvasEraserLineState }>) => {
    const { eraserLine } = action.payload;
    state.inpaintMask.objects.push(eraserLine);
    state.inpaintMask.bboxNeedsUpdate = true;
    state.layers.imageCache = null;
  },
  imRectShapeAdded: (state, action: PayloadAction<{ rectShape: CanvasRectState }>) => {
    const { rectShape } = action.payload;
    state.inpaintMask.objects.push(rectShape);
    state.inpaintMask.bboxNeedsUpdate = true;
    state.layers.imageCache = null;
  },
} satisfies SliceCaseReducers<CanvasV2State>;
