import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import type {
  CanvasBrushLineState,
  CanvasEraserLineState,
  CanvasInpaintMaskState,
  CanvasRectState,
  CanvasV2State,
  Coordinate,
  EntityRasterizedArg,
} from 'features/controlLayers/store/types';
import { imageDTOToImageWithDims } from 'features/controlLayers/store/types';
import type { ImageDTO } from 'services/api/types';

import type { RgbColor } from './types';

export const inpaintMaskReducers = {
  imReset: (state) => {
    state.inpaintMask.objects = [];
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
  imMoved: (state, action: PayloadAction<{ position: Coordinate }>) => {
    const { position } = action.payload;
    state.inpaintMask.position = position;
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
    state.layers.imageCache = null;
  },
  imEraserLineAdded: (state, action: PayloadAction<{ eraserLine: CanvasEraserLineState }>) => {
    const { eraserLine } = action.payload;
    state.inpaintMask.objects.push(eraserLine);
    state.layers.imageCache = null;
  },
  imRectAdded: (state, action: PayloadAction<{ rect: CanvasRectState }>) => {
    const { rect } = action.payload;
    state.inpaintMask.objects.push(rect);
    state.layers.imageCache = null;
  },
  inpaintMaskRasterized: (state, action: PayloadAction<EntityRasterizedArg>) => {
    const { imageObject, position } = action.payload;
    state.inpaintMask.objects = [imageObject];
    state.inpaintMask.position = position;
    state.inpaintMask.imageCache = null;
  },
} satisfies SliceCaseReducers<CanvasV2State>;
