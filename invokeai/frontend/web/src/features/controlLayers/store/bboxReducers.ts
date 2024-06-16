import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import type { BoundingBoxScaleMethod, CanvasV2State, Dimensions } from 'features/controlLayers/store/types';
import { getScaledBoundingBoxDimensions } from 'features/controlLayers/util/getScaledBoundingBoxDimensions';
import { getOptimalDimension } from 'features/parameters/util/optimalDimension';
import type { IRect } from 'konva/lib/types';

export const bboxReducers = {
  scaledBboxChanged: (state, action: PayloadAction<Partial<Dimensions>>) => {
    const { width, height } = action.payload;
    state.bbox.scaledWidth = width ?? state.bbox.scaledWidth;
    state.bbox.scaledHeight = height ?? state.bbox.scaledHeight;
  },
  bboxScaleMethodChanged: (state, action: PayloadAction<BoundingBoxScaleMethod>) => {
    state.bbox.scaleMethod = action.payload;

    if (action.payload === 'auto') {
      const bboxDims = { width: state.bbox.width, height: state.bbox.height };
      const optimalDimension = getOptimalDimension(state.params.model);
      const scaledBboxDims = getScaledBoundingBoxDimensions(bboxDims, optimalDimension);
      state.bbox.scaledWidth = scaledBboxDims.width;
      state.bbox.scaledHeight = scaledBboxDims.height;
    }
  },
  bboxChanged: (state, action: PayloadAction<IRect>) => {
    const { x, y, width, height } = action.payload;
    state.bbox.x = x;
    state.bbox.y = y;
    state.bbox.width = width;
    state.bbox.height = height;

    if (state.bbox.scaleMethod === 'auto') {
      const bboxDims = { width: state.bbox.width, height: state.bbox.height };
      const optimalDimension = getOptimalDimension(state.params.model);
      const scaledBboxDims = getScaledBoundingBoxDimensions(bboxDims, optimalDimension);
      state.bbox.scaledWidth = scaledBboxDims.width;
      state.bbox.scaledHeight = scaledBboxDims.height;
    }
  },
} satisfies SliceCaseReducers<CanvasV2State>;
