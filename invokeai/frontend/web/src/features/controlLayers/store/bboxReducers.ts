import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import type { BoundingBoxScaleMethod, CanvasV2State, Dimensions } from 'features/controlLayers/store/types';
import { getScaledBoundingBoxDimensions } from 'features/controlLayers/util/getScaledBoundingBoxDimensions';
import { getOptimalDimension } from 'features/parameters/util/optimalDimension';
import type { IRect } from 'konva/lib/types';
import { pick } from 'lodash-es';

export const bboxReducers = {
  scaledBboxChanged: (state, action: PayloadAction<Partial<Dimensions>>) => {
    state.layers.imageCache = null;
    state.bbox.scaledSize = { ...state.bbox.scaledSize, ...action.payload };
  },
  bboxScaleMethodChanged: (state, action: PayloadAction<BoundingBoxScaleMethod>) => {
    state.bbox.scaleMethod = action.payload;
    state.layers.imageCache = null;

    if (action.payload === 'auto') {
      const optimalDimension = getOptimalDimension(state.params.model);
      const size = pick(state.bbox, 'width', 'height');
      state.bbox.scaledSize = getScaledBoundingBoxDimensions(size, optimalDimension);
    }
  },
  bboxChanged: (state, action: PayloadAction<IRect>) => {
    state.bbox = { ...state.bbox, ...action.payload };
    state.layers.imageCache = null;

    if (state.bbox.scaleMethod === 'auto') {
      const optimalDimension = getOptimalDimension(state.params.model);
      const size = pick(state.bbox, 'width', 'height');
      state.bbox.scaledSize = getScaledBoundingBoxDimensions(size, optimalDimension);
    }
  },
} satisfies SliceCaseReducers<CanvasV2State>;
