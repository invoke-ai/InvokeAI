import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import type { BoundingBoxScaleMethod, CanvasV2State, Size } from 'features/controlLayers/store/types';
import { getScaledBoundingBoxDimensions } from 'features/controlLayers/util/getScaledBoundingBoxDimensions';
import { getOptimalDimension } from 'features/parameters/util/optimalDimension';
import type { IRect } from 'konva/lib/types';
import { pick } from 'lodash-es';

export const bboxReducers = {
  scaledBboxChanged: (state, action: PayloadAction<Partial<Size>>) => {
    state.layers.imageCache = null;
    state.bbox.scaledSize = { ...state.bbox.scaledSize, ...action.payload };
  },
  bboxScaleMethodChanged: (state, action: PayloadAction<BoundingBoxScaleMethod>) => {
    state.bbox.scaleMethod = action.payload;
    state.layers.imageCache = null;

    if (action.payload === 'auto') {
      const optimalDimension = getOptimalDimension(state.params.model);
      const size = pick(state.bbox.rect, 'width', 'height');
      state.bbox.scaledSize = getScaledBoundingBoxDimensions(size, optimalDimension);
    }
  },
  bboxChanged: (state, action: PayloadAction<IRect>) => {
    state.bbox.rect = action.payload;
    state.layers.imageCache = null;

    if (state.bbox.scaleMethod === 'auto') {
      const optimalDimension = getOptimalDimension(state.params.model);
      const size = pick(state.bbox.rect, 'width', 'height');
      state.bbox.scaledSize = getScaledBoundingBoxDimensions(size, optimalDimension);
    }
  },
} satisfies SliceCaseReducers<CanvasV2State>;
