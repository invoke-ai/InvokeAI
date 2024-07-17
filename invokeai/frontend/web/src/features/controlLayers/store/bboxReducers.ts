import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import { deepClone } from 'common/util/deepClone';
import { roundDownToMultiple, roundToMultiple } from 'common/util/roundDownToMultiple';
import type { BoundingBoxScaleMethod, CanvasV2State, Dimensions } from 'features/controlLayers/store/types';
import { getScaledBoundingBoxDimensions } from 'features/controlLayers/util/getScaledBoundingBoxDimensions';
import { calculateNewSize } from 'features/parameters/components/DocumentSize/calculateNewSize';
import { ASPECT_RATIO_MAP, initialAspectRatioState } from 'features/parameters/components/DocumentSize/constants';
import type { AspectRatioID } from 'features/parameters/components/DocumentSize/types';
import { getOptimalDimension } from 'features/parameters/util/optimalDimension';
import type { IRect } from 'konva/lib/types';
import { pick } from 'lodash-es';

export const bboxReducers = {
  bboxScaledSizeChanged: (state, action: PayloadAction<Partial<Dimensions>>) => {
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
  bboxWidthChanged: (state, action: PayloadAction<{ width: number; updateAspectRatio?: boolean; clamp?: boolean }>) => {
    const { width, updateAspectRatio, clamp } = action.payload;
    state.bbox.rect.width = clamp ? Math.max(roundDownToMultiple(width, 8), 64) : width;

    if (state.bbox.aspectRatio.isLocked) {
      state.bbox.rect.height = roundToMultiple(state.bbox.rect.width / state.bbox.aspectRatio.value, 8);
    }

    if (updateAspectRatio || !state.bbox.aspectRatio.isLocked) {
      state.bbox.aspectRatio.value = state.bbox.rect.width / state.bbox.rect.height;
      state.bbox.aspectRatio.id = 'Free';
      state.bbox.aspectRatio.isLocked = false;
    }

    if (!state.session.isActive) {
      if (state.initialImage.imageObject) {
        state.initialImage.imageObject.width = state.bbox.rect.width;
        state.initialImage.imageObject.height = state.bbox.rect.height;
      }
    }
  },
  bboxHeightChanged: (
    state,
    action: PayloadAction<{ height: number; updateAspectRatio?: boolean; clamp?: boolean }>
  ) => {
    const { height, updateAspectRatio, clamp } = action.payload;

    state.bbox.rect.height = clamp ? Math.max(roundDownToMultiple(height, 8), 64) : height;

    if (state.bbox.aspectRatio.isLocked) {
      state.bbox.rect.width = roundToMultiple(state.bbox.rect.height * state.bbox.aspectRatio.value, 8);
    }

    if (updateAspectRatio || !state.bbox.aspectRatio.isLocked) {
      state.bbox.aspectRatio.value = state.bbox.rect.width / state.bbox.rect.height;
      state.bbox.aspectRatio.id = 'Free';
      state.bbox.aspectRatio.isLocked = false;
    }

    if (!state.session.isActive) {
      if (state.initialImage.imageObject) {
        state.initialImage.imageObject.width = state.bbox.rect.width;
        state.initialImage.imageObject.height = state.bbox.rect.height;
      }
    }
  },
  bboxAspectRatioLockToggled: (state) => {
    state.bbox.aspectRatio.isLocked = !state.bbox.aspectRatio.isLocked;
  },
  bboxAspectRatioIdChanged: (state, action: PayloadAction<{ id: AspectRatioID }>) => {
    const { id } = action.payload;
    state.bbox.aspectRatio.id = id;
    if (id === 'Free') {
      state.bbox.aspectRatio.isLocked = false;
    } else {
      state.bbox.aspectRatio.isLocked = true;
      state.bbox.aspectRatio.value = ASPECT_RATIO_MAP[id].ratio;
      const { width, height } = calculateNewSize(
        state.bbox.aspectRatio.value,
        state.bbox.rect.width * state.bbox.rect.height
      );
      state.bbox.rect.width = width;
      state.bbox.rect.height = height;
    }
    if (!state.session.isActive) {
      if (state.initialImage.imageObject) {
        state.initialImage.imageObject.width = state.bbox.rect.width;
        state.initialImage.imageObject.height = state.bbox.rect.height;
      }
    }
  },
  bboxDimensionsSwapped: (state) => {
    state.bbox.aspectRatio.value = 1 / state.bbox.aspectRatio.value;
    if (state.bbox.aspectRatio.id === 'Free') {
      const newWidth = state.bbox.rect.height;
      const newHeight = state.bbox.rect.width;
      state.bbox.rect.width = newWidth;
      state.bbox.rect.height = newHeight;
    } else {
      const { width, height } = calculateNewSize(
        state.bbox.aspectRatio.value,
        state.bbox.rect.width * state.bbox.rect.height
      );
      state.bbox.rect.width = width;
      state.bbox.rect.height = height;
      state.bbox.aspectRatio.id = ASPECT_RATIO_MAP[state.bbox.aspectRatio.id].inverseID;
    }
    if (!state.session.isActive) {
      if (state.initialImage.imageObject) {
        state.initialImage.imageObject.width = state.bbox.rect.width;
        state.initialImage.imageObject.height = state.bbox.rect.height;
      }
    }
  },
  bboxSizeOptimized: (state) => {
    const optimalDimension = getOptimalDimension(state.params.model);
    if (state.bbox.aspectRatio.isLocked) {
      const { width, height } = calculateNewSize(state.bbox.aspectRatio.value, optimalDimension ** 2);
      state.bbox.rect.width = width;
      state.bbox.rect.height = height;
    } else {
      state.bbox.aspectRatio = deepClone(initialAspectRatioState);
      state.bbox.rect.width = optimalDimension;
      state.bbox.rect.height = optimalDimension;
    }
    if (!state.session.isActive) {
      if (state.initialImage.imageObject) {
        state.initialImage.imageObject.width = state.bbox.rect.width;
        state.initialImage.imageObject.height = state.bbox.rect.height;
      }
    }
  },
} satisfies SliceCaseReducers<CanvasV2State>;
