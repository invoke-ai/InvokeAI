import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import { deepClone } from 'common/util/deepClone';
import { roundDownToMultiple, roundToMultiple } from 'common/util/roundDownToMultiple';
import type { CanvasV2State } from 'features/controlLayers/store/types';
import { calculateNewSize } from 'features/parameters/components/DocumentSize/calculateNewSize';
import { ASPECT_RATIO_MAP, initialAspectRatioState } from 'features/parameters/components/DocumentSize/constants';
import type { AspectRatioID } from 'features/parameters/components/DocumentSize/types';
import { getOptimalDimension } from 'features/parameters/util/optimalDimension';

export const documentReducers = {
  documentWidthChanged: (
    state,
    action: PayloadAction<{ width: number; updateAspectRatio?: boolean; clamp?: boolean }>
  ) => {
    const { width, updateAspectRatio, clamp } = action.payload;
    state.document.width = clamp ? Math.max(roundDownToMultiple(width, 8), 64) : width;

    if (state.document.aspectRatio.isLocked) {
      state.document.height = roundToMultiple(state.document.width / state.document.aspectRatio.value, 8);
    }

    if (updateAspectRatio || !state.document.aspectRatio.isLocked) {
      state.document.aspectRatio.value = state.document.width / state.document.height;
      state.document.aspectRatio.id = 'Free';
      state.document.aspectRatio.isLocked = false;
    }

    if (!state.session.isActive) {
      state.bbox.width = state.document.width;
      state.bbox.height = state.document.height;
    }
  },
  documentHeightChanged: (
    state,
    action: PayloadAction<{ height: number; updateAspectRatio?: boolean; clamp?: boolean }>
  ) => {
    const { height, updateAspectRatio, clamp } = action.payload;

    state.document.height = clamp ? Math.max(roundDownToMultiple(height, 8), 64) : height;

    if (state.document.aspectRatio.isLocked) {
      state.document.width = roundToMultiple(state.document.height * state.document.aspectRatio.value, 8);
    }

    if (updateAspectRatio || !state.document.aspectRatio.isLocked) {
      state.document.aspectRatio.value = state.document.width / state.document.height;
      state.document.aspectRatio.id = 'Free';
      state.document.aspectRatio.isLocked = false;
    }

    if (!state.session.isActive) {
      state.bbox.width = state.document.width;
      state.bbox.height = state.document.height;
    }
  },
  documentAspectRatioLockToggled: (state) => {
    state.document.aspectRatio.isLocked = !state.document.aspectRatio.isLocked;
  },
  documentAspectRatioIdChanged: (state, action: PayloadAction<{ id: AspectRatioID }>) => {
    const { id } = action.payload;
    state.document.aspectRatio.id = id;
    if (id === 'Free') {
      state.document.aspectRatio.isLocked = false;
    } else {
      state.document.aspectRatio.isLocked = true;
      state.document.aspectRatio.value = ASPECT_RATIO_MAP[id].ratio;
      const { width, height } = calculateNewSize(
        state.document.aspectRatio.value,
        state.document.width * state.document.height
      );
      state.document.width = width;
      state.document.height = height;
    }
  },
  documentDimensionsSwapped: (state) => {
    state.document.aspectRatio.value = 1 / state.document.aspectRatio.value;
    if (state.document.aspectRatio.id === 'Free') {
      const newWidth = state.document.height;
      const newHeight = state.document.width;
      state.document.width = newWidth;
      state.document.height = newHeight;
    } else {
      const { width, height } = calculateNewSize(
        state.document.aspectRatio.value,
        state.document.width * state.document.height
      );
      state.document.width = width;
      state.document.height = height;
      state.document.aspectRatio.id = ASPECT_RATIO_MAP[state.document.aspectRatio.id].inverseID;
    }
  },
  documentSizeOptimized: (state) => {
    const optimalDimension = getOptimalDimension(state.params.model);
    if (state.document.aspectRatio.isLocked) {
      const { width, height } = calculateNewSize(state.document.aspectRatio.value, optimalDimension ** 2);
      state.document.width = width;
      state.document.height = height;
    } else {
      state.document.aspectRatio = deepClone(initialAspectRatioState);
      state.document.width = optimalDimension;
      state.document.height = optimalDimension;
    }
  },
} satisfies SliceCaseReducers<CanvasV2State>;
