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
    state.document.rect.width = clamp ? Math.max(roundDownToMultiple(width, 8), 64) : width;

    if (state.document.aspectRatio.isLocked) {
      state.document.rect.height = roundToMultiple(state.document.rect.width / state.document.aspectRatio.value, 8);
    }

    if (updateAspectRatio || !state.document.aspectRatio.isLocked) {
      state.document.aspectRatio.value = state.document.rect.width / state.document.rect.height;
      state.document.aspectRatio.id = 'Free';
      state.document.aspectRatio.isLocked = false;
    }

    if (!state.session.isActive) {
      state.bbox.rect.width = state.document.rect.width;
      state.bbox.rect.height = state.document.rect.height;

      if (state.initialImage.imageObject) {
        state.initialImage.imageObject.width = state.document.rect.width;
        state.initialImage.imageObject.height = state.document.rect.height;
      }
    }
  },
  documentHeightChanged: (
    state,
    action: PayloadAction<{ height: number; updateAspectRatio?: boolean; clamp?: boolean }>
  ) => {
    const { height, updateAspectRatio, clamp } = action.payload;

    state.document.rect.height = clamp ? Math.max(roundDownToMultiple(height, 8), 64) : height;

    if (state.document.aspectRatio.isLocked) {
      state.document.rect.width = roundToMultiple(state.document.rect.height * state.document.aspectRatio.value, 8);
    }

    if (updateAspectRatio || !state.document.aspectRatio.isLocked) {
      state.document.aspectRatio.value = state.document.rect.width / state.document.rect.height;
      state.document.aspectRatio.id = 'Free';
      state.document.aspectRatio.isLocked = false;
    }

    if (!state.session.isActive) {
      state.bbox.rect.width = state.document.rect.width;
      state.bbox.rect.height = state.document.rect.height;

      if (state.initialImage.imageObject) {
        state.initialImage.imageObject.width = state.document.rect.width;
        state.initialImage.imageObject.height = state.document.rect.height;
      }
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
        state.document.rect.width * state.document.rect.height
      );
      state.document.rect.width = width;
      state.document.rect.height = height;
    }
    if (!state.session.isActive) {
      state.bbox.rect.width = state.document.rect.width;
      state.bbox.rect.height = state.document.rect.height;

      if (state.initialImage.imageObject) {
        state.initialImage.imageObject.width = state.document.rect.width;
        state.initialImage.imageObject.height = state.document.rect.height;
      }
    }
  },
  documentDimensionsSwapped: (state) => {
    state.document.aspectRatio.value = 1 / state.document.aspectRatio.value;
    if (state.document.aspectRatio.id === 'Free') {
      const newWidth = state.document.rect.height;
      const newHeight = state.document.rect.width;
      state.document.rect.width = newWidth;
      state.document.rect.height = newHeight;
    } else {
      const { width, height } = calculateNewSize(
        state.document.aspectRatio.value,
        state.document.rect.width * state.document.rect.height
      );
      state.document.rect.width = width;
      state.document.rect.height = height;
      state.document.aspectRatio.id = ASPECT_RATIO_MAP[state.document.aspectRatio.id].inverseID;
    }
    if (!state.session.isActive) {
      state.bbox.rect.width = state.document.rect.width;
      state.bbox.rect.height = state.document.rect.height;

      if (state.initialImage.imageObject) {
        state.initialImage.imageObject.width = state.document.rect.width;
        state.initialImage.imageObject.height = state.document.rect.height;
      }
    }
  },
  documentSizeOptimized: (state) => {
    const optimalDimension = getOptimalDimension(state.params.model);
    if (state.document.aspectRatio.isLocked) {
      const { width, height } = calculateNewSize(state.document.aspectRatio.value, optimalDimension ** 2);
      state.document.rect.width = width;
      state.document.rect.height = height;
    } else {
      state.document.aspectRatio = deepClone(initialAspectRatioState);
      state.document.rect.width = optimalDimension;
      state.document.rect.height = optimalDimension;
    }
    if (!state.session.isActive) {
      state.bbox.rect.width = state.document.rect.width;
      state.bbox.rect.height = state.document.rect.height;

      if (state.initialImage.imageObject) {
        state.initialImage.imageObject.width = state.document.rect.width;
        state.initialImage.imageObject.height = state.document.rect.height;
      }
    }
  },
} satisfies SliceCaseReducers<CanvasV2State>;
