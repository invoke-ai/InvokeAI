import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import { isEqual } from 'lodash-es';
import type { ImageDTO } from 'services/api/types';

import type { CanvasV2State, InitialImageEntity } from './types';
import { imageDTOToImageObject } from './types';

export const initialImageReducers = {
  iiRecalled: (state, action: PayloadAction<{ data: InitialImageEntity }>) => {
    const { data } = action.payload;
    state.initialImage = data;
    state.selectedEntityIdentifier = { type: 'initial_image', id: 'initial_image' };
  },
  iiIsEnabledToggled: (state) => {
    if (!state.initialImage) {
      return;
    }
    state.initialImage.isEnabled = !state.initialImage.isEnabled;
  },
  iiReset: (state) => {
    state.initialImage.imageObject = null;
  },
  iiImageChanged: (state, action: PayloadAction<{ imageDTO: ImageDTO }>) => {
    const { imageDTO } = action.payload;
    if (!state.initialImage) {
      return;
    }
    const newImageObject = imageDTOToImageObject(imageDTO);
    if (isEqual(newImageObject, state.initialImage.imageObject)) {
      return;
    }
    state.initialImage.bbox = null;
    state.initialImage.bboxNeedsUpdate = true;
    state.initialImage.isEnabled = true;
    state.initialImage.imageObject = newImageObject;
    state.selectedEntityIdentifier = { type: 'initial_image', id: 'initial_image' };
  },
} satisfies SliceCaseReducers<CanvasV2State>;
