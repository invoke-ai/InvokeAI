import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import type { CanvasInpaintMaskState, CanvasV2State } from 'features/controlLayers/store/types';
import { imageDTOToImageWithDims } from 'features/controlLayers/store/types';
import type { ImageDTO } from 'services/api/types';

import type { RgbColor } from './types';

export const inpaintMaskReducers = {
  imRecalled: (state, action: PayloadAction<{ data: CanvasInpaintMaskState }>) => {
    const { data } = action.payload;
    state.inpaintMask = data;
    state.selectedEntityIdentifier = { type: 'inpaint_mask', id: data.id };
  },
  imFillChanged: (state, action: PayloadAction<{ fill: RgbColor }>) => {
    const { fill } = action.payload;
    state.inpaintMask.fill = fill;
  },
  imImageCacheChanged: (state, action: PayloadAction<{ imageDTO: ImageDTO | null }>) => {
    const { imageDTO } = action.payload;
    state.inpaintMask.imageCache = imageDTO ? imageDTOToImageWithDims(imageDTO) : null;
  },
} satisfies SliceCaseReducers<CanvasV2State>;
