import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import type { CanvasInpaintMaskState, CanvasV2State } from 'features/controlLayers/store/types';

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
} satisfies SliceCaseReducers<CanvasV2State>;
