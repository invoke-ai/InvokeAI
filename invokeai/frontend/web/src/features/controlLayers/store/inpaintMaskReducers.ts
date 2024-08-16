import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import type { CanvasInpaintMaskState, CanvasV2State, FillStyle } from 'features/controlLayers/store/types';
import type { RgbColor } from 'react-colorful';

export const inpaintMaskReducers = {
  imRecalled: (state, action: PayloadAction<{ data: CanvasInpaintMaskState }>) => {
    const { data } = action.payload;
    state.inpaintMask = data;
    state.selectedEntityIdentifier = { type: 'inpaint_mask', id: data.id };
  },
  imFillColorChanged: (state, action: PayloadAction<{ color: RgbColor }>) => {
    const { color } = action.payload;
    state.inpaintMask.fill.color = color;
  },
  imFillStyleChanged: (state, action: PayloadAction<{ style: FillStyle }>) => {
    const { style } = action.payload;
    state.inpaintMask.fill.style = style;
  },
} satisfies SliceCaseReducers<CanvasV2State>;
