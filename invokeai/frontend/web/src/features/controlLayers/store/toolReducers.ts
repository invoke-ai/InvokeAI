import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import type { CanvasV2State, RgbaColor, Tool } from 'features/controlLayers/store/types';

export const toolReducers = {
  brushWidthChanged: (state, action: PayloadAction<number>) => {
    state.tool.brush.width = Math.round(action.payload);
  },
  eraserWidthChanged: (state, action: PayloadAction<number>) => {
    state.tool.eraser.width = Math.round(action.payload);
  },
  fillChanged: (state, action: PayloadAction<RgbaColor>) => {
    state.tool.fill = action.payload;
  },
  invertScrollChanged: (state, action: PayloadAction<boolean>) => {
    state.tool.invertScroll = action.payload;
  },
  toolChanged: (state, action: PayloadAction<Tool>) => {
    state.tool.selected = action.payload;
  },
  toolBufferChanged: (state, action: PayloadAction<Tool | null>) => {
    state.tool.selectedBuffer = action.payload;
  },
} satisfies SliceCaseReducers<CanvasV2State>;
