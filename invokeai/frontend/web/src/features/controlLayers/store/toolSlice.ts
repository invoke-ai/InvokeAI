import { createSlice, type PayloadAction } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import type { RgbaColor } from 'features/controlLayers/store/types';

export type ToolState = {
  invertScroll: boolean;
  brush: { width: number };
  eraser: { width: number };
  fill: RgbaColor;
};

const initialState: ToolState = {
  invertScroll: false,
  fill: { r: 31, g: 160, b: 224, a: 1 }, // invokeBlue.500
  brush: {
    width: 50,
  },
  eraser: {
    width: 50,
  },
};

export const toolSlice = createSlice({
  name: 'tool',
  initialState,
  reducers: {
    brushWidthChanged: (state, action: PayloadAction<number>) => {
      state.brush.width = Math.round(action.payload);
    },
    eraserWidthChanged: (state, action: PayloadAction<number>) => {
      state.eraser.width = Math.round(action.payload);
    },
    fillChanged: (state, action: PayloadAction<RgbaColor>) => {
      state.fill = action.payload;
    },
    invertScrollChanged: (state, action: PayloadAction<boolean>) => {
      state.invertScroll = action.payload;
    },
  },
});

export const { brushWidthChanged, eraserWidthChanged, fillChanged, invertScrollChanged } = toolSlice.actions;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrate = (state: any): any => {
  return state;
};

export const toolPersistConfig: PersistConfig<ToolState> = {
  name: toolSlice.name,
  initialState,
  migrate,
  persistDenylist: [],
};

export const selectToolSlice = (state: RootState) => state.tool;
