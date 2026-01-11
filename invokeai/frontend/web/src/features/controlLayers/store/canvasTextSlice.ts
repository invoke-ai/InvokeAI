import { createSelector, createSlice, type PayloadAction } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import {
  TEXT_DEFAULT_ALIGNMENT,
  TEXT_DEFAULT_FONT_ID,
  TEXT_DEFAULT_FONT_SIZE,
  TEXT_DEFAULT_LINE_HEIGHT,
  TEXT_MAX_FONT_SIZE,
  TEXT_MAX_LINE_HEIGHT,
  TEXT_MIN_FONT_SIZE,
  TEXT_MIN_LINE_HEIGHT,
  type TextAlignment,
  type TextFontId,
  zTextAlignment,
  zTextFontId,
} from 'features/controlLayers/text/textConstants';
import { z } from 'zod';

const zCanvasTextSettingsState = z.object({
  fontId: zTextFontId,
  fontSize: z.number().int().min(TEXT_MIN_FONT_SIZE).max(TEXT_MAX_FONT_SIZE),
  bold: z.boolean(),
  italic: z.boolean(),
  underline: z.boolean(),
  strikethrough: z.boolean(),
  alignment: zTextAlignment,
  lineHeight: z.number().min(TEXT_MIN_LINE_HEIGHT).max(TEXT_MAX_LINE_HEIGHT),
});
export type CanvasTextSettingsState = z.infer<typeof zCanvasTextSettingsState>;

const getInitialState = (): CanvasTextSettingsState => ({
  fontId: TEXT_DEFAULT_FONT_ID,
  fontSize: TEXT_DEFAULT_FONT_SIZE,
  bold: false,
  italic: false,
  underline: false,
  strikethrough: false,
  alignment: TEXT_DEFAULT_ALIGNMENT,
  lineHeight: TEXT_DEFAULT_LINE_HEIGHT,
});

const slice = createSlice({
  name: 'canvasText',
  initialState: getInitialState(),
  reducers: {
    textFontChanged: (state, action: PayloadAction<TextFontId>) => {
      state.fontId = action.payload;
    },
    textFontSizeChanged: (state, action: PayloadAction<number>) => {
      const next = Math.round(action.payload);
      state.fontSize = Math.min(TEXT_MAX_FONT_SIZE, Math.max(TEXT_MIN_FONT_SIZE, next));
    },
    textBoldToggled: (state) => {
      state.bold = !state.bold;
    },
    textItalicToggled: (state) => {
      state.italic = !state.italic;
    },
    textUnderlineToggled: (state) => {
      state.underline = !state.underline;
    },
    textStrikethroughToggled: (state) => {
      state.strikethrough = !state.strikethrough;
    },
    textAlignmentChanged: (state, action: PayloadAction<TextAlignment>) => {
      state.alignment = action.payload;
    },
    textLineHeightChanged: (state, action: PayloadAction<number>) => {
      const next = action.payload;
      state.lineHeight = Math.min(TEXT_MAX_LINE_HEIGHT, Math.max(TEXT_MIN_LINE_HEIGHT, next));
    },
    textSettingsReset: () => {
      return getInitialState();
    },
  },
});

export const {
  textFontChanged,
  textFontSizeChanged,
  textBoldToggled,
  textItalicToggled,
  textUnderlineToggled,
  textStrikethroughToggled,
  textAlignmentChanged,
} = slice.actions;

export const canvasTextSliceConfig: SliceConfig<typeof slice> = {
  slice,
  schema: zCanvasTextSettingsState,
  getInitialState,
  persistConfig: {
    migrate: (state) => zCanvasTextSettingsState.parse(state),
  },
};

export const selectCanvasTextSlice = (state: RootState) => state.canvasText;
const createCanvasTextSelector = <T>(selector: (state: CanvasTextSettingsState) => T) =>
  createSelector(selectCanvasTextSlice, selector);

export const selectTextFontId = createCanvasTextSelector((state) => state.fontId);
export const selectTextFontSize = createCanvasTextSelector((state) => state.fontSize);
export const selectTextAlignment = createCanvasTextSelector((state) => state.alignment);
