import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import z from 'zod';

const zChangeBoardModalState = z.object({
  isModalOpen: z.boolean().default(false),
  image_names: z.array(z.string()).default(() => []),
  video_names: z.array(z.string()).default(() => []),
});
type ChangeBoardModalState = z.infer<typeof zChangeBoardModalState>;

const getInitialState = (): ChangeBoardModalState => zChangeBoardModalState.parse({});

const slice = createSlice({
  name: 'changeBoardModal',
  initialState: getInitialState(),
  reducers: {
    isModalOpenChanged: (state, action: PayloadAction<boolean>) => {
      state.isModalOpen = action.payload;
    },
    imagesToChangeSelected: (state, action: PayloadAction<string[]>) => {
      state.image_names = action.payload;
      state.video_names = [];
    },
    videosToChangeSelected: (state, action: PayloadAction<string[]>) => {
      state.video_names = action.payload;
      state.image_names = [];
    },
    changeBoardReset: (state) => {
      state.image_names = [];
      state.video_names = [];
      state.isModalOpen = false;
    },
  },
});

export const { isModalOpenChanged, imagesToChangeSelected, videosToChangeSelected, changeBoardReset } = slice.actions;

export const selectChangeBoardModalSlice = (state: RootState) => state.changeBoardModal;

export const changeBoardModalSliceConfig: SliceConfig<typeof slice> = {
  slice,
  schema: zChangeBoardModalState,
  getInitialState,
};
