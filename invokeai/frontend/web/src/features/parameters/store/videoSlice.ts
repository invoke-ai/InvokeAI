import type { PayloadAction, Selector } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import { isPlainObject } from 'es-toolkit';
import type { ImageWithDims } from 'features/controlLayers/store/types';
import { zImageWithDims } from 'features/controlLayers/store/types';
import { VideoOutput, zVideoOutput } from 'features/nodes/types/common';
import { assert } from 'tsafe';
import z from 'zod';

const zVideoState = z.object({
  _version: z.literal(1),
  videoFirstFrameImage: zImageWithDims.nullable(),
  videoLastFrameImage: zImageWithDims.nullable(),
  generatedVideo: zVideoOutput.nullable(),
});

export type VideoState = z.infer<typeof zVideoState>;

const getInitialState = (): VideoState => ({
  _version: 1,
  videoFirstFrameImage: null,
  videoLastFrameImage: null,
  generatedVideo: null,
});

const slice = createSlice({
  name: 'video',
  initialState: getInitialState(),
  reducers: {
    videoFirstFrameImageChanged: (state, action: PayloadAction<ImageWithDims | null>) => {
      state.videoFirstFrameImage = action.payload;
    },

    videoLastFrameImageChanged: (state, action: PayloadAction<ImageWithDims | null>) => {
      state.videoLastFrameImage = action.payload;
    },

    generatedVideoChanged: (state, action: PayloadAction<VideoOutput | null>) => {
      state.generatedVideo = action.payload;
    },

  },
});

export const {
  videoFirstFrameImageChanged,
  videoLastFrameImageChanged,
  generatedVideoChanged,
} = slice.actions;

export const videoSliceConfig: SliceConfig<typeof slice> = {
  slice,
  schema: zVideoState,
  getInitialState,
  persistConfig: {
    migrate: (state) => {
      assert(isPlainObject(state));
      if (!('_version' in state)) {
        state._version = 1;
      }
      return zVideoState.parse(state);
    },
  },
};

export const selectVideoSlice = (state: RootState) => state.video;
const createVideoSelector = <T>(selector: Selector<VideoState, T>) => createSelector(selectVideoSlice, selector);

export const selectVideoFirstFrameImage = createVideoSelector((video) => video.videoFirstFrameImage);
export const selectVideoLastFrameImage = createVideoSelector((video) => video.videoLastFrameImage);
export const selectGeneratedVideo = createVideoSelector((video) => video.generatedVideo);