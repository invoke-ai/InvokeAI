import type { PayloadAction, Selector } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import { isPlainObject } from 'es-toolkit';
import type { ImageWithDims } from 'features/controlLayers/store/types';
import { zImageWithDims } from 'features/controlLayers/store/types';
import type { VideoField } from 'features/nodes/types/common';
import { zModelIdentifierField, zVideoField } from 'features/nodes/types/common';
import { ModelIdentifier } from 'features/nodes/types/v2/common';
import { Veo3ModelConfig } from 'services/api/types';
import { assert } from 'tsafe';
import z from 'zod';

const zVideoState = z.object({
  _version: z.literal(1),
  startingFrameImage: zImageWithDims.nullable(),
  generatedVideo: zVideoField.nullable(),
  videoModel: zModelIdentifierField.nullable(),
});

export type VideoState = z.infer<typeof zVideoState>;

const getInitialState = (): VideoState => ({
  _version: 1,
  startingFrameImage: null,
  generatedVideo: null,
  videoModel: null,
});

const slice = createSlice({
  name: 'video',
  initialState: getInitialState(),
  reducers: {
    startingFrameImageChanged: (state, action: PayloadAction<ImageWithDims | null>) => {
      state.startingFrameImage = action.payload;
    },

    generatedVideoChanged: (state, action: PayloadAction<{ videoField: VideoField | null }>) => {
      const { videoField } = action.payload;
      state.generatedVideo = videoField;
    },

    videoModelChanged: (state, action: PayloadAction<Veo3ModelConfig | null>) => {
      const parsedModel = zModelIdentifierField.parse(action.payload);
      state.videoModel = parsedModel;
    },
  },
});

export const { startingFrameImageChanged, generatedVideoChanged, videoModelChanged } = slice.actions;

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

export const selectStartingFrameImage = createVideoSelector((video) => video.startingFrameImage);
export const selectGeneratedVideo = createVideoSelector((video) => video.generatedVideo);
export const selectVideoModel = createVideoSelector((video) => video.videoModel);
