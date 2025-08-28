import type { PayloadAction, Selector } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import { isPlainObject } from 'es-toolkit';
import type {
  AspectRatioID,
  ImageWithDims,
  RunwayDuration,
  RunwayResolution,
  Veo3Duration,
  Veo3Resolution,
} from 'features/controlLayers/store/types';
import {
  isRunwayAspectRatioID,
  isRunwayDurationID,
  isRunwayResolution,
  isVeo3AspectRatioID,
  isVeo3DurationID,
  isVeo3Resolution,
  zAspectRatioID,
  zImageWithDims,
  zRunwayDurationID,
  zRunwayResolution,
  zVeo3DurationID,
  zVeo3Resolution,
} from 'features/controlLayers/store/types';
import type { VideoField } from 'features/nodes/types/common';
import { zModelIdentifierField, zVideoField } from 'features/nodes/types/common';
import { modelConfigsAdapterSelectors, selectModelConfigsQuery } from 'services/api/endpoints/models';
import { isVideoModelConfig, type RunwayModelConfig, type Veo3ModelConfig } from 'services/api/types';
import { assert } from 'tsafe';
import z from 'zod';

const zVideoState = z.object({
  _version: z.literal(1),
  startingFrameImage: zImageWithDims.nullable(),
  generatedVideo: zVideoField.nullable(),
  videoModel: zModelIdentifierField.nullable(),
  videoResolution: zVeo3Resolution.or(zRunwayResolution),
  videoDuration: zVeo3DurationID.or(zRunwayDurationID),
  videoAspectRatio: zAspectRatioID,
});

export type VideoState = z.infer<typeof zVideoState>;

const getInitialState = (): VideoState => {
  return {
    _version: 1,
    startingFrameImage: null,
    generatedVideo: null,
    videoModel: null,
    videoResolution: '720p',
    videoDuration: '8',
    videoAspectRatio: '16:9',
  };
};

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

    videoModelChanged: (state, action: PayloadAction<Veo3ModelConfig | RunwayModelConfig | null>) => {
      const parsedModel = zModelIdentifierField.parse(action.payload);
      state.videoModel = parsedModel;

      if (parsedModel?.base === 'veo3') {
        if (!state.videoResolution || !isVeo3Resolution(state.videoResolution)) {
          state.videoResolution = '720p';
        }
        if (!state.videoDuration || !isVeo3DurationID(state.videoDuration)) {
          state.videoDuration = '8';
        }
        if (!state.videoAspectRatio || !isVeo3AspectRatioID(state.videoAspectRatio)) {
          state.videoAspectRatio = '16:9';
        }
      } else if (parsedModel?.base === 'runway') {
        if (!state.videoResolution || !isRunwayResolution(state.videoResolution)) {
          state.videoResolution = '720p';
        }
        if (!state.videoDuration || !isRunwayDurationID(state.videoDuration)) {
          state.videoDuration = '5';
        }
        if (!state.videoAspectRatio || !isRunwayAspectRatioID(state.videoAspectRatio)) {
          state.videoAspectRatio = '16:9';
        }
      }
    },

    videoResolutionChanged: (state, action: PayloadAction<Veo3Resolution | RunwayResolution>) => {
      state.videoResolution = action.payload;
    },

    videoDurationChanged: (state, action: PayloadAction<Veo3Duration | RunwayDuration>) => {
      state.videoDuration = action.payload;
    },

    videoAspectRatioChanged: (state, action: PayloadAction<AspectRatioID>) => {
      state.videoAspectRatio = action.payload;
    },
  },
});

export const {
  startingFrameImageChanged,
  generatedVideoChanged,
  videoModelChanged,
  videoResolutionChanged,
  videoDurationChanged,
  videoAspectRatioChanged,
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
const createVideoSelector = <T,>(selector: Selector<VideoState, T>) => createSelector(selectVideoSlice, selector);

export const selectStartingFrameImage = createVideoSelector((video) => video.startingFrameImage);
export const selectGeneratedVideo = createVideoSelector((video) => video.generatedVideo);
export const selectVideoModel = createVideoSelector((video) => video.videoModel);
export const selectVideoModelKey = createVideoSelector((video) => video.videoModel?.key);
export const selectVideoResolution = createVideoSelector((video) => video.videoResolution);
export const selectVideoDuration = createVideoSelector((video) => video.videoDuration);
export const selectVideoAspectRatio = createVideoSelector((video) => video.videoAspectRatio);
export const selectIsVeo3 = createVideoSelector((video) => video.videoModel?.base === 'veo3');
export const selectIsRunway = createVideoSelector((video) => video.videoModel?.base === 'runway');
export const selectVideoModelConfig = createSelector(
  selectModelConfigsQuery,
  selectVideoSlice,
  (modelConfigs, { videoModel }) => {
    if (!modelConfigs.data) {
      return null;
    }
    if (!videoModel) {
      return null;
    }
    const modelConfig = modelConfigsAdapterSelectors.selectById(modelConfigs.data, videoModel.key);
    if (!modelConfig) {
      return null;
    }
    if (!isVideoModelConfig(modelConfig)) {
      return null;
    }
    return modelConfig;
  }
);
