import type { PayloadAction, Selector } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import { isPlainObject } from 'es-toolkit';
import type {
  CroppableImageWithDims,
  VideoAspectRatio,
  VideoDuration,
  VideoResolution,
} from 'features/controlLayers/store/types';
import {
  isRunwayAspectRatioID,
  isRunwayDurationID,
  isRunwayResolution,
  isVeo3AspectRatioID,
  isVeo3DurationID,
  isVeo3Resolution,
  zCroppableImageWithDims,
  zVideoAspectRatio,
  zVideoDuration,
  zVideoResolution,
} from 'features/controlLayers/store/types';
import type { ModelIdentifierField } from 'features/nodes/types/common';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { REQUIRES_STARTING_FRAME_BASE_MODELS } from 'features/parameters/types/constants';
import { modelConfigsAdapterSelectors, selectModelConfigsQuery } from 'services/api/endpoints/models';
import { isVideoModelConfig } from 'services/api/types';
import { assert } from 'tsafe';
import z from 'zod';

const zVideoState = z.object({
  _version: z.literal(2),
  startingFrameImage: zCroppableImageWithDims.nullable(),
  videoModel: zModelIdentifierField.nullable(),
  videoResolution: zVideoResolution,
  videoDuration: zVideoDuration,
  videoAspectRatio: zVideoAspectRatio,
});

export type VideoState = z.infer<typeof zVideoState>;

const getInitialState = (): VideoState => {
  return {
    _version: 2,
    startingFrameImage: null,
    videoModel: null,
    videoResolution: '1080p',
    videoDuration: '8',
    videoAspectRatio: '16:9',
  };
};

const slice = createSlice({
  name: 'video',
  initialState: getInitialState(),
  reducers: {
    startingFrameImageChanged: (state, action: PayloadAction<CroppableImageWithDims | null>) => {
      state.startingFrameImage = action.payload;
    },

    videoModelChanged: (state, action: PayloadAction<{ videoModel: ModelIdentifierField | null }>) => {
      const { videoModel } = action.payload;

      state.videoModel = videoModel;

      if (videoModel?.base === 'veo3') {
        if (!state.videoResolution || !isVeo3Resolution(state.videoResolution)) {
          state.videoResolution = '1080p';
        }
        if (!state.videoDuration || !isVeo3DurationID(state.videoDuration)) {
          state.videoDuration = '8';
        }
        if (!state.videoAspectRatio || !isVeo3AspectRatioID(state.videoAspectRatio)) {
          state.videoAspectRatio = '16:9';
        }
      } else if (videoModel?.base === 'runway') {
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

    videoResolutionChanged: (state, action: PayloadAction<VideoResolution>) => {
      state.videoResolution = action.payload;
    },

    videoDurationChanged: (state, action: PayloadAction<VideoDuration>) => {
      state.videoDuration = action.payload;
    },

    videoAspectRatioChanged: (state, action: PayloadAction<VideoAspectRatio>) => {
      state.videoAspectRatio = action.payload;
    },
  },
});

export const {
  startingFrameImageChanged,
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
      if (state._version === 1) {
        state._version = 2;
        if (state.startingFrameImage) {
          // startingFrameImage changed from ImageWithDims to CroppableImageWithDims
          state.startingFrameImage = zCroppableImageWithDims.parse({ original: state.startingFrameImage });
        }
      }
      return zVideoState.parse(state);
    },
  },
};

export const selectVideoSlice = (state: RootState) => state.video;
const createVideoSelector = <T>(selector: Selector<VideoState, T>) => createSelector(selectVideoSlice, selector);

export const selectStartingFrameImage = createVideoSelector((video) => video.startingFrameImage);
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
export const selectVideoModelRequiresStartingFrame = createSelector(
  selectVideoModel,
  (model) => !!model && REQUIRES_STARTING_FRAME_BASE_MODELS.includes(model.base)
);
