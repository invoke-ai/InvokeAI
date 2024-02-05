import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice, isAnyOf } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import {
  commitStagingAreaImage,
  discardStagedImages,
  resetCanvas,
  setInitialCanvasImage,
} from 'features/canvas/store/canvasSlice';
import type { DenoiseProgress } from 'features/progress/store/types';
import { calculateStepPercentage } from 'features/system/util/calculateStepPercentage';
import { queueApi } from 'services/api/endpoints/queue';
import type { ImageDTO } from 'services/api/types';
import {
  socketConnected,
  socketDisconnected,
  socketGeneratorProgress,
  socketInvocationComplete,
  socketQueueItemStatusChanged,
} from 'services/events/actions';
import type { GeneratorProgressEvent, InvocationCompleteEvent } from 'services/events/types';

export type ProgressTab = 'linear' | 'canvas' | 'workflow';

export type LatestImageOutputEvent = InvocationCompleteEvent & {
  image_name: string;
};

export type ProgressState = {
  _version: 1;
  /**
   * The current denoise progress. Only has a value if denoising is in progress.
   */
  currentDenoiseProgress: DenoiseProgress | null;
  /**
   * The latest denoise progress, regardless of tab.
   */
  latestDenoiseProgress: DenoiseProgress | null;
  /**
   * The latest image output event.
   */
  latestImageOutputEvent: LatestImageOutputEvent | null;
  /**
   * Batch ids for batches initiated on the canvas tab.
   */
  canvasBatchIds: string[];
};

export const initialProgressState: ProgressState = {
  _version: 1,
  currentDenoiseProgress: null,
  latestDenoiseProgress: null,
  latestImageOutputEvent: null,
  canvasBatchIds: [],
};

export const progressSlice = createSlice({
  name: 'progress',
  initialState: initialProgressState,
  reducers: {
    canvasBatchEnqueued: (state, action: PayloadAction<string>) => {
      state.canvasBatchIds.push(action.payload);
    },
    imageInvocationComplete: (state, action: PayloadAction<{ data: InvocationCompleteEvent; imageDTO: ImageDTO }>) => {
      const { data, imageDTO } = action.payload;
      state.latestImageOutputEvent = { ...data, image_name: imageDTO.image_name };
    },
    latestImageLoaded: () => {
      // state.latestDenoiseProgress = null;
    },
  },
  extraReducers(builder) {
    builder.addCase(socketConnected, (state) => {
      state.latestDenoiseProgress = null;
      state.latestImageOutputEvent = null;
    });

    builder.addCase(socketDisconnected, (state) => {
      state.latestDenoiseProgress = null;
      state.latestImageOutputEvent = null;
    });

    builder.addCase(socketGeneratorProgress, (state, action) => {
      const denoiseProgress = buildDenoiseProgress(action.payload.data);
      state.latestDenoiseProgress = denoiseProgress;
      state.currentDenoiseProgress = denoiseProgress;
    });

    builder.addCase(socketInvocationComplete, (state) => {
      state.currentDenoiseProgress = null;
    });

    builder.addCase(socketQueueItemStatusChanged, (state) => {
      state.currentDenoiseProgress = null;
    });

    builder.addMatcher(
      isAnyOf(commitStagingAreaImage, discardStagedImages, resetCanvas, setInitialCanvasImage),
      (state) => {
        state.canvasBatchIds = [];
      }
    );

    builder.addMatcher(queueApi.endpoints.clearQueue.matchFulfilled, (state) => {
      // When the queue is cleared, all progress is cleared
      state.latestDenoiseProgress = null;
    });

    builder.addMatcher(queueApi.endpoints.cancelByBatchIds.matchFulfilled, (state, action) => {
      // When a batch is canceled, remove it from the list of batch ids and clear its progress if it is stored.

      const canceled_batch_ids = action.meta.arg.originalArgs.batch_ids;

      if (state.latestDenoiseProgress && canceled_batch_ids.includes(state.latestDenoiseProgress.queue_batch_id)) {
        state.latestDenoiseProgress = null;
      }
    });
  },
});

export const { imageInvocationComplete, canvasBatchEnqueued, latestImageLoaded } = progressSlice.actions;

export const selectProgressSlice = (state: RootState) => state.progress;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
export const migrateProgressState = (state: any): any => {
  if (!('_version' in state)) {
    state._version = 1;
  }
  return state;
};

const buildDenoiseProgress = (data: GeneratorProgressEvent): DenoiseProgress => ({
  ...data,
  percentage: calculateStepPercentage(data.step, data.total_steps, data.order),
});

export const progressPersistConfig: PersistConfig<ProgressState> = {
  name: progressSlice.name,
  initialState: initialProgressState,
  migrate: migrateProgressState,
  persistDenylist: ['currentDenoiseProgress', 'latestDenoiseProgress', 'latestImageOutputEvent'],
};
