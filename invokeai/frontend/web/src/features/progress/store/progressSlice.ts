import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice, isAnyOf } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import {
  addImageToStagingArea,
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

export type LatestImageData = InvocationCompleteEvent & {
  image_name: string;
};

export type ProgressState = {
  _version: 1;
  /**
   * Whether or not the system is currently generating.
   */
  isProcessing: boolean;
  /**
   * The batches that are currently being processed in the canvas, if any.
   */
  canvasBatchIds: string[];
  /**
   * The current denoise progress of the canvas, if any.
   */
  canvasDenoiseProgress: DenoiseProgress | null;
  /**
   * The latest image data for the canvas, if any.
   */
  canvasLatestImageData: LatestImageData | null;
  /**
   * The batches that are currently being processed in the linear tabs, if any.
   */
  linearBatchIds: string[];
  /**
   * The current denoise progress of the linear tabs, if any.
   */
  linearDenoiseProgress: DenoiseProgress | null;
  /**
   * The latest image data for the linear tabs, if any.
   */
  linearLatestImageData: LatestImageData | null;
  /**
   * The batches that are currently being processed in the workflow, if any.
   */
  workflowBatchIds: string[];
  /**
   * The current denoise progress of the workflow, if any.
   */
  workflowDenoiseProgress: DenoiseProgress | null;
  /**
   * The latest image data for the workflow, if any.
   */
  workflowLatestImageData: LatestImageData | null;
  /**
   * The latest denoise progress, regardless of tab.
   */
  latestDenoiseProgress: DenoiseProgress | null;
};

export const initialProgressState: ProgressState = {
  _version: 1,
  isProcessing: false,
  canvasBatchIds: [],
  canvasDenoiseProgress: null,
  canvasLatestImageData: null,
  linearBatchIds: [],
  linearDenoiseProgress: null,
  linearLatestImageData: null,
  workflowBatchIds: [],
  workflowDenoiseProgress: null,
  workflowLatestImageData: null,
  latestDenoiseProgress: null,
};

export const progressSlice = createSlice({
  name: 'progress',
  initialState: initialProgressState,
  reducers: {
    canvasBatchEnqueued: (state, action: PayloadAction<string>) => {
      state.canvasBatchIds.push(action.payload);
    },
    linearBatchEnqueued: (state, action: PayloadAction<string>) => {
      state.linearBatchIds.push(action.payload);
    },
    workflowBatchEnqueued: (state, action: PayloadAction<string>) => {
      state.workflowBatchIds.push(action.payload);
    },
    imageInvocationComplete: (state, action: PayloadAction<{ data: InvocationCompleteEvent; imageDTO: ImageDTO }>) => {
      const { data, imageDTO } = action.payload;
      if (state.canvasBatchIds.includes(data.queue_batch_id)) {
        state.canvasLatestImageData = { ...data, image_name: imageDTO.image_name };
      }
      if (state.linearBatchIds.includes(data.queue_batch_id)) {
        state.linearLatestImageData = { ...data, image_name: imageDTO.image_name };
      }
      if (state.workflowBatchIds.includes(data.queue_batch_id)) {
        state.workflowLatestImageData = { ...data, image_name: imageDTO.image_name };
      }
    },
    latestLinearImageLoaded: (state) => {
      state.linearDenoiseProgress = null;
    },
  },
  extraReducers(builder) {
    builder.addCase(socketConnected, (state) => {
      state.canvasDenoiseProgress = null;
      state.linearDenoiseProgress = null;
      state.workflowDenoiseProgress = null;
    });

    builder.addCase(socketDisconnected, (state) => {
      state.canvasDenoiseProgress = null;
      state.linearDenoiseProgress = null;
      state.workflowDenoiseProgress = null;
    });

    builder.addCase(socketInvocationComplete, (state) => {
      state.latestDenoiseProgress = null;
    });

    builder.addCase(socketGeneratorProgress, (state, action) => {
      const denoiseProgress = buildDenoiseProgress(action.payload.data);
      state.latestDenoiseProgress = denoiseProgress;
      if (state.linearBatchIds.includes(action.payload.data.queue_batch_id)) {
        state.linearDenoiseProgress = denoiseProgress;
      }
      if (state.canvasBatchIds.includes(action.payload.data.queue_batch_id)) {
        state.canvasDenoiseProgress = denoiseProgress;
      }
      if (state.workflowBatchIds.includes(action.payload.data.queue_batch_id)) {
        state.workflowDenoiseProgress = denoiseProgress;
      }
    });

    builder.addCase(socketQueueItemStatusChanged, (state, action) => {
      // This logic only applies to the linear and workflow views. Canvas progress images are linked to the staging area
      // and handled separately.

      // When the queue is empty, clear progress and batch ids.
      if (!action.payload.data.queue_status.in_progress && !action.payload.data.queue_status.pending) {
        // state.linearBatchIds = [];
        state.workflowBatchIds = [];
        // state.canvasDenoiseProgress = null;
        // state.linearDenoiseProgress = null;
        state.workflowDenoiseProgress = null;
        return;
      }

      // // If the current queue item / session has just finished *and* we are storing its progress, clear the progress.
      // const { status, session_id } = action.payload.data.queue_item;
      // if (['completed', 'canceled', 'failed'].includes(status)) {
      //   if (state.canvasDenoiseProgress?.graph_execution_state_id === session_id) {
      //     state.canvasDenoiseProgress = null;
      //   }
      //   if (state.linearDenoiseProgress?.graph_execution_state_id === session_id) {
      //     state.linearDenoiseProgress = null;
      //   }
      //   if (state.workflowDenoiseProgress?.graph_execution_state_id === session_id) {
      //     state.workflowDenoiseProgress = null;
      //   }
      // }
    });

    builder.addCase(addImageToStagingArea, (state) => {
      state.canvasDenoiseProgress = null;
    });

    // builder.addCase(imageSelected, (state, action) => {
    //   if (
    //     action.payload?.session_id &&
    //     state.linearDenoiseProgress?.graph_execution_state_id === action.payload.session_id
    //   ) {
    //     state.linearDenoiseProgress = null;
    //   }
    // });

    builder.addMatcher(
      isAnyOf(commitStagingAreaImage, discardStagedImages, resetCanvas, setInitialCanvasImage),
      (state) => {
        // These actions all should result in the canvas progress being cleared.
        state.canvasDenoiseProgress = null;
        state.canvasBatchIds = [];
      }
    );

    builder.addMatcher(queueApi.endpoints.clearQueue.matchFulfilled, (state) => {
      // When the queue is cleared, all progress is cleared
      state.canvasBatchIds = [];
      state.linearBatchIds = [];
      state.workflowBatchIds = [];
      state.canvasDenoiseProgress = null;
      state.linearDenoiseProgress = null;
      state.workflowDenoiseProgress = null;
    });

    builder.addMatcher(queueApi.endpoints.cancelByBatchIds.matchFulfilled, (state, action) => {
      // When a batch is canceled, remove it from the list of batch ids and clear its progress if it is stored.

      const canceled_batch_ids = action.meta.arg.originalArgs.batch_ids;
      state.canvasBatchIds = state.canvasBatchIds.filter((id) => !canceled_batch_ids.includes(id));
      state.linearBatchIds = state.linearBatchIds.filter((id) => !canceled_batch_ids.includes(id));
      state.workflowBatchIds = state.workflowBatchIds.filter((id) => !canceled_batch_ids.includes(id));

      if (
        state.canvasDenoiseProgress?.graph_execution_state_id &&
        canceled_batch_ids.includes(state.canvasDenoiseProgress.graph_execution_state_id)
      ) {
        state.canvasDenoiseProgress = null;
      }
      if (
        state.linearDenoiseProgress?.graph_execution_state_id &&
        canceled_batch_ids.includes(state.linearDenoiseProgress.graph_execution_state_id)
      ) {
        state.linearDenoiseProgress = null;
      }
      if (
        state.workflowDenoiseProgress?.graph_execution_state_id &&
        canceled_batch_ids.includes(state.workflowDenoiseProgress.graph_execution_state_id)
      ) {
        state.workflowDenoiseProgress = null;
      }
    });
  },
});

export const {
  canvasBatchEnqueued,
  linearBatchEnqueued,
  workflowBatchEnqueued,
  imageInvocationComplete,
  latestLinearImageLoaded,
} = progressSlice.actions;

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
  persistDenylist: [
    'latestDenoiseProgress',
    'canvasDenoiseProgress',
    'canvasLatestImageData',
    'linearDenoiseProgress',
    'linearLatestImageData',
    'workflowDenoiseProgress',
    'workflowLatestImageData',
  ],
};

