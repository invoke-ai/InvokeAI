import type { UseToastOptions } from '@invoke-ai/ui-library';
import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice, isAnyOf } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { calculateStepPercentage } from 'features/system/util/calculateStepPercentage';
import { makeToast } from 'features/system/util/makeToast';
import { t } from 'i18next';
import { startCase } from 'lodash-es';
import type { LogLevelName } from 'roarr';
import {
  socketConnected,
  socketDisconnected,
  socketGeneratorProgress,
  socketGraphExecutionStateComplete,
  socketInvocationComplete,
  socketInvocationError,
  socketInvocationRetrievalError,
  socketInvocationStarted,
  socketModelLoadCompleted,
  socketModelLoadStarted,
  socketQueueItemStatusChanged,
  socketSessionRetrievalError,
} from 'services/events/actions';

import type { Language, SystemState } from './types';

export const initialSystemState: SystemState = {
  _version: 1,
  isConnected: false,
  shouldConfirmOnDelete: true,
  enableImageDebugging: false,
  toastQueue: [],
  denoiseProgress: null,
  shouldAntialiasProgressImage: false,
  consoleLogLevel: 'debug',
  shouldLogToConsole: true,
  language: 'en',
  shouldUseNSFWChecker: false,
  shouldUseWatermarker: false,
  shouldEnableInformationalPopovers: false,
  status: 'DISCONNECTED',
};

export const systemSlice = createSlice({
  name: 'system',
  initialState: initialSystemState,
  reducers: {
    setShouldConfirmOnDelete: (state, action: PayloadAction<boolean>) => {
      state.shouldConfirmOnDelete = action.payload;
    },
    setEnableImageDebugging: (state, action: PayloadAction<boolean>) => {
      state.enableImageDebugging = action.payload;
    },
    addToast: (state, action: PayloadAction<UseToastOptions>) => {
      state.toastQueue.push(action.payload);
    },
    clearToastQueue: (state) => {
      state.toastQueue = [];
    },
    consoleLogLevelChanged: (state, action: PayloadAction<LogLevelName>) => {
      state.consoleLogLevel = action.payload;
    },
    shouldLogToConsoleChanged: (state, action: PayloadAction<boolean>) => {
      state.shouldLogToConsole = action.payload;
    },
    shouldAntialiasProgressImageChanged: (state, action: PayloadAction<boolean>) => {
      state.shouldAntialiasProgressImage = action.payload;
    },
    languageChanged: (state, action: PayloadAction<Language>) => {
      state.language = action.payload;
    },
    shouldUseNSFWCheckerChanged(state, action: PayloadAction<boolean>) {
      state.shouldUseNSFWChecker = action.payload;
    },
    shouldUseWatermarkerChanged(state, action: PayloadAction<boolean>) {
      state.shouldUseWatermarker = action.payload;
    },
    setShouldEnableInformationalPopovers(state, action: PayloadAction<boolean>) {
      state.shouldEnableInformationalPopovers = action.payload;
    },
  },
  extraReducers(builder) {
    /**
     * Socket Connected
     */
    builder.addCase(socketConnected, (state) => {
      state.isConnected = true;
      state.denoiseProgress = null;
      state.status = 'CONNECTED';
    });

    /**
     * Socket Disconnected
     */
    builder.addCase(socketDisconnected, (state) => {
      state.isConnected = false;
      state.denoiseProgress = null;
      state.status = 'DISCONNECTED';
    });

    /**
     * Invocation Started
     */
    builder.addCase(socketInvocationStarted, (state) => {
      state.denoiseProgress = null;
      state.status = 'PROCESSING';
    });

    /**
     * Generator Progress
     */
    builder.addCase(socketGeneratorProgress, (state, action) => {
      const {
        step,
        total_steps,
        order,
        progress_image,
        graph_execution_state_id: session_id,
        queue_batch_id: batch_id,
      } = action.payload.data;

      state.denoiseProgress = {
        step,
        total_steps,
        order,
        percentage: calculateStepPercentage(step, total_steps, order),
        progress_image,
        session_id,
        batch_id,
      };

      state.status = 'PROCESSING';
    });

    /**
     * Invocation Complete
     */
    builder.addCase(socketInvocationComplete, (state) => {
      state.denoiseProgress = null;
      state.status = 'CONNECTED';
    });

    /**
     * Graph Execution State Complete
     */
    builder.addCase(socketGraphExecutionStateComplete, (state) => {
      state.denoiseProgress = null;
      state.status = 'CONNECTED';
    });

    builder.addCase(socketModelLoadStarted, (state) => {
      state.status = 'LOADING_MODEL';
    });

    builder.addCase(socketModelLoadCompleted, (state) => {
      state.status = 'CONNECTED';
    });

    builder.addCase(socketQueueItemStatusChanged, (state, action) => {
      if (['completed', 'canceled', 'failed'].includes(action.payload.data.queue_item.status)) {
        state.status = 'CONNECTED';
        state.denoiseProgress = null;
      }
    });

    // *** Matchers - must be after all cases ***

    /**
     * Any server error
     */
    builder.addMatcher(isAnyServerError, (state, action) => {
      state.toastQueue.push(
        makeToast({
          title: t('toast.serverError'),
          status: 'error',
          description: startCase(action.payload.data.error_type),
        })
      );
    });
  },
});

export const {
  setShouldConfirmOnDelete,
  setEnableImageDebugging,
  addToast,
  clearToastQueue,
  consoleLogLevelChanged,
  shouldLogToConsoleChanged,
  shouldAntialiasProgressImageChanged,
  languageChanged,
  shouldUseNSFWCheckerChanged,
  shouldUseWatermarkerChanged,
  setShouldEnableInformationalPopovers,
} = systemSlice.actions;

const isAnyServerError = isAnyOf(socketInvocationError, socketSessionRetrievalError, socketInvocationRetrievalError);

export const selectSystemSlice = (state: RootState) => state.system;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
export const migrateSystemState = (state: any): any => {
  if (!('_version' in state)) {
    state._version = 1;
  }
  return state;
};

export const systemPersistConfig: PersistConfig<SystemState> = {
  name: systemSlice.name,
  initialState: initialSystemState,
  migrate: migrateSystemState,
  persistDenylist: ['isConnected', 'denoiseProgress', 'status'],
};
