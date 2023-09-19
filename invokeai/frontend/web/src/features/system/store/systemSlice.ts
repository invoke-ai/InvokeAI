import { UseToastOptions } from '@chakra-ui/react';
import { PayloadAction, createSlice, isAnyOf } from '@reduxjs/toolkit';
import { t } from 'i18next';
import { get, startCase, truncate, upperFirst } from 'lodash-es';
import { LogLevelName } from 'roarr';
import { isAnySessionRejected } from 'services/api/thunks/session';
import {
  appSocketConnected,
  appSocketDisconnected,
  appSocketGeneratorProgress,
  appSocketGraphExecutionStateComplete,
  appSocketInvocationComplete,
  appSocketInvocationError,
  appSocketInvocationRetrievalError,
  appSocketInvocationStarted,
  appSocketModelLoadCompleted,
  appSocketModelLoadStarted,
  appSocketQueueItemStatusChanged,
  appSocketSessionRetrievalError,
} from 'services/events/actions';
import { calculateStepPercentage } from '../util/calculateStepPercentage';
import { makeToast } from '../util/makeToast';
import { LANGUAGES } from './constants';
import { SystemState } from './types';
import { zPydanticValidationError } from './zodSchemas';

export const initialSystemState: SystemState = {
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
    shouldAntialiasProgressImageChanged: (
      state,
      action: PayloadAction<boolean>
    ) => {
      state.shouldAntialiasProgressImage = action.payload;
    },
    languageChanged: (state, action: PayloadAction<keyof typeof LANGUAGES>) => {
      state.language = action.payload;
    },
    shouldUseNSFWCheckerChanged(state, action: PayloadAction<boolean>) {
      state.shouldUseNSFWChecker = action.payload;
    },
    shouldUseWatermarkerChanged(state, action: PayloadAction<boolean>) {
      state.shouldUseWatermarker = action.payload;
    },
  },
  extraReducers(builder) {
    /**
     * Socket Connected
     */
    builder.addCase(appSocketConnected, (state) => {
      state.isConnected = true;
      state.denoiseProgress = null;
      state.status = 'CONNECTED';
    });

    /**
     * Socket Disconnected
     */
    builder.addCase(appSocketDisconnected, (state) => {
      state.isConnected = false;
      state.denoiseProgress = null;
      state.status = 'DISCONNECTED';
    });

    /**
     * Invocation Started
     */
    builder.addCase(appSocketInvocationStarted, (state) => {
      state.denoiseProgress = null;
      state.status = 'PROCESSING';
    });

    /**
     * Generator Progress
     */
    builder.addCase(appSocketGeneratorProgress, (state, action) => {
      const {
        step,
        total_steps,
        order,
        progress_image,
        graph_execution_state_id: session_id,
      } = action.payload.data;

      state.denoiseProgress = {
        step,
        total_steps,
        order,
        percentage: calculateStepPercentage(step, total_steps, order),
        progress_image,
        session_id,
      };

      state.status = 'PROCESSING';
    });

    /**
     * Invocation Complete
     */
    builder.addCase(appSocketInvocationComplete, (state) => {
      state.denoiseProgress = null;
      state.status = 'CONNECTED';
    });

    /**
     * Graph Execution State Complete
     */
    builder.addCase(appSocketGraphExecutionStateComplete, (state) => {
      state.denoiseProgress = null;
      state.status = 'CONNECTED';
    });

    builder.addCase(appSocketModelLoadStarted, (state) => {
      state.status = 'LOADING_MODEL';
    });

    builder.addCase(appSocketModelLoadCompleted, (state) => {
      state.status = 'CONNECTED';
    });

    builder.addCase(appSocketQueueItemStatusChanged, (state, action) => {
      if (
        ['completed', 'canceled', 'failed'].includes(action.payload.data.status)
      ) {
        state.status = 'CONNECTED';
        state.denoiseProgress = null;
      }
    });

    // *** Matchers - must be after all cases ***

    /**
     * Session Invoked - REJECTED
     * Session Created - REJECTED
     */
    builder.addMatcher(isAnySessionRejected, (state, action) => {
      let errorDescription = undefined;
      const duration = 5000;

      if (action.payload?.status === 422) {
        const result = zPydanticValidationError.safeParse(action.payload);
        if (result.success) {
          result.data.error.detail.map((e) => {
            state.toastQueue.push(
              makeToast({
                title: truncate(upperFirst(e.msg), { length: 128 }),
                status: 'error',
                description: truncate(
                  `Path:
                ${e.loc.join('.')}`,
                  { length: 128 }
                ),
                duration,
              })
            );
          });
          return;
        }
      } else if (action.payload?.error) {
        errorDescription = action.payload?.error;
      }

      state.toastQueue.push(
        makeToast({
          title: t('toast.serverError'),
          status: 'error',
          description: truncate(
            get(errorDescription, 'detail', 'Unknown Error'),
            { length: 128 }
          ),
          duration,
        })
      );
    });

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
} = systemSlice.actions;

export default systemSlice.reducer;

const isAnyServerError = isAnyOf(
  appSocketInvocationError,
  appSocketSessionRetrievalError,
  appSocketInvocationRetrievalError
);
