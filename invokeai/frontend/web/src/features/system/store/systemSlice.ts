import { UseToastOptions } from '@chakra-ui/react';
import { PayloadAction, createSlice, isAnyOf } from '@reduxjs/toolkit';
import { InvokeLogLevel } from 'app/logging/logger';
import { userInvoked } from 'app/store/actions';
import { t } from 'i18next';
import { get, startCase, upperFirst } from 'lodash-es';
import { LogLevelName } from 'roarr';
import {
  isAnySessionRejected,
  sessionCanceled,
} from 'services/api/thunks/session';
import {
  appSocketConnected,
  appSocketDisconnected,
  appSocketGeneratorProgress,
  appSocketGraphExecutionStateComplete,
  appSocketInvocationComplete,
  appSocketInvocationError,
  appSocketInvocationRetrievalError,
  appSocketInvocationStarted,
  appSocketSessionRetrievalError,
  appSocketSubscribed,
  appSocketUnsubscribed,
} from 'services/events/actions';
import { ProgressImage } from 'services/events/types';
import { makeToast } from '../util/makeToast';
import { LANGUAGES } from './constants';
import { zPydanticValidationError } from './zodSchemas';

export type CancelStrategy = 'immediate' | 'scheduled';

export interface SystemState {
  isGFPGANAvailable: boolean;
  isESRGANAvailable: boolean;
  isConnected: boolean;
  isProcessing: boolean;
  shouldConfirmOnDelete: boolean;
  currentStep: number;
  totalSteps: number;
  currentIteration: number;
  totalIterations: number;
  currentStatusHasSteps: boolean;
  isCancelable: boolean;
  enableImageDebugging: boolean;
  toastQueue: UseToastOptions[];
  /**
   * The current progress image
   */
  progressImage: ProgressImage | null;
  /**
   * The current socket session id
   */
  sessionId: string | null;
  /**
   * Cancel strategy
   */
  cancelType: CancelStrategy;
  /**
   * Whether or not a scheduled cancelation is pending
   */
  isCancelScheduled: boolean;
  /**
   * Array of node IDs that we want to handle when events received
   */
  subscribedNodeIds: string[];
  /**
   * Whether or not the available models were received
   */
  wereModelsReceived: boolean;
  /**
   * The console output logging level
   */
  consoleLogLevel: InvokeLogLevel;
  shouldLogToConsole: boolean;
  // TODO: probably better to not store keys here, should just be a string that maps to the translation key
  statusTranslationKey: string;
  /**
   * When a session is canceled, its ID is stored here until a new session is created.
   */
  canceledSession: string;
  isPersisted: boolean;
  shouldAntialiasProgressImage: boolean;
  language: keyof typeof LANGUAGES;
  isUploading: boolean;
  shouldUseNSFWChecker: boolean;
  shouldUseWatermarker: boolean;
}

export const initialSystemState: SystemState = {
  isConnected: false,
  isProcessing: false,
  isGFPGANAvailable: true,
  isESRGANAvailable: true,
  shouldConfirmOnDelete: true,
  currentStep: 0,
  totalSteps: 0,
  currentIteration: 0,
  totalIterations: 0,
  currentStatusHasSteps: false,
  isCancelable: true,
  enableImageDebugging: false,
  toastQueue: [],
  progressImage: null,
  shouldAntialiasProgressImage: false,
  sessionId: null,
  cancelType: 'immediate',
  isCancelScheduled: false,
  subscribedNodeIds: [],
  wereModelsReceived: false,
  consoleLogLevel: 'debug',
  shouldLogToConsole: true,
  statusTranslationKey: 'common.statusDisconnected',
  canceledSession: '',
  isPersisted: false,
  language: 'en',
  isUploading: false,
  shouldUseNSFWChecker: false,
  shouldUseWatermarker: false,
};

export const systemSlice = createSlice({
  name: 'system',
  initialState: initialSystemState,
  reducers: {
    setIsProcessing: (state, action: PayloadAction<boolean>) => {
      state.isProcessing = action.payload;
    },
    setCurrentStatus: (state, action: PayloadAction<string>) => {
      state.statusTranslationKey = action.payload;
    },
    setShouldConfirmOnDelete: (state, action: PayloadAction<boolean>) => {
      state.shouldConfirmOnDelete = action.payload;
    },
    setIsCancelable: (state, action: PayloadAction<boolean>) => {
      state.isCancelable = action.payload;
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
    /**
     * A cancel was scheduled
     */
    cancelScheduled: (state) => {
      state.isCancelScheduled = true;
    },
    /**
     * The scheduled cancel was aborted
     */
    scheduledCancelAborted: (state) => {
      state.isCancelScheduled = false;
    },
    /**
     * The cancel type was changed
     */
    cancelTypeChanged: (state, action: PayloadAction<CancelStrategy>) => {
      state.cancelType = action.payload;
    },
    /**
     * The array of subscribed node ids was changed
     */
    subscribedNodeIdsSet: (state, action: PayloadAction<string[]>) => {
      state.subscribedNodeIds = action.payload;
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
    isPersistedChanged: (state, action: PayloadAction<boolean>) => {
      state.isPersisted = action.payload;
    },
    languageChanged: (state, action: PayloadAction<keyof typeof LANGUAGES>) => {
      state.language = action.payload;
    },
    progressImageSet(state, action: PayloadAction<ProgressImage | null>) {
      state.progressImage = action.payload;
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
     * Socket Subscribed
     */
    builder.addCase(appSocketSubscribed, (state, action) => {
      state.sessionId = action.payload.sessionId;
      state.canceledSession = '';
    });

    /**
     * Socket Unsubscribed
     */
    builder.addCase(appSocketUnsubscribed, (state) => {
      state.sessionId = null;
    });

    /**
     * Socket Connected
     */
    builder.addCase(appSocketConnected, (state) => {
      state.isConnected = true;
      state.isCancelable = true;
      state.isProcessing = false;
      state.currentStatusHasSteps = false;
      state.currentStep = 0;
      state.totalSteps = 0;
      state.currentIteration = 0;
      state.totalIterations = 0;
      state.statusTranslationKey = 'common.statusConnected';
    });

    /**
     * Socket Disconnected
     */
    builder.addCase(appSocketDisconnected, (state) => {
      state.isConnected = false;
      state.isProcessing = false;
      state.isCancelable = true;
      state.currentStatusHasSteps = false;
      state.currentStep = 0;
      state.totalSteps = 0;
      // state.currentIteration = 0;
      // state.totalIterations = 0;
      state.statusTranslationKey = 'common.statusDisconnected';
    });

    /**
     * Invocation Started
     */
    builder.addCase(appSocketInvocationStarted, (state) => {
      state.isCancelable = true;
      state.isProcessing = true;
      state.currentStatusHasSteps = false;
      state.currentStep = 0;
      state.totalSteps = 0;
      // state.currentIteration = 0;
      // state.totalIterations = 0;
      state.statusTranslationKey = 'common.statusGenerating';
    });

    /**
     * Generator Progress
     */
    builder.addCase(appSocketGeneratorProgress, (state, action) => {
      const { step, total_steps, progress_image } = action.payload.data;

      state.isProcessing = true;
      state.isCancelable = true;
      // state.currentIteration = 0;
      // state.totalIterations = 0;
      state.currentStatusHasSteps = true;
      state.currentStep = step + 1; // TODO: step starts at -1, think this is a bug
      state.totalSteps = total_steps;
      state.progressImage = progress_image ?? null;
      state.statusTranslationKey = 'common.statusGenerating';
    });

    /**
     * Invocation Complete
     */
    builder.addCase(appSocketInvocationComplete, (state, action) => {
      const { data } = action.payload;

      // state.currentIteration = 0;
      // state.totalIterations = 0;
      state.currentStatusHasSteps = false;
      state.currentStep = 0;
      state.totalSteps = 0;
      state.statusTranslationKey = 'common.statusProcessingComplete';

      if (state.canceledSession === data.graph_execution_state_id) {
        state.isProcessing = false;
        state.isCancelable = true;
      }
    });

    /**
     * Graph Execution State Complete
     */
    builder.addCase(appSocketGraphExecutionStateComplete, (state) => {
      state.isProcessing = false;
      state.isCancelable = false;
      state.isCancelScheduled = false;
      state.currentStep = 0;
      state.totalSteps = 0;
      state.statusTranslationKey = 'common.statusConnected';
      state.progressImage = null;
    });

    /**
     * User Invoked
     */

    builder.addCase(userInvoked, (state) => {
      state.isProcessing = true;
      state.isCancelable = true;
      state.currentStatusHasSteps = false;
      state.statusTranslationKey = 'common.statusPreparing';
    });

    /**
     * Session Canceled - FULFILLED
     */
    builder.addCase(sessionCanceled.fulfilled, (state, action) => {
      state.canceledSession = action.meta.arg.session_id;
      state.isProcessing = false;
      state.isCancelable = false;
      state.isCancelScheduled = false;
      state.currentStep = 0;
      state.totalSteps = 0;
      state.statusTranslationKey = 'common.statusConnected';
      state.progressImage = null;

      state.toastQueue.push(
        makeToast({ title: t('toast.canceled'), status: 'warning' })
      );
    });

    // *** Matchers - must be after all cases ***

    /**
     * Session Invoked - REJECTED
     * Session Created - REJECTED
     */
    builder.addMatcher(isAnySessionRejected, (state, action) => {
      state.isProcessing = false;
      state.isCancelable = false;
      state.isCancelScheduled = false;
      state.currentStep = 0;
      state.totalSteps = 0;
      state.statusTranslationKey = 'common.statusConnected';
      state.progressImage = null;

      let errorDescription = undefined;
      const duration = 5000;

      if (action.payload?.status === 422) {
        const result = zPydanticValidationError.safeParse(action.payload);
        if (result.success) {
          result.data.error.detail.map((e) => {
            state.toastQueue.push(
              makeToast({
                title: upperFirst(e.msg),
                status: 'error',
                description: `Path:
                ${e.loc.slice(3).join('.')}`,
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
          description: get(errorDescription, 'detail', 'Unknown Error'),
          duration,
        })
      );
    });

    /**
     * Any server error
     */
    builder.addMatcher(isAnyServerError, (state, action) => {
      state.isProcessing = false;
      state.isCancelable = true;
      // state.currentIteration = 0;
      // state.totalIterations = 0;
      state.currentStatusHasSteps = false;
      state.currentStep = 0;
      state.totalSteps = 0;
      state.statusTranslationKey = 'common.statusError';
      state.progressImage = null;

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
  setIsProcessing,
  setShouldConfirmOnDelete,
  setCurrentStatus,
  setIsCancelable,
  setEnableImageDebugging,
  addToast,
  clearToastQueue,
  cancelScheduled,
  scheduledCancelAborted,
  cancelTypeChanged,
  subscribedNodeIdsSet,
  consoleLogLevelChanged,
  shouldLogToConsoleChanged,
  isPersistedChanged,
  shouldAntialiasProgressImageChanged,
  languageChanged,
  progressImageSet,
  shouldUseNSFWCheckerChanged,
  shouldUseWatermarkerChanged,
} = systemSlice.actions;

export default systemSlice.reducer;

const isAnyServerError = isAnyOf(
  appSocketInvocationError,
  appSocketSessionRetrievalError,
  appSocketInvocationRetrievalError
);
