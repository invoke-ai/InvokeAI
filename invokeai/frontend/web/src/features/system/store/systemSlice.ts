import { UseToastOptions } from '@chakra-ui/react';
import { PayloadAction, createSlice } from '@reduxjs/toolkit';

import { InvokeLogLevel } from 'app/logging/useLogger';
import { userInvoked } from 'app/store/actions';
import { nodeTemplatesBuilt } from 'features/nodes/store/nodesSlice';
import { t } from 'i18next';
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
  appSocketInvocationStarted,
  appSocketSubscribed,
  appSocketUnsubscribed,
} from 'services/events/actions';
import { ProgressImage } from 'services/events/types';
import { makeToast } from '../../../app/components/Toaster';
import { LANGUAGES } from '../components/LanguagePicker';

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
   * Whether or not the OpenAPI schema was received and parsed
   */
  wasSchemaParsed: boolean;
  /**
   * The console output logging level
   */
  consoleLogLevel: InvokeLogLevel;
  shouldLogToConsole: boolean;
  statusTranslationKey: any;
  /**
   * When a session is canceled, its ID is stored here until a new session is created.
   */
  canceledSession: string;
  isPersisted: boolean;
  shouldAntialiasProgressImage: boolean;
  language: keyof typeof LANGUAGES;
  isUploading: boolean;
  isNodesEnabled: boolean;
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
  wasSchemaParsed: false,
  consoleLogLevel: 'debug',
  shouldLogToConsole: true,
  statusTranslationKey: 'common.statusDisconnected',
  canceledSession: '',
  isPersisted: false,
  language: 'en',
  isUploading: false,
  isNodesEnabled: false,
};

export const systemSlice = createSlice({
  name: 'system',
  initialState: initialSystemState,
  reducers: {
    setIsProcessing: (state, action: PayloadAction<boolean>) => {
      state.isProcessing = action.payload;
    },
    setCurrentStatus: (state, action: any) => {
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
    setIsNodesEnabled(state, action: PayloadAction<boolean>) {
      state.isNodesEnabled = action.payload;
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
     * Invocation Error
     */
    builder.addCase(appSocketInvocationError, (state) => {
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
        makeToast({ title: t('toast.serverError'), status: 'error' })
      );
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

    /**
     * OpenAPI schema was parsed
     */
    builder.addCase(nodeTemplatesBuilt, (state) => {
      state.wasSchemaParsed = true;
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

      state.toastQueue.push(
        makeToast({ title: t('toast.serverError'), status: 'error' })
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
  setIsNodesEnabled,
} = systemSlice.actions;

export default systemSlice.reducer;
