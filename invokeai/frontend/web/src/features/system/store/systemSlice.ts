import { ExpandedIndex, UseToastOptions } from '@chakra-ui/react';
import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import * as InvokeAI from 'app/invokeai';
import {
  generatorProgress,
  invocationComplete,
  invocationError,
  invocationStarted,
  socketConnected,
  socketDisconnected,
  socketSubscribed,
  socketUnsubscribed,
} from 'services/events/actions';

import i18n from 'i18n';
import { isImageOutput } from 'services/types/guards';
import { ProgressImage } from 'services/events/types';
import { initialImageSelected } from 'features/parameters/store/generationSlice';
import { makeToast } from '../hooks/useToastWatcher';
import { sessionCanceled, sessionInvoked } from 'services/thunks/session';
import { InvokeTabName } from 'features/ui/store/tabMap';
import { receivedModels } from 'services/thunks/model';
import { receivedOpenAPISchema } from 'services/thunks/schema';

export type LogLevel = 'info' | 'warning' | 'error';

export interface LogEntry {
  timestamp: string;
  level: LogLevel;
  message: string;
}

export interface Log {
  [index: number]: LogEntry;
}

export type InProgressImageType = 'none' | 'full-res' | 'latents';

export type CancelType = 'immediate' | 'scheduled';

export interface SystemState
  extends InvokeAI.SystemStatus,
    InvokeAI.SystemConfig {
  shouldDisplayInProgressType: InProgressImageType;
  log: Array<LogEntry>;
  shouldShowLogViewer: boolean;
  isGFPGANAvailable: boolean;
  isESRGANAvailable: boolean;
  isConnected: boolean;
  socketId: string;
  shouldConfirmOnDelete: boolean;
  openAccordions: ExpandedIndex;
  currentStep: number;
  totalSteps: number;
  currentIteration: number;
  totalIterations: number;
  currentStatus: string;
  currentStatusHasSteps: boolean;
  shouldDisplayGuides: boolean;
  wasErrorSeen: boolean;
  isCancelable: boolean;
  saveIntermediatesInterval: number;
  enableImageDebugging: boolean;
  toastQueue: UseToastOptions[];
  searchFolder: string | null;
  foundModels: InvokeAI.FoundModel[] | null;
  openModel: string | null;
  cancelOptions: {
    cancelType: CancelType;
    cancelAfter: number | null;
  };
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
  cancelType: CancelType;
  /**
   * Whether or not a scheduled cancelation is pending
   */
  isCancelScheduled: boolean;
  /**
   * Array of node IDs that we want to handle when events received
   */
  subscribedNodeIds: string[];
  // /**
  //  * Whether or not URLs should be transformed to use a different host
  //  */
  // shouldTransformUrls: boolean;
  // /**
  //  * Array of disabled tabs
  //  */
  // disabledTabs: InvokeTabName[];
  // /**
  //  * Array of disabled features
  //  */
  // disabledFeatures: InvokeAI.AppFeature[];
  /**
   * Whether or not the available models were received
   */
  wereModelsReceived: boolean;
  /**
   * Whether or not the OpenAPI schema was received and parsed
   */
  wasSchemaParsed: boolean;
}

const initialSystemState: SystemState = {
  isConnected: false,
  isProcessing: false,
  log: [],
  shouldShowLogViewer: false,
  shouldDisplayInProgressType: 'latents',
  shouldDisplayGuides: true,
  isGFPGANAvailable: true,
  isESRGANAvailable: true,
  socketId: '',
  shouldConfirmOnDelete: true,
  openAccordions: [0],
  currentStep: 0,
  totalSteps: 0,
  currentIteration: 0,
  totalIterations: 0,
  currentStatus: i18n.isInitialized
    ? i18n.t('common.statusDisconnected')
    : 'Disconnected',
  currentStatusHasSteps: false,
  model: '',
  model_id: '',
  model_hash: '',
  app_id: '',
  app_version: '',
  model_list: {},
  infill_methods: [],
  hasError: false,
  wasErrorSeen: true,
  isCancelable: true,
  saveIntermediatesInterval: 5,
  enableImageDebugging: false,
  toastQueue: [],
  searchFolder: null,
  foundModels: null,
  openModel: null,
  cancelOptions: {
    cancelType: 'immediate',
    cancelAfter: null,
  },
  progressImage: null,
  sessionId: null,
  cancelType: 'immediate',
  isCancelScheduled: false,
  subscribedNodeIds: [],
  // shouldTransformUrls: false,
  // disabledTabs: [],
  // disabledFeatures: [],
  wereModelsReceived: false,
  wasSchemaParsed: false,
};

export const systemSlice = createSlice({
  name: 'system',
  initialState: initialSystemState,
  reducers: {
    setShouldDisplayInProgressType: (
      state,
      action: PayloadAction<InProgressImageType>
    ) => {
      state.shouldDisplayInProgressType = action.payload;
    },
    setIsProcessing: (state, action: PayloadAction<boolean>) => {
      state.isProcessing = action.payload;
    },
    setCurrentStatus: (state, action: PayloadAction<string>) => {
      state.currentStatus = action.payload;
    },
    setSystemStatus: (state, action: PayloadAction<InvokeAI.SystemStatus>) => {
      return { ...state, ...action.payload };
    },
    errorOccurred: (state) => {
      state.hasError = true;
      state.isProcessing = false;
      state.isCancelable = true;
      state.currentStep = 0;
      state.totalSteps = 0;
      state.currentIteration = 0;
      state.totalIterations = 0;
      state.currentStatusHasSteps = false;
      state.currentStatus = i18n.t('common.statusError');
      state.wasErrorSeen = false;
    },
    errorSeen: (state) => {
      state.hasError = false;
      state.wasErrorSeen = true;
      state.currentStatus = state.isConnected
        ? i18n.t('common.statusConnected')
        : i18n.t('common.statusDisconnected');
    },
    addLogEntry: (
      state,
      action: PayloadAction<{
        timestamp: string;
        message: string;
        level?: LogLevel;
      }>
    ) => {
      const { timestamp, message, level } = action.payload;
      const logLevel = level || 'info';

      const entry: LogEntry = {
        timestamp,
        message,
        level: logLevel,
      };

      state.log.push(entry);
    },
    setShouldShowLogViewer: (state, action: PayloadAction<boolean>) => {
      state.shouldShowLogViewer = action.payload;
    },
    setIsConnected: (state, action: PayloadAction<boolean>) => {
      state.isConnected = action.payload;
      state.isProcessing = false;
      state.isCancelable = true;
      state.currentStep = 0;
      state.totalSteps = 0;
      state.currentIteration = 0;
      state.totalIterations = 0;
      state.currentStatusHasSteps = false;
      state.hasError = false;
    },
    setSocketId: (state, action: PayloadAction<string>) => {
      state.socketId = action.payload;
    },
    setShouldConfirmOnDelete: (state, action: PayloadAction<boolean>) => {
      state.shouldConfirmOnDelete = action.payload;
    },
    setOpenAccordions: (state, action: PayloadAction<ExpandedIndex>) => {
      state.openAccordions = action.payload;
    },
    setSystemConfig: (state, action: PayloadAction<InvokeAI.SystemConfig>) => {
      return {
        ...state,
        ...action.payload,
      };
    },
    setShouldDisplayGuides: (state, action: PayloadAction<boolean>) => {
      state.shouldDisplayGuides = action.payload;
    },
    processingCanceled: (state) => {
      state.isProcessing = false;
      state.isCancelable = true;
      state.currentStep = 0;
      state.totalSteps = 0;
      state.currentIteration = 0;
      state.totalIterations = 0;
      state.currentStatusHasSteps = false;
      state.currentStatus = i18n.t('common.statusProcessingCanceled');
    },
    generationRequested: (state) => {
      state.isProcessing = true;
      state.isCancelable = true;
      state.currentStep = 0;
      state.totalSteps = 0;
      state.currentIteration = 0;
      state.totalIterations = 0;
      state.currentStatusHasSteps = false;
      state.currentStatus = i18n.t('common.statusPreparing');
    },
    setModelList: (
      state,
      action: PayloadAction<InvokeAI.ModelList | Record<string, never>>
    ) => {
      state.model_list = action.payload;
    },
    setIsCancelable: (state, action: PayloadAction<boolean>) => {
      state.isCancelable = action.payload;
    },
    modelChangeRequested: (state) => {
      state.currentStatus = i18n.t('common.statusLoadingModel');
      state.isCancelable = false;
      state.isProcessing = true;
      state.currentStatusHasSteps = false;
    },
    modelConvertRequested: (state) => {
      state.currentStatus = i18n.t('common.statusConvertingModel');
      state.isCancelable = false;
      state.isProcessing = true;
      state.currentStatusHasSteps = false;
    },
    modelMergingRequested: (state) => {
      state.currentStatus = i18n.t('common.statusMergingModels');
      state.isCancelable = false;
      state.isProcessing = true;
      state.currentStatusHasSteps = false;
    },
    setSaveIntermediatesInterval: (state, action: PayloadAction<number>) => {
      state.saveIntermediatesInterval = action.payload;
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
    setProcessingIndeterminateTask: (state, action: PayloadAction<string>) => {
      state.isProcessing = true;
      state.currentStatus = action.payload;
      state.currentStatusHasSteps = false;
    },
    setSearchFolder: (state, action: PayloadAction<string | null>) => {
      state.searchFolder = action.payload;
    },
    setFoundModels: (
      state,
      action: PayloadAction<InvokeAI.FoundModel[] | null>
    ) => {
      state.foundModels = action.payload;
    },
    setOpenModel: (state, action: PayloadAction<string | null>) => {
      state.openModel = action.payload;
    },
    setCancelType: (state, action: PayloadAction<CancelType>) => {
      state.cancelOptions.cancelType = action.payload;
    },
    setCancelAfter: (state, action: PayloadAction<number | null>) => {
      state.cancelOptions.cancelAfter = action.payload;
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
    cancelTypeChanged: (state, action: PayloadAction<CancelType>) => {
      state.cancelType = action.payload;
    },
    /**
     * The array of subscribed node ids was changed
     */
    subscribedNodeIdsSet: (state, action: PayloadAction<string[]>) => {
      state.subscribedNodeIds = action.payload;
    },
    // /**
    //  * `shouldTransformUrls` was changed
    //  */
    // shouldTransformUrlsChanged: (state, action: PayloadAction<boolean>) => {
    //   state.shouldTransformUrls = action.payload;
    // },
    // /**
    //  * `disabledTabs` was changed
    //  */
    // disabledTabsChanged: (state, action: PayloadAction<InvokeTabName[]>) => {
    //   state.disabledTabs = action.payload;
    // },
    // /**
    //  * `disabledFeatures` was changed
    //  */
    // disabledFeaturesChanged: (
    //   state,
    //   action: PayloadAction<InvokeAI.AppFeature[]>
    // ) => {
    //   state.disabledFeatures = action.payload;
    // },
  },
  extraReducers(builder) {
    /**
     * Socket Subscribed
     */
    builder.addCase(socketSubscribed, (state, action) => {
      state.sessionId = action.payload.sessionId;
    });

    /**
     * Socket Unsubscribed
     */
    builder.addCase(socketUnsubscribed, (state) => {
      state.sessionId = null;
    });

    /**
     * Socket Connected
     */
    builder.addCase(socketConnected, (state, action) => {
      const { timestamp } = action.payload;

      state.isConnected = true;
      state.currentStatus = i18n.t('common.statusConnected');
      state.log.push({
        timestamp,
        message: `Connected to server`,
        level: 'info',
      });
      state.toastQueue.push(
        makeToast({ title: i18n.t('toast.connected'), status: 'success' })
      );
    });

    /**
     * Socket Disconnected
     */
    builder.addCase(socketDisconnected, (state, action) => {
      const { timestamp } = action.payload;

      state.isConnected = false;
      state.currentStatus = i18n.t('common.statusDisconnected');
      state.log.push({
        timestamp,
        message: `Disconnected from server`,
        level: 'error',
      });
      state.toastQueue.push(
        makeToast({ title: i18n.t('toast.disconnected'), status: 'error' })
      );
    });

    /**
     * Invocation Started
     */
    builder.addCase(invocationStarted, (state) => {
      state.isProcessing = true;
      state.isCancelable = true;
      state.currentStatusHasSteps = false;
      state.currentStatus = i18n.t('common.statusGenerating');
    });

    /**
     * Generator Progress
     */
    builder.addCase(generatorProgress, (state, action) => {
      const {
        step,
        total_steps,
        progress_image,
        node,
        source_node_id,
        graph_execution_state_id,
      } = action.payload.data;

      state.currentStatusHasSteps = true;
      state.currentStep = step + 1; // TODO: step starts at -1, think this is a bug
      state.totalSteps = total_steps;
      state.progressImage = progress_image ?? null;
    });

    /**
     * Invocation Complete
     */
    builder.addCase(invocationComplete, (state, action) => {
      const { data, timestamp } = action.payload;

      state.isProcessing = false;
      state.currentStep = 0;
      state.totalSteps = 0;
      state.progressImage = null;
      state.currentStatus = i18n.t('common.statusProcessingComplete');

      // TODO: handle logging for other invocation types
      if (isImageOutput(data.result)) {
        state.log.push({
          timestamp,
          message: `Generated: ${data.result.image.image_name}`,
          level: 'info',
        });
      }
    });

    /**
     * Invocation Error
     */
    builder.addCase(invocationError, (state, action) => {
      const { data, timestamp } = action.payload;

      state.log.push({
        timestamp,
        message: `Server error: ${data.error}`,
        level: 'error',
      });

      state.wasErrorSeen = true;
      state.progressImage = null;
      state.isProcessing = false;

      state.toastQueue.push(
        makeToast({ title: i18n.t('toast.serverError'), status: 'error' })
      );

      state.log.push({
        timestamp,
        message: `Server error: ${data.error}`,
        level: 'error',
      });
    });

    /**
     * Session Invoked - PENDING
     */

    builder.addCase(sessionInvoked.pending, (state) => {
      state.currentStatus = i18n.t('common.statusPreparing');
    });

    /**
     * Session Canceled
     */
    builder.addCase(sessionCanceled.fulfilled, (state, action) => {
      const { timestamp } = action.payload;

      state.isProcessing = false;
      state.isCancelable = false;
      state.isCancelScheduled = false;
      state.currentStep = 0;
      state.totalSteps = 0;
      state.progressImage = null;

      state.toastQueue.push(
        makeToast({ title: i18n.t('toast.canceled'), status: 'warning' })
      );

      state.log.push({
        timestamp,
        message: `Processing canceled`,
        level: 'warning',
      });
    });

    /**
     * Initial Image Selected
     */
    builder.addCase(initialImageSelected, (state) => {
      state.toastQueue.push(makeToast(i18n.t('toast.sentToImageToImage')));
    });

    /**
     * Received available models from the backend
     */
    builder.addCase(receivedModels.fulfilled, (state, action) => {
      state.wereModelsReceived = true;
    });

    /**
     * OpenAPI schema was received and parsed
     */
    builder.addCase(receivedOpenAPISchema.fulfilled, (state, action) => {
      state.wasSchemaParsed = true;
    });
  },
});

export const {
  setShouldDisplayInProgressType,
  setIsProcessing,
  addLogEntry,
  setShouldShowLogViewer,
  setIsConnected,
  setSocketId,
  setShouldConfirmOnDelete,
  setOpenAccordions,
  setSystemStatus,
  setCurrentStatus,
  setSystemConfig,
  setShouldDisplayGuides,
  processingCanceled,
  errorOccurred,
  errorSeen,
  setModelList,
  setIsCancelable,
  modelChangeRequested,
  modelConvertRequested,
  modelMergingRequested,
  setSaveIntermediatesInterval,
  setEnableImageDebugging,
  generationRequested,
  addToast,
  clearToastQueue,
  setProcessingIndeterminateTask,
  setSearchFolder,
  setFoundModels,
  setOpenModel,
  setCancelType,
  setCancelAfter,
  cancelScheduled,
  scheduledCancelAborted,
  cancelTypeChanged,
  subscribedNodeIdsSet,
  // shouldTransformUrlsChanged,
  // disabledTabsChanged,
  // disabledFeaturesChanged,
} = systemSlice.actions;

export default systemSlice.reducer;
