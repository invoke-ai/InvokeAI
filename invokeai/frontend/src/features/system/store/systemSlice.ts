import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';
import { ExpandedIndex, UseToastOptions } from '@chakra-ui/react';
import * as InvokeAI from 'app/invokeai';
import i18n from 'i18n';

export type LogLevel = 'info' | 'warning' | 'error';

export interface LogEntry {
  timestamp: string;
  level: LogLevel;
  message: string;
}

export interface Log {
  [index: number]: LogEntry;
}

export type ReadinessPayload = {
  isReady: boolean;
  reasonsWhyNotReady: string[];
};

export type InProgressImageType = 'none' | 'full-res' | 'latents';

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
  openAccordions: [],
  currentStep: 0,
  totalSteps: 0,
  currentIteration: 0,
  totalIterations: 0,
  currentStatus: i18n.isInitialized
    ? i18n.t('common:statusDisconnected')
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
      state.currentStatus = i18n.t('common:statusError');
      state.wasErrorSeen = false;
    },
    errorSeen: (state) => {
      state.hasError = false;
      state.wasErrorSeen = true;
      state.currentStatus = state.isConnected
        ? i18n.t('common:statusConnected')
        : i18n.t('common:statusDisconnected');
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
      state.currentStatus = i18n.t('common:statusProcessingCanceled');
    },
    generationRequested: (state) => {
      state.isProcessing = true;
      state.isCancelable = true;
      state.currentStep = 0;
      state.totalSteps = 0;
      state.currentIteration = 0;
      state.totalIterations = 0;
      state.currentStatusHasSteps = false;
      state.currentStatus = i18n.t('common:statusPreparing');
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
      state.currentStatus = i18n.t('common:statusLoadingModel');
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
  setSaveIntermediatesInterval,
  setEnableImageDebugging,
  generationRequested,
  addToast,
  clearToastQueue,
  setProcessingIndeterminateTask,
  setSearchFolder,
  setFoundModels,
  setOpenModel,
} = systemSlice.actions;

export default systemSlice.reducer;
