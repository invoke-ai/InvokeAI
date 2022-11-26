import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';
import { ExpandedIndex, UseToastOptions } from '@chakra-ui/react';
import * as InvokeAI from 'app/invokeai';

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
  currentStatus: 'Disconnected',
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
      state.currentStatus = 'Error';
      state.wasErrorSeen = false;
    },
    errorSeen: (state) => {
      state.hasError = false;
      state.wasErrorSeen = true;
      state.currentStatus = state.isConnected ? 'Connected' : 'Disconnected';
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
      state.currentStatus = 'Processing canceled';
    },
    generationRequested: (state) => {
      state.isProcessing = true;
      state.isCancelable = true;
      state.currentStep = 0;
      state.totalSteps = 0;
      state.currentIteration = 0;
      state.totalIterations = 0;
      state.currentStatusHasSteps = false;
      state.currentStatus = 'Preparing';
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
      state.currentStatus = 'Loading Model';
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
} = systemSlice.actions;

export default systemSlice.reducer;
