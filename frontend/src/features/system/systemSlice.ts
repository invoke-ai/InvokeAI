import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';
import dateFormat from 'dateformat';

export interface LogEntry {
  timestamp: string;
  message: string;
}

export interface Log {
  [index: number]: LogEntry;
}

export interface SystemState {
  shouldDisplayInProgress: boolean;
  isProcessing: boolean;
  currentStep: number;
  log: Array<LogEntry>;
  shouldShowLogViewer: boolean;
  isGFPGANAvailable: boolean;
  isESRGANAvailable: boolean;
  isConnected: boolean;
  socketId: string;
  shouldConfirmOnDelete: boolean;
}

const initialSystemState = {
  isConnected: false,
  isProcessing: false,
  currentStep: 0,
  log: [],
  shouldShowLogViewer: false,
  shouldDisplayInProgress: false,
  isGFPGANAvailable: true,
  isESRGANAvailable: true,
  socketId: '',
  shouldConfirmOnDelete: true,
};

const initialState: SystemState = initialSystemState;

export const systemSlice = createSlice({
  name: 'system',
  initialState,
  reducers: {
    setShouldDisplayInProgress: (state, action: PayloadAction<boolean>) => {
      state.shouldDisplayInProgress = action.payload;
    },

    setIsProcessing: (state, action: PayloadAction<boolean>) => {
      state.isProcessing = action.payload;
      state.currentStep = 0;
    },
    setCurrentStep: (state, action: PayloadAction<number>) => {
      state.currentStep = action.payload;
    },
    addLogEntry: (state, action: PayloadAction<string>) => {
      const entry: LogEntry = {
        timestamp: dateFormat(new Date(), 'isoDateTime'),
        message: action.payload,
      };
      state.log.push(entry);
    },
    setShouldShowLogViewer: (state, action: PayloadAction<boolean>) => {
      state.shouldShowLogViewer = action.payload;
    },
    setIsConnected: (state, action: PayloadAction<boolean>) => {
      state.isConnected = action.payload;
    },
    setSocketId: (state, action: PayloadAction<string>) => {
      state.socketId = action.payload;
    },
    setShouldConfirmOnDelete: (state, action: PayloadAction<boolean>) => {
      state.shouldConfirmOnDelete = action.payload;
    },
  },
});

export const {
  setShouldDisplayInProgress,
  setIsProcessing,
  setCurrentStep,
  addLogEntry,
  setShouldShowLogViewer,
  setIsConnected,
  setSocketId,
  setShouldConfirmOnDelete,
} = systemSlice.actions;

export default systemSlice.reducer;
