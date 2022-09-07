import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';
import { testLogs } from '../../app/testingData';

export interface SystemState {
  shouldDisplayInProgress: boolean;
  isProcessing: boolean;
  currentStep: number;
  log: Array<string>;
  shouldShowLogViewer: boolean;
  isGFPGANAvailable: boolean;
  isESRGANAvailable: boolean;
  isConnected: boolean;
  socketId: string;
}

const initialSystemState = {
  isConnected: false,
  isProcessing: false,
  currentStep: 0,
  log: testLogs,
  shouldShowLogViewer: false,
  shouldDisplayInProgress: false,
  isGFPGANAvailable: false,
  isESRGANAvailable: false,
  socketId: '',
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
    appendLog: (state, action: PayloadAction<string>) => {
      state.log.push(action.payload);
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
  },
});

export const {
  setShouldDisplayInProgress,
  setIsProcessing,
  setCurrentStep,
  appendLog,
  setShouldShowLogViewer,
  setIsConnected,
  setSocketId,
} = systemSlice.actions;

export default systemSlice.reducer;
