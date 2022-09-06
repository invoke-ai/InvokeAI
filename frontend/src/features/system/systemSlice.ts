import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';
import { testLogs } from '../../app/testingData';

export interface SystemState {
  shouldDisplayInProgress: boolean;
  shouldFitToWidthHeight: boolean;
  isProcessing: boolean;
  progress: number;
  log: Array<string>;
  isGFPGANAvailable: boolean;
  isESRGANAvailable: boolean;
  isConnected: boolean;
}

const initialSystemState = {
  isConnected: false,
  isProcessing: false,
  progress: 0,
  log: testLogs,
  shouldFitToWidthHeight: false,
  shouldDisplayInProgress: false,
  isGFPGANAvailable: true,
  isESRGANAvailable: true,
};

const initialState: SystemState = initialSystemState;

export const systemSlice = createSlice({
  name: 'system',
  initialState,
  reducers: {
    setShouldDisplayInProgress: (state, action: PayloadAction<boolean>) => {
      state.shouldDisplayInProgress = action.payload;
    },
    setShouldFitToWidthHeight: (state, action: PayloadAction<boolean>) => {
      state.shouldFitToWidthHeight = action.payload;
    },
    setIsProcessing: (state, action: PayloadAction<boolean>) => {
      state.isProcessing = action.payload;
      state.progress = 0;
    },
    setProgress: (state, action: PayloadAction<number>) => {
      state.progress = action.payload;
    },
    appendLog: (state, action: PayloadAction<string>) => {
      state.log.push(action.payload);
    },
    setIsConnected: (state, action: PayloadAction<boolean>) => {
      state.isConnected = action.payload;
    },
  },
});

export const {
  setShouldDisplayInProgress,
  setShouldFitToWidthHeight,
  setIsProcessing,
  setProgress,
  appendLog,
  setIsConnected,
} = systemSlice.actions;

export default systemSlice.reducer;
