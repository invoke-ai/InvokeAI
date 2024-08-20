import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { LogNamespace } from 'app/logging/logger';
import { zLogNamespace } from 'app/logging/logger';
import type { PersistConfig, RootState } from 'app/store/store';
import { uniq } from 'lodash-es';

import type { Language, SystemState } from './types';

const initialSystemState: SystemState = {
  _version: 1,
  shouldConfirmOnDelete: true,
  shouldAntialiasProgressImage: false,
  language: 'en',
  shouldUseNSFWChecker: false,
  shouldUseWatermarker: false,
  shouldEnableInformationalPopovers: true,
  logIsEnabled: true,
  logLevel: 'debug',
  logNamespaces: [...zLogNamespace.options],
};

export const systemSlice = createSlice({
  name: 'system',
  initialState: initialSystemState,
  reducers: {
    setShouldConfirmOnDelete: (state, action: PayloadAction<boolean>) => {
      state.shouldConfirmOnDelete = action.payload;
    },
    logIsEnabledChanged: (state, action: PayloadAction<SystemState['logIsEnabled']>) => {
      state.logIsEnabled = action.payload;
    },
    logLevelChanged: (state, action: PayloadAction<SystemState['logLevel']>) => {
      state.logLevel = action.payload;
    },
    logNamespaceToggled: (state, action: PayloadAction<LogNamespace>) => {
      if (state.logNamespaces.includes(action.payload)) {
        state.logNamespaces = uniq(state.logNamespaces.filter((n) => n !== action.payload));
      } else {
        state.logNamespaces = uniq([...state.logNamespaces, action.payload]);
      }
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
});

export const {
  setShouldConfirmOnDelete,
  logIsEnabledChanged,
  logLevelChanged,
  logNamespaceToggled,
  shouldAntialiasProgressImageChanged,
  languageChanged,
  shouldUseNSFWCheckerChanged,
  shouldUseWatermarkerChanged,
  setShouldEnableInformationalPopovers,
} = systemSlice.actions;

export const selectSystemSlice = (state: RootState) => state.system;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrateSystemState = (state: any): any => {
  if (!('_version' in state)) {
    state._version = 1;
  }
  return state;
};

export const systemPersistConfig: PersistConfig<SystemState> = {
  name: systemSlice.name,
  initialState: initialSystemState,
  migrate: migrateSystemState,
  persistDenylist: [],
};
