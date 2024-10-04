import type { PayloadAction, Selector } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import type { LogNamespace } from 'app/logging/logger';
import { zLogNamespace } from 'app/logging/logger';
import { EMPTY_ARRAY } from 'app/store/constants';
import type { PersistConfig, RootState } from 'app/store/store';
import { uniq } from 'lodash-es';

import type { Language, SystemState } from './types';

const initialSystemState: SystemState = {
  _version: 1,
  shouldConfirmOnDelete: true,
  shouldAntialiasProgressImage: false,
  shouldConfirmOnNewSession: true,
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
        state.logNamespaces = uniq(state.logNamespaces.filter((n) => n !== action.payload)).toSorted();
      } else {
        state.logNamespaces = uniq([...state.logNamespaces, action.payload]).toSorted();
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
    shouldConfirmOnNewSessionToggled(state) {
      state.shouldConfirmOnNewSession = !state.shouldConfirmOnNewSession;
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
  shouldConfirmOnNewSessionToggled,
} = systemSlice.actions;

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

export const selectSystemSlice = (state: RootState) => state.system;
const createSystemSelector = <T>(selector: Selector<SystemState, T>) => createSelector(selectSystemSlice, selector);

export const selectSystemLogLevel = createSystemSelector((system) => system.logLevel);
export const selectSystemLogNamespaces = createSystemSelector((system) =>
  system.logNamespaces.length > 0 ? system.logNamespaces : EMPTY_ARRAY
);
export const selectSystemLogIsEnabled = createSystemSelector((system) => system.logIsEnabled);
export const selectSystemShouldConfirmOnDelete = createSystemSelector((system) => system.shouldConfirmOnDelete);
export const selectSystemShouldUseNSFWChecker = createSystemSelector((system) => system.shouldUseNSFWChecker);
export const selectSystemShouldUseWatermarker = createSystemSelector((system) => system.shouldUseWatermarker);
export const selectSystemShouldAntialiasProgressImage = createSystemSelector(
  (system) => system.shouldAntialiasProgressImage
);
export const selectSystemShouldEnableInformationalPopovers = createSystemSelector(
  (system) => system.shouldEnableInformationalPopovers
);
export const selectSystemShouldConfirmOnNewSession = createSystemSelector((system) => system.shouldConfirmOnNewSession);
