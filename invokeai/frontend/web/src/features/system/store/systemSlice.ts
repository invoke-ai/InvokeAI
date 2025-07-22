import type { PayloadAction, Selector } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import type { LogNamespace } from 'app/logging/logger';
import { zLogNamespace } from 'app/logging/logger';
import { EMPTY_ARRAY } from 'app/store/constants';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import { uniq } from 'es-toolkit/compat';

import type { Language, SystemState } from './types';

const getInitialState = (): SystemState => ({
  _version: 2,
  shouldConfirmOnDelete: true,
  shouldAntialiasProgressImage: false,
  shouldConfirmOnNewSession: true,
  language: 'en',
  shouldUseNSFWChecker: false,
  shouldUseWatermarker: false,
  shouldEnableInformationalPopovers: true,
  shouldEnableModelDescriptions: true,
  logIsEnabled: true,
  logLevel: 'debug',
  logNamespaces: [...zLogNamespace.options],
  shouldShowInvocationProgressDetail: false,
  shouldHighlightFocusedRegions: false,
});

export const slice = createSlice({
  name: 'system',
  initialState: getInitialState(),
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
    setShouldEnableModelDescriptions(state, action: PayloadAction<boolean>) {
      state.shouldEnableModelDescriptions = action.payload;
    },
    shouldConfirmOnNewSessionToggled(state) {
      state.shouldConfirmOnNewSession = !state.shouldConfirmOnNewSession;
    },
    setShouldShowInvocationProgressDetail(state, action: PayloadAction<boolean>) {
      state.shouldShowInvocationProgressDetail = action.payload;
    },
    setShouldHighlightFocusedRegions(state, action: PayloadAction<boolean>) {
      state.shouldHighlightFocusedRegions = action.payload;
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
  setShouldEnableModelDescriptions,
  shouldConfirmOnNewSessionToggled,
  setShouldShowInvocationProgressDetail,
  setShouldHighlightFocusedRegions,
} = slice.actions;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrate = (state: any): any => {
  if (!('_version' in state)) {
    state._version = 1;
  }
  if (state._version === 1) {
    state.language = (state as SystemState).language.replace('_', '-');
    state._version = 2;
  }
  return state;
};

export const systemSliceConfig: SliceConfig<SystemState> = {
  slice,
  getInitialState,
  persistConfig: {
    migrate,
  },
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
export const selectSystemShouldEnableModelDescriptions = createSystemSelector(
  (system) => system.shouldEnableModelDescriptions
);
export const selectSystemShouldEnableHighlightFocusedRegions = createSystemSelector(
  (system) => system.shouldHighlightFocusedRegions
);
export const selectSystemShouldConfirmOnNewSession = createSystemSelector((system) => system.shouldConfirmOnNewSession);
export const selectSystemShouldShowInvocationProgressDetail = createSystemSelector(
  (system) => system.shouldShowInvocationProgressDetail
);
