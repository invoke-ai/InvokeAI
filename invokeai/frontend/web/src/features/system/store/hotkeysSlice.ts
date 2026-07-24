import type { PayloadAction } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import { isPlainObject } from 'es-toolkit';
import { canonicalizeHotkeyString } from 'features/system/components/HotkeysModal/hotkeyStrings';
import { assert } from 'tsafe';

import { type HotkeysState, zHotkeysState } from './hotkeysTypes';

const getInitialState = (): HotkeysState => ({
  _version: 2,
  customHotkeys: {},
});

const slice = createSlice({
  name: 'hotkeys',
  initialState: getInitialState(),
  reducers: {
    hotkeyChanged: (state, action: PayloadAction<{ id: string; hotkeys: string[] }>) => {
      const { id, hotkeys } = action.payload;
      state.customHotkeys[id] = hotkeys;
    },
    hotkeyReset: (state, action: PayloadAction<string>) => {
      delete state.customHotkeys[action.payload];
    },
    allHotkeysReset: (state) => {
      state.customHotkeys = {};
    },
  },
});

export const { hotkeyChanged, hotkeyReset, allHotkeysReset } = slice.actions;

export const hotkeysSliceConfig: SliceConfig<typeof slice> = {
  slice,
  schema: zHotkeysState,
  getInitialState,
  persistConfig: {
    migrate: (state) => {
      assert(isPlainObject(state));
      if (!('_version' in state)) {
        state._version = 1;
      }
      if (state._version === 1) {
        state._version = 2;
        if (isPlainObject(state.customHotkeys)) {
          for (const [id, hotkeys] of Object.entries(state.customHotkeys)) {
            if (Array.isArray(hotkeys)) {
              state.customHotkeys[id] = hotkeys
                .filter((hotkey): hotkey is string => typeof hotkey === 'string')
                .map((hotkey) => canonicalizeHotkeyString(hotkey));
            }
          }
        }
      }
      return zHotkeysState.parse(state);
    },
  },
};

const selectHotkeysSlice = (state: RootState) => state.hotkeys;
export const selectCustomHotkeys = createSelector(selectHotkeysSlice, (hotkeys) => hotkeys.customHotkeys);
