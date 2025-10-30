import { allHotkeysReset, hotkeyChanged, hotkeyReset } from 'features/system/store/hotkeysSlice';
import { describe, expect, it } from 'vitest';

import type { HotkeysState } from './hotkeysTypes';

describe('Hotkeys Slice', () => {
  const getInitialState = (): HotkeysState => ({
    _version: 1,
    customHotkeys: {},
  });

  describe('hotkeyChanged', () => {
    it('should add a custom hotkey', () => {
      const state = getInitialState();
      const action = hotkeyChanged({ id: 'app.invoke', hotkeys: ['ctrl+shift+enter'] });
      const result = {
        ...state,
        customHotkeys: {
          'app.invoke': ['ctrl+shift+enter'],
        },
      };
      expect(result).toEqual({
        _version: 1,
        customHotkeys: {
          'app.invoke': ['ctrl+shift+enter'],
        },
      });
    });

    it('should update an existing custom hotkey', () => {
      const state: HotkeysState = {
        _version: 1,
        customHotkeys: {
          'app.invoke': ['ctrl+enter'],
        },
      };
      const action = hotkeyChanged({ id: 'app.invoke', hotkeys: ['ctrl+shift+enter'] });
      const result = {
        ...state,
        customHotkeys: {
          'app.invoke': ['ctrl+shift+enter'],
        },
      };
      expect(result).toEqual({
        _version: 1,
        customHotkeys: {
          'app.invoke': ['ctrl+shift+enter'],
        },
      });
    });
  });

  describe('hotkeyReset', () => {
    it('should remove a custom hotkey', () => {
      const state: HotkeysState = {
        _version: 1,
        customHotkeys: {
          'app.invoke': ['ctrl+shift+enter'],
          'app.cancelQueueItem': ['shift+x'],
        },
      };
      const action = hotkeyReset('app.invoke');
      const result = {
        ...state,
        customHotkeys: {
          'app.cancelQueueItem': ['shift+x'],
        },
      };
      expect(result.customHotkeys).toEqual({
        'app.cancelQueueItem': ['shift+x'],
      });
    });
  });

  describe('allHotkeysReset', () => {
    it('should clear all custom hotkeys', () => {
      const state: HotkeysState = {
        _version: 1,
        customHotkeys: {
          'app.invoke': ['ctrl+shift+enter'],
          'app.cancelQueueItem': ['shift+x'],
        },
      };
      const action = allHotkeysReset();
      const result = {
        ...state,
        customHotkeys: {},
      };
      expect(result).toEqual({
        _version: 1,
        customHotkeys: {},
      });
    });
  });
});
