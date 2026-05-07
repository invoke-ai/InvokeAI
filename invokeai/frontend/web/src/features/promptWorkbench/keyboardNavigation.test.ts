import { describe, expect, it } from 'vitest';

import {
  clampNavigationIndex,
  getNextNavigationIndex,
  getPromptWorkbenchKeyboardIntent,
} from './keyboardNavigation';

describe('prompt workbench keyboard navigation', () => {
  it('wraps arrow movement through autocomplete options', () => {
    expect(getNextNavigationIndex({ currentIndex: 0, direction: 'next', itemCount: 3 })).toBe(1);
    expect(getNextNavigationIndex({ currentIndex: 2, direction: 'next', itemCount: 3 })).toBe(0);
    expect(getNextNavigationIndex({ currentIndex: 0, direction: 'previous', itemCount: 3 })).toBe(2);
  });

  it('clamps stale active indices when option counts change', () => {
    expect(clampNavigationIndex(5, 3)).toBe(2);
    expect(clampNavigationIndex(-1, 3)).toBe(0);
    expect(clampNavigationIndex(2, 0)).toBe(0);
  });

  it('maps autocomplete enter and tab to default wildcard insertion', () => {
    expect(getPromptWorkbenchKeyboardIntent({ key: 'Enter', shiftKey: false, target: 'autocomplete' })).toBe(
      'insert_wildcard'
    );
    expect(getPromptWorkbenchKeyboardIntent({ key: 'Tab', shiftKey: false, target: 'autocomplete' })).toBe(
      'insert_wildcard'
    );
  });

  it('maps autocomplete shift-enter to fixed-value expansion', () => {
    expect(getPromptWorkbenchKeyboardIntent({ key: 'Enter', shiftKey: true, target: 'autocomplete' })).toBe(
      'open_fixed_values'
    );
  });

  it('maps fixed-value enter to concrete value insertion', () => {
    expect(getPromptWorkbenchKeyboardIntent({ key: 'Enter', shiftKey: false, target: 'fixed_values' })).toBe(
      'insert_fixed_value'
    );
  });

  it('maps escape to dismiss for both keyboard targets', () => {
    expect(getPromptWorkbenchKeyboardIntent({ key: 'Escape', shiftKey: false, target: 'autocomplete' })).toBe('dismiss');
    expect(getPromptWorkbenchKeyboardIntent({ key: 'Escape', shiftKey: false, target: 'fixed_values' })).toBe('dismiss');
  });
});
