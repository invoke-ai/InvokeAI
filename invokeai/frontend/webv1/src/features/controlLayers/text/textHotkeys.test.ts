import { describe, expect, it } from 'vitest';

import { isAllowedTextShortcut } from './textHotkeys';

describe('text hotkey suppression', () => {
  const buildEvent = (key: string, options?: Partial<KeyboardEvent>) =>
    ({
      key,
      ctrlKey: options?.ctrlKey ?? false,
      metaKey: options?.metaKey ?? false,
    }) as KeyboardEvent;

  it('allows copy/paste/undo/redo shortcuts', () => {
    expect(isAllowedTextShortcut(buildEvent('c', { ctrlKey: true }))).toBe(true);
    expect(isAllowedTextShortcut(buildEvent('v', { metaKey: true }))).toBe(true);
    expect(isAllowedTextShortcut(buildEvent('z', { ctrlKey: true }))).toBe(true);
    expect(isAllowedTextShortcut(buildEvent('y', { metaKey: true }))).toBe(true);
  });

  it('blocks other hotkeys by default', () => {
    expect(isAllowedTextShortcut(buildEvent('b', { ctrlKey: true }))).toBe(false);
    expect(isAllowedTextShortcut(buildEvent('Escape'))).toBe(false);
  });
});
