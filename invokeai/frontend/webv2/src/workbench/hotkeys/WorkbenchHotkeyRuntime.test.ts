import { describe, expect, it } from 'vitest';

import type { RegisteredHotkey } from './types';

import { getHotkeyExecutionSource } from './WorkbenchHotkeyRuntime';

const source = { instanceId: 'alpha', projectId: 'project-1', region: 'right', typeId: 'test-widget' } as const;
const activeSource = { instanceId: 'beta', projectId: 'project-1', region: 'left', typeId: 'other-widget' } as const;

const createHotkey = (overrides: Partial<RegisteredHotkey>): RegisteredHotkey => ({
  category: 'app',
  commandId: 'test.command',
  defaultKeys: ['x'],
  id: 'test.hotkey',
  keys: ['x'],
  scope: { kind: 'global' },
  title: 'Test hotkey',
  ...overrides,
});

describe('getHotkeyExecutionSource', () => {
  it('does not let source-less global hotkeys execute against the focused widget source', () => {
    expect(getHotkeyExecutionSource(createHotkey({}), activeSource)).toBeNull();
  });

  it('uses the registering source for sourced global hotkeys', () => {
    expect(getHotkeyExecutionSource(createHotkey({ source }), activeSource)).toBe(source);
  });

  it('uses the focused source for widget-scoped hotkeys', () => {
    expect(
      getHotkeyExecutionSource(createHotkey({ scope: { kind: 'widget', typeId: 'test-widget' }, source }), activeSource)
    ).toBe(activeSource);
  });
});
