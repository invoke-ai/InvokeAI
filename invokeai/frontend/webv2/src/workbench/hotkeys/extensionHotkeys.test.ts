import type { WidgetHotkeyContribution } from '@workbench/widgetContracts';

import { describe, expect, it } from 'vitest';

import { toExtensionHotkeyDefinition } from './extensionHotkeys';

describe('toExtensionHotkeyDefinition', () => {
  it('preserves the registering widget source for command execution', () => {
    const source = { instanceId: 'alpha', projectId: 'project-1', region: 'right', typeId: 'test-widget' } as const;

    expect(
      toExtensionHotkeyDefinition({
        commandId: 'test.command',
        defaultKeys: ['mod+x'],
        id: 'test.hotkey',
        scope: 'global',
        source,
        title: 'Test hotkey',
      })
    ).toMatchObject({ source });
  });

  it('scopes a focused-region shortcut to its docked region, or to its own window when it floats', () => {
    const contribution: WidgetHotkeyContribution = {
      commandId: 'test.command',
      defaultKeys: ['mod+x'],
      id: 'test.hotkey',
      scope: 'focused-region',
      title: 'Test hotkey',
    };
    const docked = { instanceId: 'alpha', projectId: 'project-1', region: 'right', typeId: 'test-widget' } as const;
    const floating = { ...docked, region: 'floating' } as const;

    expect(toExtensionHotkeyDefinition({ ...contribution, source: docked })?.scope).toEqual({
      kind: 'focused-region',
      region: 'right',
    });
    // A window is not a region other widgets share, so the shortcut follows this instance — still as a
    // focused-region shortcut, at that priority.
    expect(toExtensionHotkeyDefinition({ ...contribution, source: floating })?.scope).toEqual({
      floatingInstanceId: 'alpha',
      kind: 'focused-region',
    });
  });
});
