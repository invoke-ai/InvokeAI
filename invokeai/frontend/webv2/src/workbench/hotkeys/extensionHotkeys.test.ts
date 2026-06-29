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
});
