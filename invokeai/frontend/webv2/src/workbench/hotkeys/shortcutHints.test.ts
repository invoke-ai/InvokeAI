import { describe, expect, it } from 'vitest';

import type { HotkeyContext, HotkeyDefinition } from './types';

import { createShortcutHintSources, type ShortcutHintSource } from './hintSources';
import { MOD_KEY_LABEL, IS_MAC_OS } from './keys';
import { applyCustomHotkeys } from './resolve';
import { getVisibleShortcutHintCount, resolveShortcutHints } from './shortcutHints';

const context: HotkeyContext = {
  activeInstanceId: 'canvas',
  activeWidgetTypeId: 'canvas',
  focusedRegion: 'center',
  isModalPresent: false,
  projectId: 'one',
};
const command: HotkeyDefinition = {
  category: 'canvas',
  commandId: 'canvas.brushSizeDown',
  defaultKeys: ['['],
  id: 'canvas.brushSizeDown',
  scope: { kind: 'widget', typeId: 'canvas' },
  title: 'Smaller brush',
};

describe('contextual shortcut resolution', () => {
  it('chooses executable alternatives to guide activation/dismissal keys while preserving gesture instructions', () => {
    const hotkey = applyCustomHotkeys(command, {
      [command.id]: [' RETURN ', 'space', 'mod+Escape', 'mod+Enter'],
    });
    expect(resolveShortcutHints([{ commandId: command.commandId }], [hotkey], context, null)[0]?.parts).toEqual([
      MOD_KEY_LABEL,
      'enter',
    ]);
    expect(
      resolveShortcutHints(
        [{ commandId: command.commandId }],
        [applyCustomHotkeys(command, { [command.id]: ['enter', 'space', 'esc'] })],
        context,
        null
      )
    ).toEqual([]);
    expect(resolveShortcutHints([{ labelKey: 'apply', parts: ['enter'] }], [], context, null)[0]?.parts).toEqual([
      'enter',
    ]);
  });

  it('uses platform-normalized display parts for configured chords and preserves bare gesture modifiers', () => {
    const hotkey = applyCustomHotkeys(command, { [command.id]: [' ALT + mOd + J '] });
    expect(resolveShortcutHints([{ commandId: command.commandId }], [hotkey], context, null)[0]?.parts).toEqual([
      MOD_KEY_LABEL,
      IS_MAC_OS ? 'option' : 'alt',
      'j',
    ]);
    expect(resolveShortcutHints([{ labelKey: 'hold', parts: ['shift', 'alt'] }], [], context, null)[0]?.parts).toEqual([
      'shift',
      IS_MAC_OS ? 'option' : 'alt',
    ]);
  });
  it('shows a remapped surviving binding and omits unbound, shadowed and out-of-scope commands', () => {
    const global: HotkeyDefinition = { ...command, id: 'global', commandId: 'global', scope: { kind: 'global' } };
    const override: HotkeyDefinition = {
      ...command,
      id: 'override',
      commandId: 'override',
      scope: { kind: 'instance', instanceId: 'canvas' },
    };
    const hints = [{ commandId: command.commandId }];
    const registered = [global, command].map((definition) =>
      applyCustomHotkeys(definition, { [command.id]: ['j', '['] })
    );
    expect(resolveShortcutHints(hints, registered, context, null)).toEqual([
      { id: command.commandId, labelKey: undefined, parts: ['j'], title: 'Smaller brush' },
    ]);
    expect(resolveShortcutHints(hints, [applyCustomHotkeys(command, { [command.id]: [] })], context, null)).toEqual([]);
    expect(
      resolveShortcutHints(hints, [applyCustomHotkeys(override, {}), applyCustomHotkeys(command, {})], context, null)
    ).toEqual([]);
    expect(resolveShortcutHints(hints, registered, { ...context, activeWidgetTypeId: 'gallery' }, null)).toEqual([]);
    expect(resolveShortcutHints(hints, registered, { ...context, isModalPresent: true }, null)).toEqual([]);
  });

  it('drops entire low-priority hints and caps an abundant-width row at four', () => {
    expect(getVisibleShortcutHintCount(250, 100, [70, 70, 70])).toBe(2);
    expect(getVisibleShortcutHintCount(150, 100, [70, 20])).toBe(0);
    expect(getVisibleShortcutHintCount(1000, 100, [70, 70, 70, 70, 70])).toBe(4);
  });
});

describe('mounted hint source ownership', () => {
  it('fences project reads and cleans up subscriptions without an old release removing a replacement', () => {
    const sources = createShortcutHintSources();
    let subscriptions = 0;
    const source: ShortcutHintSource = {
      instanceId: 'canvas',
      projectId: 'one',
      getSnapshot: () => ({ hints: [], titleKey: 'brush' }),
      subscribe: () => {
        subscriptions++;
        return () => {
          subscriptions--;
        };
      },
    };
    const release = sources.register(source);
    expect(sources.get('two', 'canvas')).toBeNull();
    const replacement = { ...source, projectId: 'two' };
    const releaseReplacement = sources.register(replacement);
    release();
    expect(sources.get('two', 'canvas')).toBe(replacement);
    expect(subscriptions).toBe(1);
    releaseReplacement();
    expect(subscriptions).toBe(0);
    expect(sources.get('two', 'canvas')).toBeNull();
  });
});
