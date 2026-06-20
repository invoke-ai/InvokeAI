import { describe, expect, it, vi } from 'vitest';

import type { RegisteredHotkey } from './types';

import { applyCustomHotkeys, resolveHotkey } from './resolve';

const event = { target: null } as KeyboardEvent;
const context = {
  activeInstanceId: null,
  activeWidgetTypeId: null,
  focusedRegion: null,
  isModalLayerActive: false,
} as const;

const base = {
  category: 'app',
  defaultKeys: ['x'] as string[],
  implemented: true,
  keys: ['x'] as string[],
  preventDefault: true,
} as const;

describe('resolveHotkey', () => {
  it('prefers active widget over global', () => {
    const globalHotkey: RegisteredHotkey = {
      ...base,
      commandId: 'global',
      id: 'global',
      scope: { kind: 'global' },
      title: 'Global',
    };
    const widgetHotkey: RegisteredHotkey = {
      ...base,
      commandId: 'widget',
      id: 'widget',
      scope: { kind: 'widget', typeId: 'gallery' },
      title: 'Widget',
    };

    expect(
      resolveHotkey({
        context: { ...context, activeInstanceId: 'gallery-1', activeWidgetTypeId: 'gallery', focusedRegion: 'left' },
        event,
        hotkeys: [globalHotkey, widgetHotkey],
        matchedKey: 'x',
      })?.commandId
    ).toBe('widget');
  });

  it('prefers active instance over widget', () => {
    const widgetHotkey: RegisteredHotkey = {
      ...base,
      commandId: 'widget',
      id: 'widget',
      scope: { kind: 'widget', typeId: 'gallery' },
      title: 'Widget',
    };
    const instanceHotkey: RegisteredHotkey = {
      ...base,
      commandId: 'instance',
      id: 'instance',
      scope: { instanceId: 'gallery-1', kind: 'instance' },
      title: 'Instance',
    };

    expect(
      resolveHotkey({
        context: { ...context, activeInstanceId: 'gallery-1', activeWidgetTypeId: 'gallery', focusedRegion: 'left' },
        event,
        hotkeys: [widgetHotkey, instanceHotkey],
        matchedKey: 'x',
      })?.commandId
    ).toBe('instance');
  });

  it('resolves by the matched canonical key instead of re-serializing the event', () => {
    const hotkey: RegisteredHotkey = {
      ...base,
      commandId: 'symbol',
      id: 'symbol',
      keys: ['shift+1'],
      scope: { kind: 'global' },
      title: 'Symbol',
    };

    expect(
      resolveHotkey({
        context,
        event: { ...event, key: '!' } as KeyboardEvent,
        hotkeys: [hotkey],
        matchedKey: 'shift+1',
      })?.commandId
    ).toBe('symbol');
  });

  it('does not match region-less focused-region hotkeys without a focused region', () => {
    const globalHotkey: RegisteredHotkey = {
      ...base,
      commandId: 'global',
      id: 'global',
      scope: { kind: 'global' },
      title: 'Global',
    };
    const focusedHotkey: RegisteredHotkey = {
      ...base,
      commandId: 'focused',
      id: 'focused',
      scope: { kind: 'focused-region' },
      title: 'Focused',
    };

    expect(resolveHotkey({ context, event, hotkeys: [globalHotkey, focusedHotkey], matchedKey: 'x' })?.commandId).toBe(
      'global'
    );
    expect(
      resolveHotkey({
        context: { ...context, focusedRegion: 'left' },
        event,
        hotkeys: [globalHotkey, focusedHotkey],
        matchedKey: 'x',
      })?.commandId
    ).toBe('focused');
  });

  it('suppresses editable targets unless allowed', () => {
    class FakeHTMLElement {
      closest(): FakeHTMLElement {
        return this;
      }
    }
    vi.stubGlobal('HTMLElement', FakeHTMLElement);
    const input = new FakeHTMLElement();
    const hotkey: RegisteredHotkey = {
      ...base,
      commandId: 'global',
      id: 'global',
      scope: { kind: 'global' },
      title: 'Global',
    };

    expect(
      resolveHotkey({
        context,
        event: { target: input } as unknown as KeyboardEvent,
        hotkeys: [hotkey],
        matchedKey: 'x',
      })
    ).toBeNull();
  });

  it('suppresses hotkeys while a modal layer is active unless allowed', () => {
    const hotkey: RegisteredHotkey = {
      ...base,
      commandId: 'global',
      id: 'global',
      scope: { kind: 'global' },
      title: 'Global',
    };
    const modalHotkey: RegisteredHotkey = {
      ...hotkey,
      allowInModal: true,
      commandId: 'modal',
      id: 'modal',
      title: 'Modal',
    };

    expect(
      resolveHotkey({
        context: { ...context, isModalLayerActive: true },
        event,
        hotkeys: [hotkey],
        matchedKey: 'x',
      })
    ).toBeNull();
    expect(
      resolveHotkey({
        context: { ...context, isModalLayerActive: true },
        event,
        hotkeys: [hotkey, modalHotkey],
        matchedKey: 'x',
      })?.commandId
    ).toBe('modal');
  });

  it('keeps an empty custom key array as a disabled hotkey', () => {
    expect(applyCustomHotkeys({ defaultKeys: ['x'], id: 'app.invoke' }, { 'app.invoke': [] }).keys).toEqual([]);
  });
});
