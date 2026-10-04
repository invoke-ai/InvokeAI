import { describe, expect, it, vi } from 'vitest';

import type { RegisteredHotkey } from './types';

import { applyCustomHotkeys, resolveHotkey, toPlatformHotkey } from './resolve';

const event = { target: null } as KeyboardEvent;
const context = {
  activeInstanceId: null,
  activeWidgetTypeId: null,
  focusedRegion: null,
  isModalPresent: false,
  projectId: 'project-1',
} as const;

const base = {
  category: 'app',
  defaultKeys: ['x'] as string[],
  implemented: true,
  keys: ['x'] as string[],
  preventDefault: true,
} as const;

describe('resolveHotkey', () => {
  it('treats mod and the platform key it stands for as one press, leaving the choice to scope', () => {
    expect(toPlatformHotkey('mod+arrowup', false)).toBe('ctrl+arrowup');
    expect(toPlatformHotkey('shift+mod+a', true)).toBe('meta+shift+a');

    // The runtime hands over whichever spelling's binding fired first.
    const platformSpelling = toPlatformHotkey('mod+arrowup');
    const globalHotkey: RegisteredHotkey = {
      ...base,
      commandId: 'global',
      id: 'global',
      keys: [platformSpelling],
      scope: { kind: 'global' },
      title: 'Global',
    };
    const widgetHotkey: RegisteredHotkey = {
      ...base,
      commandId: 'widget',
      id: 'widget',
      keys: ['mod+arrowup'],
      scope: { kind: 'widget', typeId: 'gallery' },
      title: 'Widget',
    };
    const resolve = (activeWidgetTypeId: string | null) =>
      resolveHotkey({
        context: { ...context, activeInstanceId: activeWidgetTypeId && 'gallery-1', activeWidgetTypeId },
        event,
        hotkeys: [globalHotkey, widgetHotkey],
        matchedKey: platformSpelling,
      })?.commandId;

    expect(resolve('gallery')).toBe('widget');
    expect(resolve(null)).toBe('global');
  });

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

  it('keeps every scope working under a floating window, except shortcuts tied to a named docked region', () => {
    const hotkey = (id: string, scope: RegisteredHotkey['scope']): RegisteredHotkey => ({
      ...base,
      commandId: id,
      id,
      scope,
      title: id,
    });
    const floating = {
      ...context,
      activeInstanceId: 'map-1',
      activeWidgetTypeId: 'image-map',
      focusedRegion: 'floating',
    } as const;
    const global = hotkey('global', { kind: 'global' });
    const anyRegion = hotkey('any-region', { kind: 'focused-region' });
    const rightOnly = hotkey('right-only', { kind: 'focused-region', region: 'right' });
    const thisWindow = hotkey('this-window', { floatingInstanceId: 'map-1', kind: 'focused-region' });
    const otherWindow = hotkey('other-window', { floatingInstanceId: 'map-2', kind: 'focused-region' });
    const widget = hotkey('widget', { kind: 'widget', typeId: 'image-map' });
    const instance = hotkey('instance', { instanceId: 'map-1', kind: 'instance' });
    const resolve = (hotkeys: RegisteredHotkey[]) =>
      resolveHotkey({ context: floating, event, hotkeys, matchedKey: 'x' })?.commandId;

    // The dock-specific shortcut never matches; the rest keep their priority order.
    expect(resolve([global, rightOnly])).toBe('global');
    expect(resolve([global, rightOnly, anyRegion])).toBe('any-region');
    expect(resolve([global, rightOnly, anyRegion, widget])).toBe('widget');
    expect(resolve([global, rightOnly, anyRegion, widget, instance])).toBe('instance');
    expect(resolve([rightOnly])).toBeUndefined();
    // A floating widget's own focused-region shortcut follows its window, at focused-region priority: above
    // global, below a widget-scoped binding on the same key, exactly as when it is docked.
    expect(resolve([global, otherWindow, thisWindow])).toBe('this-window');
    expect(resolve([global, otherWindow])).toBe('global');
    expect(resolve([thisWindow, widget])).toBe('widget');
    // Modal suppression is unchanged by where focus is.
    expect(
      resolveHotkey({
        context: { ...floating, isModalPresent: true },
        event,
        hotkeys: [global, anyRegion, widget, instance],
        matchedKey: 'x',
      })
    ).toBeNull();
  });

  it('does not resolve hotkeys contributed by another project source', () => {
    const staleProjectHotkey: RegisteredHotkey = {
      ...base,
      commandId: 'stale',
      id: 'stale',
      scope: { kind: 'widget', typeId: 'gallery' },
      source: { instanceId: 'gallery-1', projectId: 'project-2', region: 'left', typeId: 'gallery' },
      title: 'Stale project',
    };

    expect(
      resolveHotkey({
        context: { ...context, activeInstanceId: 'gallery-1', activeWidgetTypeId: 'gallery', focusedRegion: 'left' },
        event,
        hotkeys: [staleProjectHotkey],
        matchedKey: 'x',
      })
    ).toBeNull();
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
        context: { ...context, isModalPresent: true },
        event,
        hotkeys: [hotkey],
        matchedKey: 'x',
      })
    ).toBeNull();
    expect(
      resolveHotkey({
        context: { ...context, isModalPresent: true },
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
