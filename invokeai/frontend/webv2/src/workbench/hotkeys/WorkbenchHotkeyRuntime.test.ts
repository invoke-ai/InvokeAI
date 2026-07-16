import { afterEach, describe, expect, it, vi } from 'vitest';

import type { HotkeyContext, RegisteredHotkey } from './types';

import { resolveHotkey } from './resolve';
import { getHotkeyExecutionSource, shouldPreventHotkeyDefault } from './WorkbenchHotkeyRuntime';

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

describe('shouldPreventHotkeyDefault', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('prevents the browser default for a matched binding with no explicit preventDefault', () => {
    // Regression: widget hotkeys (e.g. canvas `mod+d`) are registered without a
    // `preventDefault` flag; a claimed binding must still swallow the browser
    // default (mod+d used to open the bookmark dialog).
    expect(shouldPreventHotkeyDefault(createHotkey({}))).toBe(true);
  });

  it('honours an explicit opt-out with preventDefault: false', () => {
    expect(shouldPreventHotkeyDefault(createHotkey({ preventDefault: false }))).toBe(false);
  });

  it('does not prevent the default when no binding claimed the event', () => {
    expect(shouldPreventHotkeyDefault(null)).toBe(false);
  });

  it('prevents the default for a matched-and-run widget binding, but not when skipped by an editable target', () => {
    class FakeHTMLElement {
      closest(): FakeHTMLElement {
        return this;
      }
    }
    vi.stubGlobal('HTMLElement', FakeHTMLElement);

    // A canvas-scoped binding claimed by an active canvas widget, registered
    // without an explicit preventDefault (like the real widget registrations).
    const canvasDeselect: RegisteredHotkey = createHotkey({
      allowInEditable: false,
      commandId: 'canvas.deselect',
      id: 'canvas.deselect',
      keys: ['mod+d'],
      preventDefault: undefined,
      scope: { kind: 'widget', typeId: 'canvas' },
    });
    const context: HotkeyContext = {
      activeInstanceId: 'canvas',
      activeWidgetTypeId: 'canvas',
      focusedRegion: null,
      isModalLayerActive: false,
      projectId: 'project-1',
    };

    // Non-editable target: the binding resolves and must prevent the default.
    const matched = resolveHotkey({
      context,
      event: { target: null } as KeyboardEvent,
      hotkeys: [canvasDeselect],
      matchedKey: 'mod+d',
    });
    expect(matched?.commandId).toBe('canvas.deselect');
    expect(shouldPreventHotkeyDefault(matched)).toBe(true);

    // Editable target: the binding is skipped (allowInEditable: false) so the
    // event is left alone and the browser default is untouched.
    const skipped = resolveHotkey({
      context,
      event: { target: new FakeHTMLElement() } as unknown as KeyboardEvent,
      hotkeys: [canvasDeselect],
      matchedKey: 'mod+d',
    });
    expect(skipped).toBeNull();
    expect(shouldPreventHotkeyDefault(skipped)).toBe(false);
  });
});
