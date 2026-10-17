import { afterEach, describe, expect, it } from 'vitest';

import type { HotkeyContext, RegisteredHotkey } from './types';

import { firstPartyHotkeyCatalog } from './catalog';
import { applyCustomHotkeys, resolveHotkey } from './resolve';

/** Target selectors match real elements, so this runs where the DOM does. */
describe('resolveHotkey with target-scoped bindings', () => {
  const host = document.createElement('div');

  document.body.append(host);

  afterEach(() => {
    host.replaceChildren();
  });

  const context: HotkeyContext = {
    activeInstanceId: null,
    activeWidgetTypeId: null,
    focusedRegion: null,
    isModalPresent: false,
    projectId: 'project-1',
  };
  const weightUp = applyCustomHotkeys(
    firstPartyHotkeyCatalog.find((hotkey) => hotkey.id === 'app.promptWeightUp')!,
    {}
  );
  // A lower-priority global binding on the same chord, as a user customization could add.
  const fallback: RegisteredHotkey = {
    allowInEditable: true,
    category: 'app',
    commandId: 'fallback',
    defaultKeys: ['ctrl+arrowup'],
    id: 'fallback',
    implemented: true,
    keys: ['ctrl+arrowup'],
    scope: { kind: 'global' },
    title: 'Fallback',
  };

  const resolveOn = (html: string, hotkeys: RegisteredHotkey[]) => {
    host.innerHTML = html;
    const target = host.querySelector('[data-target]');

    return (
      resolveHotkey({ context, event: { target } as unknown as KeyboardEvent, hotkeys, matchedKey: 'ctrl+arrowup' })
        ?.commandId ?? null
    );
  };

  it('claims the chord on a prompt that takes attention weighting', () => {
    expect(resolveOn('<textarea data-target data-prompt-attention-target="true"></textarea>', [weightUp])).toBe(
      'app.promptWeightUp'
    );
  });

  it('claims nothing on any other text field, so the key keeps its native meaning', () => {
    expect(resolveOn('<textarea data-target></textarea>', [weightUp])).toBeNull();
    expect(resolveOn('<input data-target />', [weightUp])).toBeNull();
  });

  it('leaves a read-only prompt unclaimed', () => {
    expect(
      resolveOn('<textarea data-target data-prompt-attention-target="true" readonly></textarea>', [weightUp])
    ).toBeNull();
  });

  it('lets the next candidate win where the target does not match', () => {
    expect(resolveOn('<textarea data-target></textarea>', [weightUp, fallback])).toBe('fallback');
  });
});
