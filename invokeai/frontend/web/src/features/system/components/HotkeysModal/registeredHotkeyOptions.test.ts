import type { Options } from 'react-hotkeys-hook';
import { describe, expect, it, vi } from 'vitest';

import { getRegisteredHotkeyOptions } from './registeredHotkeyOptions';

type EnabledPredicate = Exclude<Options['enabled'], boolean | undefined>;

const event = {} as Parameters<EnabledPredicate>[0];
const hotkey = {} as Parameters<EnabledPredicate>[1];

describe('getRegisteredHotkeyOptions', () => {
  it('keeps a statically disabled hotkey disabled so it does not intercept a shared key', () => {
    const options: Options = { enabled: false, preventDefault: true };
    const isTextSessionActive = vi.fn(() => false);

    expect(getRegisteredHotkeyOptions(options, isTextSessionActive)).toBe(options);
  });

  it('suppresses an enabled hotkey during an uncommitted canvas text session', () => {
    const result = getRegisteredHotkeyOptions({ enabled: true }, () => true);

    expect(typeof result.enabled).toBe('function');
    expect((result.enabled as EnabledPredicate)(event, hotkey)).toBe(false);
  });

  it('preserves a configured enabled predicate outside a canvas text session', () => {
    const configuredEnabled = vi.fn(() => false);
    const result = getRegisteredHotkeyOptions({ enabled: configuredEnabled }, () => false);

    expect((result.enabled as EnabledPredicate)(event, hotkey)).toBe(false);
    expect(configuredEnabled).toHaveBeenCalledWith(event, hotkey);
  });
});
