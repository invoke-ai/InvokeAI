import type { ShortcutHint } from './hintSources';
import type { HotkeyContext, RegisteredHotkey } from './types';

import { formatHotkeyForPlatform, formatHotkeyPartForPlatform, normalizeHotkeyString } from './keys';
import { resolveHotkey } from './resolve';

export interface ResolvedShortcutHint {
  id: string;
  labelKey?: string;
  title?: string;
  parts: readonly string[];
  pointerKey?: string;
}

/** Native guide activation/dismissal wins; modified global shortcuts retain their normal runtime handling. */
export const isShortcutGuideSessionKey = (
  event: Pick<KeyboardEvent, 'altKey' | 'ctrlKey' | 'key' | 'metaKey' | 'shiftKey'>
): boolean =>
  event.key === 'Escape' ||
  ((event.key === 'Enter' || event.key === ' ') &&
    !event.altKey &&
    !event.ctrlKey &&
    !event.metaKey &&
    !event.shiftKey);

const isGuideReservedBinding = (binding: string): boolean => {
  const normalized = normalizeHotkeyString(binding);
  return normalized === 'enter' || normalized === 'space' || normalized.split('+').at(-1) === 'esc';
};

/** Advertised commands must win the resolver and remain executable when the guide owns focus. */
export const resolveShortcutHints = (
  hints: readonly ShortcutHint[],
  hotkeys: RegisteredHotkey[],
  context: HotkeyContext,
  target: EventTarget | null
): ResolvedShortcutHint[] =>
  hints.flatMap((hint): ResolvedShortcutHint[] => {
    if (!('commandId' in hint)) {
      return [
        {
          ...hint,
          parts: hint.parts.map(formatHotkeyPartForPlatform),
          id: `${hint.labelKey}:${hint.parts.join('+')}:${hint.pointerKey ?? ''}`,
        },
      ];
    }
    for (const hotkey of hotkeys) {
      if (hotkey.commandId !== hint.commandId) {
        continue;
      }
      const key = hotkey.keys.find(
        (matchedKey) =>
          !isGuideReservedBinding(matchedKey) &&
          resolveHotkey({
            context,
            event: { target } as KeyboardEvent,
            hotkeys,
            matchedKey,
          })?.commandId === hint.commandId
      );
      if (key) {
        return [
          { id: hint.commandId, labelKey: hint.labelKey, parts: formatHotkeyForPlatform(key), title: hotkey.title },
        ];
      }
    }
    return [];
  });

export const getVisibleShortcutHintCount = (available: number, reserved: number, widths: readonly number[]): number => {
  let used = reserved;
  let count = 0;
  for (const width of widths.slice(0, 4)) {
    used += width;
    if (used > available) {
      break;
    }
    count += 1;
  }
  return count;
};
