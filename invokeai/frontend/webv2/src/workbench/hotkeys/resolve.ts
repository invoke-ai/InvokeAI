import type { HotkeyContext, RegisteredHotkey } from './types';

import { IS_MAC_OS, isEditableHotkeyTarget, normalizeHotkeyString } from './keys';

/**
 * The keys a binding presses on this platform: `mod` is Cmd on macOS and Ctrl elsewhere. The runtime fires the first
 * binding a press matches, so `mod+arrowup` and `ctrl+arrowup` must resolve as one press and leave the choice to
 * scope; compared by spelling, one would shadow the other everywhere.
 */
export const toPlatformHotkey = (hotkey: string, isMacOs = IS_MAC_OS): string =>
  normalizeHotkeyString(
    hotkey
      .split('+')
      .map((part) => (part.trim().toLowerCase() === 'mod' ? (isMacOs ? 'meta' : 'ctrl') : part))
      .join('+')
  );

const getScopePriority = (hotkey: RegisteredHotkey, context: HotkeyContext): number => {
  const { scope } = hotkey;

  if (hotkey.source && hotkey.source.projectId !== context.projectId) {
    return -1;
  }

  if (scope.kind === 'instance') {
    return scope.instanceId === context.activeInstanceId ? 400 : -1;
  }

  if (scope.kind === 'widget') {
    return scope.typeId === context.activeWidgetTypeId ? 300 : -1;
  }

  if (scope.kind === 'focused-region') {
    if (scope.floatingInstanceId) {
      return context.focusedRegion === 'floating' && scope.floatingInstanceId === context.activeInstanceId ? 200 : -1;
    }

    return context.focusedRegion && (!scope.region || scope.region === context.focusedRegion) ? 200 : -1;
  }

  return 100;
};

export const applyCustomHotkeys = <Hotkey extends { defaultKeys: string[]; id: string }>(
  hotkey: Hotkey,
  customHotkeys: Record<string, string[]>
): Hotkey & { keys: string[] } => ({
  ...hotkey,
  keys: (customHotkeys[hotkey.id] ?? hotkey.defaultKeys).map(normalizeHotkeyString).filter(Boolean),
});

export const resolveHotkey = ({
  context,
  event,
  hotkeys,
  matchedKey,
}: {
  context: HotkeyContext;
  event: KeyboardEvent;
  hotkeys: RegisteredHotkey[];
  matchedKey: string;
}): RegisteredHotkey | null => {
  const pressed = toPlatformHotkey(matchedKey);
  const isEditable = isEditableHotkeyTarget(event.target);

  return (
    hotkeys
      .filter((hotkey) => hotkey.implemented !== false && hotkey.keys.some((key) => toPlatformHotkey(key) === pressed))
      .filter((hotkey) => hotkey.allowInEditable || !isEditable)
      .filter((hotkey) => hotkey.allowInModal || !context.isModalPresent)
      .map((hotkey) => ({ hotkey, priority: getScopePriority(hotkey, context) }))
      .filter(({ priority }) => priority >= 0)
      .sort((left, right) => right.priority - left.priority)[0]?.hotkey ?? null
  );
};
