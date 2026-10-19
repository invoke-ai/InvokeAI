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
    return context.activeWidgetTypeId && scope.typeIds.includes(context.activeWidgetTypeId) ? 300 : -1;
  }

  if (scope.kind === 'focused-region') {
    if (scope.floatingInstanceId) {
      return context.focusedRegion === 'floating' && scope.floatingInstanceId === context.activeInstanceId ? 200 : -1;
    }

    return context.focusedRegion && (!scope.region || scope.region === context.focusedRegion) ? 200 : -1;
  }

  return 100;
};

const isClaimedTarget = (hotkey: RegisteredHotkey, target: EventTarget | null): boolean =>
  !hotkey.targetSelector ||
  (typeof Element !== 'undefined' && target instanceof Element && target.matches(hotkey.targetSelector));

export const applyCustomHotkeys = <Hotkey extends { defaultKeys: string[]; id: string }>(
  hotkey: Hotkey,
  customHotkeys: Record<string, string[]>
): Hotkey & { customized: boolean; keys: string[] } => {
  const customKeys = customHotkeys[hotkey.id];

  return {
    ...hotkey,
    customized: customKeys !== undefined,
    keys: (customKeys ?? hotkey.defaultKeys).map(normalizeHotkeyString).filter(Boolean),
  };
};

/**
 * An open menu or listbox owns the unmodified presses inside it: arrows, Enter, Escape, Space and typeahead, Shift
 * included because it types capitals. Zag prevents the default of a typed letter but lets it bubble, and a portaled
 * popup sits outside every widget's DOM, so the press would otherwise resolve against whichever widget holds focus.
 * A Ctrl, Meta or Alt chord is never popup input and resolves as usual.
 */
const isPopupOwnedPress = (event: KeyboardEvent): boolean =>
  !event.ctrlKey &&
  !event.metaKey &&
  !event.altKey &&
  typeof Element !== 'undefined' &&
  event.target instanceof Element &&
  event.target.closest('[role="menu"], [role="listbox"]') !== null;

const customizedRank = (hotkey: RegisteredHotkey): number => (hotkey.customized ? 1 : 0);

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
  if (isPopupOwnedPress(event)) {
    return null;
  }

  const pressed = toPlatformHotkey(matchedKey);
  const isEditable = isEditableHotkeyTarget(event.target);

  return (
    hotkeys
      .filter((hotkey) => hotkey.implemented !== false && hotkey.keys.some((key) => toPlatformHotkey(key) === pressed))
      .filter((hotkey) => hotkey.allowInEditable || !isEditable)
      .filter((hotkey) => isClaimedTarget(hotkey, event.target))
      .filter((hotkey) => hotkey.allowInModal || !context.isModalPresent)
      .map((hotkey) => ({ hotkey, priority: getScopePriority(hotkey, context) }))
      .filter(({ priority }) => priority >= 0)
      .sort(
        (left, right) => right.priority - left.priority || customizedRank(right.hotkey) - customizedRank(left.hotkey)
      )[0]?.hotkey ?? null
  );
};
