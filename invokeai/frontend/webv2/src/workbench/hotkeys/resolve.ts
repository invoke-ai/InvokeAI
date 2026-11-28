import type { HotkeyContext, RegisteredHotkey } from './types';

import { IS_MAC_OS, isEditableHotkeyTarget, isNativeControlKeyPress, normalizeHotkeyString } from './keys';

/** Platform keys (`mod` is Cmd or Ctrl), so `mod+x` and `ctrl+x` resolve as one press, not shadowing each other. */
export const toPlatformHotkey = (hotkey: string, isMacOs = IS_MAC_OS): string =>
  normalizeHotkeyString(
    hotkey
      .split('+')
      .map((part) => (part.trim().toLowerCase() === 'mod' ? (isMacOs ? 'meta' : 'ctrl') : part))
      .join('+')
  );

const getScopePriority = (hotkey: RegisteredHotkey, context: HotkeyContext): number => {
  const priority = getScopeMatchPriority(hotkey, context);

  return hotkey.fallback && priority >= 0 ? 0 : priority;
};

const getScopeMatchPriority = (hotkey: RegisteredHotkey, context: HotkeyContext): number => {
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

/** An open menu or listbox owns unmodified presses inside it; Zag lets typed letters bubble past portals. */
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
  // A focused text field, radio group or slider keeps the press unless the binding opts into editable targets.
  const isOwnedByTarget = isEditableHotkeyTarget(event.target) || isNativeControlKeyPress(event);

  return (
    hotkeys
      .filter((hotkey) => hotkey.keys.some((key) => toPlatformHotkey(key) === pressed))
      .filter((hotkey) => hotkey.allowInEditable || !isOwnedByTarget)
      .filter((hotkey) => isClaimedTarget(hotkey, event.target))
      .filter((hotkey) => !hotkey.fallback || !event.defaultPrevented)
      .filter((hotkey) => hotkey.allowInModal || !context.isModalPresent)
      .map((hotkey) => ({ hotkey, priority: getScopePriority(hotkey, context) }))
      .filter(({ priority }) => priority >= 0)
      .sort(
        (left, right) => right.priority - left.priority || customizedRank(right.hotkey) - customizedRank(left.hotkey)
      )[0]?.hotkey ?? null
  );
};
