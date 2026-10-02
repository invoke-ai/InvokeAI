import type { HotkeyContext, RegisteredHotkey } from './types';

import { isEditableHotkeyTarget, normalizeHotkeyString } from './keys';

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
  const normalized = normalizeHotkeyString(matchedKey);
  const isEditable = isEditableHotkeyTarget(event.target);

  return (
    hotkeys
      .filter((hotkey) => hotkey.implemented !== false && hotkey.keys.includes(normalized))
      .filter((hotkey) => hotkey.allowInEditable || !isEditable)
      .filter((hotkey) => hotkey.allowInModal || !context.isModalLayerActive)
      .map((hotkey) => ({ hotkey, priority: getScopePriority(hotkey, context) }))
      .filter(({ priority }) => priority >= 0)
      .sort((left, right) => right.priority - left.priority)[0]?.hotkey ?? null
  );
};
