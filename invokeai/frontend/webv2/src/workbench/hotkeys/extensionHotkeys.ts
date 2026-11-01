import type { WidgetRegion } from '@workbench/layoutContracts';
import type { WidgetHotkeyContribution, WorkbenchRegion } from '@workbench/widgetContracts';

import { useOptionalWorkbenchExtensions } from '@workbench/WorkbenchContext';
import { useMemo, useSyncExternalStore } from 'react';

import type { HotkeyCategory, HotkeyDefinition } from './types';

const widgetRegions = new Set<WorkbenchRegion>(['bottom', 'center', 'left', 'right']);
const NO_EXTENSION_HOTKEYS: WidgetHotkeyContribution[] = [];
const subscribeToNothing = (): (() => void) => () => {};
const getNoExtensionHotkeys = (): WidgetHotkeyContribution[] => NO_EXTENSION_HOTKEYS;

const isWidgetRegion = (region: WorkbenchRegion | undefined): region is WidgetRegion =>
  region !== undefined && widgetRegions.has(region);

/** Settings list a contributed binding under the widget that owns it; Layers binds the Canvas's own keys. */
const CATEGORY_BY_WIDGET: Partial<Record<string, HotkeyCategory>> = {
  canvas: 'canvas',
  gallery: 'gallery',
  layers: 'canvas',
  preview: 'viewer',
  workflow: 'workflows',
};

export const toExtensionHotkeyDefinition = (hotkey: WidgetHotkeyContribution): HotkeyDefinition | null => {
  const source = hotkey.source;
  const scope = hotkey.scope ?? 'widget';
  const category = (source && CATEGORY_BY_WIDGET[source.typeId]) ?? 'app';

  if (scope === 'global') {
    return { ...hotkey, category, scope: { kind: 'global' } };
  }

  if (scope === 'instance' && source) {
    return {
      ...hotkey,
      category,
      scope: { instanceId: source.instanceId, kind: 'instance' },
    };
  }

  if (scope === 'focused-region' && isWidgetRegion(source?.region)) {
    return { ...hotkey, category, scope: { kind: 'focused-region', region: source.region } };
  }

  // A floating window is not a region other widgets share: "while my region is focused" means "while my window is".
  // It stays a focused-region shortcut, so floating a widget does not change which binding wins a shared key.
  if (scope === 'focused-region' && source?.region === 'floating') {
    return {
      ...hotkey,
      category,
      scope: { floatingInstanceId: source.instanceId, kind: 'focused-region' },
    };
  }

  if (source) {
    return { ...hotkey, category, scope: { kind: 'widget', typeIds: [source.typeId] } };
  }

  return null;
};

/**
 * One row per binding id. The catalog lists every shipped binding, mounted widget or not; a mounted widget registers
 * the same id (from each instance, and from companions such as Layers), whose translated title the row takes. Bindings
 * only a widget knows (an extension's) follow under that widget's category.
 */
export const mergeHotkeyCatalog = (
  firstParty: readonly HotkeyDefinition[],
  contributed: readonly HotkeyDefinition[]
): HotkeyDefinition[] => {
  const contributedById = new Map<string, HotkeyDefinition>();

  for (const hotkey of contributed) {
    if (!contributedById.has(hotkey.id)) {
      contributedById.set(hotkey.id, hotkey);
    }
  }

  const firstPartyIds = new Set(firstParty.map((hotkey) => hotkey.id));

  return [
    ...firstParty.map((hotkey) => {
      const title = contributedById.get(hotkey.id)?.title;

      return title ? { ...hotkey, title } : hotkey;
    }),
    ...[...contributedById.values()].filter((hotkey) => !firstPartyIds.has(hotkey.id)),
  ];
};

export const useExtensionHotkeyDefinitions = (): HotkeyDefinition[] => {
  const store = useOptionalWorkbenchExtensions()?.stores.hotkeys;
  const hotkeys = useSyncExternalStore(
    store?.subscribe ?? subscribeToNothing,
    store?.list ?? getNoExtensionHotkeys,
    store?.list ?? getNoExtensionHotkeys
  );

  return useMemo(
    () => hotkeys.map(toExtensionHotkeyDefinition).filter((hotkey): hotkey is HotkeyDefinition => hotkey !== null),
    [hotkeys]
  );
};
