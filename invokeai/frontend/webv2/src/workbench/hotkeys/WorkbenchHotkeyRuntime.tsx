import { commandApi } from '@workbench/extensions/extensionApi';
import { getFocusedRegionSnapshot } from '@workbench/focusRegions';
import { useWorkbenchPreferences } from '@workbench/settings/store';
import { useActiveProject } from '@workbench/WorkbenchContext';
import { useEffect, useEffectEvent, useMemo, useRef, useSyncExternalStore } from 'react';
import { tinykeys } from 'tinykeys';

import type { RegisteredHotkey } from './types';

import { firstPartyHotkeyCatalog } from './catalog';
import { useExtensionHotkeyDefinitions } from './extensionHotkeys';
import { useRegisterFirstPartyCommands } from './firstPartyCommands';
import { toTinykeysBinding } from './keys';
import { isHotkeyModalLayerActive, subscribeHotkeyModalLayers } from './modalLayer';
import { applyCustomHotkeys, resolveHotkey } from './resolve';
import { getHotkeyTargetWidget } from './targetWidget';

export const WorkbenchHotkeyRuntime = () => {
  useRegisterFirstPartyCommands();

  const preferences = useWorkbenchPreferences();
  const project = useActiveProject();
  const extensionHotkeys = useExtensionHotkeyDefinitions();
  const isModalLayerActive = useSyncExternalStore(
    subscribeHotkeyModalLayers,
    isHotkeyModalLayerActive,
    isHotkeyModalLayerActive
  );
  const projectRef = useRef(project);
  const registeredHotkeysRef = useRef<RegisteredHotkey[]>([]);

  projectRef.current = project;

  const registeredHotkeys = useMemo(() => {
    const customHotkeys = preferences.customHotkeys;
    const firstPartyHotkeys = firstPartyHotkeyCatalog.map((hotkey) => applyCustomHotkeys(hotkey, customHotkeys));
    const widgetHotkeys = extensionHotkeys.map((hotkey) => applyCustomHotkeys(hotkey, customHotkeys));

    return [...firstPartyHotkeys, ...widgetHotkeys];
  }, [extensionHotkeys, preferences.customHotkeys]);

  registeredHotkeysRef.current = registeredHotkeys;

  const keybindings = useMemo(() => {
    const bindings: Record<string, (event: KeyboardEvent) => void> = {};

    for (const hotkey of registeredHotkeys) {
      if (hotkey.implemented === false) {
        continue;
      }

      for (const key of hotkey.keys) {
        const tinykeysBinding = toTinykeysBinding(key);

        if (tinykeysBinding) {
          bindings[tinykeysBinding] = (event) => handleHotkey(event, key);
        }
      }
    }

    return bindings;
  }, [registeredHotkeys]);

  const executeHotkey = useEffectEvent((hotkey: RegisteredHotkey) => {
    void commandApi.execute(hotkey.commandId);
  });

  const handleHotkey = useEffectEvent((event: KeyboardEvent, matchedKey: string) => {
    if (event.isComposing || event.keyCode === 229) {
      return;
    }

    const focusedRegion = getFocusedRegionSnapshot();
    const targetWidget = getHotkeyTargetWidget(event.target);
    const activeRegion = focusedRegion ? projectRef.current.widgetRegions[focusedRegion] : null;
    const activeInstanceId = targetWidget?.instanceId ?? activeRegion?.activeInstanceId ?? null;
    const activeWidgetTypeId = activeInstanceId
      ? (targetWidget?.typeId ?? projectRef.current.widgetInstances[activeInstanceId]?.typeId ?? null)
      : null;
    const hotkey = resolveHotkey({
      context: { activeInstanceId, activeWidgetTypeId, focusedRegion, isModalLayerActive },
      event,
      hotkeys: registeredHotkeysRef.current,
      matchedKey,
    });

    if (!hotkey) {
      return;
    }

    if (hotkey.preventDefault) {
      event.preventDefault();
    }

    executeHotkey(hotkey);
  });

  useEffect(() => {
    if (Object.keys(keybindings).length === 0) {
      return;
    }

    return tinykeys(window, keybindings, { ignore: () => false });
  }, [keybindings]);

  return null;
};
