import type { WidgetContributionSource } from '@workbench/types';

import { commandApi } from '@workbench/extensions/extensionApi';
import { getFocusedRegionSnapshot } from '@workbench/focusRegions';
import { useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { areWidgetPlacementProjectsEqual, getWidgetPlacementProject } from '@workbench/widgetPlacementMeta';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { useEffect, useEffectEvent, useMemo, useRef } from 'react';
import { tinykeys } from 'tinykeys';

import type { RegisteredHotkey } from './types';

import { firstPartyHotkeyCatalog } from './catalog';
import { useExtensionHotkeyDefinitions } from './extensionHotkeys';
import { useRegisterFirstPartyCommands } from './firstPartyCommands';
import { toTinykeysBinding } from './keys';
import { useIsHotkeyModalLayerActive } from './modalLayer';
import { applyCustomHotkeys, resolveHotkey } from './resolve';
import { getHotkeyTargetWidget } from './targetWidget';

export const getHotkeyExecutionSource = (
  hotkey: Pick<RegisteredHotkey, 'scope' | 'source'>,
  activeSource: WidgetContributionSource | null
): WidgetContributionSource | null => (hotkey.scope.kind === 'global' ? (hotkey.source ?? null) : activeSource);

export const WorkbenchHotkeyRuntime = () => {
  useRegisterFirstPartyCommands();

  const customHotkeys = useWorkbenchPreferenceSelector((preferences) => preferences.customHotkeys);
  const project = useActiveProjectSelector(getWidgetPlacementProject, areWidgetPlacementProjectsEqual);
  const extensionHotkeys = useExtensionHotkeyDefinitions();
  const isModalLayerActive = useIsHotkeyModalLayerActive();
  const projectRef = useRef(project);
  const registeredHotkeysRef = useRef<RegisteredHotkey[]>([]);

  projectRef.current = project;

  const registeredHotkeys = useMemo(() => {
    const firstPartyHotkeys = firstPartyHotkeyCatalog.map((hotkey) => applyCustomHotkeys(hotkey, customHotkeys));
    const widgetHotkeys = extensionHotkeys.map((hotkey) => applyCustomHotkeys(hotkey, customHotkeys));

    return [...firstPartyHotkeys, ...widgetHotkeys];
  }, [customHotkeys, extensionHotkeys]);

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

  const executeHotkey = useEffectEvent((hotkey: RegisteredHotkey, source: WidgetContributionSource | null) => {
    void commandApi.executeForSource(hotkey.commandId, getHotkeyExecutionSource(hotkey, source));
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
    const commandSource: WidgetContributionSource | null =
      activeInstanceId && activeWidgetTypeId && (targetWidget?.region || focusedRegion)
        ? {
            instanceId: activeInstanceId,
            projectId: projectRef.current.projectId ?? '',
            region: (targetWidget?.region ?? focusedRegion)!,
            typeId: activeWidgetTypeId,
          }
        : null;
    const hotkey = resolveHotkey({
      context: {
        activeInstanceId,
        activeWidgetTypeId,
        focusedRegion,
        isModalLayerActive,
        projectId: projectRef.current.projectId ?? '',
      },
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

    executeHotkey(hotkey, commandSource);
  });

  useEffect(() => {
    if (Object.keys(keybindings).length === 0) {
      return;
    }

    return tinykeys(window, keybindings, { ignore: () => false });
  }, [keybindings]);

  return null;
};
