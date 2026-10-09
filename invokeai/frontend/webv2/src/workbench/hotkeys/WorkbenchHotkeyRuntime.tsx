import type { WidgetContributionSource } from '@workbench/widgetContracts';

import { isModalPresent } from '@platform/ui/modalPresence';
import { isToastActivationKeyEvent } from '@platform/ui/toaster';
import { useWorkbenchFocusTarget } from '@workbench/focusRegions';
import { useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { areWidgetPlacementProjectsEqual, getWidgetPlacementProject } from '@workbench/widgetPlacementMeta';
import { useActiveProjectSelector, useWorkbenchExtensions } from '@workbench/WorkbenchContext';
import { useEffect, useEffectEvent, useMemo } from 'react';
import { tinykeys } from 'tinykeys';

import type { RegisteredHotkey } from './types';

import { firstPartyHotkeyCatalog } from './catalog';
import { useExtensionHotkeyDefinitions } from './extensionHotkeys';
import { useRegisterFirstPartyCommands } from './firstPartyCommands';
import { toTinykeysBinding } from './keys';
import { applyCustomHotkeys, resolveHotkey } from './resolve';
import { getHotkeyTargetWidget, resolveHotkeyTarget } from './targetWidget';

export const getHotkeyExecutionSource = (
  hotkey: Pick<RegisteredHotkey, 'scope' | 'source'>,
  activeSource: WidgetContributionSource | null
): WidgetContributionSource | null => (hotkey.scope.kind === 'global' ? (hotkey.source ?? null) : activeSource);

/**
 * Claimed bindings suppress browser defaults unless preventDefault is explicitly false. Unclaimed events retain
 * browser behavior.
 */
export const shouldPreventHotkeyDefault = (hotkey: RegisteredHotkey | null): boolean =>
  hotkey !== null && hotkey.preventDefault !== false;

export const WorkbenchHotkeyRuntime = () => {
  useRegisterFirstPartyCommands();

  const { commands: commandApi } = useWorkbenchExtensions();
  const customHotkeys = useWorkbenchPreferenceSelector((preferences) => preferences.customHotkeys);
  const project = useActiveProjectSelector(getWidgetPlacementProject, areWidgetPlacementProjectsEqual);
  const extensionHotkeys = useExtensionHotkeyDefinitions();
  const getFocusTarget = useWorkbenchFocusTarget();

  const registeredHotkeys = useMemo(() => {
    const firstPartyHotkeys = firstPartyHotkeyCatalog.map((hotkey) => applyCustomHotkeys(hotkey, customHotkeys));
    const widgetHotkeys = extensionHotkeys.map((hotkey) => applyCustomHotkeys(hotkey, customHotkeys));

    return [...firstPartyHotkeys, ...widgetHotkeys];
  }, [customHotkeys, extensionHotkeys]);

  const executeHotkey = useEffectEvent((hotkey: RegisteredHotkey, source: WidgetContributionSource | null) => {
    void commandApi.executeForSource(hotkey.commandId, getHotkeyExecutionSource(hotkey, source));
  });

  const handleHotkey = useEffectEvent((event: KeyboardEvent, matchedKey: string) => {
    // A focused toast button keeps its activation keys; no shortcut, however scoped, takes them.
    if (event.isComposing || event.keyCode === 229 || isToastActivationKeyEvent(event)) {
      return;
    }

    const { source, ...target } = resolveHotkeyTarget({
      focusTarget: getFocusTarget(),
      project,
      targetWidget: getHotkeyTargetWidget(event.target),
    });
    const hotkey = resolveHotkey({
      context: { ...target, isModalPresent: isModalPresent(event), projectId: project.projectId ?? '' },
      event,
      hotkeys: registeredHotkeys,
      matchedKey,
    });

    if (!hotkey) {
      return;
    }

    if (shouldPreventHotkeyDefault(hotkey)) {
      event.preventDefault();
    }

    executeHotkey(hotkey, source);
  });

  useEffect(() => {
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

    if (Object.keys(bindings).length === 0) {
      return;
    }

    return tinykeys(window, bindings, { ignore: () => false });
  }, [registeredHotkeys]);

  return null;
};
