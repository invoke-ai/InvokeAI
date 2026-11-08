import type { WidgetContributionSource } from '@workbench/widgetContracts';

import { isModalPresent } from '@platform/ui/modalPresence';
import { isToastActivationKeyEvent } from '@platform/ui/toaster';
import { useWorkbenchFocusTarget } from '@workbench/focusRegions';
import { useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { areWidgetPlacementProjectsEqual, getWidgetPlacementProject } from '@workbench/widgetPlacementMeta';
import { useActiveProjectSelector, useWorkbenchExtensions } from '@workbench/WorkbenchContext';
import { useEffect, useEffectEvent, useMemo, useSyncExternalStore } from 'react';
import { tinykeys } from 'tinykeys';

import type { RegisteredHotkey } from './types';

import { firstPartyHotkeyCatalog } from './catalog';
import { useExtensionHotkeyDefinitions } from './extensionHotkeys';
import { useRegisterFirstPartyCommands } from './firstPartyCommands';
import { acceptsHotkeyPress, toTinykeysBinding } from './keys';
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

  const { commands: commandApi, stores } = useWorkbenchExtensions();
  const commandContributions = useSyncExternalStore(
    stores.commands.subscribe,
    stores.commands.list,
    stores.commands.list
  );
  const customHotkeys = useWorkbenchPreferenceSelector((preferences) => preferences.customHotkeys);
  const project = useActiveProjectSelector(getWidgetPlacementProject, areWidgetPlacementProjectsEqual);
  const extensionHotkeys = useExtensionHotkeyDefinitions();
  const getFocusTarget = useWorkbenchFocusTarget();

  const registeredHotkeys = useMemo(() => {
    // The catalog lists every binding a widget can contribute, mounted or not. A row binds only while its command has
    // a handler, so a key nothing is handling keeps its browser meaning; and a mounted widget's own contribution of the
    // same id replaces the row, so one binding answers with the widget's scope and flags.
    const handledCommandIds = new Set(commandContributions.map(({ id }) => id));
    const contributedIds = new Set(extensionHotkeys.map(({ id }) => id));
    const firstPartyHotkeys = firstPartyHotkeyCatalog
      .filter((hotkey) => handledCommandIds.has(hotkey.commandId) && !contributedIds.has(hotkey.id))
      .map((hotkey) => applyCustomHotkeys(hotkey, customHotkeys));
    const widgetHotkeys = extensionHotkeys.map((hotkey) => applyCustomHotkeys(hotkey, customHotkeys));

    return [...firstPartyHotkeys, ...widgetHotkeys];
  }, [commandContributions, customHotkeys, extensionHotkeys]);

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
      for (const key of hotkey.keys) {
        const tinykeysBinding = toTinykeysBinding(key);

        if (tinykeysBinding) {
          bindings[tinykeysBinding] = (event) => {
            if (acceptsHotkeyPress(event, key)) {
              handleHotkey(event, key);
            }
          };
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
