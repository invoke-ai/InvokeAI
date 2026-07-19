import type { WidgetContributionSource } from '@workbench/widgetContracts';

import { commandApi } from '@workbench/extensions/extensionApi';
import { getFocusedRegionSnapshot } from '@workbench/focusRegions';
import { useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { areWidgetPlacementProjectsEqual, getWidgetPlacementProject } from '@workbench/widgetPlacementMeta';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { useEffect, useEffectEvent, useMemo } from 'react';
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

/**
 * Whether the browser default should be suppressed for a resolved hotkey.
 *
 * `resolveHotkey` only returns a hotkey once it has *claimed* the event — the
 * scope is active and it is not skipped by the editable-target/modal-layer
 * rules — so a matched-and-run binding must swallow the browser default (e.g.
 * `mod+d` opening the bookmark dialog). Prevention is therefore on by default;
 * a binding opts out only by explicitly setting `preventDefault: false`. A
 * `null` hotkey (no binding claimed the event) never prevents the default.
 */
export const shouldPreventHotkeyDefault = (hotkey: RegisteredHotkey | null): boolean =>
  hotkey !== null && hotkey.preventDefault !== false;

export const WorkbenchHotkeyRuntime = () => {
  useRegisterFirstPartyCommands();

  const customHotkeys = useWorkbenchPreferenceSelector((preferences) => preferences.customHotkeys);
  const project = useActiveProjectSelector(getWidgetPlacementProject, areWidgetPlacementProjectsEqual);
  const extensionHotkeys = useExtensionHotkeyDefinitions();
  const isModalLayerActive = useIsHotkeyModalLayerActive();

  const registeredHotkeys = useMemo(() => {
    const firstPartyHotkeys = firstPartyHotkeyCatalog.map((hotkey) => applyCustomHotkeys(hotkey, customHotkeys));
    const widgetHotkeys = extensionHotkeys.map((hotkey) => applyCustomHotkeys(hotkey, customHotkeys));

    return [...firstPartyHotkeys, ...widgetHotkeys];
  }, [customHotkeys, extensionHotkeys]);

  const executeHotkey = useEffectEvent((hotkey: RegisteredHotkey, source: WidgetContributionSource | null) => {
    void commandApi.executeForSource(hotkey.commandId, getHotkeyExecutionSource(hotkey, source));
  });

  const handleHotkey = useEffectEvent((event: KeyboardEvent, matchedKey: string) => {
    if (event.isComposing || event.keyCode === 229) {
      return;
    }

    const focusedRegion = getFocusedRegionSnapshot();
    const targetWidget = getHotkeyTargetWidget(event.target);
    const activeRegion = focusedRegion ? project.widgetRegions[focusedRegion] : null;
    const activeInstanceId = targetWidget?.instanceId ?? activeRegion?.activeInstanceId ?? null;
    const activeWidgetTypeId = activeInstanceId
      ? (targetWidget?.typeId ?? project.widgetInstances[activeInstanceId]?.typeId ?? null)
      : null;
    const commandSource: WidgetContributionSource | null =
      activeInstanceId && activeWidgetTypeId && (targetWidget?.region || focusedRegion)
        ? {
            instanceId: activeInstanceId,
            projectId: project.projectId ?? '',
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
        projectId: project.projectId ?? '',
      },
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

    executeHotkey(hotkey, commandSource);
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
