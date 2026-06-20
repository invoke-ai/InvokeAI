import type { WidgetHotkeyContribution, WidgetRegion, WorkbenchRegion } from '@workbench/types';

import { extensionContributionStores } from '@workbench/extensions/extensionApi';
import { useEffect, useMemo, useState } from 'react';

import type { HotkeyDefinition } from './types';

const widgetRegions = new Set<WorkbenchRegion>(['bottom', 'center', 'left', 'right']);

const isWidgetRegion = (region: WorkbenchRegion | undefined): region is WidgetRegion =>
  region !== undefined && widgetRegions.has(region);

export const toExtensionHotkeyDefinition = (hotkey: WidgetHotkeyContribution): HotkeyDefinition | null => {
  const source = hotkey.source;
  const scope = hotkey.scope ?? 'widget';

  if (scope === 'global') {
    return { ...hotkey, category: 'app', implemented: true, scope: { kind: 'global' } };
  }

  if (scope === 'instance' && source) {
    return {
      ...hotkey,
      category: 'app',
      implemented: true,
      scope: { instanceId: source.instanceId, kind: 'instance' },
    };
  }

  if (scope === 'focused-region' && isWidgetRegion(source?.region)) {
    return { ...hotkey, category: 'app', implemented: true, scope: { kind: 'focused-region', region: source.region } };
  }

  if (source) {
    return { ...hotkey, category: 'app', implemented: true, scope: { kind: 'widget', typeId: source.typeId } };
  }

  return null;
};

export const useExtensionHotkeyDefinitions = (): HotkeyDefinition[] => {
  const [hotkeys, setHotkeys] = useState(() => extensionContributionStores.hotkeys.list());

  useEffect(
    () => extensionContributionStores.hotkeys.subscribe(() => setHotkeys(extensionContributionStores.hotkeys.list())),
    []
  );

  return useMemo(
    () => hotkeys.map(toExtensionHotkeyDefinition).filter((hotkey): hotkey is HotkeyDefinition => hotkey !== null),
    [hotkeys]
  );
};
