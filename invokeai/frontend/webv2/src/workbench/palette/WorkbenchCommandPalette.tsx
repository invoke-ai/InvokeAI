import type { WorkbenchPreferences } from '@workbench/settings/contracts';

import { getQueueQueryScope, getQueueReadModelOptions } from '@features/queue/queries';
import { useMountEffect } from '@platform/react/useMountEffect';
import { firstPartyHotkeyCatalog } from '@workbench/hotkeys/catalog';
import { formatHotkeyForPlatform } from '@workbench/hotkeys/keys';
import { registerHotkeyModalLayer } from '@workbench/hotkeys/modalLayer';
import { useWorkbenchPreferences } from '@workbench/settings/store';
import { openWidgetPlacement } from '@workbench/widgetPlacementCommands';
import { getWidgetsForRegion } from '@workbench/widgetRegistry';
import { lazy, Suspense } from 'react';

import type { SettingsEntryDeps } from './entries';

import { closeCommandPalette, commandPaletteStore } from './paletteStore';
import { SETTINGS_ENTRY_DEPS } from './settingsEntryDeps';

const LazyWorkbenchCommandPaletteDialog = lazy(() => import('./WorkbenchCommandPaletteDialog'));

/** Lightweight route host; the palette implementation is loaded only while open. */
export const WorkbenchCommandPalette = () => {
  const isOpen = commandPaletteStore.useSelector((snapshot) => snapshot.isOpen);
  const preferences = useWorkbenchPreferences();

  return isOpen ? <OpenWorkbenchCommandPalette preferences={preferences} /> : null;
};

const OpenWorkbenchCommandPalette = ({ preferences }: { preferences: WorkbenchPreferences }) => {
  useMountEffect(() => registerHotkeyModalLayer('command-palette'));

  return (
    <Suspense fallback={null}>
      <LazyWorkbenchCommandPaletteDialog
        catalog={firstPartyHotkeyCatalog}
        formatHotkey={formatHotkeyForPlatform}
        modifierKeyLabel={formatHotkeyForPlatform('mod')[0] ?? 'ctrl'}
        getWidgetsForRegion={getWidgetsForRegion}
        getQueueQueryScope={getQueueQueryScope}
        getQueueReadModelOptions={getQueueReadModelOptions}
        openWidgetPlacement={openWidgetPlacement}
        preferences={preferences}
        settingsEntryDeps={SETTINGS_ENTRY_DEPS as SettingsEntryDeps}
        onClose={closeCommandPalette}
      />
    </Suspense>
  );
};
