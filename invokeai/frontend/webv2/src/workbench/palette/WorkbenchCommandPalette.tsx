import { requestQueueItemReveal } from '@features/queue/reveal';
import { useMountEffect } from '@platform/react/useMountEffect';
import { firstPartyHotkeyCatalog } from '@workbench/hotkeys/catalog';
import { formatHotkeyForPlatform } from '@workbench/hotkeys/keys';
import { registerHotkeyModalLayer } from '@workbench/hotkeys/modalLayer';
import { useWorkbenchPreferences } from '@workbench/settings/store';
import { openWidgetPlacement } from '@workbench/widgetPlacementCommands';
import { getWidgetsForRegion } from '@workbench/widgetRegistry';
import { lazy, Suspense } from 'react';

import { closeCommandPalette, useIsCommandPaletteOpen } from './paletteStore';
import { SETTINGS_ENTRY_DEPS } from './settingsEntryDeps';

const LazyWorkbenchCommandPaletteDialog = lazy(() => import('./WorkbenchCommandPaletteDialog'));

/** Lightweight route host; the palette implementation is loaded only while open. */
export const WorkbenchCommandPalette = () => {
  const isOpen = useIsCommandPaletteOpen();

  return isOpen ? <OpenWorkbenchCommandPalette /> : null;
};

const OpenWorkbenchCommandPalette = () => {
  const preferences = useWorkbenchPreferences();

  useMountEffect(() => registerHotkeyModalLayer('command-palette'));

  return (
    <Suspense fallback={null}>
      <LazyWorkbenchCommandPaletteDialog
        catalog={firstPartyHotkeyCatalog}
        formatHotkey={formatHotkeyForPlatform}
        getWidgetsForRegion={getWidgetsForRegion}
        modifierKeyLabel={formatHotkeyForPlatform('mod')[0]!}
        openWidgetPlacement={openWidgetPlacement}
        preferences={preferences}
        requestQueueItemReveal={requestQueueItemReveal}
        settingsEntryDeps={SETTINGS_ENTRY_DEPS}
        onClose={closeCommandPalette}
      />
    </Suspense>
  );
};
