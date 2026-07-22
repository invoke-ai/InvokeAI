import type { WorkbenchPreferences } from '@workbench/settings/contracts';

import { useMountEffect } from '@platform/react/useMountEffect';
import { formatHotkeyForPlatform } from '@workbench/hotkeys/keys';
import { useWorkbenchPreferences } from '@workbench/settings/store';
import { lazy, Suspense } from 'react';
import { tinykeys } from 'tinykeys';

import type { SettingsEntryDeps } from './entries';

import { closeCommandPalette, commandPaletteStore, toggleCommandPalette } from './paletteStore';
import { SETTINGS_ENTRY_DEPS } from './settingsEntryDeps';

const LazyLaunchpadCommandPaletteDialog = lazy(() => import('./LaunchpadCommandPaletteDialog'));

/** Lightweight Launchpad runtime and lazy dialog host. */
export const LaunchpadCommandPalette = () => {
  const isOpen = commandPaletteStore.useSelector((snapshot) => snapshot.isOpen);
  const preferences = useWorkbenchPreferences();

  useMountEffect(() =>
    tinykeys(window, {
      '$mod+KeyK': (event) => {
        event.preventDefault();
        toggleCommandPalette();
      },
    })
  );

  return isOpen ? (
    <Suspense fallback={null}>
      <LazyLaunchpadCommandPaletteDialog
        preferences={preferences as WorkbenchPreferences}
        modifierKeyLabel={formatHotkeyForPlatform('mod')[0] ?? 'ctrl'}
        settingsEntryDeps={SETTINGS_ENTRY_DEPS as SettingsEntryDeps}
        onClose={closeCommandPalette}
      />
    </Suspense>
  ) : null;
};
