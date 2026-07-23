import { useMountEffect } from '@platform/react/useMountEffect';
import { OPEN_COMMAND_PALETTE_HOTKEY } from '@workbench/hotkeys/catalog';
import { formatHotkeyForPlatform, toTinykeysBinding } from '@workbench/hotkeys/keys';
import { applyCustomHotkeys } from '@workbench/hotkeys/resolve';
import { useWorkbenchPreferences, useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { lazy, Suspense } from 'react';
import { tinykeys } from 'tinykeys';

import { closeCommandPalette, toggleCommandPalette, useIsCommandPaletteOpen } from './paletteStore';
import { SETTINGS_ENTRY_DEPS } from './settingsEntryDeps';

const LazyLaunchpadCommandPaletteDialog = lazy(() => import('./LaunchpadCommandPaletteDialog'));

const LaunchpadCommandPaletteHotkeys = ({ keys }: { keys: readonly string[] }) => {
  useMountEffect(() => {
    if (keys.length === 0) {
      return;
    }

    const onHotkey = (event: KeyboardEvent): void => {
      event.preventDefault();
      toggleCommandPalette();
    };
    const bindings = Object.fromEntries(keys.map((key) => [toTinykeysBinding(key), onHotkey]));

    return tinykeys(window, bindings, { ignore: () => false });
  });

  return null;
};

/** Lightweight Launchpad runtime and lazy dialog host. */
export const LaunchpadCommandPalette = () => {
  const isOpen = useIsCommandPaletteOpen();
  const customHotkeys = useWorkbenchPreferenceSelector((preferences) => preferences.customHotkeys);
  const paletteHotkeys = applyCustomHotkeys(OPEN_COMMAND_PALETTE_HOTKEY, customHotkeys).keys;

  return (
    <>
      <LaunchpadCommandPaletteHotkeys key={paletteHotkeys.join('\n')} keys={paletteHotkeys} />
      {isOpen ? <OpenLaunchpadCommandPalette /> : null}
    </>
  );
};

const OpenLaunchpadCommandPalette = () => {
  const preferences = useWorkbenchPreferences();

  return (
    <Suspense fallback={null}>
      <LazyLaunchpadCommandPaletteDialog
        modifierKeyLabel={formatHotkeyForPlatform('mod')[0]!}
        preferences={preferences}
        settingsEntryDeps={SETTINGS_ENTRY_DEPS}
        onClose={closeCommandPalette}
      />
    </Suspense>
  );
};
