import { useWorkbenchPreferenceSelector } from '@workbench/settings/store';

import type { HotkeyDefinition } from './types';

import { firstPartyHotkeyCatalog } from './catalog';
import { formatHotkeyAriaLabel, formatHotkeyForPlatform, formatHotkeyLabel } from './keys';
import { applyCustomHotkeys } from './resolve';

const findDefinition = (commandId: string): HotkeyDefinition | undefined =>
  firstPartyHotkeyCatalog.find((hotkey) => hotkey.id === commandId);

/** Display effective bindings, including remaps; return null for unbound commands. */
export const useShortcutLabel = (commandId: string): string | null => {
  const binding = useShortcutBinding(commandId);

  return binding?.display ?? null;
};

export interface ShortcutBinding {
  aria: string;
  display: string;
  /** Canonical per-key tokens (`cmd`, `enter`, `k`, …) for glyph rendering. */
  parts: string[];
}

export const useShortcutBinding = (commandId: string): ShortcutBinding | null => {
  const customHotkeys = useWorkbenchPreferenceSelector((preferences) => preferences.customHotkeys);
  const definition = findDefinition(commandId);
  const firstKey = definition ? applyCustomHotkeys(definition, customHotkeys).keys[0] : undefined;

  return firstKey
    ? {
        aria: formatHotkeyAriaLabel(firstKey),
        display: formatHotkeyLabel(firstKey),
        parts: formatHotkeyForPlatform(firstKey),
      }
    : null;
};
