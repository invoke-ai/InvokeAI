import { Icon } from '@chakra-ui/react';
import { IconButton } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import { OPEN_COMMAND_PALETTE_HOTKEY } from '@workbench/hotkeys/catalog';
import { formatHotkeyForPlatform } from '@workbench/hotkeys/keys';
import { applyCustomHotkeys } from '@workbench/hotkeys/resolve';
import { useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { SearchIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { openCommandPalette } from './paletteStore';

/** Pointer affordance for the palette; the tooltip advertises its effective binding. */
export const PaletteButton = () => {
  const { t } = useTranslation();
  const customHotkeys = useWorkbenchPreferenceSelector((preferences) => preferences.customHotkeys);
  const firstHotkey = applyCustomHotkeys(OPEN_COMMAND_PALETTE_HOTKEY, customHotkeys).keys[0];
  const hotkeyLabel = firstHotkey ? formatHotkeyForPlatform(firstHotkey).join('+') : null;
  const tooltip = hotkeyLabel
    ? t('commandPalette.buttonTooltip', { hotkey: hotkeyLabel })
    : t('commandPalette.buttonLabel');

  return (
    <Tooltip content={tooltip}>
      <IconButton aria-label={t('commandPalette.buttonLabel')} size="sm" variant="ghost" onClick={openCommandPalette}>
        <Icon as={SearchIcon} />
      </IconButton>
    </Tooltip>
  );
};
