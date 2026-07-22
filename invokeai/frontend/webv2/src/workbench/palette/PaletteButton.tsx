import { Icon } from '@chakra-ui/react';
import { IconButton } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import { SearchIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { openCommandPalette } from './paletteStore';

const hotkeyLabel = /mac/i.test(navigator.platform) ? 'cmd+k' : 'ctrl+k';

/** Pointer affordance for the palette; the kbd hint keeps mod+K discoverable. */
export const PaletteButton = () => {
  const { t } = useTranslation();

  return (
    <Tooltip content={t('commandPalette.buttonTooltip', { hotkey: hotkeyLabel })}>
      <IconButton aria-label={t('commandPalette.buttonLabel')} size="sm" variant="ghost" onClick={openCommandPalette}>
        <Icon as={SearchIcon} />
      </IconButton>
    </Tooltip>
  );
};
