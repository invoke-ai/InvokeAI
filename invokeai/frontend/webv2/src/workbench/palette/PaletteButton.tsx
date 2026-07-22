import { Icon } from '@chakra-ui/react';
import { IconButton } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import { formatHotkeyForPlatform } from '@workbench/hotkeys/keys';
import { SearchIcon } from 'lucide-react';

import { openCommandPalette } from './paletteStore';

/** Pointer affordance for the palette; the kbd hint keeps mod+K discoverable. */
export const PaletteButton = () => (
  <Tooltip content={`Command palette (${formatHotkeyForPlatform('mod+k').join('+')})`}>
    <IconButton aria-label="Command palette" size="sm" variant="ghost" onClick={openCommandPalette}>
      <Icon as={SearchIcon} />
    </IconButton>
  </Tooltip>
);
