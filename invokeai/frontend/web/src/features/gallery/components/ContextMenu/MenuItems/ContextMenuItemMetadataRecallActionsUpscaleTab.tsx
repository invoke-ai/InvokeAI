import { Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { SubMenuButtonContent, useSubMenu } from 'common/hooks/useSubMenu';
import { useItemDTOContext } from 'features/gallery/contexts/ItemDTOContext';
import { useRecallPrompts } from 'features/gallery/hooks/useRecallPrompts';
import { useRecallSeed } from 'features/gallery/hooks/useRecallSeed';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowBendUpLeftBold, PiPlantBold, PiQuotesBold } from 'react-icons/pi';
import type { ImageDTO } from 'services/api/types';

export const ContextMenuItemMetadataRecallActionsUpscaleTab = memo(() => {
  const { t } = useTranslation();
  const subMenu = useSubMenu();

  const itemDTO = useItemDTOContext();

  // TODO: Implement video recall metadata actions
  const recallPrompts = useRecallPrompts(itemDTO as ImageDTO);
  const recallSeed = useRecallSeed(itemDTO as ImageDTO);

  return (
    <MenuItem {...subMenu.parentMenuItemProps} icon={<PiArrowBendUpLeftBold />}>
      <Menu {...subMenu.menuProps}>
        <MenuButton {...subMenu.menuButtonProps}>
          <SubMenuButtonContent label={t('parameters.recallMetadata')} />
        </MenuButton>
        <MenuList {...subMenu.menuListProps}>
          <MenuItem icon={<PiQuotesBold />} onClick={recallPrompts.recall} isDisabled={!recallPrompts.isEnabled}>
            {t('parameters.usePrompt')}
          </MenuItem>
          <MenuItem icon={<PiPlantBold />} onClick={recallSeed.recall} isDisabled={!recallSeed.isEnabled}>
            {t('parameters.useSeed')}
          </MenuItem>
        </MenuList>
      </Menu>
    </MenuItem>
  );
});

ContextMenuItemMetadataRecallActionsUpscaleTab.displayName = 'ContextMenuItemMetadataRecallActionsUpscaleTab';
