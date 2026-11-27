import { Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { SubMenuButtonContent, useSubMenu } from 'common/hooks/useSubMenu';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { useRecallPrompts } from 'features/gallery/hooks/useRecallPrompts';
import { useRecallSeed } from 'features/gallery/hooks/useRecallSeed';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowBendUpLeftBold, PiPlantBold, PiQuotesBold } from 'react-icons/pi';

export const ContextMenuItemMetadataRecallActionsUpscaleTab = memo(() => {
  const { t } = useTranslation();
  const subMenu = useSubMenu();

  const imageDTO = useImageDTOContext();

  const recallPrompts = useRecallPrompts(imageDTO);
  const recallSeed = useRecallSeed(imageDTO);

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
