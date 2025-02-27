import { Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { SubMenuButtonContent, useSubMenu } from 'common/hooks/useSubMenu';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { useImageActions } from 'features/gallery/hooks/useImageActions';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiArrowBendUpLeftBold,
  PiArrowsCounterClockwiseBold,
  PiAsteriskBold,
  PiPaintBrushBold,
  PiPlantBold,
  PiQuotesBold,
} from 'react-icons/pi';

export const ImageMenuItemMetadataRecallActions = memo(() => {
  const { t } = useTranslation();
  const imageDTO = useImageDTOContext();
  const subMenu = useSubMenu();

  const { recallAll, remix, recallSeed, recallPrompts, hasMetadata, hasSeed, hasPrompts, createAsPreset } =
    useImageActions(imageDTO);

  return (
    <MenuItem {...subMenu.parentMenuItemProps} icon={<PiArrowBendUpLeftBold />}>
      <Menu {...subMenu.menuProps}>
        <MenuButton {...subMenu.menuButtonProps}>
          <SubMenuButtonContent label={t('parameters.recallMetadata')} />
        </MenuButton>
        <MenuList {...subMenu.menuListProps}>
          <MenuItem icon={<PiArrowsCounterClockwiseBold />} onClick={remix} isDisabled={!hasMetadata}>
            {t('parameters.remixImage')}
          </MenuItem>
          <MenuItem icon={<PiQuotesBold />} onClick={recallPrompts} isDisabled={!hasPrompts}>
            {t('parameters.usePrompt')}
          </MenuItem>
          <MenuItem icon={<PiPlantBold />} onClick={recallSeed} isDisabled={!hasSeed}>
            {t('parameters.useSeed')}
          </MenuItem>
          <MenuItem icon={<PiAsteriskBold />} onClick={recallAll} isDisabled={!hasMetadata}>
            {t('parameters.useAll')}
          </MenuItem>
          <MenuItem icon={<PiPaintBrushBold />} onClick={createAsPreset} isDisabled={!hasPrompts}>
            {t('stylePresets.useForTemplate')}
          </MenuItem>
        </MenuList>
      </Menu>
    </MenuItem>
  );
});

ImageMenuItemMetadataRecallActions.displayName = 'ImageMenuItemMetadataRecallActions';
