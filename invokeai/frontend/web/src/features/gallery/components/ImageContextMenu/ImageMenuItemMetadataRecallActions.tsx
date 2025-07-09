import { Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { SubMenuButtonContent, useSubMenu } from 'common/hooks/useSubMenu';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { useCreateStylePresetFromMetadata } from 'features/gallery/hooks/useCreateStylePresetFromMetadata';
import { useRecallAll } from 'features/gallery/hooks/useRecallAll';
import { useRecallPrompts } from 'features/gallery/hooks/useRecallPrompts';
import { useRecallRemix } from 'features/gallery/hooks/useRecallRemix';
import { useRecallSeed } from 'features/gallery/hooks/useRecallSeed';
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
  const subMenu = useSubMenu();

  const imageDTO = useImageDTOContext();

  const recallAll = useRecallAll(imageDTO);
  const recallRemix = useRecallRemix(imageDTO);
  const recallPrompts = useRecallPrompts(imageDTO);
  const recallSeed = useRecallSeed(imageDTO);
  const stylePreset = useCreateStylePresetFromMetadata(imageDTO);

  return (
    <MenuItem {...subMenu.parentMenuItemProps} icon={<PiArrowBendUpLeftBold />}>
      <Menu {...subMenu.menuProps}>
        <MenuButton {...subMenu.menuButtonProps}>
          <SubMenuButtonContent label={t('parameters.recallMetadata')} />
        </MenuButton>
        <MenuList {...subMenu.menuListProps}>
          <MenuItem
            icon={<PiArrowsCounterClockwiseBold />}
            onClick={recallRemix.recall}
            isDisabled={!recallRemix.isEnabled}
          >
            {t('parameters.remixImage')}
          </MenuItem>
          <MenuItem icon={<PiQuotesBold />} onClick={recallPrompts.recall} isDisabled={!recallPrompts.isEnabled}>
            {t('parameters.usePrompt')}
          </MenuItem>
          <MenuItem icon={<PiPlantBold />} onClick={recallSeed.recall} isDisabled={!recallSeed.isEnabled}>
            {t('parameters.useSeed')}
          </MenuItem>
          <MenuItem icon={<PiAsteriskBold />} onClick={recallAll.recall} isDisabled={!recallAll.isEnabled}>
            {t('parameters.useAll')}
          </MenuItem>
          <MenuItem icon={<PiPaintBrushBold />} onClick={stylePreset.create} isDisabled={!stylePreset.isEnabled}>
            {t('stylePresets.useForTemplate')}
          </MenuItem>
        </MenuList>
      </Menu>
    </MenuItem>
  );
});

ImageMenuItemMetadataRecallActions.displayName = 'ImageMenuItemMetadataRecallActions';
