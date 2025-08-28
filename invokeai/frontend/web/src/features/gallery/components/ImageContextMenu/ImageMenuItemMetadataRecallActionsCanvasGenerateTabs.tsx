import { Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { SubMenuButtonContent, useSubMenu } from 'common/hooks/useSubMenu';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { useRecallAll } from 'features/gallery/hooks/useRecallAll';
import { useRecallCLIPSkip } from 'features/gallery/hooks/useRecallCLIPSkip';
import { useRecallDimensions } from 'features/gallery/hooks/useRecallDimensions';
import { useRecallPrompts } from 'features/gallery/hooks/useRecallPrompts';
import { useRecallRemix } from 'features/gallery/hooks/useRecallRemix';
import { useRecallSeed } from 'features/gallery/hooks/useRecallSeed';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiArrowBendUpLeftBold,
  PiArrowsCounterClockwiseBold,
  PiAsteriskBold,
  PiPlantBold,
  PiQuotesBold,
  PiRulerBold,
} from 'react-icons/pi';

export const ImageMenuItemMetadataRecallActionsCanvasGenerateTabs = memo(() => {
  const { t } = useTranslation();
  const subMenu = useSubMenu();

  const imageDTO = useImageDTOContext();

  const recallAll = useRecallAll(imageDTO);
  const recallRemix = useRecallRemix(imageDTO);
  const recallPrompts = useRecallPrompts(imageDTO);
  const recallSeed = useRecallSeed(imageDTO);
  const recallDimensions = useRecallDimensions(imageDTO);
  const recallCLIPSkip = useRecallCLIPSkip(imageDTO);

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
          <MenuItem icon={<PiRulerBold />} onClick={recallDimensions.recall} isDisabled={!recallDimensions.isEnabled}>
            {t('parameters.useSize')}
          </MenuItem>
          <MenuItem icon={<PiRulerBold />} onClick={recallCLIPSkip.recall} isDisabled={!recallCLIPSkip.isEnabled}>
            {t('parameters.useClipSkip')}
          </MenuItem>
        </MenuList>
      </Menu>
    </MenuItem>
  );
});

ImageMenuItemMetadataRecallActionsCanvasGenerateTabs.displayName =
  'ImageMenuItemMetadataRecallActionsCanvasGenerateTabs';
