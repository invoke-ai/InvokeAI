import { Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { SubMenuButtonContent, useSubMenu } from 'common/hooks/useSubMenu';
import { useItemDTOContext } from 'features/gallery/contexts/ItemDTOContext';
import { useRecallAll } from 'features/gallery/hooks/useRecallAllImageMetadata';
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
import type { ImageDTO } from 'services/api/types';

export const ContextMenuItemMetadataRecallActionsCanvasGenerateTabs = memo(() => {
  const { t } = useTranslation();
  const subMenu = useSubMenu();

  const itemDTO = useItemDTOContext();

  // TODO: Implement video recall metadata actions
  const recallAll = useRecallAll(itemDTO as ImageDTO);
  const recallRemix = useRecallRemix(itemDTO as ImageDTO);
  const recallPrompts = useRecallPrompts(itemDTO as ImageDTO);
  const recallSeed = useRecallSeed(itemDTO as ImageDTO);
  const recallDimensions = useRecallDimensions(itemDTO as ImageDTO);
  const recallCLIPSkip = useRecallCLIPSkip(itemDTO as ImageDTO);

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

ContextMenuItemMetadataRecallActionsCanvasGenerateTabs.displayName =
  'ContextMenuItemMetadataRecallActionsCanvasGenerateTabs';
