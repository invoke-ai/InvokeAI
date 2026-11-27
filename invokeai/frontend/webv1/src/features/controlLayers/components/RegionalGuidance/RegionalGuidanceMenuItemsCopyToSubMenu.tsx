import { Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { SubMenuButtonContent, useSubMenu } from 'common/hooks/useSubMenu';
import { CanvasEntityMenuItemsCopyToClipboard } from 'features/controlLayers/components/common/CanvasEntityMenuItemsCopyToClipboard';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { rgConvertedToInpaintMask } from 'features/controlLayers/store/canvasSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCopyBold } from 'react-icons/pi';

export const RegionalGuidanceMenuItemsCopyToSubMenu = memo(() => {
  const { t } = useTranslation();
  const subMenu = useSubMenu();
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext('regional_guidance');
  const isBusy = useCanvasIsBusy();

  const copyToInpaintMask = useCallback(() => {
    dispatch(rgConvertedToInpaintMask({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);

  return (
    <MenuItem {...subMenu.parentMenuItemProps} icon={<PiCopyBold />} isDisabled={isBusy}>
      <Menu {...subMenu.menuProps}>
        <MenuButton {...subMenu.menuButtonProps}>
          <SubMenuButtonContent label={t('controlLayers.copyRegionalGuidanceTo')} />
        </MenuButton>
        <MenuList {...subMenu.menuListProps}>
          <CanvasEntityMenuItemsCopyToClipboard />
          <MenuItem onClick={copyToInpaintMask} icon={<PiCopyBold />} isDisabled={isBusy}>
            {t('controlLayers.newInpaintMask')}
          </MenuItem>
        </MenuList>
      </Menu>
    </MenuItem>
  );
});

RegionalGuidanceMenuItemsCopyToSubMenu.displayName = 'RegionalGuidanceMenuItemsCopyToSubMenu';
