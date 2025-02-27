import { Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { SubMenuButtonContent, useSubMenu } from 'common/hooks/useSubMenu';
import { CanvasEntityMenuItemsCopyToClipboard } from 'features/controlLayers/components/common/CanvasEntityMenuItemsCopyToClipboard';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { inpaintMaskConvertedToRegionalGuidance } from 'features/controlLayers/store/canvasSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCopyBold } from 'react-icons/pi';

export const InpaintMaskMenuItemsCopyToSubMenu = memo(() => {
  const { t } = useTranslation();
  const subMenu = useSubMenu();
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext('inpaint_mask');
  const isBusy = useCanvasIsBusy();

  const copyToRegionalGuidance = useCallback(() => {
    dispatch(inpaintMaskConvertedToRegionalGuidance({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);

  return (
    <MenuItem {...subMenu.parentMenuItemProps} icon={<PiCopyBold />} isDisabled={isBusy}>
      <Menu {...subMenu.menuProps}>
        <MenuButton {...subMenu.menuButtonProps}>
          <SubMenuButtonContent label={t('controlLayers.copyInpaintMaskTo')} />
        </MenuButton>
        <MenuList {...subMenu.menuListProps}>
          <CanvasEntityMenuItemsCopyToClipboard />
          <MenuItem onClick={copyToRegionalGuidance} icon={<PiCopyBold />} isDisabled={isBusy}>
            {t('controlLayers.newRegionalGuidance')}
          </MenuItem>
        </MenuList>
      </Menu>
    </MenuItem>
  );
});

InpaintMaskMenuItemsCopyToSubMenu.displayName = 'InpaintMaskMenuItemsCopyToSubMenu';
