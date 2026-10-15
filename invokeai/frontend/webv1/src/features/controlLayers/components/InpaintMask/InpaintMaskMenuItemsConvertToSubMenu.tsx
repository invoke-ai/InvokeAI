import { Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { SubMenuButtonContent, useSubMenu } from 'common/hooks/useSubMenu';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useEntityIsLocked } from 'features/controlLayers/hooks/useEntityIsLocked';
import { inpaintMaskConvertedToRegionalGuidance } from 'features/controlLayers/store/canvasSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiSwapBold } from 'react-icons/pi';

export const InpaintMaskMenuItemsConvertToSubMenu = memo(() => {
  const { t } = useTranslation();
  const subMenu = useSubMenu();
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext('inpaint_mask');
  const isBusy = useCanvasIsBusy();
  const isLocked = useEntityIsLocked(entityIdentifier);

  const convertToRegionalGuidance = useCallback(() => {
    dispatch(inpaintMaskConvertedToRegionalGuidance({ entityIdentifier, replace: true }));
  }, [dispatch, entityIdentifier]);

  return (
    <MenuItem {...subMenu.parentMenuItemProps} icon={<PiSwapBold />} isDisabled={isBusy || isLocked}>
      <Menu {...subMenu.menuProps}>
        <MenuButton {...subMenu.menuButtonProps}>
          <SubMenuButtonContent label={t('controlLayers.convertInpaintMaskTo')} />
        </MenuButton>
        <MenuList {...subMenu.menuListProps}>
          <MenuItem onClick={convertToRegionalGuidance} icon={<PiSwapBold />} isDisabled={isBusy || isLocked}>
            {t('controlLayers.regionalGuidance')}
          </MenuItem>
        </MenuList>
      </Menu>
    </MenuItem>
  );
});

InpaintMaskMenuItemsConvertToSubMenu.displayName = 'InpaintMaskMenuItemsConvertToSubMenu';
