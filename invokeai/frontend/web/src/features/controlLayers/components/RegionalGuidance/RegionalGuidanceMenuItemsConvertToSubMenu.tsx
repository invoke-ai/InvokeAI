import { Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { SubMenuButtonContent, useSubMenu } from 'common/hooks/useSubMenu';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useIsEntityInteractable } from 'features/controlLayers/hooks/useEntityIsInteractable';
import { rgConvertedToInpaintMask } from 'features/controlLayers/store/canvasSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiSwapBold } from 'react-icons/pi';

export const RegionalGuidanceMenuItemsConvertToSubMenu = memo(() => {
  const { t } = useTranslation();
  const subMenu = useSubMenu();
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext('regional_guidance');
  const isInteractable = useIsEntityInteractable(entityIdentifier);

  const convertToInpaintMask = useCallback(() => {
    dispatch(rgConvertedToInpaintMask({ entityIdentifier, replace: true }));
  }, [dispatch, entityIdentifier]);

  return (
    <MenuItem {...subMenu.parentMenuItemProps} icon={<PiSwapBold />}>
      <Menu {...subMenu.menuProps}>
        <MenuButton {...subMenu.menuButtonProps}>
          <SubMenuButtonContent label={t('controlLayers.convertRegionalGuidanceTo')} />
        </MenuButton>
        <MenuList {...subMenu.menuListProps}>
          <MenuItem onClick={convertToInpaintMask} icon={<PiSwapBold />} isDisabled={!isInteractable}>
            {t('controlLayers.inpaintMask')}
          </MenuItem>
        </MenuList>
      </Menu>
    </MenuItem>
  );
});

RegionalGuidanceMenuItemsConvertToSubMenu.displayName = 'RegionalGuidanceMenuItemsConvertToSubMenu';
