import { Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { SubMenuButtonContent, useSubMenu } from 'common/hooks/useSubMenu';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useIsEntityInteractable } from 'features/controlLayers/hooks/useEntityIsInteractable';
import { inpaintMaskConvertedToRegionalGuidance } from 'features/controlLayers/store/canvasSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiSwapBold } from 'react-icons/pi';

export const InpaintMaskMenuItemsConvertToSubMenu = memo(() => {
  const { t } = useTranslation();
  const subMenu = useSubMenu();
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext('inpaint_mask');
  const isInteractable = useIsEntityInteractable(entityIdentifier);

  const convertToRegionalGuidance = useCallback(() => {
    dispatch(inpaintMaskConvertedToRegionalGuidance({ entityIdentifier, replace: true }));
  }, [dispatch, entityIdentifier]);

  return (
    <MenuItem {...subMenu.parentMenuItemProps} icon={<PiSwapBold />}>
      <Menu {...subMenu.menuProps}>
        <MenuButton {...subMenu.menuButtonProps}>
          <SubMenuButtonContent label={t('controlLayers.convertInpaintMaskTo')} />
        </MenuButton>
        <MenuList {...subMenu.menuListProps}>
          <MenuItem onClick={convertToRegionalGuidance} icon={<PiSwapBold />} isDisabled={!isInteractable}>
            {t('controlLayers.regionalGuidance')}
          </MenuItem>
        </MenuList>
      </Menu>
    </MenuItem>
  );
});

InpaintMaskMenuItemsConvertToSubMenu.displayName = 'InpaintMaskMenuItemsConvertToSubMenu';
