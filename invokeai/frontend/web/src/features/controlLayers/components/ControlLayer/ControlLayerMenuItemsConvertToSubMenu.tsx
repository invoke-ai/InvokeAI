import { Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { SubMenuButtonContent, useSubMenu } from 'common/hooks/useSubMenu';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useIsEntityInteractable } from 'features/controlLayers/hooks/useEntityIsInteractable';
import {
  controlLayerConvertedToInpaintMask,
  controlLayerConvertedToRasterLayer,
  controlLayerConvertedToRegionalGuidance,
} from 'features/controlLayers/store/canvasSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiSwapBold } from 'react-icons/pi';

export const ControlLayerMenuItemsConvertToSubMenu = memo(() => {
  const { t } = useTranslation();
  const subMenu = useSubMenu();
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext('control_layer');
  const isInteractable = useIsEntityInteractable(entityIdentifier);

  const convertToInpaintMask = useCallback(() => {
    dispatch(controlLayerConvertedToInpaintMask({ entityIdentifier, replace: true }));
  }, [dispatch, entityIdentifier]);

  const convertToRegionalGuidance = useCallback(() => {
    dispatch(controlLayerConvertedToRegionalGuidance({ entityIdentifier, replace: true }));
  }, [dispatch, entityIdentifier]);

  const convertToRasterLayer = useCallback(() => {
    dispatch(controlLayerConvertedToRasterLayer({ entityIdentifier, replace: true }));
  }, [dispatch, entityIdentifier]);

  return (
    <MenuItem {...subMenu.parentMenuItemProps} icon={<PiSwapBold />}>
      <Menu {...subMenu.menuProps}>
        <MenuButton {...subMenu.menuButtonProps}>
          <SubMenuButtonContent label={t('controlLayers.convertControlLayerTo')} />
        </MenuButton>
        <MenuList {...subMenu.menuListProps}>
          <MenuItem onClick={convertToInpaintMask} icon={<PiSwapBold />} isDisabled={!isInteractable}>
            {t('controlLayers.inpaintMask')}
          </MenuItem>
          <MenuItem onClick={convertToRegionalGuidance} icon={<PiSwapBold />} isDisabled={!isInteractable}>
            {t('controlLayers.regionalGuidance')}
          </MenuItem>
          <MenuItem onClick={convertToRasterLayer} icon={<PiSwapBold />} isDisabled={!isInteractable}>
            {t('controlLayers.rasterLayer')}
          </MenuItem>
        </MenuList>
      </Menu>
    </MenuItem>
  );
});

ControlLayerMenuItemsConvertToSubMenu.displayName = 'ControlLayerMenuItemsConvertToSubMenu';
