import { Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { SubMenuButtonContent, useSubMenu } from 'common/hooks/useSubMenu';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useEntityIsLocked } from 'features/controlLayers/hooks/useEntityIsLocked';
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
  const isBusy = useCanvasIsBusy();
  const isLocked = useEntityIsLocked(entityIdentifier);

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
    <MenuItem {...subMenu.parentMenuItemProps} icon={<PiSwapBold />} isDisabled={isLocked || isBusy}>
      <Menu {...subMenu.menuProps}>
        <MenuButton {...subMenu.menuButtonProps}>
          <SubMenuButtonContent label={t('controlLayers.convertControlLayerTo')} />
        </MenuButton>
        <MenuList {...subMenu.menuListProps}>
          <MenuItem onClick={convertToInpaintMask} icon={<PiSwapBold />} isDisabled={isLocked || isBusy}>
            {t('controlLayers.inpaintMask')}
          </MenuItem>
          <MenuItem onClick={convertToRegionalGuidance} icon={<PiSwapBold />} isDisabled={isLocked || isBusy}>
            {t('controlLayers.regionalGuidance')}
          </MenuItem>
          <MenuItem onClick={convertToRasterLayer} icon={<PiSwapBold />} isDisabled={isLocked || isBusy}>
            {t('controlLayers.rasterLayer')}
          </MenuItem>
        </MenuList>
      </Menu>
    </MenuItem>
  );
});

ControlLayerMenuItemsConvertToSubMenu.displayName = 'ControlLayerMenuItemsConvertToSubMenu';
