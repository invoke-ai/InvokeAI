import { Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { SubMenuButtonContent, useSubMenu } from 'common/hooks/useSubMenu';
import { deepClone } from 'common/util/deepClone';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useEntityIsLocked } from 'features/controlLayers/hooks/useEntityIsLocked';
import {
  rasterLayerConvertedToControlLayer,
  rasterLayerConvertedToInpaintMask,
  rasterLayerConvertedToRegionalGuidance,
} from 'features/controlLayers/store/canvasSlice';
import { initialControlNet } from 'features/controlLayers/store/util';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiSwapBold } from 'react-icons/pi';

export const RasterLayerMenuItemsConvertToSubMenu = memo(() => {
  const { t } = useTranslation();
  const subMenu = useSubMenu();

  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext('raster_layer');
  const isBusy = useCanvasIsBusy();
  const isLocked = useEntityIsLocked(entityIdentifier);

  const convertToInpaintMask = useCallback(() => {
    dispatch(rasterLayerConvertedToInpaintMask({ entityIdentifier, replace: true }));
  }, [dispatch, entityIdentifier]);

  const convertToRegionalGuidance = useCallback(() => {
    dispatch(rasterLayerConvertedToRegionalGuidance({ entityIdentifier, replace: true }));
  }, [dispatch, entityIdentifier]);

  const convertToControlLayer = useCallback(() => {
    dispatch(
      rasterLayerConvertedToControlLayer({
        entityIdentifier,
        replace: true,
        overrides: { controlAdapter: deepClone(initialControlNet) },
      })
    );
  }, [dispatch, entityIdentifier]);

  return (
    <MenuItem {...subMenu.parentMenuItemProps} icon={<PiSwapBold />} isDisabled={isBusy || isLocked}>
      <Menu {...subMenu.menuProps}>
        <MenuButton {...subMenu.menuButtonProps}>
          <SubMenuButtonContent label={t('controlLayers.convertRasterLayerTo')} />
        </MenuButton>
        <MenuList {...subMenu.menuListProps}>
          <MenuItem onClick={convertToInpaintMask} icon={<PiSwapBold />} isDisabled={isBusy || isLocked}>
            {t('controlLayers.inpaintMask')}
          </MenuItem>
          <MenuItem onClick={convertToRegionalGuidance} icon={<PiSwapBold />} isDisabled={isBusy || isLocked}>
            {t('controlLayers.regionalGuidance')}
          </MenuItem>
          <MenuItem onClick={convertToControlLayer} icon={<PiSwapBold />} isDisabled={isBusy || isLocked}>
            {t('controlLayers.controlLayer')}
          </MenuItem>
        </MenuList>
      </Menu>
    </MenuItem>
  );
});

RasterLayerMenuItemsConvertToSubMenu.displayName = 'RasterLayerMenuItemsConvertToSubMenu';
