import { Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { SubMenuButtonContent, useSubMenu } from 'common/hooks/useSubMenu';
import { deepClone } from 'common/util/deepClone';
import { CanvasEntityMenuItemsCopyToClipboard } from 'features/controlLayers/components/common/CanvasEntityMenuItemsCopyToClipboard';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import {
  rasterLayerConvertedToControlLayer,
  rasterLayerConvertedToInpaintMask,
  rasterLayerConvertedToRegionalGuidance,
} from 'features/controlLayers/store/canvasSlice';
import { initialControlNet } from 'features/controlLayers/store/util';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCopyBold } from 'react-icons/pi';

export const RasterLayerMenuItemsCopyToSubMenu = memo(() => {
  const { t } = useTranslation();
  const subMenu = useSubMenu();

  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext('raster_layer');
  const isBusy = useCanvasIsBusy();

  const copyToInpaintMask = useCallback(() => {
    dispatch(rasterLayerConvertedToInpaintMask({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);

  const copyToRegionalGuidance = useCallback(() => {
    dispatch(rasterLayerConvertedToRegionalGuidance({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);

  const copyToControlLayer = useCallback(() => {
    dispatch(
      rasterLayerConvertedToControlLayer({
        entityIdentifier,
        overrides: { controlAdapter: deepClone(initialControlNet) },
      })
    );
  }, [dispatch, entityIdentifier]);

  return (
    <MenuItem {...subMenu.parentMenuItemProps} icon={<PiCopyBold />} isDisabled={isBusy}>
      <Menu {...subMenu.menuProps}>
        <MenuButton {...subMenu.menuButtonProps}>
          <SubMenuButtonContent label={t('controlLayers.copyRasterLayerTo')} />
        </MenuButton>
        <MenuList {...subMenu.menuListProps}>
          <CanvasEntityMenuItemsCopyToClipboard />
          <MenuItem onClick={copyToInpaintMask} icon={<PiCopyBold />} isDisabled={isBusy}>
            {t('controlLayers.newInpaintMask')}
          </MenuItem>
          <MenuItem onClick={copyToRegionalGuidance} icon={<PiCopyBold />} isDisabled={isBusy}>
            {t('controlLayers.newRegionalGuidance')}
          </MenuItem>
          <MenuItem onClick={copyToControlLayer} icon={<PiCopyBold />} isDisabled={isBusy}>
            {t('controlLayers.newControlLayer')}
          </MenuItem>
        </MenuList>
      </Menu>
    </MenuItem>
  );
});

RasterLayerMenuItemsCopyToSubMenu.displayName = 'RasterLayerMenuItemsCopyToSubMenu';
