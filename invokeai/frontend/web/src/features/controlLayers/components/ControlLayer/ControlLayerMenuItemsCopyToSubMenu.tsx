import { Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { SubMenuButtonContent, useSubMenu } from 'common/hooks/useSubMenu';
import { CanvasEntityMenuItemsCopyToClipboard } from 'features/controlLayers/components/common/CanvasEntityMenuItemsCopyToClipboard';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useIsEntityInteractable } from 'features/controlLayers/hooks/useEntityIsInteractable';
import {
  controlLayerConvertedToInpaintMask,
  controlLayerConvertedToRasterLayer,
  controlLayerConvertedToRegionalGuidance,
} from 'features/controlLayers/store/canvasSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCopyBold } from 'react-icons/pi';

export const ControlLayerMenuItemsCopyToSubMenu = memo(() => {
  const { t } = useTranslation();
  const subMenu = useSubMenu();
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext('control_layer');
  const isInteractable = useIsEntityInteractable(entityIdentifier);

  const copyToInpaintMask = useCallback(() => {
    dispatch(controlLayerConvertedToInpaintMask({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);

  const copyToRegionalGuidance = useCallback(() => {
    dispatch(controlLayerConvertedToRegionalGuidance({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);

  const copyToRasterLayer = useCallback(() => {
    dispatch(controlLayerConvertedToRasterLayer({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);

  return (
    <MenuItem {...subMenu.parentMenuItemProps} icon={<PiCopyBold />}>
      <Menu {...subMenu.menuProps}>
        <MenuButton {...subMenu.menuButtonProps}>
          <SubMenuButtonContent label={t('controlLayers.copyControlLayerTo')} />
        </MenuButton>
        <MenuList {...subMenu.menuListProps}>
          <CanvasEntityMenuItemsCopyToClipboard />
          <MenuItem onClick={copyToInpaintMask} icon={<PiCopyBold />} isDisabled={!isInteractable}>
            {t('controlLayers.newInpaintMask')}
          </MenuItem>
          <MenuItem onClick={copyToRegionalGuidance} icon={<PiCopyBold />} isDisabled={!isInteractable}>
            {t('controlLayers.newRegionalGuidance')}
          </MenuItem>
          <MenuItem onClick={copyToRasterLayer} icon={<PiCopyBold />} isDisabled={!isInteractable}>
            {t('controlLayers.newRasterLayer')}
          </MenuItem>
        </MenuList>
      </Menu>
    </MenuItem>
  );
});

ControlLayerMenuItemsCopyToSubMenu.displayName = 'ControlLayerMenuItemsCopyToSubMenu';
