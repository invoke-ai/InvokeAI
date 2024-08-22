import { IconButton, Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useDefaultControlAdapter, useDefaultIPAdapter } from 'features/controlLayers/hooks/useLayerControlAdapter';
import {
  controlLayerAdded,
  inpaintMaskAdded,
  ipaAdded,
  rasterLayerAdded,
  rgAdded,
} from 'features/controlLayers/store/canvasV2Slice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';

export const AddLayerButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const defaultControlAdapter = useDefaultControlAdapter();
  const defaultIPAdapter = useDefaultIPAdapter();
  const addInpaintMask = useCallback(() => {
    dispatch(inpaintMaskAdded());
  }, [dispatch]);
  const addRegionalGuidance = useCallback(() => {
    dispatch(rgAdded());
  }, [dispatch]);
  const addRasterLayer = useCallback(() => {
    dispatch(rasterLayerAdded({ isSelected: true }));
  }, [dispatch]);
  const addControlLayer = useCallback(() => {
    dispatch(controlLayerAdded({ isSelected: true, overrides: { controlAdapter: defaultControlAdapter } }));
  }, [defaultControlAdapter, dispatch]);
  const addIPAdapter = useCallback(() => {
    dispatch(ipaAdded({ ipAdapter: defaultIPAdapter }));
  }, [defaultIPAdapter, dispatch]);

  return (
    <Menu>
      <MenuButton
        as={IconButton}
        aria-label={t('controlLayers.addLayer')}
        icon={<PiPlusBold />}
        variant="link"
        data-testid="control-layers-add-layer-menu-button"
        alignSelf="stretch"
      />
      <MenuList>
        <MenuItem onClick={addInpaintMask}>{t('controlLayers.inpaintMask', { count: 1 })}</MenuItem>
        <MenuItem onClick={addRegionalGuidance}>{t('controlLayers.regionalGuidance', { count: 1 })}</MenuItem>
        <MenuItem onClick={addRasterLayer}>{t('controlLayers.rasterLayer', { count: 1 })}</MenuItem>
        <MenuItem onClick={addControlLayer}>{t('controlLayers.controlLayer', { count: 1 })}</MenuItem>
        <MenuItem onClick={addIPAdapter}>{t('controlLayers.ipAdapter', { count: 1 })}</MenuItem>
      </MenuList>
    </Menu>
  );
});

AddLayerButton.displayName = 'AddLayerButton';
