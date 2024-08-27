import { MenuDivider, MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  allEntitiesDeleted,
  controlLayerAdded,
  inpaintMaskAdded,
  ipaAdded,
  rasterLayerAdded,
  rgAdded,
} from 'features/controlLayers/store/canvasSlice';
import { selectHasEntities } from 'features/controlLayers/store/selectors';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold, PiTrashSimpleBold } from 'react-icons/pi';

export const CanvasEntityListMenuItems = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const hasEntities = useAppSelector(selectHasEntities);
  const addInpaintMask = useCallback(() => {
    dispatch(inpaintMaskAdded({ isSelected: true }));
  }, [dispatch]);
  const addRegionalGuidance = useCallback(() => {
    dispatch(rgAdded({ isSelected: true }));
  }, [dispatch]);
  const addRasterLayer = useCallback(() => {
    dispatch(rasterLayerAdded({ isSelected: true }));
  }, [dispatch]);
  const addControlLayer = useCallback(() => {
    dispatch(controlLayerAdded({ isSelected: true }));
  }, [dispatch]);
  const addIPAdapter = useCallback(() => {
    dispatch(ipaAdded({ isSelected: true }));
  }, [dispatch]);
  const deleteAll = useCallback(() => {
    dispatch(allEntitiesDeleted());
  }, [dispatch]);

  return (
    <>
      <MenuItem icon={<PiPlusBold />} onClick={addInpaintMask}>
        {t('controlLayers.inpaintMask', { count: 1 })}
      </MenuItem>
      <MenuItem icon={<PiPlusBold />} onClick={addRegionalGuidance}>
        {t('controlLayers.regionalGuidance', { count: 1 })}
      </MenuItem>
      <MenuItem icon={<PiPlusBold />} onClick={addRasterLayer}>
        {t('controlLayers.rasterLayer', { count: 1 })}
      </MenuItem>
      <MenuItem icon={<PiPlusBold />} onClick={addControlLayer}>
        {t('controlLayers.controlLayer', { count: 1 })}
      </MenuItem>
      <MenuItem icon={<PiPlusBold />} onClick={addIPAdapter}>
        {t('controlLayers.ipAdapter', { count: 1 })}
      </MenuItem>
      <MenuDivider />
      <MenuItem onClick={deleteAll} icon={<PiTrashSimpleBold />} color="error.300" isDisabled={!hasEntities}>
        {t('controlLayers.deleteAll', { count: 1 })}
      </MenuItem>
    </>
  );
});

CanvasEntityListMenuItems.displayName = 'CanvasEntityListMenu';
