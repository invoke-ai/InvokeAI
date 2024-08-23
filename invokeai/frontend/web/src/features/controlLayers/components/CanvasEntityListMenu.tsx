import { IconButton, Menu, MenuButton, MenuDivider, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useDefaultControlAdapter, useDefaultIPAdapter } from 'features/controlLayers/hooks/useLayerControlAdapter';
import {
  allEntitiesDeleted,
  controlLayerAdded,
  inpaintMaskAdded,
  ipaAdded,
  rasterLayerAdded,
  rgAdded,
} from 'features/controlLayers/store/canvasV2Slice';
import { selectEntityCount } from 'features/controlLayers/store/selectors';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDotsThreeOutlineFill, PiPlusBold, PiTrashSimpleBold } from 'react-icons/pi';

export const CanvasEntityListMenu = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const hasEntities = useAppSelector((s) => {
    const count = selectEntityCount(s);
    return count > 0;
  });
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
  const deleteAll = useCallback(() => {
    dispatch(allEntitiesDeleted());
  }, [dispatch]);

  return (
    <Menu>
      <MenuButton
        as={IconButton}
        aria-label={t('accessibility.menu')}
        icon={<PiDotsThreeOutlineFill />}
        variant="link"
        data-testid="control-layers-add-layer-menu-button"
        alignSelf="stretch"
      />
      <MenuList>
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
      </MenuList>
    </Menu>
  );
});

CanvasEntityListMenu.displayName = 'CanvasEntityListMenu';
