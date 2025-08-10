import { MenuDivider, MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { IconMenuItemGroup } from 'common/components/IconMenuItem';
import { CanvasEntityMenuItemsArrange } from 'features/controlLayers/components/common/CanvasEntityMenuItemsArrange';
import { CanvasEntityMenuItemsCropToBbox } from 'features/controlLayers/components/common/CanvasEntityMenuItemsCropToBbox';
import { CanvasEntityMenuItemsDelete } from 'features/controlLayers/components/common/CanvasEntityMenuItemsDelete';
import { CanvasEntityMenuItemsDuplicate } from 'features/controlLayers/components/common/CanvasEntityMenuItemsDuplicate';
import { CanvasEntityMenuItemsFilter } from 'features/controlLayers/components/common/CanvasEntityMenuItemsFilter';
import { CanvasEntityMenuItemsMergeDown } from 'features/controlLayers/components/common/CanvasEntityMenuItemsMergeDown';
import { CanvasEntityMenuItemsSave } from 'features/controlLayers/components/common/CanvasEntityMenuItemsSave';
import { CanvasEntityMenuItemsSelectObject } from 'features/controlLayers/components/common/CanvasEntityMenuItemsSelectObject';
import { CanvasEntityMenuItemsTransform } from 'features/controlLayers/components/common/CanvasEntityMenuItemsTransform';
import { RasterLayerMenuItemsConvertToSubMenu } from 'features/controlLayers/components/RasterLayer/RasterLayerMenuItemsConvertToSubMenu';
import { RasterLayerMenuItemsCopyToSubMenu } from 'features/controlLayers/components/RasterLayer/RasterLayerMenuItemsCopyToSubMenu';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { rasterLayerAdjustmentsReset, rasterLayerAdjustmentsSet } from 'features/controlLayers/store/canvasSlice';
import type { CanvasRasterLayerState } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const RasterLayerMenuItems = memo(() => {
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext<'raster_layer'>();
  const { t } = useTranslation();
  const layer = useAppSelector((s) =>
    s.canvas.present.rasterLayers.entities.find((e: CanvasRasterLayerState) => e.id === entityIdentifier.id)
  );
  const hasAdjustments = Boolean(layer?.adjustments);
  const onToggleAdjustmentsPresence = useCallback(() => {
    if (hasAdjustments) {
      dispatch(rasterLayerAdjustmentsReset({ entityIdentifier }));
    } else {
      dispatch(
        rasterLayerAdjustmentsSet({
          entityIdentifier,
          adjustments: { enabled: true, collapsed: false, mode: 'simple' },
        })
      );
    }
  }, [dispatch, entityIdentifier, hasAdjustments]);

  return (
    <>
      <IconMenuItemGroup>
        <CanvasEntityMenuItemsArrange />
        <CanvasEntityMenuItemsDuplicate />
        <CanvasEntityMenuItemsDelete asIcon />
      </IconMenuItemGroup>
      <MenuDivider />
      <MenuItem onClick={onToggleAdjustmentsPresence}>
        {hasAdjustments ? t('controlLayers.removeAdjustments') : t('controlLayers.addAdjustments')}
      </MenuItem>
      <MenuDivider />
      <CanvasEntityMenuItemsTransform />
      <CanvasEntityMenuItemsFilter />
      <CanvasEntityMenuItemsSelectObject />
      <MenuDivider />
      <CanvasEntityMenuItemsMergeDown />
      <RasterLayerMenuItemsCopyToSubMenu />
      <RasterLayerMenuItemsConvertToSubMenu />
      <CanvasEntityMenuItemsCropToBbox />
      <CanvasEntityMenuItemsSave />
    </>
  );
});

RasterLayerMenuItems.displayName = 'RasterLayerMenuItems';
