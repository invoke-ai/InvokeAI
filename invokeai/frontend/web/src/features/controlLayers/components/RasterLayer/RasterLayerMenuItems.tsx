import { MenuDivider } from '@invoke-ai/ui-library';
import { CanvasEntityMenuItemsArrange } from 'features/controlLayers/components/common/CanvasEntityMenuItemsArrange';
import { CanvasEntityMenuItemsDelete } from 'features/controlLayers/components/common/CanvasEntityMenuItemsDelete';
import { CanvasEntityMenuItemsDuplicate } from 'features/controlLayers/components/common/CanvasEntityMenuItemsDuplicate';
import { CanvasEntityMenuItemsFilter } from 'features/controlLayers/components/common/CanvasEntityMenuItemsFilter';
import { CanvasEntityMenuItemsTransform } from 'features/controlLayers/components/common/CanvasEntityMenuItemsTransform';
import { RasterLayerMenuItemsRasterToControl } from 'features/controlLayers/components/RasterLayer/RasterLayerMenuItemsRasterToControl';
import { memo } from 'react';

export const RasterLayerMenuItems = memo(() => {
  return (
    <>
      <CanvasEntityMenuItemsTransform />
      <CanvasEntityMenuItemsFilter />
      <RasterLayerMenuItemsRasterToControl />
      <MenuDivider />
      <CanvasEntityMenuItemsArrange />
      <MenuDivider />
      <CanvasEntityMenuItemsDuplicate />
      <CanvasEntityMenuItemsDelete />
    </>
  );
});

RasterLayerMenuItems.displayName = 'RasterLayerMenuItems';
