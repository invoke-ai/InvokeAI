import { MenuDivider } from '@invoke-ai/ui-library';
import { CanvasEntityMenuItemsArrange } from 'features/controlLayers/components/common/CanvasEntityMenuItemsArrange';
import { CanvasEntityMenuItemsDelete } from 'features/controlLayers/components/common/CanvasEntityMenuItemsDelete';
import { CanvasEntityMenuItemsFilter } from 'features/controlLayers/components/common/CanvasEntityMenuItemsFilter';
import { CanvasEntityMenuItemsReset } from 'features/controlLayers/components/common/CanvasEntityMenuItemsReset';
import { CanvasEntityMenuItemsTransform } from 'features/controlLayers/components/common/CanvasEntityMenuItemsTransform';
import { RasterLayerMenuItemsRasterToControl } from 'features/controlLayers/components/RasterLayer/RasterLayerMenuItemsRasterToControl';
import { memo } from 'react';

export const RasterLayerMenuItems = memo(() => {
  return (
    <>
      <CanvasEntityMenuItemsFilter />
      <RasterLayerMenuItemsRasterToControl />
      <CanvasEntityMenuItemsTransform />
      <MenuDivider />
      <CanvasEntityMenuItemsArrange />
      <MenuDivider />
      <CanvasEntityMenuItemsReset />
      <CanvasEntityMenuItemsDelete />
    </>
  );
});

RasterLayerMenuItems.displayName = 'RasterLayerMenuItems';
