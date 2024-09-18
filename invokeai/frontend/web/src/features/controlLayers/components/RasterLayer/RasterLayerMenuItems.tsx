import { MenuDivider } from '@invoke-ai/ui-library';
import { CanvasEntityMenuItemsArrange } from 'features/controlLayers/components/common/CanvasEntityMenuItemsArrange';
import { CanvasEntityMenuItemsCopyToClipboard } from 'features/controlLayers/components/common/CanvasEntityMenuItemsCopyToClipboard';
import { CanvasEntityMenuItemsDelete } from 'features/controlLayers/components/common/CanvasEntityMenuItemsDelete';
import { CanvasEntityMenuItemsDuplicate } from 'features/controlLayers/components/common/CanvasEntityMenuItemsDuplicate';
import { CanvasEntityMenuItemsFilter } from 'features/controlLayers/components/common/CanvasEntityMenuItemsFilter';
import { CanvasEntityMenuItemsSave } from 'features/controlLayers/components/common/CanvasEntityMenuItemsSave';
import { CanvasEntityMenuItemsTransform } from 'features/controlLayers/components/common/CanvasEntityMenuItemsTransform';
import { RasterLayerMenuItemsConvertRasterToControl } from 'features/controlLayers/components/RasterLayer/RasterLayerMenuItemsConvertRasterToControl';
import { memo } from 'react';

export const RasterLayerMenuItems = memo(() => {
  return (
    <>
      <CanvasEntityMenuItemsTransform />
      <CanvasEntityMenuItemsFilter />
      <RasterLayerMenuItemsConvertRasterToControl />
      <MenuDivider />
      <CanvasEntityMenuItemsArrange />
      <MenuDivider />
      <CanvasEntityMenuItemsDuplicate />
      <CanvasEntityMenuItemsCopyToClipboard />
      <CanvasEntityMenuItemsSave />
      <CanvasEntityMenuItemsDelete />
    </>
  );
});

RasterLayerMenuItems.displayName = 'RasterLayerMenuItems';
