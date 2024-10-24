import { MenuDivider } from '@invoke-ai/ui-library';
import { IconMenuItemGroup } from 'common/components/IconMenuItem';
import { CanvasEntityMenuItemsArrange } from 'features/controlLayers/components/common/CanvasEntityMenuItemsArrange';
import { CanvasEntityMenuItemsCropToBbox } from 'features/controlLayers/components/common/CanvasEntityMenuItemsCropToBbox';
import { CanvasEntityMenuItemsDelete } from 'features/controlLayers/components/common/CanvasEntityMenuItemsDelete';
import { CanvasEntityMenuItemsDuplicate } from 'features/controlLayers/components/common/CanvasEntityMenuItemsDuplicate';
import { CanvasEntityMenuItemsFilter } from 'features/controlLayers/components/common/CanvasEntityMenuItemsFilter';
import { CanvasEntityMenuItemsSave } from 'features/controlLayers/components/common/CanvasEntityMenuItemsSave';
import { CanvasEntityMenuItemsSegment } from 'features/controlLayers/components/common/CanvasEntityMenuItemsSegment';
import { CanvasEntityMenuItemsTransform } from 'features/controlLayers/components/common/CanvasEntityMenuItemsTransform';
import { RasterLayerMenuItemsConvertToSubMenu } from 'features/controlLayers/components/RasterLayer/RasterLayerMenuItemsConvertToSubMenu';
import { RasterLayerMenuItemsCopyToSubMenu } from 'features/controlLayers/components/RasterLayer/RasterLayerMenuItemsCopyToSubMenu';
import { memo } from 'react';

export const RasterLayerMenuItems = memo(() => {
  return (
    <>
      <IconMenuItemGroup>
        <CanvasEntityMenuItemsArrange />
        <CanvasEntityMenuItemsDuplicate />
        <CanvasEntityMenuItemsDelete asIcon />
      </IconMenuItemGroup>
      <MenuDivider />
      <CanvasEntityMenuItemsTransform />
      <CanvasEntityMenuItemsFilter />
      <CanvasEntityMenuItemsSegment />
      <MenuDivider />
      <CanvasEntityMenuItemsCropToBbox />
      <CanvasEntityMenuItemsSave />
      <MenuDivider />
      <RasterLayerMenuItemsConvertToSubMenu />
      <RasterLayerMenuItemsCopyToSubMenu />
    </>
  );
});

RasterLayerMenuItems.displayName = 'RasterLayerMenuItems';
