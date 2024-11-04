import { MenuDivider } from '@invoke-ai/ui-library';
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
