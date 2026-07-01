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
import { ControlLayerMenuItemsConvertToSubMenu } from 'features/controlLayers/components/ControlLayer/ControlLayerMenuItemsConvertToSubMenu';
import { ControlLayerMenuItemsCopyToSubMenu } from 'features/controlLayers/components/ControlLayer/ControlLayerMenuItemsCopyToSubMenu';
import { ControlLayerMenuItemsTransparencyEffect } from 'features/controlLayers/components/ControlLayer/ControlLayerMenuItemsTransparencyEffect';
import { memo } from 'react';

export const ControlLayerMenuItems = memo(() => {
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
      <ControlLayerMenuItemsTransparencyEffect />
      <MenuDivider />
      <CanvasEntityMenuItemsMergeDown />
      <ControlLayerMenuItemsCopyToSubMenu />
      <ControlLayerMenuItemsConvertToSubMenu />
      <CanvasEntityMenuItemsCropToBbox />
      <CanvasEntityMenuItemsSave />
    </>
  );
});

ControlLayerMenuItems.displayName = 'ControlLayerMenuItems';
