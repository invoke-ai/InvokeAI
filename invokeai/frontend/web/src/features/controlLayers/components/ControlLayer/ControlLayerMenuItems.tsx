import { MenuDivider } from '@invoke-ai/ui-library';
import { CanvasEntityMenuItemsArrange } from 'features/controlLayers/components/common/CanvasEntityMenuItemsArrange';
import { CanvasEntityMenuItemsDelete } from 'features/controlLayers/components/common/CanvasEntityMenuItemsDelete';
import { CanvasEntityMenuItemsDuplicate } from 'features/controlLayers/components/common/CanvasEntityMenuItemsDuplicate';
import { CanvasEntityMenuItemsFilter } from 'features/controlLayers/components/common/CanvasEntityMenuItemsFilter';
import { CanvasEntityMenuItemsTransform } from 'features/controlLayers/components/common/CanvasEntityMenuItemsTransform';
import { ControlLayerMenuItemsControlToRaster } from 'features/controlLayers/components/ControlLayer/ControlLayerMenuItemsControlToRaster';
import { ControlLayerMenuItemsTransparencyEffect } from 'features/controlLayers/components/ControlLayer/ControlLayerMenuItemsTransparencyEffect';
import { memo } from 'react';

export const ControlLayerMenuItems = memo(() => {
  return (
    <>
      <CanvasEntityMenuItemsTransform />
      <CanvasEntityMenuItemsFilter />
      <ControlLayerMenuItemsControlToRaster />
      <ControlLayerMenuItemsTransparencyEffect />
      <MenuDivider />
      <CanvasEntityMenuItemsArrange />
      <MenuDivider />
      <CanvasEntityMenuItemsDuplicate />
      <CanvasEntityMenuItemsDelete />
    </>
  );
});

ControlLayerMenuItems.displayName = 'ControlLayerMenuItems';
