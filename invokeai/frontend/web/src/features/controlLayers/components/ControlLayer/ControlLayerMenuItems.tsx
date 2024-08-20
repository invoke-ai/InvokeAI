import { MenuDivider } from '@invoke-ai/ui-library';
import { CanvasEntityMenuItemsArrange } from 'features/controlLayers/components/common/CanvasEntityMenuItemsArrange';
import { CanvasEntityMenuItemsDelete } from 'features/controlLayers/components/common/CanvasEntityMenuItemsDelete';
import { CanvasEntityMenuItemsFilter } from 'features/controlLayers/components/common/CanvasEntityMenuItemsFilter';
import { CanvasEntityMenuItemsReset } from 'features/controlLayers/components/common/CanvasEntityMenuItemsReset';
import { ControlLayerMenuItemsControlToRaster } from 'features/controlLayers/components/ControlLayer/ControlLayerMenuItemsControlToRaster';
import { ControlLayerMenuItemsTransparencyEffect } from 'features/controlLayers/components/ControlLayer/ControlLayerMenuItemsTransparencyEffect';
import { memo } from 'react';

export const ControlLayerMenuItems = memo(() => {
  return (
    <>
      <CanvasEntityMenuItemsFilter />
      <ControlLayerMenuItemsControlToRaster />
      <ControlLayerMenuItemsTransparencyEffect />
      <MenuDivider />
      <CanvasEntityMenuItemsArrange />
      <MenuDivider />
      <CanvasEntityMenuItemsReset />
      <CanvasEntityMenuItemsDelete />
    </>
  );
});

ControlLayerMenuItems.displayName = 'ControlLayerMenuItems';
