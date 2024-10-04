import { MenuDivider } from '@invoke-ai/ui-library';
import { CanvasEntityMenuItemsArrange } from 'features/controlLayers/components/common/CanvasEntityMenuItemsArrange';
import { CanvasEntityMenuItemsCopyToClipboard } from 'features/controlLayers/components/common/CanvasEntityMenuItemsCopyToClipboard';
import { CanvasEntityMenuItemsCropToBbox } from 'features/controlLayers/components/common/CanvasEntityMenuItemsCropToBbox';
import { CanvasEntityMenuItemsDelete } from 'features/controlLayers/components/common/CanvasEntityMenuItemsDelete';
import { CanvasEntityMenuItemsDuplicate } from 'features/controlLayers/components/common/CanvasEntityMenuItemsDuplicate';
import { CanvasEntityMenuItemsFilter } from 'features/controlLayers/components/common/CanvasEntityMenuItemsFilter';
import { CanvasEntityMenuItemsSave } from 'features/controlLayers/components/common/CanvasEntityMenuItemsSave';
import { CanvasEntityMenuItemsTransform } from 'features/controlLayers/components/common/CanvasEntityMenuItemsTransform';
import { ControlLayerMenuItemsConvertControlToRaster } from 'features/controlLayers/components/ControlLayer/ControlLayerMenuItemsConvertControlToRaster';
import { ControlLayerMenuItemsTransparencyEffect } from 'features/controlLayers/components/ControlLayer/ControlLayerMenuItemsTransparencyEffect';
import { memo } from 'react';

export const ControlLayerMenuItems = memo(() => {
  return (
    <>
      <CanvasEntityMenuItemsTransform />
      <CanvasEntityMenuItemsFilter />
      <ControlLayerMenuItemsConvertControlToRaster />
      <ControlLayerMenuItemsTransparencyEffect />
      <MenuDivider />
      <CanvasEntityMenuItemsArrange />
      <MenuDivider />
      <CanvasEntityMenuItemsCropToBbox />
      <CanvasEntityMenuItemsDuplicate />
      <CanvasEntityMenuItemsCopyToClipboard />
      <CanvasEntityMenuItemsSave />
      <CanvasEntityMenuItemsDelete />
    </>
  );
});

ControlLayerMenuItems.displayName = 'ControlLayerMenuItems';
