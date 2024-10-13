import { MenuDivider } from '@invoke-ai/ui-library';
import { IconMenuItemGroup } from 'common/components/IconMenuItem';
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
      <IconMenuItemGroup>
        <CanvasEntityMenuItemsArrange />
        <CanvasEntityMenuItemsDuplicate />
        <CanvasEntityMenuItemsDelete asIcon />
      </IconMenuItemGroup>
      <MenuDivider />
      <CanvasEntityMenuItemsTransform />
      <CanvasEntityMenuItemsFilter />
      <ControlLayerMenuItemsConvertControlToRaster />
      <ControlLayerMenuItemsTransparencyEffect />
      <MenuDivider />
      <CanvasEntityMenuItemsCropToBbox />
      <CanvasEntityMenuItemsCopyToClipboard />
      <CanvasEntityMenuItemsSave />
    </>
  );
});

ControlLayerMenuItems.displayName = 'ControlLayerMenuItems';
