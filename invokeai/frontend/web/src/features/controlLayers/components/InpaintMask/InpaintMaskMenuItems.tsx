import { MenuDivider } from '@invoke-ai/ui-library';
import { CanvasEntityMenuItemsArrange } from 'features/controlLayers/components/common/CanvasEntityMenuItemsArrange';
import { CanvasEntityMenuItemsDelete } from 'features/controlLayers/components/common/CanvasEntityMenuItemsDelete';
import { CanvasEntityMenuItemsDuplicate } from 'features/controlLayers/components/common/CanvasEntityMenuItemsDuplicate';
import { CanvasEntityMenuItemsTransform } from 'features/controlLayers/components/common/CanvasEntityMenuItemsTransform';
import { memo } from 'react';

export const InpaintMaskMenuItems = memo(() => {
  return (
    <>
      <CanvasEntityMenuItemsTransform />
      <MenuDivider />
      <CanvasEntityMenuItemsArrange />
      <MenuDivider />
      <CanvasEntityMenuItemsDuplicate />
      <CanvasEntityMenuItemsDelete />
    </>
  );
});

InpaintMaskMenuItems.displayName = 'InpaintMaskMenuItems';
