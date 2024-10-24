import { MenuDivider } from '@invoke-ai/ui-library';
import { IconMenuItemGroup } from 'common/components/IconMenuItem';
import { CanvasEntityMenuItemsArrange } from 'features/controlLayers/components/common/CanvasEntityMenuItemsArrange';
import { CanvasEntityMenuItemsCropToBbox } from 'features/controlLayers/components/common/CanvasEntityMenuItemsCropToBbox';
import { CanvasEntityMenuItemsDelete } from 'features/controlLayers/components/common/CanvasEntityMenuItemsDelete';
import { CanvasEntityMenuItemsDuplicate } from 'features/controlLayers/components/common/CanvasEntityMenuItemsDuplicate';
import { CanvasEntityMenuItemsTransform } from 'features/controlLayers/components/common/CanvasEntityMenuItemsTransform';
import { InpaintMaskMenuItemsConvertToSubMenu } from 'features/controlLayers/components/InpaintMask/InpaintMaskMenuItemsConvertToSubMenu';
import { InpaintMaskMenuItemsCopyToSubMenu } from 'features/controlLayers/components/InpaintMask/InpaintMaskMenuItemsCopyToSubMenu';
import { memo } from 'react';

export const InpaintMaskMenuItems = memo(() => {
  return (
    <>
      <IconMenuItemGroup>
        <CanvasEntityMenuItemsArrange />
        <CanvasEntityMenuItemsDuplicate />
        <CanvasEntityMenuItemsDelete asIcon />
      </IconMenuItemGroup>
      <MenuDivider />
      <CanvasEntityMenuItemsTransform />
      <MenuDivider />
      <CanvasEntityMenuItemsCropToBbox />
      <MenuDivider />
      <InpaintMaskMenuItemsConvertToSubMenu />
      <InpaintMaskMenuItemsCopyToSubMenu />
    </>
  );
});

InpaintMaskMenuItems.displayName = 'InpaintMaskMenuItems';
