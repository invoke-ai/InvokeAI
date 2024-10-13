import { IconMenuItemGroup } from 'common/components/IconMenuItem';
import { CanvasEntityMenuItemsArrange } from 'features/controlLayers/components/common/CanvasEntityMenuItemsArrange';
import { CanvasEntityMenuItemsDelete } from 'features/controlLayers/components/common/CanvasEntityMenuItemsDelete';
import { CanvasEntityMenuItemsDuplicate } from 'features/controlLayers/components/common/CanvasEntityMenuItemsDuplicate';
import { memo } from 'react';

export const IPAdapterMenuItems = memo(() => {
  return (
    <IconMenuItemGroup>
      <CanvasEntityMenuItemsArrange />
      <CanvasEntityMenuItemsDuplicate />
      <CanvasEntityMenuItemsDelete asIcon />
    </IconMenuItemGroup>
  );
});

IPAdapterMenuItems.displayName = 'IPAdapterMenuItems';
