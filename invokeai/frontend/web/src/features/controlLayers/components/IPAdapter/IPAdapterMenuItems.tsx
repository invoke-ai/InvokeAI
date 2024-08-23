import { MenuDivider } from '@invoke-ai/ui-library';
import { CanvasEntityMenuItemsArrange } from 'features/controlLayers/components/common/CanvasEntityMenuItemsArrange';
import { CanvasEntityMenuItemsDelete } from 'features/controlLayers/components/common/CanvasEntityMenuItemsDelete';
import { memo } from 'react';

export const IPAdapterMenuItems = memo(() => {
  return (
    <>
      <CanvasEntityMenuItemsArrange />
      <MenuDivider />
      <CanvasEntityMenuItemsDelete />
    </>
  );
});

IPAdapterMenuItems.displayName = 'IPAdapterMenuItems';
