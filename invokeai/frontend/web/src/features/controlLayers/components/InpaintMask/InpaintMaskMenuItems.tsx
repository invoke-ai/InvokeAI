import { CanvasEntityMenuItemsReset } from 'features/controlLayers/components/common/CanvasEntityMenuItemsReset';
import { memo } from 'react';

export const InpaintMaskMenuItems = memo(() => {
  return (
    <>
      <CanvasEntityMenuItemsReset />
    </>
  );
});

InpaintMaskMenuItems.displayName = 'InpaintMaskMenuItems';
