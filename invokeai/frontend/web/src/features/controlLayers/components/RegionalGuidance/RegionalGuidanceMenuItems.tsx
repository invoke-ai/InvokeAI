import { MenuDivider } from '@invoke-ai/ui-library';
import { CanvasEntityMenuItemsArrange } from 'features/controlLayers/components/common/CanvasEntityMenuItemsArrange';
import { CanvasEntityMenuItemsDelete } from 'features/controlLayers/components/common/CanvasEntityMenuItemsDelete';
import { CanvasEntityMenuItemsReset } from 'features/controlLayers/components/common/CanvasEntityMenuItemsReset';
import { RegionalGuidanceMenuItemsAddPromptsAndIPAdapter } from 'features/controlLayers/components/RegionalGuidance/RegionalGuidanceMenuItemsAddPromptsAndIPAdapter';
import { memo } from 'react';

export const RegionalGuidanceMenuItems = memo(() => {
  return (
    <>
      <RegionalGuidanceMenuItemsAddPromptsAndIPAdapter />
      <MenuDivider />
      <CanvasEntityMenuItemsArrange />
      <MenuDivider />
      <CanvasEntityMenuItemsReset />
      <CanvasEntityMenuItemsDelete />
    </>
  );
});

RegionalGuidanceMenuItems.displayName = 'RegionalGuidanceMenuItems';
