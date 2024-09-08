import { MenuDivider } from '@invoke-ai/ui-library';
import { CanvasEntityMenuItemsArrange } from 'features/controlLayers/components/common/CanvasEntityMenuItemsArrange';
import { CanvasEntityMenuItemsDelete } from 'features/controlLayers/components/common/CanvasEntityMenuItemsDelete';
import { CanvasEntityMenuItemsDuplicate } from 'features/controlLayers/components/common/CanvasEntityMenuItemsDuplicate';
import { CanvasEntityMenuItemsTransform } from 'features/controlLayers/components/common/CanvasEntityMenuItemsTransform';
import { RegionalGuidanceMenuItemsAddPromptsAndIPAdapter } from 'features/controlLayers/components/RegionalGuidance/RegionalGuidanceMenuItemsAddPromptsAndIPAdapter';
import { RegionalGuidanceMenuItemsAutoNegative } from 'features/controlLayers/components/RegionalGuidance/RegionalGuidanceMenuItemsAutoNegative';
import { memo } from 'react';

export const RegionalGuidanceMenuItems = memo(() => {
  return (
    <>
      <RegionalGuidanceMenuItemsAddPromptsAndIPAdapter />
      <MenuDivider />
      <CanvasEntityMenuItemsTransform />
      <RegionalGuidanceMenuItemsAutoNegative />
      <MenuDivider />
      <CanvasEntityMenuItemsArrange />
      <MenuDivider />
      <CanvasEntityMenuItemsDuplicate />
      <CanvasEntityMenuItemsDelete />
    </>
  );
});

RegionalGuidanceMenuItems.displayName = 'RegionalGuidanceMenuItems';
