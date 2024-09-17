import { MenuDivider } from '@invoke-ai/ui-library';
import { CanvasEntityMenuItemsArrange } from 'features/controlLayers/components/common/CanvasEntityMenuItemsArrange';
import { CanvasEntityMenuItemsCopyToClipboard } from 'features/controlLayers/components/common/CanvasEntityMenuItemsCopyToClipboard';
import { CanvasEntityMenuItemsDelete } from 'features/controlLayers/components/common/CanvasEntityMenuItemsDelete';
import { CanvasEntityMenuItemsDuplicate } from 'features/controlLayers/components/common/CanvasEntityMenuItemsDuplicate';
import { CanvasEntityMenuItemsSave } from 'features/controlLayers/components/common/CanvasEntityMenuItemsSave';
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
      <CanvasEntityMenuItemsCopyToClipboard />
      <CanvasEntityMenuItemsSave />
      <CanvasEntityMenuItemsDelete />
    </>
  );
});

RegionalGuidanceMenuItems.displayName = 'RegionalGuidanceMenuItems';
