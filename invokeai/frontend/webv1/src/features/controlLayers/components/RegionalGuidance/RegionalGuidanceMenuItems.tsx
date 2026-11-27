import { MenuDivider } from '@invoke-ai/ui-library';
import { IconMenuItemGroup } from 'common/components/IconMenuItem';
import { CanvasEntityMenuItemsArrange } from 'features/controlLayers/components/common/CanvasEntityMenuItemsArrange';
import { CanvasEntityMenuItemsCropToBbox } from 'features/controlLayers/components/common/CanvasEntityMenuItemsCropToBbox';
import { CanvasEntityMenuItemsDelete } from 'features/controlLayers/components/common/CanvasEntityMenuItemsDelete';
import { CanvasEntityMenuItemsDuplicate } from 'features/controlLayers/components/common/CanvasEntityMenuItemsDuplicate';
import { CanvasEntityMenuItemsMergeDown } from 'features/controlLayers/components/common/CanvasEntityMenuItemsMergeDown';
import { CanvasEntityMenuItemsSave } from 'features/controlLayers/components/common/CanvasEntityMenuItemsSave';
import { CanvasEntityMenuItemsTransform } from 'features/controlLayers/components/common/CanvasEntityMenuItemsTransform';
import { RegionalGuidanceMenuItemsAddPromptsAndIPAdapter } from 'features/controlLayers/components/RegionalGuidance/RegionalGuidanceMenuItemsAddPromptsAndIPAdapter';
import { RegionalGuidanceMenuItemsAutoNegative } from 'features/controlLayers/components/RegionalGuidance/RegionalGuidanceMenuItemsAutoNegative';
import { RegionalGuidanceMenuItemsConvertToSubMenu } from 'features/controlLayers/components/RegionalGuidance/RegionalGuidanceMenuItemsConvertToSubMenu';
import { RegionalGuidanceMenuItemsCopyToSubMenu } from 'features/controlLayers/components/RegionalGuidance/RegionalGuidanceMenuItemsCopyToSubMenu';
import { memo } from 'react';

export const RegionalGuidanceMenuItems = memo(() => {
  return (
    <>
      <IconMenuItemGroup>
        <CanvasEntityMenuItemsArrange />
        <CanvasEntityMenuItemsDuplicate />
        <CanvasEntityMenuItemsDelete asIcon />
      </IconMenuItemGroup>
      <MenuDivider />
      <RegionalGuidanceMenuItemsAddPromptsAndIPAdapter />
      <MenuDivider />
      <CanvasEntityMenuItemsTransform />
      <RegionalGuidanceMenuItemsAutoNegative />
      <MenuDivider />
      <CanvasEntityMenuItemsMergeDown />
      <RegionalGuidanceMenuItemsCopyToSubMenu />
      <RegionalGuidanceMenuItemsConvertToSubMenu />
      <CanvasEntityMenuItemsCropToBbox />
      <CanvasEntityMenuItemsSave />
    </>
  );
});

RegionalGuidanceMenuItems.displayName = 'RegionalGuidanceMenuItems';
