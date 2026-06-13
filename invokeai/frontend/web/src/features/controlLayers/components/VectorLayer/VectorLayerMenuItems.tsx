import { IconMenuItemGroup } from 'common/components/IconMenuItem';
import { CanvasEntityMenuItemsArrange } from 'features/controlLayers/components/common/CanvasEntityMenuItemsArrange';
import { CanvasEntityMenuItemsDelete } from 'features/controlLayers/components/common/CanvasEntityMenuItemsDelete';
import { CanvasEntityMenuItemsDuplicate } from 'features/controlLayers/components/common/CanvasEntityMenuItemsDuplicate';
import { CanvasEntityMenuItemsMergeDown } from 'features/controlLayers/components/common/CanvasEntityMenuItemsMergeDown';
import { VectorLayerMenuItemsEdit } from 'features/controlLayers/components/VectorLayer/VectorLayerMenuItemsEdit';
import { VectorLayerMenuItemsTraceAll } from 'features/controlLayers/components/VectorLayer/VectorLayerMenuItemsTraceAll';
import { memo } from 'react';

export const VectorLayerMenuItems = memo(() => {
  return (
    <>
      <VectorLayerMenuItemsEdit />
      <VectorLayerMenuItemsTraceAll />
      <CanvasEntityMenuItemsMergeDown />
      <IconMenuItemGroup>
        <CanvasEntityMenuItemsArrange />
        <CanvasEntityMenuItemsDuplicate />
        <CanvasEntityMenuItemsDelete asIcon />
      </IconMenuItemGroup>
    </>
  );
});

VectorLayerMenuItems.displayName = 'VectorLayerMenuItems';
