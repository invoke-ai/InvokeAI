import { createSelector } from '@reduxjs/toolkit';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasEntityGroupList } from 'features/controlLayers/components/common/CanvasEntityGroupList';
import { InpaintMask } from 'features/controlLayers/components/InpaintMask/InpaintMask';
import { mapId } from 'features/controlLayers/konva/util';
import { selectCanvasSlice, selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { memo } from 'react';

const selectEntityIds = createMemoizedSelector(selectCanvasSlice, (canvas) => {
  return canvas.inpaintMasks.entities.map(mapId).reverse();
});

const selectIsSelected = createSelector(selectSelectedEntityIdentifier, (selectedEntityIdentifier) => {
  return selectedEntityIdentifier?.type === 'inpaint_mask';
});

export const InpaintMaskList = memo(() => {
  const isSelected = useAppSelector(selectIsSelected);
  const entityIds = useAppSelector(selectEntityIds);

  if (entityIds.length === 0) {
    return null;
  }

  if (entityIds.length > 0) {
    return (
      <CanvasEntityGroupList type="inpaint_mask" isSelected={isSelected}>
        {entityIds.map((id) => (
          <InpaintMask key={id} id={id} />
        ))}
      </CanvasEntityGroupList>
    );
  }
});

InpaintMaskList.displayName = 'InpaintMaskList';
