import { createSelector } from '@reduxjs/toolkit';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasEntityGroupList } from 'features/controlLayers/components/CanvasEntityList/CanvasEntityGroupList';
import { InpaintMask } from 'features/controlLayers/components/InpaintMask/InpaintMask';
import { selectCanvasSlice, selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { getEntityIdentifier } from 'features/controlLayers/store/types';
import { memo } from 'react';

const selectEntityIdentifiers = createMemoizedSelector(selectCanvasSlice, (canvas) => {
  return canvas.inpaintMasks.entities.map(getEntityIdentifier).toReversed();
});

const selectIsSelected = createSelector(selectSelectedEntityIdentifier, (selectedEntityIdentifier) => {
  return selectedEntityIdentifier?.type === 'inpaint_mask';
});

export const InpaintMaskList = memo(() => {
  const isSelected = useAppSelector(selectIsSelected);
  const entityIdentifiers = useAppSelector(selectEntityIdentifiers);

  if (entityIdentifiers.length === 0) {
    return null;
  }

  if (entityIdentifiers.length > 0) {
    return (
      <CanvasEntityGroupList type="inpaint_mask" isSelected={isSelected} entityIdentifiers={entityIdentifiers}>
        {entityIdentifiers.map((entityIdentifier) => (
          <InpaintMask key={entityIdentifier.id} id={entityIdentifier.id} />
        ))}
      </CanvasEntityGroupList>
    );
  }
});

InpaintMaskList.displayName = 'InpaintMaskList';
