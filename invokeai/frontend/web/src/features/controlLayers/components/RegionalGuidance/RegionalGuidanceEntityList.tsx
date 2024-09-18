import { createSelector } from '@reduxjs/toolkit';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasEntityGroupList } from 'features/controlLayers/components/common/CanvasEntityGroupList';
import { RegionalGuidance } from 'features/controlLayers/components/RegionalGuidance/RegionalGuidance';
import { mapId } from 'features/controlLayers/konva/util';
import { selectCanvasSlice, selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { memo } from 'react';

const selectEntityIds = createMemoizedSelector(selectCanvasSlice, (canvas) => {
  return canvas.regionalGuidance.entities.map(mapId).reverse();
});
const selectIsSelected = createSelector(selectSelectedEntityIdentifier, (selectedEntityIdentifier) => {
  return selectedEntityIdentifier?.type === 'regional_guidance';
});

export const RegionalGuidanceEntityList = memo(() => {
  const isSelected = useAppSelector(selectIsSelected);
  const rgIds = useAppSelector(selectEntityIds);

  if (rgIds.length === 0) {
    return null;
  }

  if (rgIds.length > 0) {
    return (
      <CanvasEntityGroupList type="regional_guidance" isSelected={isSelected}>
        {rgIds.map((id) => (
          <RegionalGuidance key={id} id={id} />
        ))}
      </CanvasEntityGroupList>
    );
  }
});

RegionalGuidanceEntityList.displayName = 'RegionalGuidanceEntityList';
