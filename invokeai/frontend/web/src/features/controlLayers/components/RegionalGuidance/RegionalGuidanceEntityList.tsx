import { createSelector } from '@reduxjs/toolkit';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasEntityGroupList } from 'features/controlLayers/components/CanvasEntityList/CanvasEntityGroupList';
import { RegionalGuidance } from 'features/controlLayers/components/RegionalGuidance/RegionalGuidance';
import { selectCanvasSlice, selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { getEntityIdentifier } from 'features/controlLayers/store/types';
import { memo } from 'react';

const selectEntityIdentifiers = createMemoizedSelector(selectCanvasSlice, (canvas) => {
  return canvas.regionalGuidance.entities.map(getEntityIdentifier).toReversed();
});
const selectIsSelected = createSelector(selectSelectedEntityIdentifier, (selectedEntityIdentifier) => {
  return selectedEntityIdentifier?.type === 'regional_guidance';
});

export const RegionalGuidanceEntityList = memo(() => {
  const isSelected = useAppSelector(selectIsSelected);
  const entityIdentifiers = useAppSelector(selectEntityIdentifiers);

  if (entityIdentifiers.length === 0) {
    return null;
  }

  if (entityIdentifiers.length > 0) {
    return (
      <CanvasEntityGroupList type="regional_guidance" isSelected={isSelected} entityIdentifiers={entityIdentifiers}>
        {entityIdentifiers.map((entityIdentifier) => (
          <RegionalGuidance key={entityIdentifier.id} id={entityIdentifier.id} />
        ))}
      </CanvasEntityGroupList>
    );
  }
});

RegionalGuidanceEntityList.displayName = 'RegionalGuidanceEntityList';
