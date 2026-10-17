import { createSelector } from '@reduxjs/toolkit';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasEntityGroupList } from 'features/controlLayers/components/CanvasEntityList/CanvasEntityGroupList';
import { VectorLayer } from 'features/controlLayers/components/VectorLayer/VectorLayer';
import { selectCanvasSlice, selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { getEntityIdentifier } from 'features/controlLayers/store/types';
import { memo } from 'react';

const selectEntityIdentifiers = createMemoizedSelector(selectCanvasSlice, (canvas) => {
  return canvas.vectorLayers.entities.map(getEntityIdentifier).toReversed();
});
const selectIsSelected = createSelector(selectSelectedEntityIdentifier, (selectedEntityIdentifier) => {
  return selectedEntityIdentifier?.type === 'vector_layer';
});

export const VectorLayerEntityList = memo(() => {
  const isSelected = useAppSelector(selectIsSelected);
  const entityIdentifiers = useAppSelector(selectEntityIdentifiers);

  if (entityIdentifiers.length === 0) {
    return null;
  }

  return (
    <CanvasEntityGroupList type="vector_layer" isSelected={isSelected} entityIdentifiers={entityIdentifiers}>
      {entityIdentifiers.map((entityIdentifier) => (
        <VectorLayer key={entityIdentifier.id} id={entityIdentifier.id} />
      ))}
    </CanvasEntityGroupList>
  );
});

VectorLayerEntityList.displayName = 'VectorLayerEntityList';
