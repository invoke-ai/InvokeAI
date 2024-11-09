import { createSelector } from '@reduxjs/toolkit';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasEntityGroupList } from 'features/controlLayers/components/CanvasEntityList/CanvasEntityGroupList';
import { ControlLayer } from 'features/controlLayers/components/ControlLayer/ControlLayer';
import { selectCanvasSlice, selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { getEntityIdentifier } from 'features/controlLayers/store/types';
import { memo } from 'react';

const selectEntityIdentifiers = createMemoizedSelector(selectCanvasSlice, (canvas) => {
  return canvas.controlLayers.entities.map(getEntityIdentifier).toReversed();
});

const selectIsSelected = createSelector(selectSelectedEntityIdentifier, (selectedEntityIdentifier) => {
  return selectedEntityIdentifier?.type === 'control_layer';
});

export const ControlLayerEntityList = memo(() => {
  const isSelected = useAppSelector(selectIsSelected);
  const entityIdentifiers = useAppSelector(selectEntityIdentifiers);

  if (entityIdentifiers.length === 0) {
    return null;
  }

  if (entityIdentifiers.length > 0) {
    return (
      <CanvasEntityGroupList type="control_layer" isSelected={isSelected} entityIdentifiers={entityIdentifiers}>
        {entityIdentifiers.map((entityIdentifier) => (
          <ControlLayer key={entityIdentifier.id} id={entityIdentifier.id} />
        ))}
      </CanvasEntityGroupList>
    );
  }
});

ControlLayerEntityList.displayName = 'ControlLayerEntityList';
