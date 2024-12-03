/* eslint-disable i18next/no-literal-string */
import { createSelector } from '@reduxjs/toolkit';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasEntityGroupList } from 'features/controlLayers/components/CanvasEntityList/CanvasEntityGroupList';
import { IPAdapter } from 'features/controlLayers/components/IPAdapter/IPAdapter';
import { selectCanvasSlice, selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { getEntityIdentifier } from 'features/controlLayers/store/types';
import { memo } from 'react';

const selectEntityIdentifiers = createMemoizedSelector(selectCanvasSlice, (canvas) => {
  return canvas.referenceImages.entities.map(getEntityIdentifier).toReversed();
});
const selectIsSelected = createSelector(selectSelectedEntityIdentifier, (selectedEntityIdentifier) => {
  return selectedEntityIdentifier?.type === 'reference_image';
});

export const IPAdapterList = memo(() => {
  const isSelected = useAppSelector(selectIsSelected);
  const entityIdentifiers = useAppSelector(selectEntityIdentifiers);

  if (entityIdentifiers.length === 0) {
    return null;
  }

  if (entityIdentifiers.length > 0) {
    return (
      <CanvasEntityGroupList type="reference_image" isSelected={isSelected} entityIdentifiers={entityIdentifiers}>
        {entityIdentifiers.map((entityIdentifiers) => (
          <IPAdapter key={entityIdentifiers.id} id={entityIdentifiers.id} />
        ))}
      </CanvasEntityGroupList>
    );
  }
});

IPAdapterList.displayName = 'IPAdapterList';
