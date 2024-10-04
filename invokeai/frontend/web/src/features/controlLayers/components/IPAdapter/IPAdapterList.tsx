/* eslint-disable i18next/no-literal-string */
import { createSelector } from '@reduxjs/toolkit';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasEntityGroupList } from 'features/controlLayers/components/common/CanvasEntityGroupList';
import { IPAdapter } from 'features/controlLayers/components/IPAdapter/IPAdapter';
import { mapId } from 'features/controlLayers/konva/util';
import { selectCanvasSlice, selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { memo } from 'react';

const selectEntityIds = createMemoizedSelector(selectCanvasSlice, (canvas) => {
  return canvas.referenceImages.entities.map(mapId).reverse();
});
const selectIsSelected = createSelector(selectSelectedEntityIdentifier, (selectedEntityIdentifier) => {
  return selectedEntityIdentifier?.type === 'reference_image';
});

export const IPAdapterList = memo(() => {
  const isSelected = useAppSelector(selectIsSelected);
  const ipaIds = useAppSelector(selectEntityIds);

  if (ipaIds.length === 0) {
    return null;
  }

  if (ipaIds.length > 0) {
    return (
      <CanvasEntityGroupList type="reference_image" isSelected={isSelected}>
        {ipaIds.map((id) => (
          <IPAdapter key={id} id={id} />
        ))}
      </CanvasEntityGroupList>
    );
  }
});

IPAdapterList.displayName = 'IPAdapterList';
