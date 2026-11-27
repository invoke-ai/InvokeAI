import { createSelector } from '@reduxjs/toolkit';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasEntityGroupList } from 'features/controlLayers/components/CanvasEntityList/CanvasEntityGroupList';
import { RasterLayer } from 'features/controlLayers/components/RasterLayer/RasterLayer';
import { selectCanvasSlice, selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { getEntityIdentifier } from 'features/controlLayers/store/types';
import { memo } from 'react';

const selectEntityIdentifiers = createMemoizedSelector(selectCanvasSlice, (canvas) => {
  return canvas.rasterLayers.entities.map(getEntityIdentifier).toReversed();
});
const selectIsSelected = createSelector(selectSelectedEntityIdentifier, (selectedEntityIdentifier) => {
  return selectedEntityIdentifier?.type === 'raster_layer';
});

export const RasterLayerEntityList = memo(() => {
  const isSelected = useAppSelector(selectIsSelected);
  const entityIdentifiers = useAppSelector(selectEntityIdentifiers);

  if (entityIdentifiers.length === 0) {
    return null;
  }

  if (entityIdentifiers.length > 0) {
    return (
      <CanvasEntityGroupList type="raster_layer" isSelected={isSelected} entityIdentifiers={entityIdentifiers}>
        {entityIdentifiers.map((entityIdentifier) => (
          <RasterLayer key={entityIdentifier.id} id={entityIdentifier.id} />
        ))}
      </CanvasEntityGroupList>
    );
  }
});

RasterLayerEntityList.displayName = 'RasterLayerEntityList';
