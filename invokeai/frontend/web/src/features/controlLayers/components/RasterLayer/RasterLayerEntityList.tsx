import { createSelector } from '@reduxjs/toolkit';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasEntityGroupList } from 'features/controlLayers/components/common/CanvasEntityGroupList';
import { RasterLayer } from 'features/controlLayers/components/RasterLayer/RasterLayer';
import { mapId } from 'features/controlLayers/konva/util';
import { selectCanvasSlice, selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { memo } from 'react';

const selectEntityIds = createMemoizedSelector(selectCanvasSlice, (canvas) => {
  return canvas.rasterLayers.entities.map(mapId).reverse();
});
const selectIsSelected = createSelector(selectSelectedEntityIdentifier, (selectedEntityIdentifier) => {
  return selectedEntityIdentifier?.type === 'raster_layer';
});

export const RasterLayerEntityList = memo(() => {
  const isSelected = useAppSelector(selectIsSelected);
  const layerIds = useAppSelector(selectEntityIds);

  if (layerIds.length === 0) {
    return null;
  }

  if (layerIds.length > 0) {
    return (
      <CanvasEntityGroupList type="raster_layer" isSelected={isSelected}>
        {layerIds.map((id) => (
          <RasterLayer key={id} id={id} />
        ))}
      </CanvasEntityGroupList>
    );
  }
});

RasterLayerEntityList.displayName = 'RasterLayerEntityList';
