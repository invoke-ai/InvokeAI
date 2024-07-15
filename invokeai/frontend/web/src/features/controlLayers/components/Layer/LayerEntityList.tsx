/* eslint-disable i18next/no-literal-string */
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { Layer } from 'features/controlLayers/components/Layer/Layer';
import { mapId } from 'features/controlLayers/konva/util';
import { selectCanvasV2Slice } from 'features/controlLayers/store/canvasV2Slice';
import { memo } from 'react';

const selectEntityIds = createMemoizedSelector(selectCanvasV2Slice, (canvasV2) => {
  return canvasV2.layers.entities.map(mapId).reverse();
});

export const LayerEntityList = memo(() => {
  const layerIds = useAppSelector(selectEntityIds);

  if (layerIds.length === 0) {
    return null;
  }

  if (layerIds.length > 0) {
    return (
      <>
        {layerIds.map((id) => (
          <Layer key={id} id={id} />
        ))}
      </>
    );
  }
});

LayerEntityList.displayName = 'LayerEntityList';
