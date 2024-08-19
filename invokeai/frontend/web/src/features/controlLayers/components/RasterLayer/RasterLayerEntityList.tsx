import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasEntityGroupList } from 'features/controlLayers/components/common/CanvasEntityGroupList';
import { RasterLayer } from 'features/controlLayers/components/RasterLayer/RasterLayer';
import { mapId } from 'features/controlLayers/konva/util';
import { selectCanvasV2Slice } from 'features/controlLayers/store/canvasV2Slice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const selectEntityIds = createMemoizedSelector(selectCanvasV2Slice, (canvasV2) => {
  return canvasV2.rasterLayers.entities.map(mapId).reverse();
});

export const RasterLayerEntityList = memo(() => {
  const { t } = useTranslation();
  const isSelected = useAppSelector((s) => Boolean(s.canvasV2.selectedEntityIdentifier?.type === 'raster_layer'));
  const layerIds = useAppSelector(selectEntityIds);

  if (layerIds.length === 0) {
    return null;
  }

  if (layerIds.length > 0) {
    return (
      <CanvasEntityGroupList
        type="raster_layer"
        title={t('controlLayers.rasterLayers_withCount', { count: layerIds.length })}
        isSelected={isSelected}
      >
        {layerIds.map((id) => (
          <RasterLayer key={id} id={id} />
        ))}
      </CanvasEntityGroupList>
    );
  }
});

RasterLayerEntityList.displayName = 'RasterLayerEntityList';
