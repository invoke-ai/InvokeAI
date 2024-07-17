import { Text } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { Layer } from 'features/controlLayers/components/Layer/Layer';
import { mapId } from 'features/controlLayers/konva/util';
import { selectCanvasV2Slice } from 'features/controlLayers/store/canvasV2Slice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const selectEntityIds = createMemoizedSelector(selectCanvasV2Slice, (canvasV2) => {
  return canvasV2.layers.entities.map(mapId).reverse();
});

export const LayerEntityList = memo(() => {
  const { t } = useTranslation();
  const isTypeSelected = useAppSelector((s) => Boolean(s.canvasV2.selectedEntityIdentifier?.type === 'layer'));
  const layerIds = useAppSelector(selectEntityIds);

  if (layerIds.length === 0) {
    return null;
  }

  if (layerIds.length > 0) {
    return (
      <>
        <Text
          color={isTypeSelected ? 'base.100' : 'base.300'}
          fontWeight={isTypeSelected ? 'semibold' : 'normal'}
          userSelect="none"
        >
          {t('controlLayers.layers_withCount', { count: layerIds.length })}
        </Text>
        {layerIds.map((id) => (
          <Layer key={id} id={id} />
        ))}
      </>
    );
  }
});

LayerEntityList.displayName = 'LayerEntityList';
