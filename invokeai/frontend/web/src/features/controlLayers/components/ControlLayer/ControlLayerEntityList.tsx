import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasEntityGroupList } from 'features/controlLayers/components/common/CanvasEntityGroupList';
import { ControlLayer } from 'features/controlLayers/components/ControlLayer/ControlLayer';
import { mapId } from 'features/controlLayers/konva/util';
import { selectCanvasV2Slice } from 'features/controlLayers/store/canvasV2Slice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const selectEntityIds = createMemoizedSelector(selectCanvasV2Slice, (canvasV2) => {
  return canvasV2.controlLayers.entities.map(mapId).reverse();
});

export const ControlLayerEntityList = memo(() => {
  const { t } = useTranslation();
  const isSelected = useAppSelector((s) => Boolean(s.canvasV2.selectedEntityIdentifier?.type === 'control_layer'));
  const layerIds = useAppSelector(selectEntityIds);

  if (layerIds.length === 0) {
    return null;
  }

  if (layerIds.length > 0) {
    return (
      <CanvasEntityGroupList
        type="control_layer"
        title={t('controlLayers.controlLayers_withCount', { count: layerIds.length })}
        isSelected={isSelected}
      >
        {layerIds.map((id) => (
          <ControlLayer key={id} id={id} />
        ))}
      </CanvasEntityGroupList>
    );
  }
});

ControlLayerEntityList.displayName = 'ControlLayerEntityList';
