import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasEntityGroupList } from 'features/controlLayers/components/common/CanvasEntityGroupList';
import { ControlLayer } from 'features/controlLayers/components/ControlLayer/ControlLayer';
import { mapId } from 'features/controlLayers/konva/util';
import { selectCanvasSlice, selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { memo } from 'react';

const selectEntityIds = createMemoizedSelector(selectCanvasSlice, (canvas) => {
  return canvas.controlLayers.entities.map(mapId).reverse();
});

export const ControlLayerEntityList = memo(() => {
  const isSelected = useAppSelector((s) => selectSelectedEntityIdentifier(s)?.type === 'control_layer');
  const layerIds = useAppSelector(selectEntityIds);

  if (layerIds.length === 0) {
    return null;
  }

  if (layerIds.length > 0) {
    return (
      <CanvasEntityGroupList type="control_layer" isSelected={isSelected}>
        {layerIds.map((id) => (
          <ControlLayer key={id} id={id} />
        ))}
      </CanvasEntityGroupList>
    );
  }
});

ControlLayerEntityList.displayName = 'ControlLayerEntityList';
