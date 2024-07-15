/* eslint-disable i18next/no-literal-string */
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { RG } from 'features/controlLayers/components/RegionalGuidance/RG';
import { mapId } from 'features/controlLayers/konva/util';
import { selectCanvasV2Slice } from 'features/controlLayers/store/canvasV2Slice';
import { memo } from 'react';

const selectEntityIds = createMemoizedSelector(selectCanvasV2Slice, (canvasV2) => {
  return canvasV2.regions.entities.map(mapId).reverse();
});

export const RGEntityList = memo(() => {
  const rgIds = useAppSelector(selectEntityIds);

  if (rgIds.length === 0) {
    return null;
  }

  if (rgIds.length > 0) {
    return (
      <>
        {rgIds.map((id) => (
          <RG key={id} id={id} />
        ))}
      </>
    );
  }
});

RGEntityList.displayName = 'RGEntityList';
