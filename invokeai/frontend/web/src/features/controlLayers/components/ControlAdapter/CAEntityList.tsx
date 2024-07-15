/* eslint-disable i18next/no-literal-string */
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { CA } from 'features/controlLayers/components/ControlAdapter/CA';
import { mapId } from 'features/controlLayers/konva/util';
import { selectCanvasV2Slice } from 'features/controlLayers/store/canvasV2Slice';
import { memo } from 'react';

const selectEntityIds = createMemoizedSelector(selectCanvasV2Slice, (canvasV2) => {
  return canvasV2.controlAdapters.entities.map(mapId).reverse();
});

export const CAEntityList = memo(() => {
  const caIds = useAppSelector(selectEntityIds);

  if (caIds.length === 0) {
    return null;
  }

  if (caIds.length > 0) {
    return (
      <>
        {caIds.map((id) => (
          <CA key={id} id={id} />
        ))}
      </>
    );
  }
});

CAEntityList.displayName = 'CAEntityList';
