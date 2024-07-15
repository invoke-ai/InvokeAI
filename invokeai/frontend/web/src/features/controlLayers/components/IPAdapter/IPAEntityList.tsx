/* eslint-disable i18next/no-literal-string */
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { IPA } from 'features/controlLayers/components/IPAdapter/IPA';
import { mapId } from 'features/controlLayers/konva/util';
import { selectCanvasV2Slice } from 'features/controlLayers/store/canvasV2Slice';
import { memo } from 'react';

const selectEntityIds = createMemoizedSelector(selectCanvasV2Slice, (canvasV2) => {
  return canvasV2.ipAdapters.entities.map(mapId).reverse();
});

export const IPAEntityList = memo(() => {
  const ipaIds = useAppSelector(selectEntityIds);

  if (ipaIds.length === 0) {
    return null;
  }

  if (ipaIds.length > 0) {
    return (
      <>
        {ipaIds.map((id) => (
          <IPA key={id} id={id} />
        ))}
      </>
    );
  }
});

IPAEntityList.displayName = 'IPAEntityList';
