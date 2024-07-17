/* eslint-disable i18next/no-literal-string */
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasEntityGroupTitle } from 'features/controlLayers/components/common/CanvasEntityGroupTitle';
import { IPA } from 'features/controlLayers/components/IPAdapter/IPA';
import { mapId } from 'features/controlLayers/konva/util';
import { selectCanvasV2Slice } from 'features/controlLayers/store/canvasV2Slice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const selectEntityIds = createMemoizedSelector(selectCanvasV2Slice, (canvasV2) => {
  return canvasV2.ipAdapters.entities.map(mapId).reverse();
});

export const IPAEntityList = memo(() => {
  const { t } = useTranslation();
  const isSelected = useAppSelector((s) => Boolean(s.canvasV2.selectedEntityIdentifier?.type === 'ip_adapter'));
  const ipaIds = useAppSelector(selectEntityIds);

  if (ipaIds.length === 0) {
    return null;
  }

  if (ipaIds.length > 0) {
    return (
      <>
        <CanvasEntityGroupTitle
          title={t('controlLayers.ipAdapters_withCount', { count: ipaIds.length })}
          isSelected={isSelected}
        />
        {ipaIds.map((id) => (
          <IPA key={id} id={id} />
        ))}
      </>
    );
  }
});

IPAEntityList.displayName = 'IPAEntityList';
