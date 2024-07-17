/* eslint-disable i18next/no-literal-string */
import { Text } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
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
  const isTypeSelected = useAppSelector((s) => Boolean(s.canvasV2.selectedEntityIdentifier?.type === 'ip_adapter'));
  const ipaIds = useAppSelector(selectEntityIds);

  if (ipaIds.length === 0) {
    return null;
  }

  if (ipaIds.length > 0) {
    return (
      <>
        <Text
          userSelect="none"
          color={isTypeSelected ? 'base.100' : 'base.300'}
          fontWeight={isTypeSelected ? 'semibold' : 'normal'}
        >
          {t('controlLayers.ipAdapters_withCount', { count: ipaIds.length })}
        </Text>
        {ipaIds.map((id) => (
          <IPA key={id} id={id} />
        ))}
      </>
    );
  }
});

IPAEntityList.displayName = 'IPAEntityList';
