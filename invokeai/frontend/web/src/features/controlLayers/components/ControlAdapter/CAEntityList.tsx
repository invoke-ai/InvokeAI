import { Text } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { CA } from 'features/controlLayers/components/ControlAdapter/CA';
import { mapId } from 'features/controlLayers/konva/util';
import { selectCanvasV2Slice } from 'features/controlLayers/store/canvasV2Slice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const selectEntityIds = createMemoizedSelector(selectCanvasV2Slice, (canvasV2) => {
  return canvasV2.controlAdapters.entities.map(mapId).reverse();
});

export const CAEntityList = memo(() => {
  const { t } = useTranslation();
  const isTypeSelected = useAppSelector((s) =>
    Boolean(s.canvasV2.selectedEntityIdentifier?.type === 'control_adapter')
  );

  const caIds = useAppSelector(selectEntityIds);

  if (caIds.length === 0) {
    return null;
  }

  if (caIds.length > 0) {
    return (
      <>
        <Text
          color={isTypeSelected ? 'base.100' : 'base.300'}
          fontWeight={isTypeSelected ? 'semibold' : 'normal'}
          userSelect="none"
        >
          {t('controlLayers.controlAdapters_withCount', { count: caIds.length })}
        </Text>
        {caIds.map((id) => (
          <CA key={id} id={id} />
        ))}
      </>
    );
  }
});

CAEntityList.displayName = 'CAEntityList';
