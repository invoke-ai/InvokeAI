import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasEntityGroupTitle } from 'features/controlLayers/components/common/CanvasEntityGroupTitle';
import { ControlAdapter } from 'features/controlLayers/components/ControlAdapter/ControlAdapter';
import { mapId } from 'features/controlLayers/konva/util';
import { selectCanvasV2Slice } from 'features/controlLayers/store/canvasV2Slice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const selectEntityIds = createMemoizedSelector(selectCanvasV2Slice, (canvasV2) => {
  return canvasV2.controlAdapters.entities.map(mapId).reverse();
});

export const ControlAdapterList = memo(() => {
  const { t } = useTranslation();
  const isSelected = useAppSelector((s) => Boolean(s.canvasV2.selectedEntityIdentifier?.type === 'control_adapter'));
  const caIds = useAppSelector(selectEntityIds);

  if (caIds.length === 0) {
    return null;
  }

  if (caIds.length > 0) {
    return (
      <>
        <CanvasEntityGroupTitle
          title={t('controlLayers.controlAdapters_withCount', { count: caIds.length })}
          isSelected={isSelected}
        />
        {caIds.map((id) => (
          <ControlAdapter key={id} id={id} />
        ))}
      </>
    );
  }
});

ControlAdapterList.displayName = 'ControlAdapterList';
