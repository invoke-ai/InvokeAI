import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasEntityGroupTitle } from 'features/controlLayers/components/common/CanvasEntityGroupTitle';
import { RG } from 'features/controlLayers/components/RegionalGuidance/RG';
import { mapId } from 'features/controlLayers/konva/util';
import { selectCanvasV2Slice } from 'features/controlLayers/store/canvasV2Slice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const selectEntityIds = createMemoizedSelector(selectCanvasV2Slice, (canvasV2) => {
  return canvasV2.regions.entities.map(mapId).reverse();
});

export const RGEntityList = memo(() => {
  const { t } = useTranslation();
  const isSelected = useAppSelector((s) => Boolean(s.canvasV2.selectedEntityIdentifier?.type === 'regional_guidance'));
  const rgIds = useAppSelector(selectEntityIds);

  if (rgIds.length === 0) {
    return null;
  }

  if (rgIds.length > 0) {
    return (
      <>
        <CanvasEntityGroupTitle
          title={t('controlLayers.regionalGuidance_withCount', { count: rgIds.length })}
          isSelected={isSelected}
        />
        {rgIds.map((id) => (
          <RG key={id} id={id} />
        ))}
      </>
    );
  }
});

RGEntityList.displayName = 'RGEntityList';
