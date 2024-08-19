import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasEntityGroupList } from 'features/controlLayers/components/common/CanvasEntityGroupList';
import { RegionalGuidance } from 'features/controlLayers/components/RegionalGuidance/RegionalGuidance';
import { mapId } from 'features/controlLayers/konva/util';
import { selectCanvasV2Slice } from 'features/controlLayers/store/canvasV2Slice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const selectEntityIds = createMemoizedSelector(selectCanvasV2Slice, (canvasV2) => {
  return canvasV2.regions.entities.map(mapId).reverse();
});

export const RegionalGuidanceEntityList = memo(() => {
  const { t } = useTranslation();
  const isSelected = useAppSelector((s) => Boolean(s.canvasV2.selectedEntityIdentifier?.type === 'regional_guidance'));
  const rgIds = useAppSelector(selectEntityIds);

  if (rgIds.length === 0) {
    return null;
  }

  if (rgIds.length > 0) {
    return (
      <CanvasEntityGroupList
        type="regional_guidance"
        title={t('controlLayers.regionalGuidance_withCount', { count: rgIds.length })}
        isSelected={isSelected}
      >
        {rgIds.map((id) => (
          <RegionalGuidance key={id} id={id} />
        ))}
      </CanvasEntityGroupList>
    );
  }
});

RegionalGuidanceEntityList.displayName = 'RegionalGuidanceEntityList';
