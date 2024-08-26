import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasEntityGroupList } from 'features/controlLayers/components/common/CanvasEntityGroupList';
import { RegionalGuidance } from 'features/controlLayers/components/RegionalGuidance/RegionalGuidance';
import { mapId } from 'features/controlLayers/konva/util';
import { selectCanvasV2Slice } from 'features/controlLayers/store/selectors';
import { memo } from 'react';

const selectEntityIds = createMemoizedSelector(selectCanvasV2Slice, (canvasV2) => {
  return canvasV2.regions.entities.map(mapId).reverse();
});

export const RegionalGuidanceEntityList = memo(() => {
  const isSelected = useAppSelector((s) => Boolean(s.canvasV2.selectedEntityIdentifier?.type === 'regional_guidance'));
  const rgIds = useAppSelector(selectEntityIds);

  if (rgIds.length === 0) {
    return null;
  }

  if (rgIds.length > 0) {
    return (
      <CanvasEntityGroupList type="regional_guidance" isSelected={isSelected}>
        {rgIds.map((id) => (
          <RegionalGuidance key={id} id={id} />
        ))}
      </CanvasEntityGroupList>
    );
  }
});

RegionalGuidanceEntityList.displayName = 'RegionalGuidanceEntityList';
