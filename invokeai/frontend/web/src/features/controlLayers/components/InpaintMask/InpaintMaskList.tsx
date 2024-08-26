import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasEntityGroupList } from 'features/controlLayers/components/common/CanvasEntityGroupList';
import { InpaintMask } from 'features/controlLayers/components/InpaintMask/InpaintMask';
import { mapId } from 'features/controlLayers/konva/util';
import { selectCanvasV2Slice } from 'features/controlLayers/store/selectors';
import { memo } from 'react';

const selectEntityIds = createMemoizedSelector(selectCanvasV2Slice, (canvasV2) => {
  return canvasV2.inpaintMasks.entities.map(mapId).reverse();
});

export const InpaintMaskList = memo(() => {
  const isSelected = useAppSelector((s) => Boolean(s.canvasV2.selectedEntityIdentifier?.type === 'inpaint_mask'));
  const entityIds = useAppSelector(selectEntityIds);

  if (entityIds.length === 0) {
    return null;
  }

  if (entityIds.length > 0) {
    return (
      <CanvasEntityGroupList type="inpaint_mask" isSelected={isSelected}>
        {entityIds.map((id) => (
          <InpaintMask key={id} id={id} />
        ))}
      </CanvasEntityGroupList>
    );
  }
});

InpaintMaskList.displayName = 'InpaintMaskList';
