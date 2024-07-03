import { useDisclosure } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIDroppable from 'common/components/IAIDroppable';
import { CanvasEntityContainer } from 'features/controlLayers/components/common/CanvasEntityContainer';
import { LayerHeader } from 'features/controlLayers/components/Layer/LayerHeader';
import { LayerSettings } from 'features/controlLayers/components/Layer/LayerSettings';
import { entitySelected } from 'features/controlLayers/store/canvasV2Slice';
import type { LayerImageDropData } from 'features/dnd/types';
import { memo, useCallback, useMemo } from 'react';

type Props = {
  id: string;
};

export const Layer = memo(({ id }: Props) => {
  const dispatch = useAppDispatch();
  const isSelected = useAppSelector((s) => s.canvasV2.selectedEntityIdentifier?.id === id);
  const { isOpen, onToggle } = useDisclosure({ defaultIsOpen: true });
  const onSelect = useCallback(() => {
    dispatch(entitySelected({ id, type: 'layer' }));
  }, [dispatch, id]);
  const droppableData = useMemo<LayerImageDropData>(
    () => ({ id, actionType: 'ADD_LAYER_IMAGE', context: { id } }),
    [id]
  );

  return (
    <CanvasEntityContainer isSelected={isSelected} onSelect={onSelect}>
      <LayerHeader id={id} onToggleVisibility={onToggle} />
      {isOpen && <LayerSettings id={id} />}
      <IAIDroppable data={droppableData} />
    </CanvasEntityContainer>
  );
});

Layer.displayName = 'Layer';
