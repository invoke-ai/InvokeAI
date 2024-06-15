import { useDisclosure } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { CanvasEntityContainer } from 'features/controlLayers/components/common/CanvasEntityContainer';
import { LayerHeader } from 'features/controlLayers/components/Layer/LayerHeader';
import { LayerSettings } from 'features/controlLayers/components/Layer/LayerSettings';
import { entitySelected } from 'features/controlLayers/store/controlLayersSlice';
import { memo, useCallback } from 'react';

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

  return (
    <CanvasEntityContainer isSelected={isSelected} onSelect={onSelect}>
      <LayerHeader id={id} onToggleVisibility={onToggle} />
      {isOpen && <LayerSettings id={id} />}
    </CanvasEntityContainer>
  );
});

Layer.displayName = 'Layer';
