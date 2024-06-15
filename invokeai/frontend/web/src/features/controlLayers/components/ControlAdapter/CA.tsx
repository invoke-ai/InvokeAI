import { useDisclosure } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { CanvasEntityContainer } from 'features/controlLayers/components/common/CanvasEntityContainer';
import { CAHeader } from 'features/controlLayers/components/ControlAdapter/CAEntityHeader';
import { CASettings } from 'features/controlLayers/components/ControlAdapter/CASettings';
import { entitySelected } from 'features/controlLayers/store/controlLayersSlice';
import { memo, useCallback } from 'react';

type Props = {
  id: string;
};

export const CA = memo(({ id }: Props) => {
  const dispatch = useAppDispatch();
  const isSelected = useAppSelector((s) => s.canvasV2.selectedEntityIdentifier?.id === id);
  const { isOpen, onToggle } = useDisclosure({ defaultIsOpen: true });
  const onSelect = useCallback(() => {
    dispatch(entitySelected({ id, type: 'control_adapter' }));
  }, [dispatch, id]);

  return (
    <CanvasEntityContainer isSelected={isSelected} onSelect={onSelect}>
      <CAHeader id={id} onToggleVisibility={onToggle} />
      {isOpen && <CASettings id={id} />}
    </CanvasEntityContainer>
  );
});

CA.displayName = 'CA';
