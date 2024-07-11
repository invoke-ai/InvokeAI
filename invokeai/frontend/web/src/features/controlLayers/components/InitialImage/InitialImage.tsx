import { useDisclosure } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { CanvasEntityContainer } from 'features/controlLayers/components/common/CanvasEntityContainer';
import { InitialImageHeader } from 'features/controlLayers/components/InitialImage/InitialImageHeader';
import { InitialImageSettings } from 'features/controlLayers/components/InitialImage/InitialImageSettings';
import { entitySelected } from 'features/controlLayers/store/canvasV2Slice';
import { memo, useCallback } from 'react';

export const InitialImage = memo(() => {
  const dispatch = useAppDispatch();
  const isSelected = useAppSelector((s) => s.canvasV2.selectedEntityIdentifier?.id === 'initial_image');
  const { isOpen, onToggle } = useDisclosure({ defaultIsOpen: true });
  const onSelect = useCallback(() => {
    dispatch(entitySelected({ id: 'initial_image', type: 'initial_image' }));
  }, [dispatch]);

  return (
    <CanvasEntityContainer isSelected={isSelected} onSelect={onSelect}>
      <InitialImageHeader onToggleVisibility={onToggle} />
      {isOpen && <InitialImageSettings />}
    </CanvasEntityContainer>
  );
});

InitialImage.displayName = 'InitialImage';
