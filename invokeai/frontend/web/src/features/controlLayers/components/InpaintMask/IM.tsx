import { useDisclosure } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { rgbColorToString } from 'common/util/colorCodeTransformers';
import { CanvasEntityContainer } from 'features/controlLayers/components/common/CanvasEntityContainer';
import { IMHeader } from 'features/controlLayers/components/InpaintMask/IMHeader';
import { IMSettings } from 'features/controlLayers/components/InpaintMask/IMSettings';
import { entitySelected } from 'features/controlLayers/store/canvasV2Slice';
import { memo, useCallback } from 'react';

export const IM = memo(() => {
  const dispatch = useAppDispatch();
  const selectedBorderColor = useAppSelector((s) => rgbColorToString(s.canvasV2.inpaintMask.fill));
  const isSelected = useAppSelector((s) => s.canvasV2.selectedEntityIdentifier?.id === 'inpaint_mask');
  const { isOpen, onToggle } = useDisclosure({ defaultIsOpen: true });
  const onSelect = useCallback(() => {
    dispatch(entitySelected({ id: 'inpaint_mask', type: 'inpaint_mask' }));
  }, [dispatch]);
  return (
    <CanvasEntityContainer isSelected={isSelected} onSelect={onSelect} selectedBorderColor={selectedBorderColor}>
      <IMHeader onToggleVisibility={onToggle} />
      {isOpen && <IMSettings />}
    </CanvasEntityContainer>
  );
});

IM.displayName = 'IM';
