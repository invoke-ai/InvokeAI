import { useDisclosure } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { rgbColorToString } from 'common/util/colorCodeTransformers';
import { CanvasEntityContainer } from 'features/controlLayers/components/common/CanvasEntityContainer';
import { RGHeader } from 'features/controlLayers/components/RegionalGuidance/RGHeader';
import { RGSettings } from 'features/controlLayers/components/RegionalGuidance/RGSettings';
import { entitySelected } from 'features/controlLayers/store/canvasV2Slice';
import { selectRGOrThrow } from 'features/controlLayers/store/regionsReducers';
import { memo, useCallback } from 'react';

type Props = {
  id: string;
};

export const RG = memo(({ id }: Props) => {
  const dispatch = useAppDispatch();
  const selectedBorderColor = useAppSelector((s) => rgbColorToString(selectRGOrThrow(s.canvasV2, id).fill));
  const isSelected = useAppSelector((s) => s.canvasV2.selectedEntityIdentifier?.id === id);
  const { isOpen, onToggle } = useDisclosure({ defaultIsOpen: true });
  const onSelect = useCallback(() => {
    dispatch(entitySelected({ id, type: 'regional_guidance' }));
  }, [dispatch, id]);
  return (
    <CanvasEntityContainer isSelected={isSelected} onSelect={onSelect} selectedBorderColor={selectedBorderColor}>
      <RGHeader id={id} onToggleVisibility={onToggle} />
      {isOpen && <RGSettings id={id} />}
    </CanvasEntityContainer>
  );
});

RG.displayName = 'RG';
