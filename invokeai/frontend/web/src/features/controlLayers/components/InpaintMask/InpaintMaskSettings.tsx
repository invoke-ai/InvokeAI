import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasEntitySettingsWrapper } from 'features/controlLayers/components/common/CanvasEntitySettingsWrapper';
import { InpaintMaskAddButtons } from 'features/controlLayers/components/InpaintMask/InpaintMaskAddButtons';
import { InpaintMaskDenoiseLimitSlider } from 'features/controlLayers/components/InpaintMask/InpaintMaskDenoiseLimitSlider';
import { InpaintMaskNoiseSlider } from 'features/controlLayers/components/InpaintMask/InpaintMaskNoiseSlider';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { selectCanvasSlice, selectEntityOrThrow } from 'features/controlLayers/store/selectors';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { memo, useMemo } from 'react';

const buildSelectFlags = (entityIdentifier: CanvasEntityIdentifier<'inpaint_mask'>) =>
  createMemoizedSelector(selectCanvasSlice, (canvas) => {
    const entity = selectEntityOrThrow(canvas, entityIdentifier, 'InpaintMaskSettings');
    return {
      hasNoiseLevel: entity.noiseLevel !== null,
      hasDenoiseLimit: entity.denoiseLimit !== null,
    };
  });

export const InpaintMaskSettings = memo(() => {
  const entityIdentifier = useEntityIdentifierContext('inpaint_mask');
  const selectFlags = useMemo(() => buildSelectFlags(entityIdentifier), [entityIdentifier]);
  const flags = useAppSelector(selectFlags);

  return (
    <CanvasEntitySettingsWrapper>
      {!flags.hasNoiseLevel && !flags.hasDenoiseLimit && <InpaintMaskAddButtons />}
      {flags.hasNoiseLevel && <InpaintMaskNoiseSlider />}
      {flags.hasDenoiseLimit && <InpaintMaskDenoiseLimitSlider />}
    </CanvasEntitySettingsWrapper>
  );
});

InpaintMaskSettings.displayName = 'InpaintMaskSettings';
